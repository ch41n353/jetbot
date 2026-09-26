#!/usr/bin/env python3
"""Search for a named object and drive to it. One loop, one file.

    look -> if not visible, turn and look again
         -> if visible, aim at it, take one short step, look again
         -> stop when the measured range reaches the standoff

Written deliberately small. The measurement that matters is the distance to the
target, and the camera supplies that directly on every look, so the drive does
not need odometry: a step that comes out short or long is simply measured again
from the new position. That is why there is no visual-odometry, map, planner or
tracking layer here, and so no way for any of them to end a run.

What it does keep, because none of it is recoverable by looking again: the power
guard, the expiring motor lease, the control generation, a tilt limit, and a
stall check. Those stop the robot. Nothing else does.

Geometry: x right, z forward, centimetres, measured from the camera lens.
Requires a running local_nav/service.py and OPENAI_API_KEY in the environment.
"""
import argparse
import base64
import fcntl
import json
import math
import os
import socket
import sys
import time
import urllib.request

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOCKET = '/tmp/jetbot-local-nav/control.sock'
SERVICE_LOCK = '/tmp/jetbot-local-nav/lock'
MODEL = 'gpt-5.6-sol'

TURN_POWER = .14           # validated band is 0.14-0.16
DRIVE_POWER = .16
LEASE_SECONDS = .15        # renew well inside the service's ~200 ms expiry
TILT_LIMIT_DEG = 20.       # catches being tipped or lifted, not carpet slope
YAW_RATE_LIMIT = 260.      # Runaway guard only. Measured turn rates vary hugely
                           # with surface: ~85 deg/s on carpet but 143 mean and
                           # 169 peak on a hard floor at the same 0.14 duty, so a
                           # limit tuned to carpet stops normal turns elsewhere.
YAW_RATE_PERSIST = 3       # consecutive over-limit samples before it is a runaway.
                           # One sample is a knock or a gyro glitch, and a single
                           # 315 deg/s reading killed an otherwise healthy leg.
                           # At ~30 Hz this is about 100 ms, so a genuine runaway
                           # is still caught inside ~30 degrees of rotation.
STEP_MAX_CM = 15.
STEP_MIN_CM = 4.
ARRIVAL_CM = 6.            # how close to the standoff counts as arrived
AIM_TOLERANCE_DEG = 10.
COAST_SECONDS = .10        # measured: a cut at 92 deg/s coasted 9.2 deg
COAST_QUIET_DEG_S = 1.5
COAST_TIMEOUT_S = 1.2
TURN_TOLERANCE_DEG = 1.5
TURN_APPROACH_GAIN = 2.2   # deg/s of rate asked for, per degree still to go
TURN_MIN_RATE = 22.        # slow enough to stop accurately, fast enough to move
TURN_MAX_RATE = 110.
TURN_MIN_DUTY = .08        # measured stiction wall: 0.07 gives no motion at
TURN_MAX_DUTY = .17        # all, 0.08 jumps straight to ~50 deg/s
PULSE_DUTY = .15           # comfortably above stiction on carpet and hard floor
PULSE_MIN_S = .04
PULSE_MAX_S = .30
PULSE_AIM = .7             # close most of the gap per burst, never overshoot
PULSE_SETTLE_S = .22       # let it stop before measuring the step it took
FINE_TURN_DEG = 14.        # hand over to pulses with about this much to go
FINE_DRIVE_CM = 5.
DRIVE_TOLERANCE_CM = 1.
DRIVE_APPROACH_GAIN = 1.6  # cm/s of speed asked for, per cm still to go
DRIVE_MIN_RATE = 6.
DRIVE_MAX_RATE = 16.
DRIVE_MIN_DUTY = .09
DRIVE_MAX_DUTY = .18
DRIVE_GAIN_CM_S_PER_DUTY = 60.    # ~8.5 cm/s at 0.14; learned per surface
BREAKAWAY_START_DUTY = .105       # cautious opening guess; raised by evidence
BREAKAWAY_MAX_DUTY = .17
STICTION_PATIENCE_S = .25         # humming this long without moving is stuck
TURN_GAIN_DEG_S_PER_DUTY = 700.   # ~100 deg/s at 0.14; learned per surface
FINE_STEPS = 6             # bounded: a move that will not close says so
SPEED_CM_S = 7.5           # measured on carpet at 0.16 duty: 17.9 cm in 2.4 s.
                           # Only a starting value -- the odometer corrects it.
ROUTE_RANGE_CM = 140.      # past this a pixel of contact error is worth too much
ROUTE_MAX_CM = 200.
ROUTE_MIN_LEG_CM = 4.
INITIAL_TURN_LIMIT_DEG = 20. # planner routes must begin near the current heading
LEAD_POINT_CM = 10.        # a first waypoint nearer than this steers nothing
CORRIDOR_HALF_CM = 6.      # physical half-width of the 12 cm chassis
OBSTACLE_RADIUS_CM = 6.    # half-extent assumed for an obstacle given as ONE contact
# Ceiling on any obstacle's modelled half-extent, measured or assumed. A disc
# is a poor stand-in for a real footprint: it is too big on the diagonal and
# too small across a wide flat face, and the error grows with the object. Two
# base pixels projected near the horizon have produced half-extents of 74 cm,
# which fence off the room. Capping keeps one bad reading from doing that --
# at the cost of under-modelling anything genuinely wider than 10 cm, which is
# the same direction as the 4 cm constant that drove into a 12.3 cm carton on
# 2026-09-24. Interim, pending real polygons.
OBSTACLE_MAX_RADIUS_CM = 5.
                           # pixel. The model returns a single point per obstacle and
                           # this stands in for the object's real size. 4 cm was too
                           # small for anything in this room: measured 2026-09-24, a
                           # milk carton's near edge is 12.3 cm long on the floor, so
                           # a route cleared to 14.6 cm from its contact point still
                           # passed 11.6 cm from its left corner and the chassis edge
                           # went by at 5.6 cm. Stopgap until obstacles carry an
                           # extent (two base pixels) instead of a radius.
                           #
                           # 8 was too large in the other direction. A turn moves an
                           # obstacle to at most its own distance from the new
                           # heading, so anything nearer than the exclusion floor
                           # cannot be escaped by turning at all: with the floor at
                           # 14 cm a Lego baseplate 10.7 cm ahead deadlocked the
                           # robot completely. 6 keeps the floor at 12 cm, still
                           # covering the ~12 cm objects in this room, while leaving
                           # turning a viable escape. Reversing is not an option --
                           # there is no rear sensor.
KEEP_BACK_CM = 6.          # also yields a 10 cm forward exclusion threshold
OBSTACLE_CORNER_X_PX = 96. # outer 15 percent of a 640 px fisheye frame
OBSTACLE_CORNER_Y_PX = 360.# lower quarter: ground range is ill-conditioned
HEADING_TOLERANCE_DEG = 6.
STEER_GAIN = .004          # duty per degree of heading error, bounded below
VISION_INTERVAL_S = .12    # measured: fits land 9/9 at 0.12 s and 5/5 at 0.20 s,
                           # but only 1/3 at 0.30 s, where the carpet has moved
                           # too far between frames for optical flow to latch.
                           # A frame costs 23 ms, so sampling this often is cheap.
TURN_MAX_PER_LEG_DEG = 60. # one turn is chunked to this; a larger bearing is
                           # closed by several, not abandoned half-turned
WAYPOINT_MAX_LEGS = 24     # bound on chunks spent reaching one waypoint: a 2 m
                           # route at 15 cm a leg, with turns, fits inside this
AIM_PASSES = 12            # bound on the aim fixed point; it settles in 4-6
AIM_SETTLED_DEG = .02      # two orders inside TURN_TOLERANCE_DEG: no point
                           # iterating past what the wheels can hold
FACE_ATTEMPTS = 4          # consecutive turn chunks before a waypoint is called
                           # unfaceable; 180 degrees needs three, so this is
                           # headroom, not a working limit
STALL_ATTEMPTS = 2         # drives that buy no ground before giving up
STALL_GAIN_CM = .5         # what counts as having bought ground
GOAL_ADJACENT_CM = 22.     # an obstacle this near the target is part of the
                           # scene around it, not something to route away from:
                           # a bottle among toys cannot be reached otherwise
REPAIR_PASSES = 3          # nudging one waypoint clear can push it into the
                           # next obstacle, so the sweep is repeated
REPAIR_MARGIN_CM = 6.      # a vertex placed exactly on the clearance circle
                           # still lets the two segments either side cut across
                           # it as a chord, so the push goes a little beyond
POINT = {'type': 'object', 'additionalProperties': False,
         'required': ['x', 'y'],
         'properties': {'x': {'type': 'number'}, 'y': {'type': 'number'}}}

# Metric contract (2026-09-24). The model answers in centimetres in the robot
# frame -- the same frame the controller drives in -- so no projection stands
# between what it drew and what is executed. The pixel schema below it is kept
# for the callers that still use it and for replaying stored runs.
VEC = {'type': 'object', 'additionalProperties': False,
       'required': ['right', 'forward'],
       'properties': {'right': {'type': 'number'}, 'forward': {'type': 'number'}}}

METRIC_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['goal', 'all_done', 'visible', 'target_px', 'route_cm',
                 'obstacles', 'turn_degrees', 'note', 'motion'],
    'properties': {
        'motion': {'type': 'string', 'enum': ['follow', 'turn', 'hold']},
        'goal': {'type': 'string'},
        'all_done': {'type': 'boolean'},
        # Whether the target was identified in image 1 this call. Distinct from
        # target_cm, which can be known while the target is out of frame.
        'visible': {'type': 'boolean'},
        # Where the target meets the floor in image 1, when it was found there
        # this call. Null when it was not: the supplied target_cm then stands,
        # and visible=false says so. Same reasoning as obstacles -- a fresh
        # sighting is a pixel, not a number that could have come from memory.
        'target_px': {'anyOf': [POINT, {'type': 'null'}]},
        'route_cm': {'type': 'array', 'items': VEC},
        'turn_degrees': {'anyOf': [{'type': 'number'}, {'type': 'null'}]},
        # radius_cm is the half-extent of the real object, not a constant the
        # controller invents. A 12.3 cm carton modelled as a 4 cm disc is what
        # drove this robot into one on 2026-09-24.
        # Obstacles are seen in image 1, so they are reported where they are
        # seen: the two pixels where the object meets the floor at its left and
        # right edges. The controller projects both and takes the separation as
        # the real width. That keeps perception with the layer that has eyes and
        # measurement with the layer that has the calibration, and it means a
        # reported obstacle is evidence that the picture was looked at -- a
        # metric position can be echoed back from memory, a pixel cannot.
        'obstacles': {'type': 'array', 'items': {
            'type': 'object', 'additionalProperties': False,
            'required': ['label', 'base_left_px', 'base_right_px'],
            'properties': {'label': {'type': 'string'},
                           'base_left_px': POINT, 'base_right_px': POINT}}},
        'note': {'type': 'string'},
    },
}

with open(os.path.join(ROOT, 'local_nav/prompts/metric-20260924.txt')) as _metric_file:
    METRIC_PROMPT = _metric_file.read()

SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['goal', 'all_done', 'visible', 'contact_pixel', 'route_pixels',
                 'obstacles', 'turn_degrees', 'note', 'motion'],
    'properties': {
        'motion': {'type': 'string', 'enum': ['follow', 'turn', 'turn_then_follow', 'hold']},
        # What the instruction asked for, read back in the model's own words, so
        # a misread shows up in the log rather than in the wheels.
        'goal': {'type': 'string'},
        # An instruction can have several steps. Without this the robot arrives
        # at the first one and stops, because arriving is all the loop knew how
        # to recognise.
        'all_done': {'type': 'boolean'},
        'visible': {'type': 'boolean'},
        'contact_pixel': {'anyOf': [POINT, {'type': 'null'}]},
        # The trajectory, as floor contact points. Pixels, not centimetres: the
        # model has no scale, and the calibration supplies every distance.
        'route_pixels': {'type': 'array', 'items': POINT},
        # The one motion the model may ask for. Rotating is the only move that
        # is safe when something is close: it shifts the corridor without
        # carrying the robot into anything. Everything else is still pixels.
        'turn_degrees': {'anyOf': [{'type': 'number'}, {'type': 'null'}]},
        'obstacles': {'type': 'array', 'items': {
            'type': 'object', 'additionalProperties': False,
            'required': ['label', 'contact_pixel'],
            'properties': {'label': {'type': 'string'}, 'contact_pixel': POINT}}},
        'note': {'type': 'string'},
    },
}

with open(os.path.join(ROOT, 'local_nav/prompts/trajectory-20260920.txt')) as _prompt_file:
    PROMPT = _prompt_file.read()

if os.environ.get('JETBOT_CLEARANCE_PROFILE') == 'extreme_reference':
    with open(os.path.join(ROOT, 'local_nav/prompts/clearance-candidate-20260922.txt')) as _prompt_file:
        PROMPT += '\n\n' + _prompt_file.read()


class Stop(Exception):
    """A condition the robot must not drive through."""


class PlannerHold(Stop):
    """Explicit no-motion result; raised before old-route fallback can run."""
    def __init__(self, answer):
        self.answer = answer
        super().__init__('planner hold: ' + str(answer.get('note', 'no supported route')))


def wait_for_service_release(timeout=15., socket_path=SOCKET,
                             lock_path=SERVICE_LOCK):
    """Wait until shutdown has released both its API and hardware lock.

    The shutdown reply is intentionally sent before camera/IMU worker cleanup.
    Treating that reply, or even disappearance of the socket, as process exit
    races the next service against the still-held flock.  Acquiring the flock
    is the authoritative lifecycle boundary.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        handle = None
        try:
            handle = open(lock_path, 'a')
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not os.path.exists(socket_path):
                fcntl.flock(handle, fcntl.LOCK_UN)
                handle.close()
                return
            fcntl.flock(handle, fcntl.LOCK_UN)
        except (IOError, OSError):
            pass
        finally:
            if handle is not None and not handle.closed:
                handle.close()
        time.sleep(.05)
    raise Stop('service shutdown timed out before hardware lock release')


def call(action, timeout=2., **fields):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as link:
        link.settimeout(timeout)
        link.connect(SOCKET)
        link.sendall(json.dumps(dict(action=action, **fields)).encode() + b'\n')
        data = b''
        while b'\n' not in data:
            chunk = link.recv(65536)
            if not chunk:
                raise Stop('service disconnected')
            data += chunk
    reply = json.loads(data)
    if 'error' in reply:
        raise Stop(reply['error'])
    if action == 'shutdown' and reply.get('accepted'):
        wait_for_service_release()
    return reply


class Robot:
    """Hardware access plus the few checks that must end a run."""

    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        status = call('status')
        self.token = {k: status[k] for k in ('session_id', 'control_epoch')}
        self.up = self._gravity(status['imu'])
        with open(os.path.join(ROOT, 'calibration/floor_geometry.json')) as handle:
            profile = json.load(handle)
        with open(profile['intrinsics_path']) as handle:
            intrinsics = json.load(handle)
        self.K = np.asarray(intrinsics['K'], dtype=float)
        self.D = np.asarray(intrinsics['D'], dtype=float)
        self.height = float(profile['camera_height_cm'])
        self.pitch = math.radians(profile['pitch_degrees'])
        self.speed = SPEED_CM_S
        self.coast_s = COAST_SECONDS   # measured per surface as it turns
        self.turn_gain = TURN_GAIN_DEG_S_PER_DUTY
        self.duty = TURN_MIN_DUTY
        # Measured, not guessed: a burst averages about 60 deg/s including the
        # time it spends breaking loose, well under the ~120 deg/s of a turn
        # already in motion. Starting from the higher figure made every burst
        # half the length it needed, so each one closed half the gap and the
        # robot chattered through several where one would do.
        self.pulse_rate = 60.          # deg/s a burst achieves; learned
        self.pulse_speed = 5.          # cm/s a burst achieves; learned
        self.drive_gain = DRIVE_GAIN_CM_S_PER_DUTY
        # Lowest duty that actually breaks the wheels loose here. Below it the
        # motor is energised and humming but the robot does not move, which
        # wastes the pack and heats a stalled motor. It differs by surface --
        # 0.08 on a hard floor, near 0.14 on carpet -- so it is learned, never
        # assumed, and only ever raised by evidence of not moving.
        self.breakaway = BREAKAWAY_START_DUTY
        self._silent_since = None

    def floor_duty(self, duty):
        """Never command into the stiction band: move, or command nothing."""
        return max(self.breakaway, duty)

    def note_motion(self, duty, moving):
        """Raise the breakaway estimate when a real command produced nothing."""
        now = time.monotonic()
        if moving or duty <= .001:
            self._silent_since = None
            return
        if self._silent_since is None:
            self._silent_since = now
        elif now - self._silent_since > STICTION_PATIENCE_S:
            self.breakaway = min(BREAKAWAY_MAX_DUTY, max(self.breakaway, duty) + .01)
            self._silent_since = None

    @staticmethod
    def _gravity(imu):
        vector = np.asarray(imu['acceleration'], dtype=float)
        norm = np.linalg.norm(vector)
        if not 3 < norm < 20:
            raise Stop('implausible accelerometer reading %.1f m/s^2' % norm)
        return vector / norm

    def check(self, status=None):
        """Power, health and control generation. Raises Stop, never returns bad."""
        status = status or call('status')
        for key, value in self.token.items():
            if status.get(key) != value:
                raise Stop('control generation changed (cancelled elsewhere)')
        if not status.get('healthy'):
            raise Stop('sensors unhealthy')
        power = status.get('power') or {}
        if power.get('stop_latched') or power.get('motion_allowed') is False:
            raise Stop('power guard: %.2f V pack' % power.get('pack_voltage_v', 0.))
        if power.get('stale') is True:
            raise Stop('power telemetry stale')
        return status

    @staticmethod
    def heading(imu):
        """The IMU's own fused heading, degrees.

        Preferred over integrating the gyro here: the chip fuses at 100 Hz while
        this loop samples near 30 Hz, and multiplying a sampled rate by dt
        over-counts during spin-up. Measured against it, integration read 90
        degrees for a turn the chip put at 85. Magnetometer calibration is 0 on
        this unit, so treat it as a relative heading, never as north.
        """
        return float(imu['euler'][0])

    @staticmethod
    def unwrap(delta):
        return (delta + 180.) % 360. - 180.

    def yaw_rate(self, imu):
        """Rotation about gravity, degrees per second, positive to the right."""
        gyro = np.asarray(imu['gyro'], dtype=float)
        if imu.get('gyro_units') == 'rad/s':
            gyro = np.degrees(gyro)
        return -float(gyro.dot(self.up))

    def tilt(self, imu):
        """Tilt from the starting attitude, or None while accelerating.

        An accelerometer measures gravity plus whatever the robot is doing, so
        during a turn the vector swings and a naive comparison reads tens of
        degrees of tilt that do not exist. Only a near-1g reading is gravity.
        """
        vector = np.asarray(imu['acceleration'], dtype=float)
        if abs(np.linalg.norm(vector) - 9.80665) > .6:
            return None
        measured = vector / np.linalg.norm(vector)
        angle = math.degrees(math.acos(max(-1., min(1., measured.dot(self.up)))))
        # Follow the floor slowly while it is quiet, so a sloping carpet does
        # not accumulate into an apparent tilt over a long run. Being picked up
        # or tipped happens far faster than this filter can follow.
        self.up = self.up + .05 * (measured - self.up)
        self.up /= np.linalg.norm(self.up)
        return angle

    def ground(self, x, y):
        """Floor position of an image pixel, as (right_cm, forward_cm)."""
        xy = cv2.fisheye.undistortPoints(np.array([[[float(x), float(y)]]]),
                                         self.K, self.D).reshape(2)
        ray = np.array([xy[0], xy[1], 1.])
        down = np.array([0., math.cos(self.pitch), math.sin(self.pitch)])
        forward = np.array([0., 0., 1.]) - down * down[2]
        forward /= np.linalg.norm(forward)
        right = np.cross(down, forward)
        denominator = ray.dot(down)
        if denominator < .05:
            raise Stop('pixel is at or above the horizon, not on the floor')
        scale = self.height / denominator
        return float(ray.dot(right) * scale), float(ray.dot(forward) * scale)

    def pixel(self, right, forward):
        """Image pixel a floor position appears at: the inverse of ground().

        Needed to show the model what it said last time. A route is planned in
        the frame of one photograph and the robot then drives part of it, so
        the only way to put that plan in front of the model again is to move it
        into the new pose and re-project it through the same lens.

        Returns None for floor behind the camera, which has no pixel -- the
        fisheye model will still return coordinates there, and they are wrong.
        """
        down = np.array([0., math.cos(self.pitch), math.sin(self.pitch)])
        ahead = np.array([0., 0., 1.]) - down * down[2]
        ahead /= np.linalg.norm(ahead)
        side = np.cross(down, ahead)
        ray = side * right + ahead * forward + down * self.height
        length = np.linalg.norm(ray)
        if length < 1e-6 or ray[2] <= 0.:
            return None
        spot, _ = cv2.fisheye.projectPoints((ray / length).reshape(1, 1, 3),
                                            np.zeros(3), np.zeros(3), self.K, self.D)
        x, y = spot.reshape(2)
        return (float(x), float(y))

    def frame(self):
        observation = call('observation', timeout=5.)
        self.check(observation if 'healthy' in observation else None)
        image = cv2.imdecode(np.frombuffer(
            base64.b64decode(observation['jpeg_base64']), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise Stop('camera frame could not be decoded')
        return image, observation

    def hold(self, left, right):
        if self.dry_run:
            return
        call('motors_hold', left=float(left), right=float(right), **self.token)

    def halt(self):
        try:
            call('stop')
        except Exception:
            pass
        self.token['control_epoch'] = call('status')['control_epoch']

    def resync(self):
        """Adopt the service's current control generation without stopping.

        The service bumps the generation on *any* request it rejects, and until
        now `halt` was the only thing that ever re-read it. So a single
        transient rejection left this object holding a dead lease, and every
        later command failed with 'control generation changed (cancelled
        elsewhere)' -- forever, or until an operator happened to press stop.
        Measured: twenty consecutive trajectories refused in 0.2 s each, 0 cm
        driven, after one stalled run hours earlier.

        Only call this when the robot is known idle and this process is the one
        that will drive next. Re-syncing mid-run would let a cancelled
        controller resume, which is the thing the generation exists to prevent.
        """
        self.token['control_epoch'] = call('status')['control_epoch']
        return self.token['control_epoch']

    def run_steered(self, command, done, limit):
        """Like _run, but `command()` supplies (left, right) afresh each cycle.

        That is what lets a leg steer: the trim is recomputed from the heading
        error while the wheels are turning, rather than fixed at the start.
        """
        return self._run(command, None, done, limit)

    def _run(self, left, right, done, limit, settle=True):
        """Hold a differential command until `done(state)`, then stop.

        `done` sees a dict of measured state each cycle. Every exit path stops
        the motors: the lease would expire anyway, but waiting for that would
        leave the robot coasting.
        """
        # Tilt is only measurable standing still: once the wheels push, the
        # accelerometer reads gravity plus the robot's own acceleration, and
        # the direction swings without the magnitude changing much. So check it
        # here, before committing to the motion, and not inside the loop.
        tilt = self.tilt(call('status')['imu'])
        if tilt is not None and tilt > TILT_LIMIT_DEG:
            raise Stop('tilt %.1f degrees before moving' % tilt)
        if self.dry_run:
            return      # no command was sent, so there is no motion to measure
        started = last = time.monotonic()
        over = 0                # consecutive over-limit yaw samples
        try:
            while True:
                now = time.monotonic()
                if now - started > limit:
                    raise Stop('motion exceeded its %.1f s bound' % limit)
                status = self.check()
                imu = status['imu']
                rate = self.yaw_rate(imu)
                if abs(rate) > YAW_RATE_LIMIT:
                    over += 1
                    if over >= YAW_RATE_PERSIST:
                        raise Stop('yaw rate %.0f deg/s sustained over %d samples'
                                   % (rate, over))
                else:
                    over = 0
                step = now - last
                last = now
                if done(dict(elapsed=now - started, dt=step, rate=rate, imu=imu)):
                    return
                self.hold(*(left() if callable(left) else (left, right)))
                time.sleep(LEASE_SECONDS / 3)
        finally:
            self.hold(0., 0.)
            if settle:
                time.sleep(.25)      # let it settle before the next measurement

    def _sweep(self, direction, target, budget, continuous=False):
        """Turn `target` degrees, regulating rate instead of predicting coast.

        The rate asked for falls as the target approaches, so the robot arrives
        slowly and whatever it coasts is small. That is what makes the result
        stop depending on the surface: a slick floor simply gets less duty for
        the same requested rate, and the duty-to-rate gain is learned as it
        goes. The earlier scheme ran flat out and cut at a predicted point,
        which is why moving the robot onto a harder floor wrecked it.

        Returns degrees actually swept, coast included.
        """
        previous = [self.heading(self.check()['imu'])]
        swept = [0.]
        started = last = time.monotonic()
        moved = False
        over = 0                # consecutive over-limit yaw samples
        try:
            while True:
                now = time.monotonic()
                if now - started > budget:
                    raise Stop('turn exceeded its %.1f s bound' % budget)
                status = self.check()
                imu = status['imu']
                heading = self.heading(imu)
                swept[0] += direction * self.unwrap(heading - previous[0])
                previous[0] = heading
                rate = abs(self.yaw_rate(imu))
                if rate > YAW_RATE_LIMIT:
                    over += 1
                    if over >= YAW_RATE_PERSIST:
                        raise Stop('yaw rate %.0f deg/s sustained over %d samples'
                                   % (rate, over))
                else:
                    over = 0
                if swept[0] > 2.:
                    moved = True
                if now - started > 1.2 and not moved:
                    raise Stop('wheels are not turning the robot (stalled)')
                remaining = target - swept[0]
                # Hand over to bursts once the rest is finer than this loop can
                # steer: the robot cannot turn slower than stiction allows.
                cutoff=max(TURN_TOLERANCE_DEG,rate*self.coast_s) if continuous else FINE_TURN_DEG
                if remaining <= cutoff:
                    break
                # Learn how much rate this surface gives for the duty applied.
                if rate > 15. and self.duty > .01:
                    self.turn_gain = max(100., min(2000.,
                        .8 * self.turn_gain + .2 * rate / self.duty))
                wanted = max(TURN_MIN_RATE, min(TURN_MAX_RATE,
                                                remaining * TURN_APPROACH_GAIN))
                self.duty = self.floor_duty(min(TURN_MAX_DUTY,
                                                wanted / self.turn_gain))
                self.note_motion(self.duty, rate > 4.)
                self.hold(self.duty * direction, -self.duty * direction)
                last = now
                time.sleep(.015)
        finally:
            self.hold(0., 0.)
        quiet = 0
        last = time.monotonic()
        while time.monotonic() - last < COAST_TIMEOUT_S:
            imu = self.check()['imu']
            heading = self.heading(imu)
            swept[0] += direction * self.unwrap(heading - previous[0])
            previous[0] = heading
            quiet = quiet + 1 if abs(self.yaw_rate(imu)) < COAST_QUIET_DEG_S else 0
            if quiet >= 3:
                break
            time.sleep(.02)
        return swept[0]

    def _pulse_once(self, left, right, seconds, measure):
        """One short burst, then stop and measure what it produced.

        Below about 0.08 duty the wheels never break stiction, so the robot has
        no slow continuous speed to decelerate into. Bursts give back the fine
        control that costs: each one is small, the result is measured while
        stopped, and the next is sized from what the last actually did.
        """
        started = time.monotonic()
        try:
            while time.monotonic() - started < seconds:
                self.check()
                self.hold(left, right)
                time.sleep(.012)
        finally:
            self.hold(0., 0.)
        time.sleep(PULSE_SETTLE_S)
        return measure()

    def _creep_drive(self, remaining, measure):
        """Close the last few centimetres with bursts, learning cm per second.

        Translation hits the same stiction wall as rotation: there is no slow
        continuous crawl, so the fine approach is made of small measured steps.
        """
        for _ in range(FINE_STEPS):
            if remaining <= DRIVE_TOLERANCE_CM:
                break
            seconds = max(PULSE_MIN_S, min(PULSE_MAX_S,
                          PULSE_AIM * remaining / max(3., self.pulse_speed)))
            before = measure()
            duty = max(PULSE_DUTY, self.breakaway)
            after = self._pulse_once(duty, duty, seconds, measure)
            step = after - before
            if step > .2:
                self.pulse_speed = max(2., min(40.,
                    .6 * self.pulse_speed + .4 * step / seconds))
            remaining -= step
        return remaining

    def _creep_turn(self, direction, remaining, measure):
        """Close the last few degrees with bursts, learning degrees per second."""
        for _ in range(FINE_STEPS):
            if remaining <= TURN_TOLERANCE_DEG:
                break
            # Aim a burst at most of what is left, not all of it. What a burst
            # achieves varies with how completely the robot had stopped, so
            # asking for the whole gap overshoots; undershooting converges in
            # one more burst instead, which is quieter than correcting back.
            seconds = max(PULSE_MIN_S, min(PULSE_MAX_S,
                          PULSE_AIM * remaining / max(20., self.pulse_rate)))
            before = measure()
            duty = max(PULSE_DUTY, self.breakaway)
            after = self._pulse_once(duty * direction, -duty * direction,
                                     seconds, measure)
            step = after - before
            if step > .5:
                self.pulse_rate = max(20., min(400.,
                    .6 * self.pulse_rate + .4 * step / seconds))
            remaining -= step
        return remaining

    def turn_continuous(self, degrees):
        """One uninterrupted regulated sweep, then stop and measure coast.

        No fine burst train or hidden reversal; executor checks final error.
        """
        if abs(degrees)<.5:return 0.
        if self.dry_run:return degrees
        direction=1. if degrees>0 else -1.
        return direction*self._sweep(direction,abs(degrees),3.+abs(degrees)/12.,continuous=True)

    def turn(self, degrees):
        """Rotate by `degrees` (positive right), closed loop on the IMU.

        One regulated sweep is usually enough; a bounded correction follows if
        the robot still finished outside tolerance. A turn that will not close
        reports the angle it actually reached rather than pretending.
        """
        if abs(degrees) < .5:
            return 0.
        if self.dry_run:
            return degrees
        direction = 1. if degrees > 0 else -1.
        target = abs(degrees)
        swept = [0.]
        if target > FINE_TURN_DEG:
            swept[0] += self._sweep(direction, target, 3. + target / 12.)
        base = self.heading(self.check()['imu'])
        previous = [base]

        def measure():
            heading = self.heading(self.check()['imu'])
            swept[0] += direction * self.unwrap(heading - previous[0])
            previous[0] = heading
            return swept[0]
        self._creep_turn(direction, target - measure(), measure)
        return direction * swept[0]

    def step(self, centimetres):
        """Drive forward roughly `centimetres`, open loop, bounded.

        Distance is not measured here on purpose: the next look measures the
        range to the target, which is what actually matters, and corrects any
        error in this estimate.
        """
        seconds = max(.1, min(2.5, centimetres / max(4., self.speed)))
        heading = [0.]

        def done(state):
            heading[0] += state['rate'] * state['dt']
            if abs(heading[0]) > 25.:
                raise Stop('drifted %.0f degrees off heading while driving' % heading[0])
            return state['elapsed'] >= seconds
        self._run(DRIVE_POWER, DRIVE_POWER, done, limit=seconds + 1.5)
        return seconds


def where_words(bearing):
    """A bearing said in words, because that is what survives leaving frame."""
    side = 'left' if bearing < 0 else 'right'
    size = abs(bearing)
    if size < 12.:
        return 'straight ahead'
    if size < 50.:
        return 'ahead and a little to your ' + side
    if size < 100.:
        return 'off to your ' + side
    if size < 150.:
        return 'behind you, to the ' + side
    return 'directly behind you'


def recall(lens, memory, pose, width=640, height=480):
    """The part of the last plan not yet driven, redrawn in the frame taken now.

    Only the plan, and where the goal was. Obstacles are deliberately not sent:
    the model is looking at a photograph of the floor and can see them again,
    and a re-detection from here is better than a remembered position carried
    through motion it did not measure. The route is the one thing it cannot
    recover, because it is the model's own earlier intent -- ask for an approach
    from an object's left and that choice lives in the shape of the route and
    nowhere else.

    The executor keeps its own obstacle memory regardless. That is what stops
    the robot driving into something it has already passed, and it is a
    different question from what is worth putting in a prompt.

    `memory` is the previous answer in floor coordinates of the frame it was
    planned in -- {'route': [...], 'goal': (x, z), 'done': [...]} -- and `pose`
    is where the robot ended up in that same frame. Waypoints already driven
    past fall behind the camera and have no pixel, so what survives is what is
    still to come.

    Returns None when nothing survives, so the caller sends no prior at all
    rather than an empty one.
    """
    # Where the goal was, carried across any amount of turning. A pixel stops
    # existing the moment the target leaves the frame, and a bearing does not.
    target = None
    if memory.get('goal') is not None:
        spot = rebase([memory['goal']], pose)[0]
        bearing = math.degrees(math.atan2(spot[0], spot[1]))
        place = lens.pixel(spot[0], spot[1]) if spot[1] > 0 else None
        inside = bool(place and 0. <= place[0] <= width and 0. <= place[1] <= height)
        target = dict(
            bearing_degrees=round(bearing, 1),
            where=where_words(bearing),
            contact_pixel=(dict(x=round(place[0], 1), y=round(place[1], 1))
                           if inside else None),
            still_in_view=inside)

    route = []
    for spot in rebase(memory.get('route') or [], pose):
        if spot[1] <= 0.:
            continue                    # behind the robot: already driven past
        place = lens.pixel(spot[0], spot[1])
        if place and 0. <= place[0] <= width and 0. <= place[1] <= height:
            route.append(dict(x=round(place[0], 1), y=round(place[1], 1)))

    if not route and target is None and not memory.get('done'):
        return None

    travelled = math.hypot(pose[0], pose[1])
    return dict(
        route_pixels=route, target_was=target,
        already_reached=list(memory.get('done') or []),
        since_then='drove %.0f cm and turned %+.0f degrees'
                   % (travelled, pose[2]))


def annotate(image, prior):
    """Draw the unexecuted part of the last route into the picture being sent.

    The model is far better at reading marks in a photograph than at holding a
    list of coordinates in its head and projecting them onto what it sees, and
    the leftover route is the one thing it cannot recover any other way: if it
    chose to come at something from the left, that choice lives in the shape of
    those points and nowhere else. When the target has just gone out of frame
    this is the only thing left to steer by.

    Returns a new image; the caller's frame is never drawn on, because that
    same frame is what the calibration and the odometry read.
    """
    if not prior:
        return image
    shot = image.copy()
    previous = None
    for index, point in enumerate(prior.get('route_pixels') or []):
        at = (int(point['x']), int(point['y']))
        if previous:
            cv2.line(shot, previous, at, (255, 210, 90), 2)
        # Hollow crosses rather than filled discs: this ink lands on carpet the
        # model is also reading for obstacles, so it has to mark a spot without
        # hiding what is under it.
        cv2.drawMarker(shot, at, (255, 210, 90), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(shot, str(index + 1), (at[0] + 7, at[1] - 7),
                    cv2.FONT_HERSHEY_PLAIN, 1., (255, 210, 90), 1)
        previous = at
    target = prior.get('target_was')
    if target:
        spot = target.get('contact_pixel')
        if spot:
            cv2.drawMarker(shot, (int(spot['x']), int(spot['y'])),
                           (60, 220, 220), cv2.MARKER_STAR, 20, 2)
        else:
            # Out of frame. Say which way and how far round, because that is
            # exactly what the model is being told in words as well.
            edge = 6 if target['bearing_degrees'] < 0 else shot.shape[1] - 6
            cv2.arrowedLine(shot, (int(shot.shape[1] / 2), 40),
                            (edge, 40), (60, 220, 220), 2, tipLength=.3)
            cv2.putText(shot, 'target %+.0f deg: %s'
                        % (target['bearing_degrees'], target['where']),
                        (8, 62), cv2.FONT_HERSHEY_PLAIN, 1., (60, 220, 220), 1)
    return shot


def recognize(image, instruction, prior=None, timeout=20., paired=None, metric=False):
    """Read a plain-words instruction against one photograph.

    `instruction` is plain words rather than a noun phrase -- "reach the can of
    nuts without hitting anything", "go to the pink bin by the desk" -- and the
    model reads the goal out of it and reports what it understood, so a misread
    shows in the log instead of in the wheels.

    Obstacle clearance is not negotiable and is enforced here, not by the model.
    Otherwise perception only: it answers a question about pixels, and every
    centimetre comes from the local calibration.

    `prior` is what recall() built from the previous answer, so the model sees
    its own last route and obstacle list drawn in the photograph in front of it
    rather than starting from nothing each look.
    """
    ok, encoded = cv2.imencode('.jpg', annotate(image, prior))
    if not ok:
        raise Stop('could not encode the camera frame')
    # Effort is settable so it can be measured rather than argued about.
    # 'none' is the default because a FLOW look is issued every second and the
    # wheels wait on it; whether that is the right trade for a one-shot ASK is
    # a different question, and the answer should come from a comparison.
    effort = os.environ.get('JETBOT_REASONING_EFFORT', 'none')
    body = dict(model=MODEL, reasoning=dict(effort=effort), store=False,
                # Reasoning tokens come out of this budget, so a cap tuned
                # for no reasoning truncates the answer the moment there is
                # any. The route is a few hundred tokens either way.
                max_output_tokens=600 if effort == 'none' else 4000,
                instructions=PROMPT,
                input=[dict(role='user', content=[
                    dict(type='input_text', text=json.dumps(
                        dict(instruction=instruction, last_time=prior) if prior
                        else dict(instruction=instruction))),
                    dict(type='input_image', detail='high',
                         image_url='data:image/jpeg;base64,'
                                   + base64.b64encode(encoded).decode('ascii'))])],
                text=dict(format=dict(type='json_schema', name='sighting',
                                      strict=True, schema=SCHEMA)))
    if paired is not None and metric:
        # Metric contract: one prompt written for the inputs actually sent, a
        # schema in centimetres, and nothing drawn on either picture. Replaces
        # the whole stack of appended override blocks rather than adding to it.
        body['instructions'] = METRIC_PROMPT
        body['text'] = dict(format=dict(type='json_schema', name='plan',
                                        strict=True, schema=METRIC_SCHEMA))
        content = body['input'][0]['content']
        content[0]['text'] = json.dumps(dict(instruction=instruction, **paired[2]))
        content[1]['image_url'] = ('data:image/jpeg;base64,'
            + base64.b64encode(cv2.imencode('.jpg', paired[0])[1]).decode('ascii'))
        content.append(dict(type='input_image', detail='high',
            image_url='data:image/jpeg;base64,'
                      + base64.b64encode(cv2.imencode('.jpg', paired[1])[1]).decode('ascii')))
    elif paired is not None:
        from paired_planning import INSTRUCTIONS
        body['instructions'] += '\n'+INSTRUCTIONS
        content=body['input'][0]['content']
        clean_prior=dict(prior or {})
        # Pixel hints must match the metric previous route shown in both inputs.
        from evaluate_gpt_routes import Lens
        lens=Lens();pixels=[]
        for x,z in paired[2].get('previous_route_cm',[]):
            q=Robot.pixel(lens,x,z) if z>0 else None
            if q and 0<=q[0]<640 and 0<=q[1]<480:pixels.append(dict(x=round(q[0],1),y=round(q[1],1)))
        clean_prior['route_pixels']=pixels
        public_map={k:v for k,v in paired[2].items() if not k.startswith('_')}
        if paired[2].get('target_requires_fresh_rgb'):clean_prior.pop('target_was',None)
        content[0]['text']=json.dumps(dict(instruction=instruction,last_time=clean_prior,local_map=public_map))
        content[1]['image_url']='data:image/jpeg;base64,'+base64.b64encode(cv2.imencode('.jpg',paired[0])[1]).decode('ascii')
        content.append(dict(type='input_image',detail='high',image_url='data:image/jpeg;base64,'+base64.b64encode(cv2.imencode('.jpg',paired[1])[1]).decode('ascii')))
    if not (paired is not None and metric):
        # Apply the tested turn-specific clarification after paired-view rules.
        with open(os.path.join(ROOT, 'local_nav/prompts/turn-clearance-candidate-20260923.txt')) as turn_prompt:
            body['instructions'] += '\n\n' + turn_prompt.read()
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        raise Stop('OPENAI_API_KEY is not set')
    request = urllib.request.Request(
        'https://api.openai.com/v1/responses', data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
    # Thinking takes wall-clock time: measured on this robot, 4.5 s at 'none'
    # and 11.4 s at 'low'. A timeout tuned for no reasoning simply cancels the
    # call, which reads as a model failure and is not one.
    if effort != 'none':
        timeout = max(timeout, 90.)
    from gpt_audit import Capture
    audit = Capture(body)
    if audit.path:
        if paired is not None:(audit.path / 'local-map.json').write_text(json.dumps(paired[2]))
        ok, clean = cv2.imencode(".jpg", image)
        if ok:
            (audit.path / "clean.jpg").write_bytes(clean.tobytes())
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            response_bytes = reply.read()
        audit.response(response_bytes)
        raw = json.loads(response_bytes)
    except Exception as exc:
        audit.event('error', error=str(exc))
        raise
    if raw.get('status') != 'completed':
        raise Stop('recognizer returned %s' % raw.get('status'))
    text = ''.join(part.get('text', '')
                   for item in raw.get('output', []) if item.get('type') == 'message'
                   for part in item.get('content', []) if part.get('type') == 'output_text')
    answer = json.loads(text)
    if audit.path:
        answer['_audit_path'] = str(audit.path)
    if answer.get('motion') == 'hold':
        raise PlannerHold(answer)
    if answer.get('motion') == 'turn':
        answer['route_pixels'] = []
    elif answer.get('motion') == 'follow':
        answer['turn_degrees'] = None
    if answer.get('motion') == 'turn_then_follow':
        turn = answer.get('turn_degrees')
        if (not isinstance(turn, (int, float)) or isinstance(turn, bool)
                or not math.isfinite(turn) or not 0 < abs(turn) <= 30
                or not answer.get('route_pixels')):
            raise Stop('turn_then_follow needs a nonempty route and a turn within 30 degrees')
    pixel = answer.get('contact_pixel')
    if answer.get('visible') and isinstance(pixel, dict):
        x, y = float(pixel['x']), float(pixel['y'])
        if not (0 <= x <= 640 and 0 <= y <= 480):
            answer['visible'] = False   # outside the frame is not a sighting
    return answer


def fetch(target, standoff=20., scan_step=15., max_looks=40, dry_run=False,
          log=None):
    """Find `target`, then drive the trajectory the recognizer proposes.

    One look produces a whole obstacle-avoiding path; the local controller
    executes it as a maneuver with IMU heading hold and visual distance. After
    each maneuver the robot looks again, because the only honest check on a
    trajectory is where it actually left the robot.
    """
    robot = Robot(dry_run=dry_run)
    odometer = Odometer(robot)
    events = []
    started = time.monotonic()

    def record(event, **fields):
        row = dict(event=event, t=round(time.monotonic() - started, 2), **fields)
        events.append(row)
        print(json.dumps(row), flush=True)
        if log:
            with open(log, 'w') as handle:
                json.dump(dict(target=target, standoff_cm=standoff,
                               events=events), handle, indent=2)
        return row

    stuck = 0
    record('start', target=target, standoff_cm=standoff)
    try:
        for attempt in range(max_looks):
            robot.check()
            image, _ = robot.frame()
            answer = recognize(image, target)
            record('look', n=attempt, visible=bool(answer.get('visible')),
                   route_points=len(answer.get('route_pixels') or []),
                   obstacles=[o.get('label') for o in answer.get('obstacles') or []][:4],
                   note=str(answer.get('note', ''))[:80])
            if not answer.get('visible'):
                return record('not_visible',
                              note=str(answer.get('note', ''))[:80])

            # Measure where it is before planning how to get there: once the
            # robot is inside the standoff there is no route left to build, and
            # asking for one reported an empty route instead of an arrival.
            contact = answer.get('contact_pixel')
            if not isinstance(contact, dict):
                raise Stop('no target contact point to drive to')
            goal = robot.ground(contact['x'], contact['y'])
            distance = math.hypot(*goal)
            if distance <= standoff + ARRIVAL_CM:
                return record('reached', range_cm=round(distance, 1),
                              standoff_cm=standoff)
            route, goal = project_route(robot, answer, standoff)
            obstacles = project_obstacles(robot, answer)
            record('trajectory', range_cm=round(distance, 1),
                   legs=len(route),
                   route_cm=[[round(v, 1) for v in p] for p in route],
                   obstacles_cm=[[o[0], round(o[1][0], 1), round(o[1][1], 1)]
                                 for o in obstacles])
            # One chunk per waypoint, then look again: the next measurement of
            # the range is worth more than more open-loop travel on this one.
            pose = follow(robot, odometer, route, record, obstacles,
                          complete=False, initial_turn_limit_deg=INITIAL_TURN_LIMIT_DEG)
            record('maneuver_done', pose=[round(v, 1) for v in pose])
            if math.hypot(pose[0], pose[1]) < 1.:
                stuck += 1
                if stuck >= 2:
                    return record('blocked', range_cm=round(distance, 1),
                                  note='cannot find a way past what is in front')
            else:
                stuck = 0
        return record('gave_up', attempts=max_looks)
    except Stop as exc:
        robot.halt()
        return record('stopped', reason=str(exc))
    except KeyboardInterrupt:
        robot.halt()
        return record('stopped', reason='interrupted')
    finally:
        try:
            robot.hold(0., 0.)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target')
    parser.add_argument('--standoff-cm', type=float, default=20.)
    parser.add_argument('--scan-step-deg', type=float, default=15.)
    parser.add_argument('--max-looks', type=int, default=40)
    parser.add_argument('--log')
    parser.add_argument('--dry-run', action='store_true',
                        help='perceive and decide, but never command the motors')
    args = parser.parse_args()
    result = fetch(args.target, args.standoff_cm, args.scan_step_deg,
                   args.max_looks, args.dry_run, args.log)
    return 0 if result['event'] == 'reached' else 1



def project_obstacles(robot, answer):
    """Floor positions of the things the recognizer says are in the way.

    A contact pixel is one point, so it is treated as a small disc, not as a
    box stretched to the horizon. Getting that wrong before turned a single
    cable into a wall the robot could not get around.
    """
    spots = []
    for obstacle in answer.get('obstacles') or []:
        point = obstacle.get('contact_pixel')
        if not isinstance(point, dict):
            continue
        try:
            x, y = float(point['x']), float(point['y'])
        except (Stop, KeyError, TypeError, ValueError):
            continue
        # The lower fisheye corners turn a few pixels of model error into a
        # large and often inverted floor-range error.  In the failed Advil run
        # GPT placed an offscreen can at (639,478), which projected 12.5 cm from
        # the robot and blocked a clear route.  Do not create or refresh an
        # obstacle from those corners; a prior metric observation, if any,
        # remains available through the separate obstacle-memory path.
        if (y >= OBSTACLE_CORNER_Y_PX
                and (x <= OBSTACLE_CORNER_X_PX
                     or x >= 640. - OBSTACLE_CORNER_X_PX)):
            continue
        try:
            spot = robot.ground(x, y)
        except Stop:
            continue        # not on visible floor: it cannot be placed, so skip
        if math.hypot(*spot) <= ROUTE_RANGE_CM:
            spots.append((obstacle.get('label', '?'), spot))
    return spots


def clear_distance(obstacles, start, heading, wanted):
    """How far this leg may run before something blocks the corridor.

    The robot sweeps a corridor of its own width along the leg. Anything whose
    disc reaches into that corridor ahead of the robot cuts the leg short, so
    it stops before contact instead of discovering the obstacle with its wheels.
    Returns the distance and whatever is responsible for limiting it.
    """
    limit, blame = wanted, None
    angle = math.radians(heading)
    for item in obstacles:
        label, (ox, oz) = item[0], item[1]
        # An obstacle may carry the half-extent the planner measured for it.
        # Where it does not, fall back on the constant -- which is a guess, and
        # was the reason a 12.3 cm carton was modelled as an 8 cm disc and
        # driven into on 2026-09-24.
        radius = float(item[2]) if len(item) > 2 and item[2] else OBSTACLE_RADIUS_CM
        radius = min(radius, OBSTACLE_MAX_RADIUS_CM)
        dx, dz = ox - start[0], oz - start[1]
        along = math.cos(angle) * dz + math.sin(angle) * dx
        across = math.cos(angle) * dx - math.sin(angle) * dz
        if along <= 0:
            continue        # behind the robot: not in the way of this leg
        if abs(across) > CORRIDOR_HALF_CM + radius:
            continue        # passes to one side
        room = along - radius - KEEP_BACK_CM
        if room < limit:
            limit, blame = room, label
    return max(0., limit), blame


def project_route(robot, answer, standoff):
    """Turn the model's pixel path into a checked metric polyline.

    The model proposes; this decides. Every point must land on usable floor and
    move the robot forward along the path, and the whole thing is bounded. A
    route that fails any of that is refused rather than repaired, because a
    quietly patched trajectory is one nobody chose.
    """
    points = answer.get('route_pixels') or []
    target = answer.get('contact_pixel')
    if not isinstance(target, dict):
        raise Stop('no target contact point to drive to')
    goal = robot.ground(target['x'], target['y'])
    route, previous, total = [], (0., 0.), 0.
    for index, point in enumerate(points):
        try:
            x, y = float(point['x']), float(point['y'])
        except (KeyError, TypeError, ValueError):
            raise Stop('route point %d is not a pixel' % index)
        if not (0 <= x <= 640 and 0 <= y <= 480):
            break        # off frame: the usable part of the route ends here
        try:
            spot = robot.ground(x, y)
        except Stop:
            break        # above the floor line: same, stop the route here
        if math.hypot(*spot) > ROUTE_RANGE_CM:
            # End the route here rather than refusing all of it. The near part
            # is well conditioned and worth driving; the far part gets re-planned
            # from closer up, where the projection is far more trustworthy.
            break
        leg = math.hypot(spot[0] - previous[0], spot[1] - previous[1])
        if leg < ROUTE_MIN_LEG_CM:
            continue          # crowded points are dropped, not fatal
        if not route and math.hypot(*spot) < LEAD_POINT_CM:
            previous = spot   # a formality point right under the nose: it says
            continue          # nothing about which way to go, and driving it
                              # first commits the robot straight ahead
        total += leg
        if total > ROUTE_MAX_CM:
            raise Stop('route is longer than the %.0f cm bound' % ROUTE_MAX_CM)
        route.append(spot)
        previous = spot
    # Trim the tail back to the standoff so the robot stops short of the object
    # rather than driving into it -- and drop anything already past that point,
    # or the path doubles back on itself to reach the standoff from beyond it.
    reach = math.hypot(*goal)
    keep = max(0., reach - standoff)
    while route and math.hypot(*route[-1]) > keep:
        route.pop()
    if reach > standoff:
        scale = keep / reach
        route.append((goal[0] * scale, goal[1] * scale))
    if not route:
        raise Stop('route had no usable points')
    return route, goal


def _near_segment(start, end, point):
    """Closest point on the travelled segment to `point`, and the distance."""
    dx, dz = end[0] - start[0], end[1] - start[1]
    span = dx * dx + dz * dz
    if span < 1e-9:
        return start, math.hypot(point[0] - start[0], point[1] - start[1])
    along = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dz) / span
    along = max(0., min(1., along))
    close = (start[0] + along * dx, start[1] + along * dz)
    return close, math.hypot(point[0] - close[0], point[1] - close[1])


def rebase(points, pose):
    """Floor points from an earlier frame, expressed in the robot's frame now.

    `pose` is what follow() returns: where the robot ended up, in the frame the
    route was planned in. Everything the robot knew before that motion is still
    true about the room; it is only the robot's own origin that moved. So the
    memory is kept by moving the points, not by re-asking for them.
    """
    px, pz, heading = pose[0], pose[1], math.radians(pose[2])
    moved = []
    for x, z in points:
        dx, dz = x - px, z - pz
        moved.append((math.cos(heading) * dx - math.sin(heading) * dz,
                      math.cos(heading) * dz + math.sin(heading) * dx))
    return moved


def escape_heading(obstacles, wanted=STEP_MAX_CM, sweep=150., step=15.):
    """Which way to turn when the robot is boxed in, or None if it is not.

    clear_distance needs OBSTACLE_RADIUS + KEEP_BACK of room, so a robot that
    has closed to within that of something cannot legally move along any heading
    whose corridor still contains it -- and at 10 cm a 9 cm corridor spans a
    cone of about 42 degrees, so that is most of them. Measured on a live run:
    six waypoints refused in a row, twice, then the mission stopped with the
    robot a centimetre from where it started.

    Turning is the way out, because rotation moves the corridor without moving
    the robot into anything. This picks the heading with the most room, and
    returns None when straight ahead is already the best answer -- so it only
    fires when the robot really is stuck.
    """
    ahead = clear_distance(obstacles, (0., 0.), 0., wanted)[0]
    if ahead >= wanted:
        return None                     # not stuck; nothing to escape from

    # The smallest turn that frees the corridor, not the one with the most room.
    # Maximising room picks a near-U-turn and throws away every centimetre of
    # progress toward the target; the point is to get moving again while still
    # facing roughly the right way.
    fallback, most = None, ahead
    for size in [step * n for n in range(1, int(sweep / step) + 1)]:
        for turn in (size, -size):
            room = clear_distance(obstacles, (0., 0.), turn, wanted)[0]
            if room >= wanted:
                return turn
            if room > most + .5:
                fallback, most = turn, room
    return fallback


def merge_obstacles(remembered, pose, fresh, keep_cm=60.):
    """Everything known to be on the floor, in the frame of the look just taken.

    A photograph only reports what is in it, and the things the robot is about
    to touch are the first to leave the frame: at 9.5 cm of camera height the
    nearest visible floor is about 6 cm away, so an obstacle seen at 22 cm is
    invisible by the time the robot is 15 cm closer to it. Dropping it there
    means the clearance check forgets an obstacle exactly when it matters most,
    which is how the robot ends up touching things it had already been told
    about.

    So remembered obstacles are carried into the new frame and merged with the
    fresh ones. A fresh sighting of the same thing wins -- it is measured from
    closer -- and anything now behind the robot or past `keep_cm` is let go,
    since carrying stale positions forever would eventually wall the robot in.
    """
    kept = list(fresh)
    # fresh carries a measured half-extent as a third element under the metric
    # contract. Unpacking two names here raised on every cycle that had memory
    # to merge -- which is every cycle after the first -- and the mission died
    # silently inside its own thread, always after exactly one leg.
    spots = [item[1] for item in fresh]
    # Remembered obstacles are three-tuples since the metric contract; carry
    # the measured half-extent through rather than dropping it on every rebase.
    extents = [item[2] if len(item) > 2 else None for item in remembered]
    labels = [item[0] for item in remembered]
    moved = rebase([item[1] for item in remembered], pose)
    for label, spot, extent in zip(labels, moved, extents):
        if spot[1] < -OBSTACLE_RADIUS_CM or math.hypot(*spot) > keep_cm:
            continue                      # behind, or far enough to re-see
        if any(math.hypot(spot[0] - s[0], spot[1] - s[1]) < OBSTACLE_RADIUS_CM * 2.
               for s in spots):
            continue                      # the same thing, seen again just now
        # Do not re-suffix something already carried: it compounded into
        # "blue block (remembered) (remembered) (remembered)" over four cycles.
        kept.append((label if label.endswith(')') else label + ' (remembered)',
                     spot, extent))
    return kept


CONTACT_PIXEL_SLOP = 12.   # how wrong the recognizer's contact point tends to
                           # be, vertically, in pixels. Measured from a live
                           # sighting whose range was out by 51 cm: about 17 px.


def nearest_range(lens, x, y, slop=CONTACT_PIXEL_SLOP):
    """The closest the target could plausibly be, given a contact pixel.

    A pixel does not carry one range, it carries a bracket, and near the horizon
    that bracket is enormous: three pixels is 4 per cent of the range at 9 cm and
    21 per cent at 1 m, and the recognizer is wrong by rather more than three.
    Taking the middle of the bracket and subtracting a standoff from it is how
    the robot drove into a bin: it believed 73 cm, the bin was about 25, and
    "stop 20 cm short" became a 53 cm commitment.

    So the approach is planned against the near edge instead. Being too cautious
    costs another look, which is cheap; being too bold costs a collision.
    """
    try:
        return lens.ground(float(x), float(y) + slop)[1]
    except Stop:
        return None


def cap_path(route, limit):
    """Cut a route back to `limit` cm of travel, whatever it was aiming at.

    Range from a single contact pixel is only as good as the pixel, and near
    the horizon it is very poor: 34 pixels of error measured as 80 cm of range
    on this camera. So a standoff computed from a long-range sighting can be
    wrong by more than the standoff itself -- "stop 20 cm short of a thing
    73 cm away" drove 53 cm into a bin that was really about 25 cm away.

    No standoff survives an error like that, because the error scales with the
    quantity it is subtracted from. A flat cap on how far the robot commits
    before looking again does survive it: a wildly wrong range then costs one
    short leg and a fresh measurement, instead of a collision.
    """
    if limit <= 0.:
        return list(route)
    kept = []
    previous = (0., 0.)
    travelled = 0.
    for spot in route:
        span = math.hypot(spot[0] - previous[0], spot[1] - previous[1])
        if travelled + span <= limit:
            kept.append(spot)
            travelled += span
            previous = spot
            continue
        room = limit - travelled
        if room >= ROUTE_MIN_LEG_CM and span > 1e-6:
            t = room / span
            kept.append((previous[0] + (spot[0] - previous[0]) * t,
                         previous[1] + (spot[1] - previous[1]) * t))
        break
    return kept


def stop_short(route, goal, standoff):
    """End the route `standoff` cm from the target, on the side it approached.

    The model is asked to end its route at the object's contact pixel, which is
    the right answer to the question it was asked and the wrong thing to drive:
    arriving there puts the wheels where the object is.

    Cutting the path where it first crosses the standoff ring is the obvious
    trim and it is wrong, because which side the robot arrives on is carried by
    the LAST segment -- exactly the part that cut discards. Asked to reach a can
    from its left, the model returned an eight point curve coming in from the
    left and the trim kept two points of it, aimed straight ahead; the robot
    then drove at the can and called it arrival. "From the left" never survived
    to the wheels.

    So the stopping point is placed on the ring along the approach the route
    ends on, and the points still outside the ring are kept. Where the route
    says nothing about a side, that reduces to stopping short along the way it
    was already going.
    """
    if goal is None or not route:
        return list(route)

    # The direction the route arrives from, taken from its final segment.
    tail = route[-1]
    before = route[-2] if len(route) > 1 else (0., 0.)
    dx, dz = tail[0] - before[0], tail[1] - before[1]
    span = math.hypot(dx, dz)
    if span < 1e-6:
        dx, dz = tail[0] - goal[0], tail[1] - goal[1]
        span = math.hypot(dx, dz)
        if span < 1e-6:
            return []                  # already on it, nothing to drive
        dx, dz = -dx, -dz              # point at the goal, not away from it
    stop = (goal[0] - dx / span * standoff, goal[1] - dz / span * standoff)

    if stop[1] <= 0.:
        return []      # the standoff point is behind the robot: it is there

    kept = []
    previous = (0., 0.)
    for spot in route:
        if math.hypot(spot[0] - goal[0], spot[1] - goal[1]) <= standoff:
            break      # inside the ring; the stopping point replaces the rest
        if math.hypot(spot[0] - previous[0], spot[1] - previous[1]) >= ROUTE_MIN_LEG_CM:
            kept.append(spot)
            previous = spot
    if math.hypot(stop[0] - previous[0], stop[1] - previous[1]) >= ROUTE_MIN_LEG_CM:
        kept.append(stop)
    elif kept:
        kept[-1] = stop
    else:
        kept = [stop]
    return kept


def clip_standoff(route, goal, standoff):
    """Clip at first entry into the target disk without inventing a new approach.

    Unlike stop_short, this is safe on an already-trimmed or widened route.
    """
    if goal is None:
        return list(route)
    if math.hypot(*goal) <= standoff:
        return []
    kept, previous = [], (0., 0.)
    for point in route:
        dx, dz = point[0]-previous[0], point[1]-previous[1]
        ox, oz = previous[0]-goal[0], previous[1]-goal[1]
        a = dx*dx + dz*dz
        b = 2*(ox*dx + oz*dz)
        c = ox*ox + oz*oz - standoff*standoff
        disc = b*b - 4*a*c
        if a > 1e-12 and disc >= 0:
            t = (-b-math.sqrt(disc))/(2*a)
            if -1e-8 <= t <= 1+1e-8:
                t = min(1., max(0., t))
                stop = (previous[0]+t*dx, previous[1]+t*dz)
                if math.hypot(stop[0]-previous[0], stop[1]-previous[1]) > 1e-6:
                    kept.append(stop)
                return kept
        kept.append(point)
        previous = point
    return kept


def round_corners(route, keep_last=True, passes=2):
    """Cut the corners off a route so it reads as an arc, not a dog-leg.

    Chaikin's corner cutting: each corner is replaced by two points a quarter
    and three quarters along its legs, which pulls the path into a curve while
    keeping it on the same side of everything. Asking the model for a smooth
    arc helps a little -- measured, mean bend 8.1 to 6.9 degrees -- but it has
    no scale, so the curve it draws is a shape, not a distance. Doing it here
    means the arc is made out of centimetres.

    The last point is the target and does not move.
    """
    if len(route) < 3:
        return list(route)
    path = [tuple(p) for p in route]
    goal = path[-1] if keep_last else None
    for _ in range(passes):
        cut = [path[0]]
        for i in range(len(path) - 1):
            ax, az = path[i]
            bx, bz = path[i + 1]
            cut.append((ax + .25 * (bx - ax), az + .25 * (bz - az)))
            cut.append((ax + .75 * (bx - ax), az + .75 * (bz - az)))
        cut.append(path[-1])
        path = cut
    if goal is not None:
        path[-1] = goal
    # Corner cutting multiplies points; the executor skips anything closer than
    # ROUTE_MIN_LEG_CM anyway, so thin them back out to legs it will actually
    # drive rather than handing it a cloud of near-duplicates.
    thinned = [path[0]]
    for spot in path[1:]:
        if math.hypot(spot[0] - thinned[-1][0], spot[1] - thinned[-1][1]) >= ROUTE_MIN_LEG_CM:
            thinned.append(spot)
    if thinned[-1] != path[-1]:
        thinned.append(path[-1])
    return thinned


def avoid(route, obstacles, goal=None):
    """Widen a route to the clearance the wheels need, keeping its shape.

    The recognizer is asked to leave room for the robot, and it tries, but it
    works in pixels with no scale: measured over stored room photographs it
    left 3 to 14 cm where 15 is needed, in every scene with clutter, on every
    sample. One image cannot tell it how wide 15 cm is. The distances are ours,
    so the correction belongs here rather than in the prompt.

    Two things are fixed, because a route can foul an obstacle in two ways. A
    waypoint inside the corridor is pushed straight out to its edge. A segment
    that passes too close between two clear waypoints -- the common case, and
    the one a first version of this missed -- gets a new waypoint inserted at
    the pinch, pushed out along the same outward normal. Both keep the side the
    model chose to pass on, and change only the margin.

    Obstacles within GOAL_ADJACENT_CM of the target are left alone: the thing
    being fetched is usually sitting among them, and routing away from them
    means never arriving.
    """
    # Each obstacle keeps its own clearance, from its own measured half-extent
    # where the planner supplied one.
    relevant = [(item[1],
                 (float(item[2]) if len(item) > 2 and item[2] else OBSTACLE_RADIUS_CM)
                 + CORRIDOR_HALF_CM + REPAIR_MARGIN_CM)
                for item in obstacles
                if goal is None
                or math.hypot(item[1][0] - goal[0], item[1][1] - goal[1]) > GOAL_ADJACENT_CM]
    if not relevant or not route:
        return list(route), 0

    # Widen to the padded figure, not the bare limit. Testing against the bare
    # limit here meant any route that merely scraped past was returned
    # untouched: measured 2026-09-24, the planner drew a route passing an
    # obstacle at 11.6 cm, which clears the 10 cm floor, so this exited early
    # and the padding was never applied -- the chassis edge went by 5.6 cm from
    # the box. The obstacles themselves are declared with 5 cm of position
    # uncertainty, so the floor alone is not a margin. Routes already outside
    # the padded figure are still returned untouched.
    legs = [(0., 0.)] + list(route)
    if all(_near_segment(start, end, spot)[1] >= clearance
           for start, end in zip(legs, legs[1:]) for spot, clearance in relevant):
        return list(route), 0

    def push(point, spot, clearance):
        dx, dz = point[0] - spot[0], point[1] - spot[1]
        gap = math.hypot(dx, dz)
        if gap < 1e-6:            # sitting on it: no outward direction to use
            return None
        return (spot[0] + dx / gap * clearance, spot[1] + dz / gap * clearance)

    def usable(points):
        """Drop what pushing produced that the robot cannot drive to.

        Shoving a waypoint clear of something close and to one side can put it
        behind the wheels, or on top of its neighbour. Either is worse than the
        point it replaced, so they come back out here rather than being handed
        to the executor.
        """
        kept = []
        for point in points:
            if point[1] < 0.:
                continue                     # behind the robot: not a way round
            if kept and math.hypot(point[0] - kept[-1][0],
                                   point[1] - kept[-1][1]) < ROUTE_MIN_LEG_CM:
                continue                     # on top of the one before
            kept.append(point)
        return kept

    changed = 0
    fixed = [tuple(point) for point in route]
    last = len(fixed) - 1 if goal is not None else None
    for _ in range(REPAIR_PASSES):
        touched = False
        for index, point in enumerate(fixed):
            if index == last:
                continue          # the last point is the target; do not move it
            for spot, clearance in relevant:
                if math.hypot(point[0] - spot[0], point[1] - spot[1]) >= clearance:
                    continue
                moved = push(point, spot, clearance)
                if moved:
                    point, touched = moved, True
            if point != fixed[index]:
                fixed[index] = point
                changed += 1

        # Now the segments. Insert at the tightest pinch only, then re-sweep:
        # moving one point changes the two segments either side of it.
        pinch = None
        legs = [(0., 0.)] + fixed
        for leg in range(len(legs) - 1):
            for spot, clearance in relevant:
                close, gap = _near_segment(legs[leg], legs[leg + 1], spot)
                if gap < clearance and (pinch is None or gap < pinch[0]):
                    pinch = (gap, leg, close, spot, clearance)
        if pinch is not None:
            _, leg, close, spot, clearance = pinch
            moved = push(close, spot, clearance)
            if moved:
                fixed.insert(leg, moved)   # legs[0] is the robot, so leg == index
                if last is not None:
                    last += 1
                changed += 1
                touched = True
        if not touched:
            break
    goal_point = fixed[last] if last is not None and last < len(fixed) else None
    fixed = usable(fixed)
    if goal_point is not None and (not fixed or fixed[-1] != goal_point):
        fixed.append(goal_point)             # never drop the target itself

    # Round last, not first. Pushing to clearance leaves kinks; rounding takes
    # them out and moves each point at most a quarter of the way to its
    # neighbour, which on legs this short is under a centimetre -- already
    # covered by REPAIR_MARGIN_CM. Doing it the other way round and then
    # pushing again re-made every corner: measured, mean bend went up from 6
    # degrees to 10 while clearance stayed the same.
    curved = round_corners(fixed, keep_last=goal is not None)

    # Then fix only what is genuinely too close, and only as far as the real
    # limit rather than the padded one, so a correction here is a nudge and not
    # a new corner.
    for index, point in enumerate(curved):
        if goal is not None and index == len(curved) - 1:
            continue
        for spot, clearance in relevant:
            limit = clearance - REPAIR_MARGIN_CM      # the real limit, unpadded
            gap = math.hypot(point[0] - spot[0], point[1] - spot[1])
            if gap >= limit or gap < 1e-6:
                continue
            point = (spot[0] + (point[0] - spot[0]) / gap * limit,
                     spot[1] + (point[1] - spot[1]) / gap * limit)
        curved[index] = point
    curved = usable(curved)

    # Drop leading waypoints the robot cannot face. Pushing a point clear of
    # something close to the robot moves it radially, and for a point only a
    # few centimetres out "radially" means sideways: measured, a first waypoint
    # at (-0.1, 6.3) came back at (7.8, 2.5), turning a straight-ahead start
    # into a 72 degree pivot. Inside the pivot orbit follow skips it anyway --
    # after paying for the turn -- so it is better not to offer it at all.
    orbit = pivot_radius() + ROUTE_MIN_LEG_CM
    while len(curved) > 1 and math.hypot(curved[0][0], curved[0][1]) < orbit:
        curved.pop(0)

    if goal_point is not None:
        # Rounding can leave a point almost on top of the target. The executor
        # skips legs under ROUTE_MIN_LEG_CM, so such a point is not a waypoint,
        # it is just something for the aim to chatter at.
        while curved and math.hypot(curved[-1][0] - goal_point[0],
                                    curved[-1][1] - goal_point[1]) < ROUTE_MIN_LEG_CM:
            curved.pop()
        curved.append(goal_point)
    return curved, changed


class Odometer:
    """Distance travelled, from carpet features, with an honest failure mode.

    Visual odometry on plain carpet does fail. When it does this says so and the
    caller falls back to timing, rather than ending a maneuver that is going
    fine -- the trajectory's real check is the next look at the target.
    """

    def __init__(self, robot):
        from point_controller import FloorTracker, TURN_FLOW_WINDOW
        with open(os.path.join(ROOT, 'calibration/floor_geometry.json')) as handle:
            profile = json.load(handle)
        with open(profile['intrinsics_path']) as handle:
            intrinsics = json.load(handle)
        self.tracker = FloorTracker(profile, intrinsics, 250,
                                    flow_window=TURN_FLOW_WINDOW)
        self.robot = robot
        self.reference = None

    def mark(self, image):
        self.reference = image

    def travelled(self, image):
        """Centimetres moved since mark(), or None when the carpet is unreadable."""
        if self.reference is None:
            return None
        try:
            _, translation, _ = self.tracker.motion(self.reference, image)
            return float(np.linalg.norm(translation))
        except Exception:
            return None


def drive_leg(robot, odometer, centimetres, record, interrupt=None):
    """Drive one straight leg: regulated approach, then measured bursts.

    Same shape as a turn, for the same reason. Speed is servoed down as the
    target nears so the robot arrives slowly, and the last centimetre is closed
    with short bursts because stiction means there is no slow crawl to ramp
    into. Heading is held throughout by trimming the wheels against the IMU,
    and distance comes from carpet optical flow sampled every 0.12 s.
    """
    if robot.dry_run:
        record('leg', wanted_cm=round(centimetres, 1), moved_cm=round(centimetres, 1),
               by='dry-run', drift_deg=0., speed_cm_s=round(robot.speed, 1))
        return centimetres
    tilt = robot.tilt(call('status')['imu'])
    if tilt is not None and tilt > TILT_LIMIT_DEG:
        raise Stop('tilt %.1f degrees before moving' % tilt)

    travelled = [0.]
    visual = [0]
    timed = [0]
    heading = [0.]
    moved_by = [0.]
    image, _ = robot.frame()
    odometer.mark(image)
    started = last = sampled = time.monotonic()
    limit = max(3., min(12., centimetres / max(3., robot.speed) * 3. + 2.))

    def sample():
        """Fold in however far the carpet says the robot has moved since last."""
        frame, _ = robot.frame()
        step = odometer.travelled(frame)
        now = time.monotonic()
        if step is None:
            timed[0] += 1
            travelled[0] += robot.speed * (now - sampled)
        else:
            visual[0] += 1
            travelled[0] += step
        odometer.mark(frame)
        record('moving', moved_cm=round(travelled[0], 1))
        return now

    try:
        while travelled[0] < centimetres - FINE_DRIVE_CM:
            now = time.monotonic()
            if interrupt is not None and interrupt():
                # A fresher plan has landed. Finishing this leg first would
                # spend up to a second and a half driving the old one, which is
                # most of what the pipeline was built to avoid.
                record('handed_over', moved_cm=round(travelled[0], 1))
                break
            if now - started > limit:
                raise Stop('leg exceeded its %.1f s bound' % limit)
            status = robot.check()
            rate = robot.yaw_rate(status['imu'])
            if abs(rate) > YAW_RATE_LIMIT:
                raise Stop('yaw rate %.0f deg/s' % rate)
            heading[0] += rate * (now - last)
            last = now
            if abs(heading[0]) > 30.:
                raise Stop('drifted %.0f degrees off the leg' % heading[0])
            remaining = centimetres - travelled[0]
            wanted = max(DRIVE_MIN_RATE, min(DRIVE_MAX_RATE,
                                             remaining * DRIVE_APPROACH_GAIN))
            duty = robot.floor_duty(min(DRIVE_MAX_DUTY, wanted / robot.drive_gain))
            robot.note_motion(duty, travelled[0] > moved_by[0] + .2)
            moved_by[0] = travelled[0]
            trim = max(-.03, min(.03, -heading[0] * STEER_GAIN))
            robot.hold(duty + trim, duty - trim)
            if now - sampled >= VISION_INTERVAL_S:
                before = travelled[0]
                sampled = sample()
                robot.hold(duty + trim, duty - trim)
                last = time.monotonic()
                gap = sampled - (last - VISION_INTERVAL_S)
                if travelled[0] - before > .3 and duty > .01:
                    robot.drive_gain = max(20., min(200., .8 * robot.drive_gain
                        + .2 * ((travelled[0] - before) / max(.05, VISION_INTERVAL_S)) / duty))
            time.sleep(.015)
    finally:
        robot.hold(0., 0.)
        time.sleep(PULSE_SETTLE_S)

    def measure():
        nonlocal sampled
        sampled = sample()
        return travelled[0]
    measure()
    robot._creep_drive(centimetres - travelled[0], measure)
    if visual[0] and travelled[0] > 1.:
        elapsed = max(.3, time.monotonic() - started)
        robot.speed = max(3., min(30., .7 * robot.speed + .3 * travelled[0] / elapsed))
    record('leg', wanted_cm=round(centimetres, 1), moved_cm=round(travelled[0], 1),
           by='visual %d/%d' % (visual[0], visual[0] + timed[0]),
           drift_deg=round(heading[0], 1), speed_cm_s=round(robot.speed, 1))
    return travelled[0]


def pivot_shift(degrees):
    """Where the lens lands after turning `degrees`, in the pre-turn body frame.

    The robot does not spin about its camera. calibration/turn_pivot.json puts
    the pivot 7.3 cm behind the lens (measured over 28 turns, 0.8 mm RMS, and
    the user independently said 7 cm), so a large turn slides the pose
    sideways: 105 degrees moves it about 12 cm. Ignoring that is fine for the
    10-degree turns of a scan and badly wrong for a drawn trajectory.
    """
    global _PIVOT
    if _PIVOT is None:
        try:
            with open(os.path.join(ROOT, 'calibration/turn_pivot.json')) as handle:
                fitted = json.load(handle)
            _PIVOT = (float(fitted['pivot_x_cm']), float(fitted['pivot_z_cm']))
        except (IOError, OSError, ValueError, KeyError):
            _PIVOT = (0., 0.)      # unmeasured: spin about the lens, as before
    px, pz = _PIVOT
    turn = math.radians(degrees)
    cos, sin = math.cos(turn), math.sin(turn)
    return (px * (1. - cos) - pz * sin, px * sin + pz * (1. - cos))


_PIVOT = None


def pivot_radius():
    """How far turning alone moves the lens: the radius it swings through."""
    pivot_shift(0.)                     # make sure the calibration is loaded
    return math.hypot(_PIVOT[0], _PIVOT[1])


def aim_turn(pose, target):
    """The turn that leaves the robot facing `target` -- after the pivot slides it.

    Turning moves the lens (see pivot_shift), so the naive bearing is stale by
    the time the turn finishes: aiming at a waypoint 20 cm away can be several
    degrees out, and the robot then zigzags in to it. Solving for the angle that
    is correct after the shift is a fixed point over pure arithmetic, so it is
    iterated to convergence rather than to a fixed count -- a close waypoint,
    where the shift is large next to the range, needs the most passes.
    """
    heading = math.radians(pose[2])
    turn = 0.
    for _ in range(AIM_PASSES):
        shift = pivot_shift(turn)
        x = pose[0] + math.cos(heading) * shift[0] + math.sin(heading) * shift[1]
        z = pose[1] + math.cos(heading) * shift[1] - math.sin(heading) * shift[0]
        dx, dz = target[0] - x, target[1] - z
        if abs(dx) < 1e-9 and abs(dz) < 1e-9:
            break     # the turn has put the lens on the waypoint; nothing to aim at
        solved = (math.degrees(math.atan2(dx, dz)) - pose[2] + 180.) % 360. - 180.
        settled = abs(solved - turn) < AIM_SETTLED_DEG
        turn = solved
        if settled:
            break
    return turn


def follow(robot, odometer, route, record, obstacles=(), complete=True,
           deadline=None, interrupt=None, initial_turn_limit_deg=None):
    """Execute the trajectory: reach each waypoint in turn, then the next.

    Turns and drives are both chunked -- at most TURN_MAX_PER_LEG_DEG and
    STEP_MAX_CM at a time -- so that obstacles and odometry are re-checked
    often. A chunk is a step toward the waypoint, not a substitute for it: the
    loop keeps issuing chunks until the robot is actually there. It used to do
    one of each and move on, which quietly dropped every waypoint needing a
    turn past 60 degrees or a leg past 15 cm.

    `complete=False` restores that single chunk per waypoint on purpose, for the
    caller that re-looks afterwards: in `fetch` the camera re-measures the range
    from wherever the robot ended up, so committing further open loop buys
    nothing and spends the accuracy that closing in was supposed to gain.

    The recognizer's route is meant to avoid obstacles already, but it is a
    proposal made from one image and the robot is the thing that has to live
    with it. So every leg is checked against what the recognizer said is on the
    floor, and cut short of anything in the corridor rather than driven into.
    """
    pose = [0., 0., 0.]          # where we think we are, in the start frame
    driven = 0.
    for index, (x, z) in enumerate(route):
        if interrupt is not None and interrupt():
            return pose
        if deadline is not None and time.monotonic() >= deadline:
            # A recurrent planner wants the wheels back on a fixed cadence so it
            # can look again. Stopping between chunks rather than mid-leg keeps
            # the pose honest: every chunk that started has finished and been
            # accounted for.
            record('out_of_time', waypoint=index, of=len(route))
            return pose
        reached = False
        blocked = False      # this waypoint is unreachable; the next may not be
        faces = 0            # turn chunks spent since the last drive
        stalls = 0           # drives that bought no ground
        closest = None       # best range to this waypoint so far
        orbit = pivot_radius() + ROUTE_MIN_LEG_CM
        for chunk in range(WAYPOINT_MAX_LEGS if complete else 1):
            if interrupt is not None and interrupt():
                return pose
            if deadline is not None and time.monotonic() >= deadline:
                record('out_of_time', waypoint=index, of=len(route))
                return pose
            dx, dz = x - pose[0], z - pose[1]
            angle = math.radians(pose[2])
            ahead = math.cos(angle) * dz + math.sin(angle) * dx
            across = math.cos(angle) * dx - math.sin(angle) * dz
            leg = math.hypot(ahead, across)
            if leg < ROUTE_MIN_LEG_CM:
                if chunk == 0:
                    # Perspective bunches clicks: near the bottom of the frame
                    # 30 px is under 2 cm of floor. Say so rather than dropping
                    # it silently.
                    record('skipped', waypoint=index, leg_cm=round(leg, 1),
                           minimum_cm=ROUTE_MIN_LEG_CM)
                reached = True
                break
            bearing = aim_turn(pose, (x, z))
            # Check the actual executable point, including after skipped/blocked
            # points and async rebasing. A token point under the lens must not
            # license a sharp turn on the next point. Do not clamp or skip the
            # offending point: that would invent an unchecked connecting path.
            if (initial_turn_limit_deg is not None and driven < LEAD_POINT_CM
                    and (ahead <= 0. or
                         abs(math.degrees(math.atan2(across, ahead))) > initial_turn_limit_deg
                         or abs(bearing) > initial_turn_limit_deg)):
                robot.hold(0., 0.)
                record('route_start_rejected', waypoint=index,
                       bearing_deg=round(bearing, 1),
                       limit_deg=initial_turn_limit_deg, driven_cm=round(driven, 1))
                return pose
            if abs(bearing) > HEADING_TOLERANCE_DEG and leg < orbit:
                # Turning swings the lens through a circle of `orbit` cm about
                # the pivot. A waypoint nearer than that cannot be faced: every
                # turn carries the lens further than the gap it was closing, so
                # the robot orbits the point instead of arriving. Measured as a
                # period-6 cycle of 1440 degrees before this check existed.
                record('too_close_to_face', waypoint=index,
                       leg_cm=round(leg, 1), bearing_deg=round(bearing, 1),
                       orbit_cm=round(orbit, 1))
                reached = True
                break
            if abs(bearing) > HEADING_TOLERANCE_DEG:
                if faces >= FACE_ATTEMPTS:
                    record('cannot_face', waypoint=index,
                           bearing_deg=round(bearing, 1), turns=faces)
                    break
                faces += 1
                asked = max(-TURN_MAX_PER_LEG_DEG,
                            min(TURN_MAX_PER_LEG_DEG, bearing))
                turned = robot.turn(asked)
                shift = pivot_shift(turned)
                heading = math.radians(pose[2])
                pose[0] += math.cos(heading) * shift[0] + math.sin(heading) * shift[1]
                pose[1] += math.cos(heading) * shift[1] - math.sin(heading) * shift[0]
                pose[2] += turned
                record('pose', x=round(pose[0], 1), z=round(pose[1], 1),
                       heading=round(pose[2], 1))
                record('faced', waypoint=index, bearing_deg=round(bearing, 1),
                       asked_deg=round(asked, 1), turned_deg=round(turned, 1),
                       remaining_deg=round(bearing - turned, 1))
                if abs(bearing - turned) > HEADING_TOLERANCE_DEG:
                    continue     # still not facing it; turn again before driving
            wanted = min(leg, STEP_MAX_CM)
            room, blame = clear_distance(obstacles, (pose[0], pose[1]), pose[2], wanted)
            if room < wanted - .5:
                record('shortened', waypoint=index, wanted_cm=round(wanted, 1),
                       clear_cm=round(room, 1), blocked_by=blame)
            if room < ROUTE_MIN_LEG_CM:
                record('leg_blocked', waypoint=index, clear_cm=round(room, 1),
                       blocked_by=blame,
                       trying_next=index + 1 < len(route))
                # The route was drawn to go around this obstacle, so the leg
                # that runs into it is usually the direct one and the way past
                # is the waypoint after it. Abandoning the whole route here
                # threw away the detour and left the robot inching forward a
                # few centimetres per look. The next leg is clearance-checked
                # in its own right, so trying it is not a way round the guard.
                blocked = True
                break
            # Relay progress as a pose, so a viewer can redraw the route from
            # where the robot is *now* rather than where it last stopped. The
            # leg is the long part of a run; without this the overlay only moves
            # once a waypoint, which is exactly when it is least interesting.
            start = list(pose)

            def relay(kind, **fields):
                if kind != 'moving':
                    return record(kind, **fields)
                along = math.radians(start[2])
                record('pose',
                       x=round(start[0] + math.sin(along) * fields['moved_cm'], 1),
                       z=round(start[1] + math.cos(along) * fields['moved_cm'], 1),
                       heading=round(start[2], 1))

            moved = drive_leg(robot, odometer, room, relay, interrupt)
            driven += max(0., moved)
            heading = math.radians(pose[2])
            pose[0] += math.sin(heading) * moved
            pose[1] += math.cos(heading) * moved
            faces = 0
            left = math.hypot(x - pose[0], z - pose[1])
            if closest is None or left < closest - STALL_GAIN_CM:
                closest = left
                stalls = 0
            else:
                stalls += 1
                if stalls >= STALL_ATTEMPTS:
                    record('no_progress', waypoint=index,
                           closest_cm=round(closest, 1), drives=stalls)
                    break
            if moved < 1. and room > ROUTE_MIN_LEG_CM:
                record('stalled', waypoint=index, asked_cm=round(room, 1),
                       moved_cm=round(moved, 1),
                       short_by_cm=round(leg - moved, 1))
                return pose
        else:
            if complete:     # stopping short is the contract when complete=False
                record('unreached', waypoint=index, legs=WAYPOINT_MAX_LEGS,
                       short_by_cm=round(math.hypot(x - pose[0], z - pose[1]), 1))
        if blocked and index + 1 >= len(route):
            record('route_blocked', waypoint=index)
            return pose      # nothing left to try; the next look re-plans
        record('waypoint', n=index, of=len(route), reached=reached,
               short_by_cm=round(math.hypot(x - pose[0], z - pose[1]), 1),
               pose=[round(v, 1) for v in pose])
    return pose


if __name__ == '__main__':
    sys.exit(main())
