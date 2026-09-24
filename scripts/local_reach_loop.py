#!/usr/bin/env python3
"""Chained randomised trajectories for the local executor, inside a 1 m radius.

The local planner owns a 1 m neighbourhood and nothing beyond it, so every goal
here is sampled inside that disc and a fresh one is issued as soon as the last
finishes. The robot is never returned to a start pose: the interesting states
are the ones it wanders into.

It is not a collision checker, so this picks goals against a floor scan of the
frame taken immediately before each submission (see floor_scan). When no bearing
has room -- nose to a wall, boxed into a corner -- the next goal is deliberately
placed *behind* the robot, which is the case the executor is supposed to resolve
by pivoting in place rather than refusing. A stalled or failed run counts as
blocked too, because the scan can miss a cable the wheels have already found.

Records land in the USB run root: one JSONL row and one viz section per cycle.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, os.pardir, 'local_nav'))

import floor_scan
import local_reach_trial as trial

RADIUS_CM = 100.           # the local planner's stated neighbourhood
STANDOFF_CM = 30.          # chassis half-length plus the range bracket a near-horizon pixel carries
MIN_USEFUL_CM = 34.        # a goal shorter than this is not worth a cycle
BEARINGS = list(range(-60, 61, 5))
# The service's own guard at 10.8 V is the authority; this sits just above it
# so the loop bows out first rather than pre-empting a pack that still drives.
# Measured: a clean 24.9-degree turn for 25 asked at 10.93 V. An earlier floor
# of 11.0 V would have called that flat, and a stall was in fact the robot
# sitting lifted off its wheels, not the battery.
PACK_FLOOR_V = 10.85


def pack_voltage():
    import fetch
    return float(fetch.call('status')['power']['pack_voltage_v'])


def camera_frame():
    import urllib.request
    raw = urllib.request.urlopen(
        'http://127.0.0.1:8770/api/frame?view=camera', timeout=10).read()
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def reachable(image):
    """(bearing, allowed_cm) for every bearing with usable room."""
    room = []
    for bearing, seen in floor_scan.profile(image, BEARINGS, RADIUS_CM + STANDOFF_CM):
        allowed = min(RADIUS_CM, seen - STANDOFF_CM)
        if allowed >= MIN_USEFUL_CM:
            room.append((bearing, allowed))
    return room


def forward_goal(rng, room):
    """A random goal on open floor, biased to the bearings with most room."""
    bearing, allowed = rng.choice(room)
    distance = rng.uniform(MIN_USEFUL_CM, allowed)
    a = math.radians(bearing)
    return [round(distance * math.sin(a), 2), round(distance * math.cos(a), 2)]


def backward_goal(rng):
    """A goal behind the robot, which it can only reach by pivoting first.

    Deliberately unscanned: the camera cannot see behind, and the point is to
    make the executor turn. It is bounded short so that the move stays inside
    floor the robot has just driven over.
    """
    bearing = rng.choice([rng.uniform(130., 180.), rng.uniform(-180., -130.)])
    distance = rng.uniform(25., 45.)
    a = math.radians(bearing)
    return [round(distance * math.sin(a), 2), round(distance * math.cos(a), 2)]


def build(rng, room, blocked, image=None, attempts=12):
    """The next trajectory, and a word for why it has that shape.

    Every candidate is re-checked against the frame with clear_path, which walks
    each leg's corridor. Picking bearings off the profile is not enough on its
    own: a dogleg's second leg runs from the first waypoint in a direction the
    profile never looked at, and that is floor nothing has verified.
    """
    for _ in range(attempts):
        shape, traj, why = _propose(rng, room, blocked)
        if shape == 'backward' or image is None:
            return shape, traj, why
        if floor_scan.clear_path(image, traj['waypoints_cm']):
            return shape, traj, why
    # Nothing forward survived the corridor check; back out instead of guessing.
    return 'backward', dict(waypoints_cm=[backward_goal(rng)], initial_turn_deg=0.), \
        'no candidate path cleared the corridor check; goal placed behind'


def _propose(rng, room, blocked):
    if blocked or not room:
        return 'backward', dict(waypoints_cm=[backward_goal(rng)],
                                initial_turn_deg=0.), 'boxed in; goal placed behind'
    goal = forward_goal(rng, room)
    # Sometimes bend the approach, staying inside the same scanned cone so the
    # dogleg does not wander into floor that was never looked at.
    if rng.random() < .35 and len(room) > 2:
        mid = forward_goal(rng, room)
        scale = rng.uniform(.35, .7)
        bend = [round(mid[0] * scale, 2), round(mid[1] * scale, 2)]
        return 'dogleg', dict(waypoints_cm=[bend, goal], initial_turn_deg=0.), \
            'two-leg path inside the scanned cone'
    if rng.random() < .3:
        turn = round(math.degrees(math.atan2(goal[0], goal[1])), 2)
        return 'turn_then_go', dict(waypoints_cm=[goal], initial_turn_deg=turn), \
            'face the goal first, then translate'
    return 'direct', dict(waypoints_cm=[goal], initial_turn_deg=0.), \
        'straight reach, executor works out its own pivot'


LEASE_FAULTS = ('control generation changed',
                'Control cancelled or service restarted')


def lease_fault(result):
    """A dead control lease, which says nothing about the floor."""
    return any(text in str(result.get('error') or '') for text in LEASE_FAULTS)


def blocked_by(result):
    """Did this run end in a way that means the robot cannot go that way?

    A lease fault is explicitly not blockage. Counting it as such sent twenty
    consecutive goals backwards while the real problem was that no goal in any
    direction could start at all.
    """
    if result.get('phase') in ('completed', 'paused') or lease_fault(result):
        return False
    error = str(result.get('error') or '')
    return ('no progress' in error or 'waypoint timeout' in error
            or 'stalled' in error or (result.get('travelled_cm') or 0.) < 5.)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cycles', type=int, default=20)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    if not os.path.ismount('/mnt/robotlogs'):
        raise SystemExit('robot log volume not mounted')
    with open('/mnt/robotlogs/current-search.json') as handle:
        root = json.load(handle)['root']

    seed = args.seed if args.seed is not None else random.randrange(1 << 30)
    rng = random.Random(seed)
    record = open(os.path.join(root, 'local-reach-chain.jsonl'), 'a')

    print('seed %d | %d cycle(s) | radius %.0f cm' % (seed, args.cycles, RADIUS_CM))
    blocked = False
    tally = {}
    for cycle in range(args.cycles):
        pack = pack_voltage()
        if pack < PACK_FLOOR_V:
            print('stopping: pack at %.2f V, below the %.2f V loop floor'
                  % (pack, PACK_FLOOR_V))
            break
        if not trial.wait_until_free():
            print('stopping: controller never went idle')
            break
        trial.post('/api/capture', {})
        image = camera_frame()
        room = reachable(image)
        shape, traj, why = build(rng, room, blocked, image)
        widest = max(room, key=lambda r: r[1]) if room else None
        print('\n%2d %-12s %-42s pack %.2f V' % (cycle, shape, why, pack))
        print('   scan: %d bearing(s) usable%s'
              % (len(room),
                 '' if not widest else ', widest %+d deg at %.0f cm'
                 % (widest[0], widest[1])))
        print('   %s' % json.dumps(traj))

        started = time.time()
        result = trial.execute(traj)
        if lease_fault(result):
            # Re-syncing is the executor's job now, but keep the recovery here
            # too: this loop must not spend its cycles re-reporting one fault.
            print('   lease fault; halting to re-sync and retrying once')
            trial.post('/api/halt', {})
            result = trial.execute(traj)
        phase = result.get('phase')
        error = result.get('error')
        blocked = blocked_by(result)
        tally[phase] = tally.get(phase, 0) + 1
        print('   -> %s%s  %.1f cm driven, %d/%d waypoint(s), %d rejection(s), %.1f s'
              % (phase, ': ' + error if error else '',
                 result.get('travelled_cm') or 0., result.get('completed_waypoints') or 0,
                 len(traj['waypoints_cm']), result.get('tracking_rejections') or 0,
                 time.time() - started))
        if blocked:
            print('   treating as blocked; next goal goes behind')

        entry = dict(cycle=cycle, seed=seed, shape=shape, rationale=why,
                     trajectory=traj, usable_bearings=len(room),
                     widest=widest, pack_v=pack, phase=phase, error=error,
                     travelled_cm=result.get('travelled_cm'),
                     completed_waypoints=result.get('completed_waypoints'),
                     tracking_rejections=result.get('tracking_rejections'),
                     tracking_rejection_streak=result.get('tracking_rejection_streak'),
                     pose_cm_deg=result.get('pose_cm_deg'),
                     execution_id=result.get('execution_id'),
                     blocked_after=blocked, at=time.time())
        record.write(json.dumps(entry) + '\n')
        record.flush()
        trial.narrate(root, 'Chain cycle %d (%s)' % (cycle, shape),
                      '%s. Scan found %d usable bearing(s); pack %.2f V.\n\n'
                      '```json\n%s\n```\n\nResult: **%s**%s, %.1f cm driven, '
                      '%d/%d waypoint(s), %d tracking rejection(s).'
                      % (why, len(room), pack, json.dumps(traj), phase,
                         ' - ' + error if error else '',
                         result.get('travelled_cm') or 0.,
                         result.get('completed_waypoints') or 0,
                         len(traj['waypoints_cm']),
                         result.get('tracking_rejections') or 0))

    record.close()
    print('\n--- chain summary (seed %d) ---' % seed)
    for phase in sorted(tally):
        print('%-12s %d' % (phase, tally[phase]))


if __name__ == '__main__':
    main()
