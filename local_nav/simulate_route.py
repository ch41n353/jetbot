#!/usr/bin/env python3
"""Offline route-controller simulation with a motor plant and synthetic carpet.

Uses the production route executor, tracker and planar estimator. Replaces robot
socket calls, camera frames, IMU attitude and clock inside a scoped mock context.
No hardware is opened. Simulation parameters are hypotheses, not calibration.
"""
import argparse
import base64
import json
import math
import os
import tempfile
from unittest.mock import patch
import cv2
import numpy as np
from point_controller import ROOT, FloorTracker
from route_executor import execute, load_json
from route_geometry import StraightRoute


class Plant:
    def __init__(self, speed=12., coast=.08, camera_hz=12., latency=.025,
                 compute=.045, stall=False, seed=1, noise=0.,
                 cancel_after=None, frame_fault_after=None, blocked_frame_after=None,
                 height_scale=1., imu_yaw_bias_dps=0., low_texture=False, exposure_jump=0.):
        self.time = 100.
        self.x = self.z = self.yaw = self.velocity = self.yaw_rate = 0.
        self.left = self.right = 0.
        self.expiry = 0.
        self.speed, self.coast, self.period = speed, coast, 1. / camera_hz
        self.latency, self.compute, self.stall, self.noise = latency, compute, stall, noise
        self.rng = np.random.RandomState(seed)
        self.commands = []
        self.max_power = 0.
        self.powered_start = None
        self.last_capture = self.time - self.period
        self.stop_position = None
        self.watchdog_stops = 0
        self.sequence = 0
        self.imu_yaw_bias_dps = imu_yaw_bias_dps
        self.exposure_jump = exposure_jump
        self.token = dict(session_id='simulation', control_epoch=0)
        self.external_cancel_at = None
        self.cancel_after = cancel_after
        self.frame_fault_after = frame_fault_after
        self.blocked_frame_after = blocked_frame_after
        self.block_injected = False
        self.cancelled = False
        self.initial_up = np.array([0., 1., 0.])
        p = load_json(os.path.join(ROOT, 'calibration/floor_geometry.json'))
        i = load_json(p['intrinsics_path'])
        pitch = math.radians(p['pitch_degrees'])
        self.down = np.array([0., math.cos(pitch), math.sin(pitch)])
        xy = np.indices((480, 640)).transpose(1, 2, 0)[:, :, ::-1].astype(float)
        rays = cv2.fisheye.undistortPoints(xy.reshape(-1, 1, 2), np.array(i['K']), np.array(i['D'])).reshape(-1, 2)
        rays = np.column_stack((rays, np.ones(len(rays))))
        forward = np.array([0., -math.sin(pitch), math.cos(pitch)])
        denominator = rays @ self.down
        self.floor_valid = denominator > .08
        safe_denominator = np.maximum(denominator, .08)
        self.floor_x = (rays[:, 0] * p['camera_height_cm'] * height_scale / safe_denominator).reshape(480, 640)
        self.floor_z = ((rays @ forward) * p['camera_height_cm'] * height_scale / safe_denominator).reshape(480, 640)
        # Reproducible random carpet with useful structure at several scales.
        texture_rng = np.random.RandomState(seed)
        tex = texture_rng.randint(45, 210, (1536, 1536)).astype(np.uint8)
        self.texture = cv2.GaussianBlur(tex, (3, 3), .55)
        if low_texture:
            self.texture.fill(128)

    def advance(self, duration):
        destination = self.time + duration
        while self.time < destination - 1e-9:
            dt = min(.002, destination - self.time)
            if self.external_cancel_at is not None and self.time >= self.external_cancel_at and not self.cancelled:
                self.token['control_epoch'] += 1
                self.left = self.right = 0.
                self.cancelled = True
            if self.time >= self.expiry and (self.left or self.right):
                self.left = self.right = 0.
                self.watchdog_stops += 1
            desired_v = 0. if self.stall else self.speed * (self.left + self.right) / .32
            desired_yaw = 0. if self.stall else math.radians(70) * (self.left - self.right) / .28
            tau = .07 if desired_v else self.coast
            self.velocity += (desired_v - self.velocity) * (1 - math.exp(-dt / tau))
            self.yaw_rate += (desired_yaw - self.yaw_rate) * (1 - math.exp(-dt / .05))
            self.yaw += self.yaw_rate * dt
            self.x += math.sin(self.yaw) * self.velocity * dt
            self.z += math.cos(self.yaw) * self.velocity * dt
            self.time += dt

    def render(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        x = self.x + c * self.floor_x + s * self.floor_z
        z = self.z - s * self.floor_x + c * self.floor_z
        gray = cv2.remap(self.texture, (768 + 8 * x).astype(np.float32),
                         (384 + 8 * z).astype(np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
        gray.reshape(-1)[~self.floor_valid] = 128
        if self.exposure_jump:
            gray = np.clip(gray.astype(float) + self.exposure_jump * (1 if self.sequence % 2 else -1), 0, 255).astype(np.uint8)
        if self.noise:
            gray = np.clip(gray.astype(float) + self.rng.normal(0, self.noise, gray.shape), 0, 255).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def frame(self, timeline, settled=False):
        elapsed = 0. if self.powered_start is None else self.time - self.powered_start
        if self.frame_fault_after is not None and elapsed >= self.frame_fault_after:
            raise RuntimeError('Simulated camera/IMU failure')
        if self.blocked_frame_after is not None and elapsed >= self.blocked_frame_after and not self.block_injected:
            self.advance(.35)
            self.block_injected = True
        self.advance(max(0., self.last_capture + self.period - self.time))
        self.last_capture = self.time
        ts = self.time
        image = self.render()
        attitude = dict(time=ts, down_camera=self.down,
                        yaw=self.yaw + math.radians(self.imu_yaw_bias_dps) * (ts - 100.),
                        variance=math.radians(.25) ** 2, extrapolation_ms=0.)
        self.advance(self.latency)
        self.sequence += 1
        return image, ts, attitude

    def call(self, action, **fields):
        self.commands.append(dict(time=self.time, action=action, fields=fields))
        if action == 'status':
            return dict(healthy=True, motion_enabled=True, motor={'output': [self.left, self.right]}, **self.token)
        if action == 'stop':
            if self.stop_position is None and self.powered_start is not None:
                self.stop_position = [self.x, self.z]
            self.left = self.right = 0.
            self.token['control_epoch'] += 1
            return dict(accepted=True, **self.token)
        if action == 'motors_hold':
            if any(fields.get(k) != v for k, v in self.token.items()):
                raise RuntimeError('Control cancelled or service restarted; replan')
            self.left, self.right = fields['left'], fields['right']
            self.max_power = max(self.max_power, abs(self.left), abs(self.right))
            self.expiry = self.time + .2
            if self.powered_start is None:
                self.powered_start = self.time
                if self.cancel_after is not None:
                    self.external_cancel_at = self.time + self.cancel_after
            return dict(accepted=True, **self.token)
        raise RuntimeError('Simulation prohibits socket action: ' + action)


def run_case(parameters, predictive=True, feature_budget=250, target_cm=5.):
    plant = Plant(**parameters)
    target = target_cm
    profile = load_json(os.path.join(ROOT, 'calibration/floor_geometry.json'))
    tracker = FloorTracker(profile, load_json(profile['intrinsics_path']), max_features=feature_budget)
    original_motion = tracker.motion

    def motion(*args):
        try:
            return original_motion(*args)
        finally:
            plant.advance(plant.compute)

    class Timeline:
        def __init__(self, mount):
            self.up = plant.initial_up.copy()

    with tempfile.TemporaryDirectory(prefix='jetbot-simulation-') as directory:
        path = os.path.join(directory, 'anchor.png')
        cv2.imwrite(path, plant.render())
        plan = dict(waypoints_cm=[v for v in (1., 3.) if v < target] + [target], inspected_free_rectangle_cm=[-20, -30, 20, 30],
                    obstacle_rectangles_cm=[], captured_monotonic=plant.time, image_path=path, **plant.token)
        with patch('point_controller.call', side_effect=plant.call), \
                patch('point_controller.frame', side_effect=plant.frame), \
                patch('point_controller.FloorTracker', return_value=tracker), \
                patch.object(tracker, 'motion', side_effect=motion), \
                patch('state_estimator.AttitudeTimeline', Timeline), \
                patch('route_executor.time.monotonic', side_effect=lambda: plant.time), \
                patch('route_executor.time.sleep', side_effect=plant.advance):
            result = execute(plan, StraightRoute(plan), os.path.join(directory, 'result.json'), predictive, feature_budget)
        # Independently integrate remaining coast to compare true final positions.
        plant.advance(1.)
    first_stop = next((i for i, c in enumerate(plant.commands) if c['action'] == 'stop'), len(plant.commands))
    return dict(parameters=parameters, predictive=predictive, feature_budget=feature_budget, target_cm=target, result=result,
                true_final_position_cm=[plant.x, plant.z], true_error_cm=plant.z-target,
                true_stop_position_cm=plant.stop_position, max_power=plant.max_power,
                renewed_after_stop=any(c['action'] == 'motors_hold' for c in plant.commands[first_stop:]),
                watchdog_stops=plant.watchdog_stops)


def run_batch_case(parameters, target_cm=30., cancel_between_segments=False, displacement_between_segments=0.):
    """Same independent plant, production batch/segment controllers, no hardware."""
    from approach_batch import execute_batch
    plant = Plant(**parameters)
    profile = load_json(os.path.join(ROOT, 'calibration/floor_geometry.json'))
    tracker = FloorTracker(profile, load_json(profile['intrinsics_path']), max_features=125)
    original_motion = tracker.motion

    def motion(*args):
        try:
            return original_motion(*args)
        finally:
            plant.advance(plant.compute)

    def call(action, **fields):
        if action != 'observation':
            return plant.call(action, **fields)
        if cancel_between_segments:
            plant.token['control_epoch'] += 1
        plant.z += displacement_between_segments
        image, timestamp, _ = plant.frame(None)
        encoded = cv2.imencode('.jpg', image)[1].tobytes()
        return dict(time=timestamp, jpeg_base64=base64.b64encode(encoded).decode(), **plant.token)

    class Timeline:
        def __init__(self, mount):
            self.up = plant.initial_up.copy()

    with tempfile.TemporaryDirectory(prefix='jetbot-batch-simulation-') as directory:
        path = os.path.join(directory, 'anchor.png')
        cv2.imwrite(path, plant.render())
        plan = dict(approach_distance_cm=target_cm,
                    inspected_free_rectangle_cm=[-22, -30, 22, target_cm+16],
                    obstacle_rectangles_cm=[], captured_monotonic=plant.time,
                    image_path=path, **plant.token)
        with patch('point_controller.call', side_effect=call), \
                patch('point_controller.frame', side_effect=plant.frame), \
                patch('point_controller.FloorTracker', return_value=tracker), \
                patch.object(tracker, 'motion', side_effect=motion), \
                patch('state_estimator.AttitudeTimeline', Timeline), \
                patch('route_executor.time.monotonic', side_effect=lambda: plant.time), \
                patch('route_executor.time.sleep', side_effect=plant.advance):
            result = execute_batch(plan, os.path.join(directory, 'batch.json'))
        for segment in result['segments']:
            with open(segment.pop('log_path')) as source:
                segment['result'] = json.load(source)
        plant.advance(1.)
    return dict(parameters=parameters, result=result, true_final_position_cm=[plant.x, plant.z],
                true_error_cm=plant.z-target_cm, motor_commands=plant.commands,
                cancel_between_segments=cancel_between_segments)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cases', type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.cases <= 500:
        parser.error('cases must be 1–500')
    rng = np.random.RandomState(20260914)
    cases = []
    for seed in range(args.cases):
        parameters = dict(speed=float(rng.uniform(8, 16)), coast=float(rng.uniform(.035, .12)),
                          camera_hz=float(rng.uniform(10, 15)), latency=float(rng.uniform(.005, .035)),
                          compute=.045, seed=seed + 1, noise=float(rng.uniform(0, 2)))
        for predictive in (False, True):
            case = run_case(parameters, predictive)
            cases.append(case)
            print(json.dumps(dict(case=seed, predictive=predictive, outcome=case['result']['outcome'],
                                  true_error_cm=case['true_error_cm'])), flush=True)
        with open(args.output + '.tmp', 'w') as out:
            json.dump(dict(cases=cases, complete=seed == args.cases - 1), out, indent=2)
        os.replace(args.output + '.tmp', args.output)


if __name__ == '__main__':
    main()
