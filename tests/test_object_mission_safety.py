"""Deterministic state-machine faults; no camera rendering or hardware access."""
import math
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch
import cv2
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'local_nav'))
import object_mission
from target_tracker import TargetLost


class MissionHarness:
    def __init__(self, outcomes=None, target=(0., 30.), velocity=0., cancel_at=None,
                 yaw_step=0., stop_error=False, clear_drive=True, search_delay=0., search_drift=0.):
        self.time = 0.
        self.tick = -1
        self.commands = []
        self.output = [0., 0.]
        self.outcomes = outcomes or {}
        self.target = np.array(target)
        self.velocity = velocity
        self.cancel_at = cancel_at
        self.yaw_step = yaw_step
        self.yaw = 0.
        self.stop_error = stop_error
        self.clear_drive = clear_drive
        self.search_delay = search_delay
        self.search_drift = search_drift
        self.search_done = False
        self.image = np.zeros((32, 32, 3), np.uint8)
        self.token = dict(session_id='test', control_epoch=1)

    def call(self, action, **fields):
        self.commands.append((self.tick, action, fields))
        if action == 'status':
            return dict(healthy=True, motion_enabled=True, motor=dict(output=self.output),
                        power={}, **self.token)
        if action == 'stop':
            self.output = [0., 0.]
            if self.stop_error:
                raise RuntimeError('injected stop failure')
        if action == 'motors_hold':
            if self.cancel_at is not None and self.tick >= self.cancel_at:
                self.output = [0., 0.]
                raise RuntimeError('controller cancelled')
            self.output = [fields['left'], fields['right']]
        return {}

    def frame(self, *args, **kwargs):
        self.tick += 1
        self.time += .1
        return self.image, self.time, {}

    def run(self):
        harness = self
        class Floor:
            def __init__(self, *args, **kwargs): pass
            def motion(self, *args):
                angle = math.radians(harness.yaw_step if harness.tick > 0 else 0.)
                r = np.array([[math.cos(angle), -math.sin(angle)],
                              [math.sin(angle), math.cos(angle)]])
                t = np.array([harness.search_drift if harness.search_done else 0., 0.])
                return r, t, {}
            def ground(self, *args):
                a = math.radians(harness.yaw)
                x, z = harness.target
                return np.array([[math.cos(a)*x-math.sin(a)*z,
                                  math.sin(a)*x+math.cos(a)*z]])
        class State:
            def update(self, *args):
                harness.yaw += harness.yaw_step
                return dict(position_cm=[0., 0.], velocity_cm_s=[harness.velocity, 0.],
                            yaw_degrees=harness.yaw)
        class Target:
            box = [0, 0, 20, 20]
            def __init__(self, *args, **kwargs): pass
            def update(self, *args):
                if harness.outcomes.get(harness.tick) == 'lost':
                    raise TargetLost('injected target loss')
                return dict(base_pixel=[0, 0], confidence=.99)
            reacquire = update
        class Timeline:
            def __init__(self, *args): self.up = np.array([0., 0., 1.])
        class Planner:
            def __init__(self, *args, **kwargs): pass
            def clear(self, *args): return True
            def search(self, *args, **kwargs):
                harness.time += harness.search_delay
                harness.search_done = True
                return dict(outcome='route_found', actions=[dict(kind='drive', value=10,
                           predicted_end_pose=[*harness.target, 0.])])
        def load(path):
            if path.endswith('robot_footprint.json'):
                return dict(width_cm=12, length_cm=15, camera_location='front_center')
            return dict(intrinsics_path='unused')
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            patches = [patch('object_mission.validate', return_value=(self.image, 20.)),
                       patch('object_mission.TargetTracker', Target),
                       patch('object_mission.SpatialPlanner', Planner),
                       patch('object_mission.drive_clear', return_value=self.clear_drive),
                       patch('object_mission.time.monotonic', side_effect=lambda: self.time),
                       patch('point_controller.call', side_effect=self.call),
                       patch('point_controller.frame', side_effect=self.frame),
                       patch('point_controller.FloorTracker', Floor),
                       patch('state_estimator.PlanarState', State),
                       patch('state_estimator.AttitudeTimeline', Timeline),
                       patch('route_executor.load_json', side_effect=load)]
            for item in patches: stack.enter_context(item)
            plan = dict(captured_monotonic=0., target_box=Target.box,
                        obstacle_rectangles_cm=[], **self.token)
            self.result = object_mission.execute(plan, os.path.join(directory, 'result.json'))
        return self.result


class ObjectMissionSafetyTests(unittest.TestCase):
    def test_projection_limit_accounts_for_target_footprint_but_is_capped(self):
        # radius term: radius+4, capped at 20
        self.assertEqual(object_mission.target_projection_limit([0,50],7),11.)
        self.assertEqual(object_mission.target_projection_limit([0,50],20),20.)
        # range term dominates further out, where projection error grows
        self.assertAlmostEqual(object_mission.target_projection_limit([0,80],1),14.4)
        # and it is still bounded, not open-ended
        self.assertEqual(object_mission.target_projection_limit([0,10],1),6.)

    def test_recovery_requires_consecutive_observations(self):
        # Two good reads, a loss, then two good reads must not renew drive.
        harness = MissionHarness(outcomes={1: 'lost', 4: 'lost'}, cancel_at=7)
        result = harness.run()
        self.assertIn('cancelled', result['reason'])
        self.assertNotIn('target_reacquired', [e['event'] for e in result['events']])
        self.assertFalse(any(action == 'motors_hold' and fields['left'] != 0
                             for _, action, fields in harness.commands))
        self.assertEqual(harness.output, [0., 0.])

    def test_successful_reads_cannot_extend_recovery_deadline(self):
        # Confidence returns, but a moving robot never satisfies stationary recovery.
        harness = MissionHarness(outcomes={1: 'lost'}, velocity=1.)
        result = harness.run()
        self.assertIn('recovery deadline', result['reason'])
        self.assertLess(result['elapsed_seconds'], 2.5)
        self.assertNotIn('target_reacquired', [e['event'] for e in result['events']])
        self.assertEqual(harness.output, [0., 0.])

    def test_wrong_turn_direction_stops_before_renewing(self):
        harness = MissionHarness(target=(20., 30.))
        # Introduce wrong-way rotation only after the first turning command.
        original_frame = harness.frame
        def frame(*args, **kwargs):
            if any(action == 'motors_hold' and fields['left'] * fields['right'] < 0
                   for _, action, fields in harness.commands):
                harness.yaw_step = -4.
            return original_frame(*args, **kwargs)
        harness.frame = frame
        result = harness.run()
        self.assertIn('wrong direction', result['reason'])
        powered = [(tick, fields) for tick, action, fields in harness.commands
                   if action == 'motors_hold' and fields['left'] != 0]
        self.assertEqual(len(powered), 1)
        self.assertEqual(harness.output, [0., 0.])

    def test_blocked_final_approach_has_bounded_local_retries(self):
        harness = MissionHarness(clear_drive=False)
        result = harness.run()
        self.assertIn('Forward clearance recovery exhausted', result['reason'])
        # Bounded, not unbounded: the clearance retry ceiling is 10 attempts.
        self.assertLess(result["elapsed_seconds"], 4.)
        self.assertEqual(harness.output, [0., 0.])
        self.assertFalse(any(action == 'motors_hold' and fields['left'] != 0
                             for _, action, fields in harness.commands))

    def test_stationary_search_latency_does_not_trigger_powered_gap_guard(self):
        harness = MissionHarness(target=(0., 40.), search_delay=.3)
        result = harness.run()
        self.assertNotIn('camera gap', result['reason'])
        self.assertTrue(any(action == 'motors_hold' and fields['left'] != 0
                            for _, action, fields in harness.commands))
        self.assertEqual(harness.output, [0., 0.])

    def test_excessive_search_gap_stops_before_power(self):
        harness = MissionHarness(target=(0., 40.), search_delay=.6)
        result = harness.run()
        self.assertIn('Stopped search refresh exceeded600ms', result['reason'])
        self.assertFalse(any(action == 'motors_hold' and fields['left'] != 0
                             for _, action, fields in harness.commands))
        self.assertEqual(harness.output, [0., 0.])

    def test_motion_during_stationary_search_is_rejected(self):
        harness = MissionHarness(target=(0., 40.), search_delay=.3, search_drift=.2)
        result = harness.run()
        self.assertIn('Robot moved during stopped search', result['reason'])
        self.assertFalse(any(action == 'motors_hold' and fields['left'] != 0
                             for _, action, fields in harness.commands))
        self.assertEqual(harness.output, [0., 0.])

    def test_stop_failure_cannot_report_reached(self):
        harness = MissionHarness(target=(0., 20.), stop_error=True)
        result = harness.run()
        self.assertEqual(result['outcome'], 'stopped')
        self.assertTrue(result['requires_planner'])
        self.assertIn('stop failure', result['stop_error'])


if __name__ == '__main__':
    cv2.setNumThreads(1)
    unittest.main()
