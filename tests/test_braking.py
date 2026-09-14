import os
import sys
import unittest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'local_nav'))
from braking import braking_distance, SettlingCheck


class BrakingTests(unittest.TestCase):
    def test_faster_motion_and_older_frames_stop_earlier(self):
        self.assertGreater(braking_distance(12, .04, .1), braking_distance(6, .04, .1))
        self.assertGreater(braking_distance(6, .10, .1), braking_distance(6, .01, .1))
        self.assertEqual(braking_distance(-2, .04, .1), 0)
        self.assertLessEqual(braking_distance(100, .1, .1), 2)

    def test_stale_or_nonfinite_state_rejected(self):
        for args in ((10, .2, .1), (10, -.1, .1), (10, .01, .3), (float('nan'), .01, .1)):
            with self.assertRaises(RuntimeError):
                braking_distance(*args)

    def test_crossing_target_while_coasting_is_not_arrival(self):
        check = SettlingCheck(5)
        for ts, z in ((1, 4.7), (1.1, 5), (1.2, 5.5), (1.3, 6.2)):
            self.assertIsNone(check.update(ts, 0, z))
        result = None
        for ts in (1.4, 1.5, 1.6, 1.7):
            result = check.update(ts, 0, 6.2)
        self.assertEqual(result['outcome'], 'overshot_goal')

    def test_settled_position_reports_short_or_reached(self):
        for z, outcome in ((3.5, 'stopped_short'), (4.7, 'goal_reached')):
            check = SettlingCheck(5)
            result = None
            for ts in (1, 1.1, 1.2, 1.3):
                result = check.update(ts, 0, z)
            self.assertEqual(result['outcome'], outcome)
            self.assertAlmostEqual(result['final_error_cm'], z - 5)

    def test_duplicate_frames_cannot_establish_settling(self):
        check = SettlingCheck(5)
        check.update(1, 0, 5)
        with self.assertRaises(RuntimeError):
            check.update(1, 0, 5)


class BrakingExecutorTests(unittest.TestCase):
    def run_route(self, fail_after_stop=False):
        import tempfile
        from unittest.mock import patch
        import numpy as np
        from route_geometry import StraightRoute
        from route_executor import execute
        p = dict(waypoints_cm=[1, 3, 5], inspected_free_rectangle_cm=[-20, -30, 20, 30],
                 obstacle_rectangles_cm=[], session_id='test', control_epoch=0,
                 captured_monotonic=100., image_path='mock.jpg')
        clock = [100.]
        times = iter([100. + n / 10. for n in range(12)])
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        translations = iter([0., 0., 1., 2., 1.3, .4, .1, 0., 0., 0., 0., 0.])
        commands = []

        def frame(*args, **kwargs):
            ts = next(times)
            clock[0] = ts + .02
            return image, ts, {}

        def motion(*args):
            if fail_after_stop and any(a == 'stop' for a, _ in commands):
                raise RuntimeError('Simulated post-stop tracking loss')
            return np.eye(2), np.array([0., -next(translations)]), {'residual_cm': .02}

        def call(action, **fields):
            commands.append((action, fields))
            return dict(session_id='test', control_epoch=0, healthy=True, motion_enabled=True,
                        motor={'output': [0, 0]})

        with tempfile.TemporaryDirectory() as directory, \
                patch('route_executor.time.monotonic', side_effect=lambda: clock[0]), \
                patch('point_controller.call', side_effect=call), \
                patch('point_controller.frame', side_effect=frame), \
                patch('point_controller.FloorTracker') as tracker, \
                patch('state_estimator.AttitudeTimeline') as timeline, \
                patch('cv2.imread', return_value=image):
            tracker.return_value.motion.side_effect = motion
            timeline.return_value.up = np.array([0., 0., 1.])
            result = execute(p, StraightRoute(p), directory + '/run.json', predictive_braking=True)
        first_stop = [a for a, _ in commands].index('stop')
        self.assertFalse(any(a == 'motors_hold' for a, _ in commands[first_stop:]))
        return result

    def test_brakes_early_then_measures_coasting_without_restarting(self):
        result = self.run_route()
        self.assertEqual(result['outcome'], 'goal_reached', result)
        self.assertAlmostEqual(result['brake_position_cm'][1], 4.3)
        self.assertAlmostEqual(result['final_position_cm'][1], 4.8)
        self.assertAlmostEqual(result['final_error_cm'], -.2)
        self.assertLess(result['powered_seconds'], .4)

    def test_failed_post_stop_tracking_does_not_claim_arrival(self):
        result = self.run_route(fail_after_stop=True)
        self.assertEqual(result['outcome'], 'stopped')
        self.assertIn('tracking loss', result['reason'])
        self.assertNotIn('final_position_cm', result)
