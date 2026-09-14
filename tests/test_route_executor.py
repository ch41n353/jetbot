import os
import sys
import unittest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'local_nav'))
from core import ControlGeneration
from route_geometry import StraightRoute, ProgressGuard


def plan():
    return dict(waypoints_cm=[5, 10, 15], inspected_free_rectangle_cm=[-20, -30, 20, 30],
                obstacle_rectangles_cm=[])


class RouteTests(unittest.TestCase):
    def test_all_waypoints_share_one_sweep(self):
        route = StraightRoute(plan())
        self.assertEqual(route.corridor, [-16, -24, 16, 28])
        route.check_pose(.8, 12, 4)

    def test_offscreen_side_and_rear_obstacles_retained(self):
        for obstacle in ([7, -8, 9, -5], [-2, -20, 2, -18], [-1, 7.01, 1, 7.02]):
            p = plan()
            p['obstacle_rectangles_cm'] = [obstacle]
            with self.assertRaises(ValueError):
                StraightRoute(p)

    def test_front_braking_space_required(self):
        p = plan()
        p['inspected_free_rectangle_cm'][3] = 20
        with self.assertRaises(ValueError):
            StraightRoute(p)

    def test_reverse_checks_rear_body_and_rejects_wrong_direction(self):
        p = plan()
        p.update(travel_direction='reverse', inspected_free_rectangle_cm=[-20,-50,20,10])
        route = StraightRoute(p)
        self.assertEqual(route.corridor, [-16,-43,16,9])
        route.check_pose(0,-10,0)
        with self.assertRaises(RuntimeError):
            route.check_pose(0,1,0)
        p['obstacle_rectangles_cm'] = [[-2,-40,2,-38]]
        with self.assertRaises(ValueError):
            StraightRoute(p)

    def test_narrow_or_unknown_space_rejected(self):
        p = plan()
        p['inspected_free_rectangle_cm'][0] = -10
        with self.assertRaises(ValueError):
            StraightRoute(p)

    def test_invalid_waypoints(self):
        for values in ([], [5, 4], [-1], [16], [float('nan')], [True]):
            p = plan()
            p['waypoints_cm'] = values
            with self.assertRaises(ValueError):
                StraightRoute(p)

    def test_pose_escape_stops(self):
        route = StraightRoute(plan())
        for pose in ((1.1, 5, 0), (0, 5, 6), (0, -1, 0), (0, 20, 0), (float('nan'), 0, 0)):
            with self.assertRaises((RuntimeError, ValueError)):
                route.check_pose(*pose)

    def test_stall_and_healthy_progress(self):
        guard = ProgressGuard()
        with self.assertRaises(RuntimeError):
            guard.check(.41, .1)
        guard = ProgressGuard()
        for elapsed in (.1, .2, .4, .6, .8):
            guard.check(elapsed, elapsed * 10)
        with self.assertRaises(RuntimeError):
            guard.check(1.3, 8)


class CancellationTests(unittest.TestCase):
    def test_repeated_renewal_preserves_token(self):
        generation = ControlGeneration()
        token = generation.token()
        for _ in range(100):
            generation.validate(token)
        self.assertEqual(token, generation.token())

    def test_stop_and_legacy_command_cancel_route(self):
        for legacy in (False, True):
            generation = ControlGeneration()
            token = generation.token()
            if legacy:
                generation.validate({})
            else:
                generation.invalidate()
            with self.assertRaises(ValueError):
                generation.validate(token)

    def test_restart_and_partial_token_rejected(self):
        generation = ControlGeneration()
        for token in (ControlGeneration().token(), {'control_epoch': 0}):
            with self.assertRaises(ValueError):
                generation.validate(token)


class ExecutorTests(unittest.TestCase):
    def test_multiple_waypoints_without_intermediate_stop(self):
        import time
        import tempfile
        from unittest.mock import patch
        import numpy as np
        from route_executor import execute
        p = plan()
        p.update(waypoints_cm=[1, 2, 3], session_id='test', control_epoch=0,
                 captured_monotonic=time.monotonic(), image_path='mock.jpg')
        route = StraightRoute(p)
        commands = []
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_times = iter([1, 1.05, 1.10, 1.15])
        motion = iter([(np.eye(2), np.zeros(2), {})] +
                      [(np.eye(2), np.array([0., -1.]), {'residual_cm': .02})] * 3)

        def call(action, **fields):
            commands.append((action, fields))
            return dict(session_id='test', control_epoch=0, healthy=True, motion_enabled=True,
                        motor={'output': [0, 0]})

        with tempfile.NamedTemporaryFile() as log, \
                patch('point_controller.call', side_effect=call), \
                patch('point_controller.frame', side_effect=lambda *a, **k: (image, next(frame_times), {})), \
                patch('point_controller.FloorTracker') as tracker, \
                patch('state_estimator.AttitudeTimeline') as timeline, \
                patch('cv2.imread', return_value=image):
            tracker.return_value.motion.side_effect = lambda *a: next(motion)
            timeline.return_value.up = np.array([0., 0., 1.])
            timeline.return_value.route_reference_up = None
            result = execute(p, route, log.name)
        self.assertEqual(result['outcome'], 'distance_threshold_reached', result)
        self.assertEqual(result['passed_waypoints_cm'], [1, 2, 3])
        self.assertEqual([a for a, _ in commands], ['status', 'motors_hold', 'motors_hold', 'stop'])
        self.assertTrue(all(fields['control_epoch'] == 0 for a, fields in commands if a == 'motors_hold'))
