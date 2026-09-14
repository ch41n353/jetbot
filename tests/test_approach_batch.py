import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'local_nav'))
from approach_batch import BatchRoute
from simulate_route import run_batch_case


class BatchTests(unittest.TestCase):
    def plan(self):
        return dict(approach_distance_cm=30,
                    inspected_free_rectangle_cm=[-22,-30,22,46],
                    obstacle_rectangles_cm=[])

    def test_full_route_and_rear_obstacle_rejected(self):
        for obstacle in ([-1,35,1,37], [8,-10,9,-8]):
            plan = self.plan()
            plan['obstacle_rectangles_cm'] = [obstacle]
            with self.assertRaises(ValueError):
                BatchRoute(plan)

    def test_pose_rotation_cannot_infer_free_space(self):
        batch = BatchRoute(self.plan())
        capture = dict(session_id='test',control_epoch=0,captured_monotonic=0,image_path='test.jpg')
        self.assertIsNotNone(batch.segment(capture, [.5,15], 1))
        with self.assertRaises(RuntimeError):
            batch.segment(capture, [3,15], 0)

    def test_production_batch_finishes_without_planner(self):
        case = run_batch_case(dict(speed=12,coast=.05,seed=3))
        self.assertEqual(case['result']['outcome'], 'approach_distance_reached_estimate', case['result'])
        self.assertEqual(len(case['result']['segments']), 2)
        self.assertEqual(case['result']['intermediate_planner_calls'], 0)
        self.assertLess(abs(case['true_error_cm']), 2)

    def test_external_stop_between_segments_prevents_restart(self):
        case = run_batch_case(dict(speed=12,coast=.05,seed=3), cancel_between_segments=True)
        self.assertEqual(case['result']['outcome'], 'stopped')
        self.assertIn('cancelled between', case['result']['reason'])
        self.assertEqual(len(case['result']['segments']), 1)
        commands = case['motor_commands']
        first_stop = next(i for i,c in enumerate(commands) if c['action']=='stop')
        self.assertFalse(any(c['action']=='motors_hold' for c in commands[first_stop:]))

    def test_uncommanded_movement_between_segments_prevents_restart(self):
        case = run_batch_case(dict(speed=12,coast=.05,seed=3), displacement_between_segments=1.)
        self.assertEqual(case['result']['outcome'], 'stopped')
        self.assertIn('moved since preview', case['result']['segments'][-1]['result']['reason'])
        commands = case['motor_commands']
        first_stop = next(i for i,c in enumerate(commands) if c['action']=='stop')
        self.assertFalse(any(c['action']=='motors_hold' for c in commands[first_stop:]))
