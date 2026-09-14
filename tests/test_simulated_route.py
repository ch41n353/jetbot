import os
import sys
import unittest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'local_nav'))
from simulate_route import run_case


class SimulatedRouteTests(unittest.TestCase):
    def test_tracking_failure_never_reports_arrival(self):
        r = run_case(dict(frame_fault_after=.25))
        self.assertEqual(r['result']['outcome'], 'stopped')
        self.assertIn('failure', r['result']['reason'])
        self.assertFalse(r['renewed_after_stop'])

    def test_stall_has_bounded_motor_interval(self):
        r = run_case(dict(stall=True))
        self.assertEqual(r['result']['outcome'], 'stopped')
        self.assertIn('progress', r['result']['reason'])
        self.assertLess(r['result']['powered_seconds'], .6)
        self.assertAlmostEqual(r['true_final_position_cm'][1], 0)

    def test_external_stop_cancels_next_renewal(self):
        r = run_case(dict(cancel_after=.20))
        self.assertEqual(r['result']['outcome'], 'stopped')
        self.assertIn('cancelled', r['result']['reason'])
        self.assertFalse(r['renewed_after_stop'])

    def test_blocked_camera_trips_independent_lease(self):
        r = run_case(dict(blocked_frame_after=.20))
        self.assertEqual(r['result']['outcome'], 'stopped')
        self.assertGreaterEqual(r['watchdog_stops'], 1)
        self.assertFalse(r['renewed_after_stop'])

    def test_low_texture_prevents_any_power_command(self):
        r = run_case(dict(low_texture=True), feature_budget=125)
        self.assertEqual(r['result']['outcome'], 'stopped')
        self.assertEqual(r['max_power'], 0)
        self.assertIsNone(r['result']['powered_seconds'])

    def test_lower_feature_budget_reaches_rendered_goal(self):
        r = run_case(dict(seed=404), feature_budget=125, target_cm=10.)
        self.assertEqual(r['result']['outcome'], 'goal_reached', r['result'])
        self.assertLess(abs(r['true_error_cm']), 1)
        self.assertFalse(r['renewed_after_stop'])
