import os
import sys
import unittest

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'local_nav'))
from evaluate_autonomous_stack import evaluate


class AutonomousStackSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(1)
        cls.report = evaluate()

    def test_every_scenario_passes_without_api_or_hardware(self):
        self.assertEqual(self.report['hardware_calls'], 0)
        self.assertEqual(self.report['api_calls'], 0)
        self.assertTrue(self.report['passed'], self.report['cases'])

    def test_required_fault_and_mission_scenarios_are_present(self):
        names = {case['name'] for case in self.report['cases']}
        self.assertEqual(names, {
            'target_visible', 'target_absent', 'blocked_cable',
            'api_latency_failure', 'stale_reply', 'cancellation',
            'low_battery', 'multi_room_frontier_checkpoint', 'final_approach'})

    def test_no_failed_case_is_hidden_by_aggregate(self):
        failed = [case['name'] for case in self.report['cases'] if not case['passed']]
        self.assertEqual(failed, [])


if __name__ == '__main__':
    unittest.main()
