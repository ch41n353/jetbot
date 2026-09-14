import os
import sys
import unittest

sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','local_nav'))
from simulate_spatial import run_spatial_case
from maneuver_session import route_command


class SpatialExecutionTests(unittest.TestCase):
    def test_full_turn_and_drive_loop_with_unmeasured_pivot_error(self):
        case = run_spatial_case(dict(speed=12,coast=.05,seed=3,pivot_cm=5.))
        result = case['result']
        self.assertEqual(result['outcome'],'target_reached_estimate',result.get('reason'))
        self.assertLess(case['true_goal_error_cm'],4.)
        self.assertEqual(result['intermediate_model_calls'],0)
        self.assertEqual(len(case['timeline_initializations']),1)
        self.assertEqual(case['final_motor_output'],[0,0])
        self.assertEqual(set(a['plan']['kind'] for a in result['actions']),{'drive','turn'})
        self.assertLessEqual(len(result['actions']),12)

    def test_external_stop_between_actions_prevents_restart(self):
        case = run_spatial_case(dict(speed=12,coast=.05,seed=3,pivot_cm=7.5),cancel_between=True)
        self.assertEqual(case['result']['outcome'],'stopped')
        self.assertIn('cancelled between',case['result']['reason'])
        self.assertEqual(len(case['result']['actions']),1)
        commands=case['motor_commands']
        first_stop=next(i for i,c in enumerate(commands) if c['action']=='stop')
        self.assertFalse(any(c['action']=='motors_hold' for c in commands[first_stop:]))

    def test_external_stop_during_turn_prevents_pulse_restart(self):
        case = run_spatial_case(dict(speed=12,coast=.05,seed=3,pivot_cm=7.5,cancel_after=.15),goal=(30,0))
        self.assertEqual(case['result']['outcome'],'stopped')
        self.assertEqual(case['final_motor_output'],[0,0])
        self.assertEqual(len(case['result']['actions']),1)

    def test_spatial_session_dispatch_and_ambiguous_mode_rejected(self):
        command=route_command('plan.json','result.json',spatial=True)
        self.assertTrue(any(v.endswith('spatial_executor.py') for v in command))
        self.assertNotIn('--feature-budget',command)
        with self.assertRaises(ValueError):
            route_command('plan.json','result.json',spatial=True,batch=True)

    def test_out_and_back_mission_without_model_between_targets(self):
        case=run_spatial_case(dict(speed=12,coast=.05,seed=3,pivot_cm=7.5),goals=[[0,20],[0,0]])
        result=case['result']
        self.assertEqual(result['outcome'],'target_reached_estimate',result.get('reason'))
        self.assertEqual(len(result['reached_targets']),2)
        self.assertEqual(result['intermediate_model_calls'],0)
        self.assertLess(case['true_goal_error_cm'],4)


if __name__=='__main__':
    unittest.main()
