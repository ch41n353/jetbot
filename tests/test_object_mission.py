import os
import sys
import unittest
import cv2
import numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','local_nav'))
from target_tracker import TargetTracker,TargetLost
from simulate_object_mission import run_case
from maneuver_session import route_command


class ObjectMissionTests(unittest.TestCase):
    def test_continues_through_waypoints_for_more_than_four_seconds(self):
        run=run_case(dict(speed=12,coast=.05,seed=3),target=(0,100))
        self.assertEqual(run['result']['outcome'],'object_reached_estimate',run['result'].get('reason'))
        self.assertLess(abs(run['true_target_range_cm']-20),3)
        commands=run['commands']
        start=None;longest=0.
        for command in commands:
            if command['action']=='motors_hold' and command['fields']['left']!=0:
                if start is None:start=command['time']
                longest=max(longest,command['time']-start)
            elif command['action']=='stop' or command['action']=='motors_hold':start=None
        self.assertGreater(longest,6)
        self.assertEqual(run['clearance_violations'],[])
        self.assertEqual(run['watchdog_stops'],0)
        self.assertEqual(run['final_motor_output'],[0,0])

    def test_angled_object_is_reached_without_remote_alignment(self):
        run=run_case(dict(speed=12,coast=.05,seed=3,pivot_cm=7.5),target=(-17,60))
        self.assertEqual(run['result']['outcome'],'object_reached_estimate',run['result'].get('reason'))
        self.assertLess(abs(run['true_target_range_cm']-20),3)
        self.assertEqual(run['clearance_violations'],[])
        self.assertEqual(run['result']['intermediate_model_calls'],0)

    def test_temporary_occlusion_stops_reacquires_and_finishes_locally(self):
        run=run_case(dict(speed=12,coast=.05,seed=3),occlude=(1.,1.5))
        self.assertEqual(run['result']['outcome'],'object_reached_estimate',run['result'].get('reason'))
        self.assertGreaterEqual(run['result']['local_recoveries'],1)
        self.assertLessEqual(run['result']['local_recoveries'],2)
        self.assertIn('target_reacquired',[e['event'] for e in run['result']['events']])
        self.assertNotIn('remote_planner_required',[e['event'] for e in run['result']['events']])

    def test_persistent_target_loss_escalates_and_stops(self):
        run=run_case(dict(speed=12,coast=.05,seed=3),occlude=(1.,20.))
        self.assertEqual(run['result']['outcome'],'stopped')
        self.assertIn('recovery exhausted',run['result']['reason'])
        self.assertEqual(run['final_motor_output'],[0,0])
        self.assertLess(run["result"]["elapsed_seconds"],10)  # 6 recoveries, 6 s window

    def test_cancel_cannot_be_overwritten_by_local_recovery(self):
        run=run_case(dict(speed=12,coast=.05,seed=3,cancel_after=1.2),occlude=(1.,1.5))
        self.assertEqual(run['result']['outcome'],'stopped')
        self.assertIn('cancelled',run['result']['reason'])
        self.assertEqual(run['final_motor_output'],[0,0])

    def test_ambiguous_reacquisition_is_rejected(self):
        image=np.full((240,320,3),128,np.uint8)
        patch=np.random.RandomState(5).randint(0,255,(50,30,3)).astype(np.uint8)
        image[100:150,120:150]=patch
        tracker=TargetTracker(image,[120,100,150,150])
        ambiguous=image.copy();ambiguous[100:150,180:210]=patch
        with self.assertRaises(TargetLost):tracker.reacquire(ambiguous)

    def test_mission_dispatch_is_explicit_and_exclusive(self):
        self.assertTrue(route_command('plan.json','log.json',mission=True)[1].endswith('object_mission.py'))
        with self.assertRaises(ValueError):route_command('plan.json','log.json',mission=True,spatial=True)


if __name__=='__main__':unittest.main()
