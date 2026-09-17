"""End-to-end geometry alignment with a biased synthetic camera attitude."""
import os
import sys
import unittest
import cv2
import numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','local_nav'))
from simulate_object_mission import run_case
from object_mission import approach_goal


class AlignedMissionTests(unittest.TestCase):
    def test_shallow_goal_reaches_standoff_inside_heading_scope(self):
        target=np.array([-18.,100.]);goal=approach_goal(target,20,True)
        self.assertAlmostEqual(np.linalg.norm(target-goal),20.)
        self.assertLessEqual(abs(np.degrees(np.arctan2(goal[0],goal[1]))),7.00001)
        with self.assertRaises(ValueError):approach_goal([-60.,60.],20,True)

    def test_uncompensated_camera_bias_stops_before_motion(self):
        run=run_case(dict(speed=12,coast=.05,seed=3),target=(0,60),camera_pitch_bias_degrees=4.)
        self.assertEqual(run['result']['outcome'],'stopped')
        self.assertIn('projection disagrees',run['result']['reason'])
        self.assertEqual(run['final_motor_output'],[0,0])
        self.assertAlmostEqual(run['true_pose'][1],0.)

    def test_aligned_projective_contact_mission_reaches_without_remote_recovery(self):
        alignment=cv2.Rodrigues(np.array([np.radians(-4.),0.,0.]))[0]
        run=run_case(dict(speed=12,coast=.05,seed=3),target=(0,60),camera_pitch_bias_degrees=4.,
                     mission_options=dict(projective_contact=True,floor_alignment_rotation=alignment.tolist()))
        self.assertEqual(run['result']['outcome'],'object_reached_estimate',run['result'].get('reason'))
        self.assertLess(abs(run['true_target_range_cm']-20),3)
        self.assertEqual(run['result']['intermediate_model_calls'],0)
        self.assertEqual(run['clearance_violations'],[])
        self.assertEqual(run['final_motor_output'],[0,0])


if __name__=='__main__':unittest.main()
