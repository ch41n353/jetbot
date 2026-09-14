import os
import sys
import unittest
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','local_nav'))
from mapped_drive import MappedDriveRoute
from spatial_planner import SpatialPlanner
from simulate_spatial import run_spatial_case


class MappedDriveTests(unittest.TestCase):
    def test_return_from_actual_live_offset(self):
        case=run_spatial_case(dict(speed=12,coast=.05,seed=3,pivot_cm=7.5,drive_yaw_bias_dps=-4),
                              goal=(0,0),free=[-18,-30,18,34],measured_map_drive=True,
                              initial_pose=[2.4597813589,20.8447391402,1.607047403])
        self.assertEqual(case['result']['outcome'],'target_reached_estimate',case['result'].get('reason'))
        self.assertLess(case['true_goal_error_cm'],4)
        self.assertEqual(case['clearance_violations'],[])
        self.assertEqual(case['final_motor_output'],[0,0])

    def test_side_drift_and_rear_braking_space_are_checked(self):
        planner=SpatialPlanner(dict(goal_cm=[0,-10],inspected_free_rectangle_cm=[-18,-40,18,15],
                                    obstacle_rectangles_cm=[],measured_map_drive=True))
        route=MappedDriveRoute(planner,(0,0,0),-10)
        route.check_pose(1,-5,1)
        with self.assertRaises(RuntimeError):route.check_pose(6,-5,1)
        blocked=SpatialPlanner(dict(goal_cm=[0,-10],inspected_free_rectangle_cm=[-18,-40,18,15],
                                    obstacle_rectangles_cm=[[-4,-28,4,-26]],measured_map_drive=True))
        with self.assertRaises(ValueError):MappedDriveRoute(blocked,(0,0,0),-10)


if __name__=='__main__':unittest.main()
