import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'local_nav'))
from spatial_planner import SpatialPlanner, sweep, transform, predict


class SpatialPlannerTests(unittest.TestCase):
    def planner(self, goal, obstacles=None, free=None):
        return SpatialPlanner(dict(goal_cm=goal, obstacle_rectangles_cm=obstacles or [],
                                   inspected_free_rectangle_cm=free or [-120,-100,120,160]))

    def test_long_straight_needs_no_model_at_internal_boundaries(self):
        planner = self.planner([0,60])
        result = planner.search(timeout_seconds=3)
        self.assertEqual(result['outcome'], 'route_found')
        self.assertGreaterEqual(len(result['actions']), 4)
        self.assertEqual(result['model_calls'], 0)
        self.assertTrue(planner.arrived(result['predicted_final_pose']))

    def test_side_goal_contains_turn_and_drive(self):
        planner = self.planner([50,30])
        result = planner.search(timeout_seconds=5)
        self.assertEqual(result['outcome'], 'route_found')
        self.assertEqual(set(a['kind'] for a in result['actions']), {'turn','drive'})
        for action in result['actions']:
            self.assertTrue(planner.clear(action['swept_bounds_cm']))

    def test_retained_obstacle_blocks_direct_route(self):
        planner = self.planner([0,75], [[-10,28,10,45]])
        self.assertFalse(planner.clear(sweep((0,0,0), ('drive',15))))
        result = planner.search(timeout_seconds=5)
        self.assertEqual(result['outcome'], 'route_found')
        self.assertIn('turn', [a['kind'] for a in result['actions']])
        for action in result['actions']:
            self.assertTrue(planner.clear(action['swept_bounds_cm']))

    def test_unknown_space_and_rear_obstacle_not_assumed_clear(self):
        planner = self.planner([0,60], free=[-16,-24,16,20])
        self.assertEqual(planner.search()['outcome'], 'no_checked_route')
        blocked = self.planner([0,60], [[-4,-14,4,-10]])
        self.assertEqual(blocked.search()['outcome'], 'blocked_start')

    def test_measured_pose_changes_next_route(self):
        planner = self.planner([0,30])
        result = planner.search((1,15,2), timeout_seconds=3)
        self.assertEqual(result['outcome'], 'route_found')
        self.assertEqual(result['actions'][0]['predicted_start_pose'], [1,15,2])

    def test_turn_envelope_contains_dense_pivot_and_angle_samples(self):
        for yaw in (-170, -35, 0, 80, 179):
            for turn in (-30,30):
                pose = (3,-2,yaw)
                envelope = sweep(pose, ('turn',turn))
                for px in (-6,0,6):
                    for pz in (-15,-7.5,0):
                        for bx in (-6,6):
                            for bz in (-15,0):
                                for i in range(101):
                                    a = math.radians((turn+math.copysign(5,turn))*i/100)
                                    x = px+math.cos(a)*(bx-px)+math.sin(a)*(bz-pz)
                                    z = pz-math.sin(a)*(bx-px)+math.cos(a)*(bz-pz)
                                    x,z = transform(pose,x,z)
                                    self.assertLessEqual(envelope[0], x-7+1e-8)
                                    self.assertLessEqual(envelope[1], z-7+1e-8)
                                    self.assertGreaterEqual(envelope[2], x+7-1e-8)
                                    self.assertGreaterEqual(envelope[3], z+7-1e-8)

    def test_budget_exhaustion_never_returns_partial_executable_path(self):
        result = self.planner([100,100]).search(max_nodes=1)
        self.assertEqual(result['outcome'], 'search_budget_exhausted')
        self.assertEqual(result['actions'], [])

    def test_small_heading_drift_can_be_corrected_locally(self):
        planner=self.planner([0,20],free=[-18,-30,18,34])
        pose=(1.1296377377,15.2171916114,-2.1668960013)
        self.assertFalse(planner.clear(planner.envelope(pose,('drive',5.))))
        result=planner.search(pose)
        self.assertEqual(result['outcome'],'route_found')
        self.assertEqual(result['actions'][0]['kind'],'turn')
        self.assertLess(abs(result['actions'][0]['value']),5.)

    def test_precise_turn_bounds_cover_pivots_and_angles_in_world_frame(self):
        for yaw in (-35,0,80):
            pose=(3,-2,yaw)
            envelope=sweep(pose,('turn',30),precise_turn=True)
            for px in (-6,0,6):
                for pz in (-15,-7.5,0):
                    for bx in (-6,6):
                        for bz in (-15,0):
                            for i in range(101):
                                a=math.radians(-2+39*i/100)
                                x=px+math.cos(a)*(bx-px)+math.sin(a)*(bz-pz)
                                z=pz-math.sin(a)*(bx-px)+math.cos(a)*(bz-pz)
                                x,z=transform(pose,x,z)
                                self.assertLessEqual(envelope[0],x-7+1e-8)
                                self.assertLessEqual(envelope[1],z-7+1e-8)
                                self.assertGreaterEqual(envelope[2],x+7-1e-8)
                                self.assertGreaterEqual(envelope[3],z+7-1e-8)


if __name__ == '__main__':
    unittest.main()
