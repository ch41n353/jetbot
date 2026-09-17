import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'local_nav'))
from spatial_planner import SpatialPlanner, sweep, transform, predict, uncovered, PIVOT_X, PIVOT_Z


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

    def test_turn_sweep_uses_the_measured_pivot_not_every_chassis_corner(self):
        """Deliberate narrowing: the pivot is measured, so stop sweeping corners.

        Fitted from 28 recorded turns to 0.8 mm RMS (calibration/turn_pivot.json).
        Assuming the robot might spin about its front-left or rear-right corner
        inflated the worst-case rotation radius from 10.5 cm to 19.2 cm, which
        refused turns the robot can physically make. The uncertainty box is
        still swept, so this is a narrower assumption, not a dropped check.
        """
        self.assertTrue(-6 <= min(PIVOT_X) and max(PIVOT_X) <= 6)
        self.assertTrue(-15 <= min(PIVOT_Z) and max(PIVOT_Z) <= 0)
        # The pivot box must stay a box, not collapse to a point.
        self.assertGreater(max(PIVOT_X) - min(PIVOT_X), 0.)
        self.assertGreater(max(PIVOT_Z) - min(PIVOT_Z), 0.)
        radius = lambda px, pz: max(math.hypot(bx-px, bz-pz)
                                    for bx in (-6, 6) for bz in (-15, 0))
        measured = max(radius(px, pz) for px in PIVOT_X for pz in PIVOT_Z)
        corners = max(radius(px, pz) for px in (-6, 6) for pz in (-15, 0))
        self.assertLess(measured, corners)
        # A turn envelope must be smaller than the every-corner one it replaced.
        envelope = sweep((0, 0, 0), ('turn', 30), precise_turn=True)
        self.assertLess(envelope[2] - envelope[0], 2 * corners + 14)

    def test_turn_envelope_contains_dense_pivot_and_angle_samples(self):
        for yaw in (-170, -35, 0, 80, 179):
            for turn in (-30,30):
                pose = (3,-2,yaw)
                envelope = sweep(pose, ('turn',turn))
                for px in PIVOT_X+(sum(PIVOT_X)/2,):
                    for pz in PIVOT_Z+(sum(PIVOT_Z)/2,):
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
            for px in PIVOT_X+(sum(PIVOT_X)/2,):
                for pz in PIVOT_Z+(sum(PIVOT_Z)/2,):
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


class FreeUnionTests(unittest.TestCase):
    def planner(self, plan):
        base = dict(goal_cm=[0,20], obstacle_rectangles_cm=[])
        base.update(plan)
        return SpatialPlanner(base)

    def covered(self, planner, box):
        """Dense sampling: every sampled point must sit in some free rectangle."""
        for i in range(21):
            for j in range(21):
                x = box[0]+(box[2]-box[0])*i/20.
                z = box[1]+(box[3]-box[1])*j/20.
                if not any(r[0] <= x <= r[2] and r[1] <= z <= r[3] for r in planner.free):
                    return False
        return True

    def test_singular_key_still_defines_the_whole_free_space(self):
        planner = self.planner(dict(inspected_free_rectangle_cm=[-30,-20,30,40]))
        self.assertEqual(planner.free, [[-30,-20,30,40]])
        self.assertIsNotNone(planner.single_free)
        self.assertTrue(planner.clear([-30,-20,30,40]))
        self.assertFalse(planner.clear([-31,-20,30,40]))
        self.assertFalse(planner.clear([-30,-20,30,41]))

    def test_envelope_straddling_two_adjacent_rectangles_is_accepted(self):
        planner = self.planner(dict(inspected_free_rectangles_cm=[[-40,-20,0,40],[0,-20,40,40]]))
        self.assertIsNone(planner.single_free)
        straddle = [-10,-5,10,15]
        self.assertTrue(planner.clear(straddle))
        self.assertTrue(self.covered(planner, straddle))
        for single in planner.free:
            self.assertFalse(self.planner(dict(inspected_free_rectangle_cm=single)).clear(straddle))

    def test_envelope_straddling_an_l_shaped_union_is_accepted(self):
        planner = self.planner(dict(inspected_free_rectangles_cm=[[-40,-20,10,10],[-10,10,10,60]]))
        self.assertTrue(planner.clear([-5,0,10,20]))
        self.assertTrue(planner.clear([-40,-20,10,10]))
        # The concave corner is outside the union even though each edge is inside.
        self.assertFalse(planner.clear([-20,0,10,20]))

    def test_envelope_poking_outside_the_union_is_rejected(self):
        planner = self.planner(dict(inspected_free_rectangles_cm=[[-40,-20,0,40],[0,-20,40,40]]))
        for envelope in ([30,-5,45,15], [-10,35,10,45], [-45,-5,-30,5], [-10,-25,10,-5]):
            self.assertFalse(planner.clear(envelope))
        # A gap between members must not be bridged.
        gapped = self.planner(dict(inspected_free_rectangles_cm=[[-40,-20,-1,40],[0,-20,40,40]]))
        self.assertFalse(gapped.clear([-10,-5,10,15]))

    def test_obstacle_inside_the_union_still_rejects(self):
        plan = dict(inspected_free_rectangles_cm=[[-40,-20,0,40],[0,-20,40,40]],
                    obstacle_rectangles_cm=[[-2,0,2,5]])
        planner = self.planner(plan)
        self.assertFalse(planner.clear([-10,-5,10,15]))
        self.assertFalse(planner.clear([1.5,4,20,20]))
        self.assertTrue(planner.clear([-10,10,10,30]))

    def test_both_keys_are_unioned(self):
        planner = self.planner(dict(inspected_free_rectangle_cm=[-40,-20,0,40],
                                    inspected_free_rectangles_cm=[[0,-20,40,40]]))
        self.assertEqual(len(planner.free), 2)
        self.assertTrue(planner.clear([-10,-5,10,15]))

    def test_plan_without_any_inspected_free_space_is_rejected(self):
        with self.assertRaises(ValueError):
            self.planner(dict(inspected_free_rectangles_cm=[]))
        with self.assertRaises(ValueError):
            self.planner(dict(inspected_free_rectangles_cm=[[10,0,0,10]]))
        with self.assertRaises(ValueError):
            self.planner(dict(inspected_free_rectangles_cm=dict(a=[0,0,10,10])))

    def test_uncovered_reports_the_exact_remainder(self):
        self.assertEqual(uncovered([0,0,10,10], [[0,0,10,10]]), [])
        self.assertEqual(uncovered([0,0,10,10], [[0,0,4,10],[4,0,10,10]]), [])
        self.assertEqual(uncovered([0,0,10,10], [[0,0,4,10]]), [[4,0,10,10]])
        self.assertEqual(uncovered([0,0,10,10], []), [[0,0,10,10]])

    def test_route_is_searched_across_a_seam_between_free_rectangles(self):
        planner = self.planner(dict(goal_cm=[0,60],
                                    inspected_free_rectangles_cm=[[-40,-30,40,20],[-40,20,40,90]]))
        result = planner.search(timeout_seconds=5)
        self.assertEqual(result['outcome'], 'route_found')
        for action in result['actions']:
            self.assertTrue(planner.clear(action['swept_bounds_cm']))
            self.assertTrue(self.covered(planner, action['swept_bounds_cm']))

    def test_search_stays_fast_with_many_free_rectangles(self):
        tiles = [[x,z,x+10,z+10] for x in range(-60,60,10) for z in range(-40,90,10)]
        planner = self.planner(dict(goal_cm=[0,60], inspected_free_rectangles_cm=tiles))
        self.assertGreater(len(tiles), 150)
        result = planner.search(timeout_seconds=3)
        self.assertEqual(result['outcome'], 'route_found')
        self.assertLess(result['elapsed_seconds'], 3.)
        for action in result['actions']:
            self.assertTrue(self.covered(planner, action['swept_bounds_cm']))

    def test_tiled_union_matches_the_equivalent_single_rectangle(self):
        tiles = [[x,z,x+10,z+10] for x in range(-60,60,10) for z in range(-40,90,10)]
        union = self.planner(dict(inspected_free_rectangles_cm=tiles))
        whole = self.planner(dict(inspected_free_rectangle_cm=[-60,-40,60,90]))
        for x in (-70,-55,-6,0,12,55,70):
            for z in (-50,-35,0,17,88,95):
                envelope = [x-8,z-9,x+8,z+9]
                self.assertEqual(union.clear(envelope), whole.clear(envelope), envelope)


if __name__ == '__main__':
    unittest.main()
