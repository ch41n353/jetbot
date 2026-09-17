import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'local_nav'))
from free_space import FreeSpace, free_rectangles, local_to_map, merge
from spatial_planner import SpatialPlanner, transform


def quad(pose, local):
    x0, z0, x1, z1 = local
    return [transform(pose, x, z) for x, z in ((x0,z0),(x1,z0),(x1,z1),(x0,z1))]


def inside(polygon, point, tolerance=1e-6):
    """Convex point test: every edge cross product keeps the same sign."""
    sign = 0
    for i in range(len(polygon)):
        a, b = polygon[i], polygon[(i+1) % len(polygon)]
        cross = (b[0]-a[0])*(point[1]-a[1])-(b[1]-a[1])*(point[0]-a[0])
        if abs(cross) < tolerance:
            continue
        current = 1 if cross > 0 else -1
        if sign and current != sign:
            return False
        sign = current
    return True


def covers(rects, point):
    return any(r[0] <= point[0] <= r[2] and r[1] <= point[1] <= r[3] for r in rects)


def area(rects):
    return sum((r[2]-r[0])*(r[3]-r[1]) for r in rects)


class FreeSpaceTests(unittest.TestCase):
    def test_axis_aligned_observation_is_reproduced_exactly(self):
        observation = ([0,0,0], [-20,0,20,60])
        self.assertEqual(free_rectangles([observation], grid=1.), [[-20,0,20,60]])
        slabs = local_to_map(observation[0], observation[1], slabs=6, grid=1.)
        self.assertEqual(len(slabs), 6)
        self.assertEqual(area(slabs), 40*60)

    def test_translated_and_reversed_observation_lands_where_the_camera_looked(self):
        rects = free_rectangles([([10,-5,180], [-8,0,8,30])], grid=1.)
        self.assertTrue(covers(rects, (10,-20)))    # 180 degrees looks toward -z
        self.assertFalse(covers(rects, (10,10)))
        rects = free_rectangles([([0,0,90], [-8,0,8,30])], grid=1.)
        self.assertTrue(covers(rects, (20,0)))      # 90 degrees looks toward +x
        self.assertFalse(covers(rects, (-20,0)))

    def test_rotated_cover_never_certifies_space_outside_the_true_footprint(self):
        local = [-14,2,9,55]
        for yaw in (-170,-91,-45,-7,0,13,45,90,137,179):
            for pose in ([0,0,yaw], [12,-30,yaw], [-40,55,yaw]):
                footprint = quad(pose, local)
                rects = local_to_map(pose, local)
                self.assertTrue(rects)
                for r in rects:
                    for corner in ((r[0],r[1]),(r[2],r[1]),(r[2],r[3]),(r[0],r[3])):
                        self.assertTrue(inside(footprint, corner), (yaw, pose, r, corner))
                self.assertLessEqual(area(rects), (9+14)*(55-2)+1e-9)

    def test_rotated_cover_is_inscribed_not_the_bounding_box(self):
        pose, local = [0,0,45], [-10,0,10,40]
        footprint = quad(pose, local)
        rects = local_to_map(pose, local)
        box = [min(p[0] for p in footprint), min(p[1] for p in footprint),
               max(p[0] for p in footprint), max(p[1] for p in footprint)]
        outside = 0
        for i in range(41):
            for j in range(41):
                point = (box[0]+(box[2]-box[0])*i/40., box[1]+(box[3]-box[1])*j/40.)
                if inside(footprint, point, 1e-9):
                    continue
                outside += 1
                self.assertFalse(covers(rects, point), point)
        self.assertGreater(outside, 100)

    def test_finer_slabs_recover_more_of_the_rotated_footprint(self):
        pose, local = [0,0,30], [-12,0,12,50]
        coarse = area(local_to_map(pose, local, slabs=4, grid=.5))
        fine = area(local_to_map(pose, local, slabs=40, grid=.5))
        self.assertGreater(fine, coarse)
        self.assertLess(fine, 24*50)

    def test_merge_coalesces_slabs_and_preserves_coverage(self):
        rects = (local_to_map([0,0,0], [-20,0,20,60], slabs=12, grid=1.)
                 + local_to_map([0,60,0], [-20,0,20,40], slabs=12, grid=1.))
        merged = merge(rects)
        self.assertEqual(merged, [[-20,0,20,100]])
        for i in range(31):
            for j in range(31):
                point = (-20+40*i/30., 100*j/30.)
                self.assertEqual(covers(rects, point), covers(merged, point), point)

    def test_merge_drops_contained_duplicates_without_growing_the_union(self):
        rects = [[0,0,10,10],[2,2,4,4],[0,0,10,10],[10,0,20,10],[0,20,10,30]]
        merged = merge(rects)
        self.assertEqual(merged, [[0,0,20,10],[0,20,10,30]])
        for x in (-1,0,5,15,20,21):
            for z in (-1,0,5,10,19,25,31):
                self.assertEqual(covers(rects, (x,z)), covers(merged, (x,z)), (x,z))

    def test_merge_caps_the_list_and_only_discards_inspected_space(self):
        rects = [[4*i,4*j,4*i+2,4*j+2] for i in range(12) for j in range(12)]
        merged = merge(rects, cap=40)
        self.assertEqual(len(merged), 40)
        for r in merged:
            self.assertIn(r, rects)

    def test_accumulated_observations_stay_bounded_and_inscribed(self):
        space = FreeSpace(cap=64)
        local = [-12,0,12,45]
        poses = [[3*k,2*k,17*k] for k in range(-9,10)]
        for pose in poses:
            space.add(pose, local)
            self.assertLessEqual(len(space.rectangles()), 64)
        rects = space.rectangles()
        self.assertTrue(rects)
        for r in rects:
            for corner in ((r[0],r[1]),(r[2],r[1]),(r[2],r[3]),(r[0],r[3])):
                self.assertTrue(any(inside(quad(pose, local), corner) for pose in poses), (r, corner))

    def test_batch_helper_matches_the_accumulator(self):
        observations = [([0,0,0], [-15,0,15,40]), ([0,20,35], [-10,0,10,50])]
        space = FreeSpace()
        for pose, local in observations:
            space.add(pose, local)
        self.assertEqual(space.rectangles(), free_rectangles(observations))

    def test_invalid_observations_are_rejected(self):
        with self.assertRaises(ValueError):
            local_to_map([0,0], [-10,0,10,10])
        with self.assertRaises(ValueError):
            local_to_map([0,0,float('nan')], [-10,0,10,10])
        with self.assertRaises(ValueError):
            local_to_map([0,0,0], [10,0,-10,10])
        with self.assertRaises(ValueError):
            local_to_map([0,0,0], [-10,0,10,10], grid=0)

    def test_planner_drives_through_free_space_assembled_from_two_headings(self):
        observations = [([0,0,0], [-35,-30,35,70]), ([0,60,90], [-35,-30,35,70])]
        rects = free_rectangles(observations, grid=1.)
        plan = dict(goal_cm=[45,60], obstacle_rectangles_cm=[],
                    inspected_free_rectangles_cm=rects)
        planner = SpatialPlanner(plan)
        self.assertEqual(rects, [[-35,-30,35,70],[-30,25,70,95]])
        self.assertTrue(planner.clear([-16,-10,16,40]))      # inside the first fan
        self.assertTrue(planner.clear([-34,30,50,40]))       # straddles both fans
        self.assertFalse(planner.clear([-34,75,50,85]))      # left half never inspected
        self.assertFalse(planner.clear([-16,100,16,130]))    # never inspected
        result = planner.search(timeout_seconds=5)
        self.assertEqual(result['outcome'], 'route_found')
        for action in result['actions']:
            for corner in ((action['swept_bounds_cm'][0], action['swept_bounds_cm'][1]),
                           (action['swept_bounds_cm'][2], action['swept_bounds_cm'][3])):
                self.assertTrue(covers(rects, corner), corner)


if __name__ == '__main__':
    unittest.main()
