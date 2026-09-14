import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'local_nav'))
from approach_plan import prepare
from point_controller import FloorTracker, ROOT


class ApproachTests(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(ROOT, 'calibration/floor_geometry.json')) as source:
            profile = json.load(source)
        with open(profile['intrinsics_path']) as source:
            self.tracker = FloorTracker(profile, json.load(source))
        self.capture = dict(image_path='fresh.jpg', session_id='s', control_epoch=2,
                            captured_monotonic=10)
        self.request = dict(image_path='fresh.jpg', target_base_pixel=[365, 263],
                            inspected_free_rectangle_cm=[-18, -24, 18, 29],
                            obstacle_rectangles_cm=[[-1, 29, 12, 40]])

    def test_recorded_approach_and_final_view(self):
        result = prepare(self.capture, self.request, self.tracker)
        self.assertEqual(result['outcome'], 'approach_plan_prepared')
        self.assertEqual(result['plan']['waypoints_cm'], [14])
        self.assertGreaterEqual(result['target_ground_cm'][1]-14, 18)
        self.request['target_base_pixel'] = [414, 354]
        result = prepare(self.capture, self.request, self.tracker)
        self.assertEqual(result['outcome'], 'within_standoff_band_estimate')
        self.assertNotIn('plan', result)

    def test_obstacle_or_unknown_space_blocks_route(self):
        self.request['obstacle_rectangles_cm'] = [[-2, 10, 2, 12]]
        with self.assertRaises(ValueError):
            prepare(self.capture, self.request, self.tracker)
        self.request['obstacle_rectangles_cm'] = []
        self.request['inspected_free_rectangle_cm'] = [-10, -24, 10, 30]
        with self.assertRaises(ValueError):
            prepare(self.capture, self.request, self.tracker)

    def test_wrong_image_or_small_standoff_rejected(self):
        self.request['image_path'] = 'old.jpg'
        with self.assertRaises(ValueError):
            prepare(self.capture, self.request, self.tracker)
        self.request['image_path'] = 'fresh.jpg'
        self.request['standoff_cm'] = 5
        with self.assertRaises(ValueError):
            prepare(self.capture, self.request, self.tracker)

    def test_off_axis_target_requires_alignment(self):
        self.request['target_base_pixel'] = [224, 245]
        result = prepare(self.capture, self.request, self.tracker)
        self.assertEqual(result['outcome'], 'alignment_required')
        self.assertNotIn('plan', result)

    def test_batch_prepares_whole_approach(self):
        self.request.update(batch=True, target_base_pixel=[345,238],
                            inspected_free_rectangle_cm=[-22,-30,22,60],
                            obstacle_rectangles_cm=[])
        result = prepare(self.capture, self.request, self.tracker)
        self.assertGreater(result['plan']['approach_distance_cm'], 15)
        self.assertLessEqual(result['plan']['approach_distance_cm'], 30)
        self.assertNotIn('waypoints_cm', result['plan'])
