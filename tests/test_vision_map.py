import json
import math
import os
import sys
import unittest

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, 'local_nav'))
from point_controller import FloorTracker
import vision_map as vm


def tracker():
    profile = json.load(open(os.path.join(ROOT, 'calibration/floor_geometry.json')))
    intrinsics = json.load(open(profile['intrinsics_path']))
    return FloorTracker(profile, intrinsics, 125), profile, intrinsics


def to_pixels(profile, intrinsics, points):
    """Inverse of the floor projection, for checking claimed floor was seen."""
    angle = math.radians(profile['pitch_degrees'])
    down = np.array([0., math.cos(angle), math.sin(angle)])
    forward = np.array([0., 0., 1.]) - down * down[2]
    forward /= np.linalg.norm(forward)
    right = np.cross(down, forward)
    rays = []
    for r, f in points:
        rays.append(right * r + forward * f + down * profile['camera_height_cm'])
    rays = np.asarray(rays, dtype=float)
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    out, _ = cv2.fisheye.projectPoints(rays.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                       np.asarray(intrinsics['K'], dtype=float),
                                       np.asarray(intrinsics['D'], dtype=float))
    return out.reshape(-1, 2)


class ProjectionTests(unittest.TestCase):
    def test_rejects_pixels_that_cannot_be_trusted_as_floor(self):
        t, _, _ = tracker()
        for bad in ((-1., 300.), (641., 300.), (320., -5.), (320., 481.)):
            with self.assertRaises(ValueError):
                vm.project(t, [bad])
        # Near and above the horizon the ray never meets the floor in front.
        with self.assertRaises(ValueError):
            vm.project(t, [(320., 40.)])

    def test_far_projections_are_refused_rather_than_guessed(self):
        t, _, _ = tracker()
        # Row 198 projects to 130 cm and is allowed; 196 exceeds the limit.
        self.assertLess(vm.project(t, [(320., 198.)])[0][1], vm.MAX_PROJECTION_CM)
        with self.assertRaisesRegex(ValueError, 'not trusted'):
            vm.project(t, [(320., 196.)])

    def test_known_contact_pixel_projects_to_the_recorded_range(self):
        # The live run that recognised the bottle reported contact (457, 216).
        t, _, _ = tracker()
        right, forward = vm.project(t, [(457., 216.)])[0]
        self.assertAlmostEqual(right, 36.5, delta=.5)
        self.assertAlmostEqual(forward, 77.4, delta=.5)


class ObstacleTests(unittest.TestCase):
    def test_footprint_depth_is_bounded_not_run_to_the_horizon(self):
        t, _, _ = tracker()
        box = {'x0': 300., 'y0': 240., 'x1': 360., 'y1': 300.}
        x0, z0, x1, z1 = vm.obstacle_footprint(t, box)
        near = vm.project(t, [(300., 300.), (360., 300.)])[0][1]
        self.assertLess(z0, near)
        self.assertLess(x0, x1)
        # A box carries no depth, so the assumption is bounded. Filling to the
        # range limit turned a cable's wide bounding box into a wall that
        # blocked every turn without making the robot any safer.
        self.assertLess(z1, vm.MAX_PROJECTION_CM)
        self.assertLessEqual(z1 - z0, vm.OBSTACLE_DEPTH_CM + 3 * vm.OBSTACLE_MARGIN_CM
                             + (near - z0) + 30.)

    def test_a_wide_shallow_box_does_not_swallow_the_map(self):
        """The cable case: a long diagonal object in an axis-aligned box."""
        t, _, _ = tracker()
        cable = {'x0': 134., 'y0': 230., 'x1': 640., 'y1': 470.}
        x0, z0, x1, z1 = vm.obstacle_footprint(t, cable)
        self.assertLess(z1 - z0, 60., 'footprint still reaches across the room')

    def test_footprint_is_padded_beyond_the_visible_face(self):
        t, _, _ = tracker()
        box = {'x0': 300., 'y0': 240., 'x1': 360., 'y1': 300.}
        contact = vm.project(t, [(300., 300.), (360., 300.)])
        padded = vm.obstacle_footprint(t, box)
        self.assertLess(padded[0], min(contact[:, 0]))
        self.assertGreater(padded[2], max(contact[:, 0]))

    def test_malformed_boxes_are_refused(self):
        t, _, _ = tracker()
        for bad in ({'x0': 360., 'y0': 240., 'x1': 300., 'y1': 300.},
                    {'x0': 300., 'y0': 300., 'x1': 360., 'y1': 240.},
                    {'x0': -5., 'y0': 240., 'x1': 360., 'y1': 300.}):
            with self.assertRaises(ValueError):
                vm.obstacle_footprint(t, bad)


class FanTests(unittest.TestCase):
    def test_bands_never_claim_floor_the_camera_did_not_see(self):
        """The safety-critical invariant: certified floor must be visible."""
        t, profile, intrinsics = tracker()
        for x0, z0, x1, z1 in vm.fan_rectangles(t, bands=12):
            corners = [(x, z) for x in (x0, x1) for z in (z0, z1)]
            for u, v in to_pixels(profile, intrinsics, corners):
                self.assertTrue(0. <= u <= 640., 'x %.1f outside frame' % u)
                self.assertTrue(0. <= v <= 480., 'y %.1f outside frame' % v)

    def test_bands_are_ordered_and_do_not_overlap_in_depth(self):
        t, _, _ = tracker()
        bands = vm.fan_rectangles(t, bands=12)
        self.assertGreater(len(bands), 3)
        for (_, _, _, prev_far), (_, near, _, _) in zip(bands, bands[1:]):
            self.assertAlmostEqual(prev_far, near, places=6)

    def test_fan_widens_with_distance_and_stays_inside_the_limit(self):
        t, _, _ = tracker()
        bands = vm.fan_rectangles(t, bands=12)
        widths = [b[2] for b in bands]
        self.assertEqual(widths, sorted(widths))
        self.assertLessEqual(max(b[3] for b in bands), vm.MAX_PROJECTION_CM)


class FrameTests(unittest.TestCase):
    def test_world_transform_matches_a_hand_rotation(self):
        point = vm.to_world([0., 0., 90.], [(0., 10.)])[0]
        self.assertAlmostEqual(point[0], 10., places=6)
        self.assertAlmostEqual(point[1], 0., places=6)

    def test_world_rectangle_bounds_the_rotated_corners(self):
        box = vm.world_rectangle([0., 0., 45.], [-10., 0., 10., 10.])
        self.assertAlmostEqual(box[0], -10. * math.cos(math.radians(45.)), places=4)
        self.assertGreater(box[2], 0.)


if __name__ == '__main__':
    unittest.main()
