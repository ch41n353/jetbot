"""The floor scan must refuse a wall, and must not calibrate itself to one.

This is a regression test for a scan that drove the robot into walls. The
threshold separating carpet from paint used to be a fraction of a reference
patch at the bottom of the frame -- "what the robot is standing on". Nose to a
wall that patch IS the wall, so the reference collapsed and every bearing came
back at the 120 cm ceiling. Measured on the live frame: lower-third median
texture 2.0 against 6-7 on real carpet.

Synthetic frames here, so the test carries its own ground truth and does not
depend on captured images staying around.
"""
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'scripts'))
import floor_scan


def carpet(shape=(480, 640)):
    """High-frequency speckle, like the weave the tracker keys on."""
    rng = np.random.RandomState(7)
    grey = rng.randint(90, 210, shape, dtype=np.uint8)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


def painted_wall(shape=(480, 640)):
    """A smooth, bright, faintly graded surface: paint under even light."""
    ramp = np.linspace(196, 214, shape[0], dtype=np.float32)
    flat = np.repeat(ramp[:, None], shape[1], axis=1).astype(np.uint8)
    return cv2.cvtColor(flat, cv2.COLOR_GRAY2BGR)


class WallRefusalTests(unittest.TestCase):
    def test_open_carpet_reads_clear(self):
        image = carpet()
        self.assertTrue(floor_scan.standing_on_floor(image))
        ranges = [r for _, r in floor_scan.profile(image)]
        self.assertGreater(max(ranges), 90.)

    def test_wall_filling_the_frame_is_refused_everywhere(self):
        image = painted_wall()
        self.assertFalse(floor_scan.standing_on_floor(image))
        for bearing, reach in floor_scan.profile(image):
            self.assertEqual(reach, 0., 'bearing %+d called clear against a wall'
                             % bearing)

    def test_wall_does_not_become_its_own_floor_reference(self):
        # The heart of the bug: a frame that is entirely wall must not be read
        # as entirely floor just because nothing in it looks different.
        self.assertFalse(floor_scan.clear_path(painted_wall(), [[0., 60.]]))

    def test_wall_across_the_top_shortens_the_range(self):
        # Carpet near, paint beyond: the range must stop, not run to the cap.
        image = carpet()
        image[:250, :] = painted_wall()[:250, :]
        self.assertTrue(floor_scan.standing_on_floor(image))
        ahead = floor_scan.free_range(image, 0.)
        self.assertLess(ahead, 120.)

    def test_clear_path_checks_the_second_leg_of_a_dogleg(self):
        # Both waypoints sit on carpet; the leg between them crosses paint. A
        # check that only looked at endpoints from the origin would pass this.
        image = carpet()
        image[200:260, :] = painted_wall()[200:260, :]
        self.assertFalse(floor_scan.clear_path(image, [[-45., 45.], [45., 45.]]))


if __name__ == '__main__':
    unittest.main()
