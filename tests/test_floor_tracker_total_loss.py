"""Total feature loss must raise the recoverable error, not AttributeError.

Losing every tracked feature is exactly what a bump does -- the robot crosses a
cable, the floor blurs, and forward/backward optical flow agrees on nothing.
FloorTracker.motion has a guard for it that raises 'Lost carpet tracking', and
trajectory_executor.recoverable_tracking_error() absorbs that as one rejected
camera pair while keeping the motor lease.

The guard never ran. `ground()` is called one line earlier with the surviving
points, cv2.fisheye.undistortPoints returns None for an empty input on this
OpenCV build, and `.reshape` on None raised AttributeError -- which matches no
recoverable prefix, so the whole segment failed. Measured live: a 273 cm
trajectory died at 114 cm this way.
"""
import json
import math
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'local_nav'))
from point_controller import FloorTracker, ROOT
import trajectory_executor


class TotalFeatureLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(ROOT, 'calibration/floor_geometry.json')) as f:
            cls.profile = json.load(f)
        with open(cls.profile['intrinsics_path']) as f:
            cls.intrinsics = json.load(f)

    def tracker(self):
        return FloorTracker(self.profile, self.intrinsics, 125)

    def test_ground_of_no_points_is_empty_not_a_crash(self):
        empty = self.tracker().ground(np.zeros((0, 2), dtype=float))
        self.assertEqual(empty.shape, (0, 2))

    def test_ground_still_projects_a_single_pixel(self):
        # The empty-input shortcut must not disturb the ordinary path, which
        # every other caller uses with one pixel at a time.
        spot = self.tracker().ground([[320., 400.]])
        self.assertEqual(spot.shape, (1, 2))
        self.assertTrue(np.isfinite(spot).all())

    def test_untrackable_pair_is_a_recoverable_tracking_error(self):
        # Pure noise against pure noise: features are found in the first frame
        # and none of them survive the forward/backward check.
        rng = np.random.RandomState(4)
        before = rng.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        after = rng.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        with self.assertRaises(RuntimeError) as caught:
            self.tracker().motion(before, after)
        self.assertTrue(
            trajectory_executor.recoverable_tracking_error(caught.exception),
            'executor would abort the segment on: %s' % caught.exception)

    def test_attribute_error_is_not_recoverable(self):
        # Guards the diagnosis rather than the fix: if ground() ever returns to
        # crashing, the executor still has no reason to treat it as transient,
        # so this test failing together with the one above is the signature.
        self.assertFalse(trajectory_executor.recoverable_tracking_error(
            AttributeError("'NoneType' object has no attribute 'reshape'")))


if __name__ == '__main__':
    unittest.main()
