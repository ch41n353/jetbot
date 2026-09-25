"""The tracker's floor ROI follows the calibration, not fixed image rows.

Rows 280..460 silently meant "27 cm to 7 cm" at the original 14.25 degree
mount. Tilting the camera to 30.40 degrees left those same rows looking at
13.3..3.5 cm, where the same wheel speed pushes roughly twice the pixel
displacement through a 21 px LK window. Measured immediately afterwards, on
carpet that had been tracking fine an hour before: 5 of 250 features survived
and the leg failed with "Lost carpet tracking".

Deriving the rows from the calibration keeps the band fixed in centimetres,
which is the thing the tracker actually depends on.
"""
import copy
import json
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'local_nav'))
import point_controller as pc

ORIGINAL_PITCH = 14.2466          # the mount the hardcoded rows were written for
ORIGINAL_ROWS = (280., 460.)


class FloorBandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(pc.ROOT, 'calibration/floor_geometry.json')) as f:
            cls.profile = json.load(f)
        with open(cls.profile['intrinsics_path']) as f:
            cls.intrinsics = json.load(f)

    def at(self, pitch):
        profile = dict(self.profile, pitch_degrees=pitch)
        return pc.band_rows(profile, self.intrinsics)

    def reach(self, row, pitch):
        """Metric range of a centre-column row, independent of the tracker."""
        import cv2
        K = np.asarray(self.intrinsics['K'], float)
        D = np.asarray(self.intrinsics['D'], float)
        height = float(self.profile['camera_height_cm'])
        a = math.radians(pitch)
        down = np.array([0., math.cos(a), math.sin(a)])
        forward = np.array([0., 0., 1.]) - down * down[2]
        forward /= np.linalg.norm(forward)
        xy = cv2.fisheye.undistortPoints(np.array([[[320., float(row)]]]), K, D).reshape(2)
        ray = np.array([xy[0], xy[1], 1.])
        return height / ray.dot(down) * ray.dot(forward)

    def test_the_original_mount_reproduces_the_hardcoded_rows(self):
        # The derivation must be faithful to what the constants encoded, or it
        # is a new guess wearing the old one's clothes.
        top, bottom = self.at(ORIGINAL_PITCH)
        self.assertAlmostEqual(top, ORIGINAL_ROWS[0], delta=1.)
        self.assertAlmostEqual(bottom, ORIGINAL_ROWS[1], delta=1.)

    def test_the_band_stays_put_in_centimetres_across_pitches(self):
        near, far = min(pc.FLOOR_BAND_CM), max(pc.FLOOR_BAND_CM)
        for pitch in (14.2466, 20., 25., 30.4008, 35.):
            top, bottom = self.at(pitch)
            self.assertAlmostEqual(self.reach(top, pitch), far, delta=.5,
                                   msg='far edge drifted at %.1f deg' % pitch)
            self.assertAlmostEqual(self.reach(bottom, pitch), near, delta=.5,
                                   msg='near edge drifted at %.1f deg' % pitch)

    def test_the_tilt_that_broke_tracking_moves_the_rows(self):
        # If this did not move, the fix would be doing nothing.
        was = self.at(ORIGINAL_PITCH)
        now = self.at(30.4008)
        self.assertGreater(was[0] - now[0], 50.,
                           'rows barely moved; the ROI would still be misplaced')
        # And the old rows really did look at near floor at the new pitch.
        self.assertLess(self.reach(ORIGINAL_ROWS[0], 30.4008), 20.)

    def test_the_rows_stay_inside_the_cropped_region(self):
        # motion() crops to rows 168..480 before masking; a band outside that is
        # silently empty rather than merely misplaced.
        for pitch in (14.2466, 20., 25., 30.4008):
            top, bottom = self.at(pitch)
            self.assertGreaterEqual(top, 168., 'band above the crop at %.1f deg' % pitch)
            self.assertLessEqual(bottom, 480., 'band below the crop at %.1f deg' % pitch)

    def test_a_malformed_calibration_falls_back_rather_than_failing(self):
        self.assertEqual(pc.band_rows({}, self.intrinsics), ORIGINAL_ROWS)
        self.assertEqual(pc.band_rows({'pitch_degrees': 'nonsense',
                                       'camera_height_cm': 9.5}, self.intrinsics),
                         ORIGINAL_ROWS)

    def test_the_landing_window_follows_the_source_box(self):
        tracker = pc.FloorTracker(self.profile, self.intrinsics, 250)
        x0, y0, x1, y1 = tracker.flow_window
        self.assertLess(y0, tracker.floor_top, 'window must not clip the source box')
        self.assertGreater(y1, tracker.floor_bottom)
        self.assertLess(x0, x1)

    def test_an_explicit_window_is_still_honoured(self):
        # Rotation-heavy callers pass TURN_FLOW_WINDOW deliberately.
        tracker = pc.FloorTracker(self.profile, self.intrinsics, 250,
                                  flow_window=pc.TURN_FLOW_WINDOW)
        self.assertEqual(tracker.flow_window, tuple(float(v) for v in pc.TURN_FLOW_WINDOW))


if __name__ == '__main__':
    unittest.main()
