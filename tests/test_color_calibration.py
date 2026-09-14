import importlib.util
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'jetbot', 'camera'))
import color_balance
spec = importlib.util.spec_from_file_location('calibrate', os.path.join(ROOT, 'scripts', 'calibrate_camera_color.py'))
calibrate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibrate)


class CalibrationTests(unittest.TestCase):
    def test_fit_neutral_and_preserve_input(self):
        frame = np.full((30, 30, 3), [90, 100, 140], dtype=np.uint8)
        before = frame.copy()
        med = calibrate.reference(frame, [0, 0, 30, 30])
        output = color_balance.apply_gains(frame, med.mean()/med)
        self.assertLess(calibrate.neutral_error(np.median(output, axis=(0, 1))), 0.02)
        np.testing.assert_array_equal(frame, before)

    def test_reject_clipping_and_bad_roi(self):
        for value in (0, 255):
            with self.assertRaises(ValueError):
                calibrate.reference(np.full((30,30,3), value, np.uint8), [0,0,30,30])
        with self.assertRaises(ValueError):
            calibrate.reference(np.full((30,30,3), 100, np.uint8), [-1,0,30,30])

    def test_reject_uneven_reference(self):
        frame = np.full((30,30,3), 60, np.uint8)
        frame[15:] = 160
        with self.assertRaises(ValueError):
            calibrate.reference(frame, [0,0,30,30])

    def test_override_and_invalid_gains(self):
        with patch.dict(os.environ, {'JETBOT_COLOR_GAINS': '1,1,1'}):
            np.testing.assert_array_equal(color_balance.load_gains(), [1,1,1])
        for values in ([1,1], [1,1,float('nan')], [1,1,-1], [1,1,8]):
            with self.assertRaises(ValueError):
                color_balance.validate_gains(values)


if __name__ == '__main__':
    unittest.main()
