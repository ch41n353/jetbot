"""Measured neutral-reference correction; environment gains override the profile."""
import json
import os

import cv2
import numpy as np

DEFAULT_PROFILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'calibration', 'color_profile.json'))


def validate_gains(values):
    gains = np.asarray(values, dtype=float)
    if gains.shape != (3,) or not np.all(np.isfinite(gains)) or np.any(gains < 0.25) or np.any(gains > 4):
        raise ValueError('Expected three finite B,G,R gains between 0.25 and 4')
    return gains


def load_gains():
    override = os.environ.get('JETBOT_COLOR_GAINS')
    if override is not None:
        return validate_gains([float(v) for v in override.split(',')])
    path = os.environ.get('JETBOT_COLOR_PROFILE', DEFAULT_PROFILE)
    if not os.path.exists(path):
        if 'JETBOT_COLOR_PROFILE' in os.environ:
            raise ValueError('Requested color profile does not exist: ' + path)
        return np.ones(3)
    with open(path) as stream:
        profile = json.load(stream)
    if profile.get('version') != 1 or profile.get('channel_order') != 'BGR':
        raise ValueError('Unsupported color profile')
    return validate_gains(profile['gains'])


def apply_gains(frame, gains):
    gains = validate_gains(gains)
    lut = np.clip(np.arange(256)[:, None] * gains, 0, 255).astype(np.uint8).reshape(256, 1, 3)
    return cv2.LUT(frame, lut)


_GAINS = load_gains()
_LUT = np.clip(np.arange(256)[:, None] * _GAINS, 0, 255).astype(np.uint8).reshape(256, 1, 3)


def correct_bgr(frame):
    """Correct a frame without modifying it. Restart capture after changing a profile."""
    return cv2.LUT(frame, _LUT)
