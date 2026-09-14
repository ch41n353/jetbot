#!/usr/bin/env python3
"""Capture uncorrected frames and fit/validate neutral-reference color gains."""
import argparse
import datetime
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'jetbot', 'camera'))
from color_balance import apply_gains, validate_gains, DEFAULT_PROFILE


def reference(frame, roi):
    x, y, w, h = roi
    if x < 0 or y < 0 or w < 16 or h < 16 or x+w > frame.shape[1] or y+h > frame.shape[0]:
        raise ValueError('ROI must be inside the image and at least 16 by 16 pixels')
    patch = frame[y:y+h, x:x+w].reshape(-1, 3).astype(float)
    valid = np.all((patch > 15) & (patch < 240), axis=1)
    if valid.mean() < 0.95:
        raise ValueError('Reference is too dark or clipped; adjust lighting/exposure')
    patch = patch[valid]
    med = np.median(patch, axis=0)
    spread = np.percentile(patch, 90, axis=0) - np.percentile(patch, 10, axis=0)
    if np.any(spread / med > 0.20):
        raise ValueError('Reference is uneven or textured; select a uniformly lit neutral patch')
    return med


def neutral_error(med):
    return float((max(med)-min(med))/np.mean(med))


def read(path):
    frame = cv2.imread(path)
    if frame is None:
        raise ValueError('Cannot read ' + path)
    return frame


def save_image(path, frame):
    if not cv2.imwrite(path, frame):
        raise ValueError('Could not write ' + path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command')
    cap = sub.add_parser('capture', help='Uncorrected BGR PNG after 3 seconds of settling; never moves motors')
    cap.add_argument('--output', required=True)
    fit = sub.add_parser('fit', help='Fit gains from a known neutral gray/white patch')
    fit.add_argument('--image', required=True)
    fit.add_argument('--roi', nargs=4, type=int, required=True, metavar=('X','Y','W','H'))
    fit.add_argument('--output', required=True, help='Candidate JSON profile (not installed automatically)')
    fit.add_argument('--lighting', required=True)
    fit.add_argument('--camera', required=True, help='Camera label/identity supplied by the operator')
    check = sub.add_parser('validate', help='Validate with a separate reference capture')
    check.add_argument('--image', required=True)
    check.add_argument('--roi', nargs=4, type=int, required=True)
    check.add_argument('--profile', required=True)
    check.add_argument('--preview', required=True)
    check.add_argument('--report', required=True)
    check.add_argument('--install', action='store_true', help='Install only if neutral-reference validation passes')
    args = parser.parse_args()
    if args.command == 'capture':
        pipeline = 'nvarguscamerasrc sensor-mode=3 wbmode=1 awblock=false ! video/x-raw(memory:NVMM), width=1640, height=1232, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! appsink max-buffers=1 drop=true'
        camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        try:
            deadline = time.monotonic() + 3
            while True:
                ok, frame = camera.read()
                if not ok:
                    raise ValueError('Camera capture failed')
                if time.monotonic() >= deadline:
                    break
            save_image(args.output, frame)
        finally:
            camera.release()
    elif args.command == 'fit':
        frame = read(args.image)
        med = reference(frame, args.roi)
        gains = validate_gains(np.mean(med)/med)
        profile = dict(version=1, channel_order='BGR', gains=gains.tolist(),
                       camera=args.camera, lighting=args.lighting, wbmode=1,
                       created_utc=datetime.datetime.utcnow().isoformat()+'Z',
                       source_image=os.path.abspath(args.image), roi=args.roi,
                       reference_bgr=med.tolist(), image_shape=list(frame.shape))
        with open(args.output, 'w') as stream:
            json.dump(profile, stream, indent=2)
        print(json.dumps(profile, indent=2))
    elif args.command == 'validate':
        with open(args.profile) as stream:
            profile = json.load(stream)
        if profile.get('version') != 1 or profile.get('channel_order') != 'BGR':
            raise ValueError('Unsupported profile')
        frame = read(args.image)
        if list(frame.shape) != profile['image_shape']:
            raise ValueError('Validation resolution differs from calibration')
        source = read(profile['source_image'])
        if np.array_equal(source, frame):
            raise ValueError('Use a separate capture for validation, not the fitting image')
        before = reference(frame, args.roi)
        corrected = apply_gains(frame, profile['gains'])
        after = reference(corrected, args.roi)
        passed = neutral_error(after) <= 0.05 and neutral_error(after) <= neutral_error(before)+0.01
        report = dict(passed=passed, before_error=neutral_error(before), after_error=neutral_error(after),
                      before_bgr=before.tolist(), after_bgr=after.tolist(),
                      note='Neutrality check only; does not prove full color accuracy or correct spatial tint.')
        save_image(args.preview, corrected)
        with open(args.report, 'w') as stream:
            json.dump(report, stream, indent=2)
        print(json.dumps(report, indent=2))
        if not passed:
            raise ValueError('Validation failed; profile was not installed')
        if args.install:
            profile['validation'] = report
            os.makedirs(os.path.dirname(DEFAULT_PROFILE), exist_ok=True)
            temp = DEFAULT_PROFILE + '.tmp'
            with open(temp, 'w') as stream:
                json.dump(profile, stream, indent=2)
            os.replace(temp, DEFAULT_PROFILE)
            print('Installed ' + DEFAULT_PROFILE + '; restart camera consumers')
    else:
        parser.error('Choose capture, fit, or validate')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as error:
        sys.exit(str(error))
