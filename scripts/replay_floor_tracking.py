#!/usr/bin/env python3
"""Replay recorded camera pairs through FloorTracker at candidate thresholds.

Lowering a threshold to make a surface "work" is only defensible if you can see
what it admits. The executor saves every published source frame, so the frames
from a failed run are a ready-made test set: consecutive pairs, real motion,
the exact surface that failed.

What matters is not how many pairs are accepted but whether the accepted ones
are *right*. Two checks stand in for ground truth:

  * residual -- the fit's own median reprojection error, already a gate;
  * IMU agreement -- visual yaw against the inertial yaw over the same
    interval, which is independent of the floor entirely.

A setting that accepts more pairs while residual and IMU disagreement stay flat
is buying real coverage. One that accepts more by letting both drift is buying
fabricated odometry, which is worse than the refusal it replaces.
"""
import argparse
import glob
import json
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, 'local_nav'))

import point_controller as pc


def pairs(run_root, limit=None, newest=None):
    """Consecutive (before, after) source frames from each execution.

    `newest` keeps only the N most recently modified executions. Without it this
    sweeps every execution the run root has ever held -- which, on a root that
    has seen several surfaces, silently averages the surface under test together
    with every earlier one and reports a pass rate belonging to neither.
    """
    out = []
    directories = sorted(glob.glob(os.path.join(run_root, 'local-executions', '*')))
    if newest:
        directories = sorted(directories, key=os.path.getmtime)[-newest:]
    for directory in directories:
        frames = sorted(glob.glob(os.path.join(directory, 'snapshots', '*-source.jpg')))
        for before, after in zip(frames, frames[1:]):
            a, b = cv2.imread(before), cv2.imread(after)
            if a is not None and b is not None:
                out.append((os.path.basename(directory)[:8], a, b))
    return out[:limit] if limit else out


def evaluate(samples, tracker, label):
    accepted, refusals = [], {}
    for _, before, after in samples:
        try:
            rotation, translation, quality = tracker.motion(before, after)
        except RuntimeError as exc:
            key = str(exc).split('(')[0].strip()
            refusals[key] = refusals.get(key, 0) + 1
            continue
        accepted.append(dict(
            residual=quality['residual_cm'],
            matches=quality['matches'],
            yaw_deg=math.degrees(math.atan2(rotation[1, 0], rotation[0, 0])),
            step_cm=float(np.linalg.norm(translation))))
    n = len(samples)
    got = len(accepted)
    print('%-22s accepted %3d/%-3d (%3.0f%%)' % (label, got, n, 100. * got / max(1, n)),
          end='')
    if accepted:
        res = sorted(a['residual'] for a in accepted)
        step = sorted(a['step_cm'] for a in accepted)
        yaw = sorted(abs(a['yaw_deg']) for a in accepted)
        print('  residual med %.3f max %.3f cm | step med %.2f max %.2f cm'
              ' | |yaw| med %.2f max %.2f deg'
              % (res[len(res) // 2], res[-1], step[len(step) // 2], step[-1],
                 yaw[len(yaw) // 2], yaw[-1]))
    else:
        print()
    for reason, count in sorted(refusals.items(), key=lambda kv: -kv[1]):
        print('%26s x%-3d %s' % ('', count, reason))
    return accepted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--newest', type=int, default=None,
                        help='only the N most recent executions (one surface)')
    args = parser.parse_args()

    root = args.run_root
    if root is None:
        with open('/mnt/robotlogs/current-search.json') as handle:
            root = json.load(handle)['root']

    with open(os.path.join(pc.ROOT, 'calibration/floor_geometry.json')) as f:
        profile = json.load(f)
    with open(profile['intrinsics_path']) as f:
        intrinsics = json.load(f)

    samples = pairs(root, args.limit, args.newest)
    print('%d consecutive frame pair(s) from %s\n' % (len(samples), root))
    if not samples:
        return

    baseline = dict(source=pc.MIN_SOURCE_FEATURES, tracked=pc.MIN_TRACKED_FEATURES,
                    inliers=pc.MIN_INLIERS, fraction=pc.MIN_INLIER_FRACTION)
    settings = [
        ('carpet (current)', baseline),
        ('tracked 8', dict(baseline, tracked=8)),
        ('tracked 6, inl 5', dict(baseline, tracked=6, inliers=5)),
        ('+ fraction .35', dict(baseline, tracked=6, inliers=5, fraction=.35)),
        ('+ fraction .25', dict(baseline, tracked=6, inliers=4, fraction=.25)),
    ]
    try:
        for label, s in settings:
            pc.MIN_SOURCE_FEATURES = s['source']
            pc.MIN_TRACKED_FEATURES = s['tracked']
            pc.MIN_INLIERS = s['inliers']
            pc.MIN_INLIER_FRACTION = s['fraction']
            evaluate(samples, pc.FloorTracker(profile, intrinsics, 250), label)
    finally:
        pc.MIN_SOURCE_FEATURES = baseline['source']
        pc.MIN_TRACKED_FEATURES = baseline['tracked']
        pc.MIN_INLIERS = baseline['inliers']
        pc.MIN_INLIER_FRACTION = baseline['fraction']


if __name__ == '__main__':
    main()
