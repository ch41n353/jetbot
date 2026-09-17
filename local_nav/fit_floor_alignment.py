#!/usr/bin/env python3
"""Fit a scene-scoped pitch candidate from recorded floor tracks; never install it.

This offline tool does not use targets, motor commands or live sensor access.
A temporal held-out half evaluates the pitch chosen using training frames only.
Floor slope and camera mounting error remain observationally ambiguous.
"""
import argparse
import glob
import json
import math
import os
import cv2
import numpy as np
from point_controller import FloorTracker, ROOT
from state_estimator import AttitudeTimeline


def validate_scan(pitch_limit, step):
    if not math.isfinite(pitch_limit) or not 0 < pitch_limit <= 8:
        raise ValueError('Pitch limit must be positive and at most eight degrees')
    if not math.isfinite(step) or not .25 <= step <= pitch_limit:
        raise ValueError('Pitch step must be between 0.25 degrees and the limit')


def validate_config(directory, output, pitch_limit, step):
    if not os.path.isdir(directory):
        raise ValueError('Observation directory does not exist')
    validate_scan(pitch_limit, step)
    destination = os.path.realpath(output)
    protected = os.path.realpath(os.path.join(ROOT, 'calibration'))
    observations = os.path.realpath(directory)
    if any(os.path.commonpath([destination, p]) == p for p in (protected, observations)):
        raise ValueError('Candidate output must not overwrite calibration or observations')


def collect_pairs(directory, profile, intrinsics, mount):
    paths = sorted(glob.glob(os.path.join(directory, '*.json')))
    if len(paths) < 10:
        raise ValueError('Insufficient recorded frames: need at least ten')
    if len(paths) > 1000:
        raise ValueError('Select a bounded observation directory of at most 1000 frames')
    timeline = AttitudeTimeline(mount)
    tracker = FloorTracker(profile, intrinsics, 125)
    ground = tracker.ground
    points = []
    def collect(pixel, attitude=None):
        points.append((np.asarray(pixel).copy(), attitude))
        return ground(pixel, attitude)
    tracker.ground = collect
    # A segment can inherit its live timeline and omit an initialization marker.
    # Merge two initial records, then demand the ordinary stationary initializer.
    initial = []
    for path in paths[:2]:
        with open(path) as source:
            initial.append(json.load(source))
    samples = {sample['time']: sample for item in initial for sample in item['imu_samples']}
    timeline.initialize_stationary([samples[t] for t in sorted(samples)], initial[-1]['time'])
    previous = None
    pairs, rejected = [], []
    for path in paths[1:]:
        with open(path) as source:
            observation = json.load(source)
        timestamp = observation['time']
        timeline.feed(observation['imu_samples'])
        attitude = timeline.at(timestamp)
        image = cv2.imread(path[:-5] + '.jpg')
        if image is None or image.shape[:2] != (480, 640):
            raise ValueError('Missing or invalid recorded image: ' + path[:-5] + '.jpg')
        if previous is not None:
            points[:] = []
            try:
                _, translation, _ = tracker.motion(previous[0], image, previous[1], attitude)
                if len(points) == 2 and np.linalg.norm(translation) > .3:
                    pairs.append(dict(points=points[:], sources=[previous[2], path]))
            except RuntimeError as exc:
                rejected.append(dict(source=path, reason=str(exc)))
        previous = image, attitude, path
    if len(pairs) < 8:
        raise ValueError('Insufficient moving floor pairs: need eight, found %d' % len(pairs))
    return pairs, rejected


def score(pairs, ground, delta):
    rotation = cv2.Rodrigues(np.array([math.radians(delta), 0., 0.]))[0]
    values = []
    for pair in pairs:
        planes = []
        try:
            for pixels, attitude in pair['points']:
                planes.append(ground(pixels, dict(attitude, down_camera=rotation @ attitude['down_camera'])))
        except RuntimeError:
            return None
        cv2.setRNGSeed(0)
        transform, mask = cv2.estimateAffinePartial2D(*planes, method=cv2.RANSAC,
            ransacReprojThreshold=.35, maxIters=1000, confidence=.99)
        if transform is None or mask is None:
            return None
        use = mask.ravel() > 0
        if use.sum() < 20 or use.mean() < .65:
            return None
        scale = float(np.linalg.norm(transform[:, 0]))
        if not math.isfinite(scale) or scale <= 0:
            return None
        residual_vectors = planes[1][use] - planes[0][use] @ (transform[:, :2] / scale).T
        residual = float(np.median(np.linalg.norm(residual_vectors - np.median(residual_vectors, axis=0), axis=1)))
        values.append([abs(scale - 1), residual])
    mean = np.mean(values, axis=0)
    return dict(pair_count=len(values), mean_absolute_scale_error=float(mean[0]),
                mean_rigid_residual_cm=float(mean[1]))


def fit(directory, pitch_limit=8., step=.25):
    validate_scan(pitch_limit, step)
    with open(os.path.join(ROOT, 'calibration/floor_geometry.json')) as source:
        profile = json.load(source)
    with open(profile['intrinsics_path']) as source:
        intrinsics = json.load(source)
    with open(os.path.join(ROOT, 'calibration/imu_mount.json')) as source:
        mount = json.load(source)
    pairs, rejected = collect_pairs(directory, profile, intrinsics, mount)
    split = len(pairs) // 2
    ground = FloorTracker(profile, intrinsics, 125).ground
    deltas = sorted(set([0.] + [float(v) for v in np.arange(-pitch_limit, pitch_limit + 1e-9, step)]))
    rows = [dict(pitch_delta_degrees=delta, training=score(pairs[:split], ground, delta),
                 heldout=score(pairs[split:], ground, delta)) for delta in deltas]
    valid = [row for row in rows if row['training'] is not None]
    if not valid:
        raise ValueError('No pitch candidate fit all training pairs')
    # Select only by training residual; held-out data never select the candidate.
    chosen = min(valid, key=lambda row: (row['training']['mean_rigid_residual_cm'],
                                        abs(row['pitch_delta_degrees'])))
    baseline = next(row for row in rows if row['pitch_delta_degrees'] == 0.)
    heldout = chosen['heldout']
    improved = bool(heldout is not None and baseline['heldout'] is not None and
        heldout['mean_rigid_residual_cm'] < baseline['heldout']['mean_rigid_residual_cm'] and
        heldout['mean_absolute_scale_error'] <= baseline['heldout']['mean_absolute_scale_error'])
    delta = chosen['pitch_delta_degrees']
    q = cv2.Rodrigues(np.array([math.radians(delta), 0., 0.]))[0]
    return dict(candidate_only=True, installed=False, scene_scoped=True,
        observation_directory=os.path.abspath(directory), pair_count=len(pairs),
        training_sources=[p['sources'] for p in pairs[:split]],
        heldout_sources=[p['sources'] for p in pairs[split:]], rejected_pairs=rejected,
        selection_objective='training mean rigid floor residual; no target observations',
        pitch_delta_degrees=delta, rotation_matrix=q.tolist(),
        selection_at_search_boundary=abs(abs(delta)-pitch_limit) < 1e-8,
        heldout_improves_both_metrics=improved, selected_metrics=chosen,
        baseline_metrics=baseline, scan=rows,
        limitations='Local visible-plane fit only; floor slope and fixed mounting error are not distinguishable. '
        'No metric ground truth, target arrival validation, yaw calibration, or authorization for motion. '
        'A fixed camera-frame correction need not remain valid after turning or entering another floor patch. '
        'No calibration files or controller guards are modified.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('observations')
    parser.add_argument('output_candidate')
    parser.add_argument('--pitch-limit', type=float, default=8.)
    parser.add_argument('--step', type=float, default=.25)
    args = parser.parse_args()
    try:
        validate_config(args.observations, args.output_candidate, args.pitch_limit, args.step)
        cv2.setNumThreads(1)
        result = fit(args.observations, args.pitch_limit, args.step)
        with open(args.output_candidate, 'w') as destination:
            json.dump(result, destination, indent=2)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps({k: result[k] for k in ('pair_count', 'pitch_delta_degrees',
        'heldout_improves_both_metrics', 'selection_at_search_boundary', 'installed')}))


if __name__ == '__main__':
    main()
