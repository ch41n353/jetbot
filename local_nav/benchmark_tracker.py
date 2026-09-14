#!/usr/bin/env python3
"""Offline, fixed-input tracker timing; never opens hardware or robot sockets."""
import argparse
import glob
import json
import os
import time
import cv2
import numpy as np
from point_controller import ROOT, FloorTracker
from state_estimator import AttitudeTimeline


def load_pairs(directories, limit=80):
    pairs = []
    for directory in directories:
        with open(os.path.join(ROOT, 'calibration/imu_mount.json')) as source:
            timeline = AttitudeTimeline(json.load(source))
        previous = None
        for path in sorted(glob.glob(os.path.join(directory, '*.json'))):
            with open(path) as source:
                observation = json.load(source)
            timeline.feed(observation['imu_samples'])
            attitude = timeline.at(observation['time'])
            image = cv2.imread(path[:-5] + '.jpg')
            if image is None:
                raise RuntimeError('Missing recorded image: ' + path)
            current = image, attitude
            if previous:
                pairs.append((previous[0], image, previous[1], attitude))
                if len(pairs) >= limit:
                    return pairs
            previous = current
    if not pairs:
        raise RuntimeError('No recorded frame pairs')
    return pairs


def measure(pairs, threads, repeats=2):
    cv2.setNumThreads(threads)
    with open(os.path.join(ROOT, 'calibration/floor_geometry.json')) as source:
        profile = json.load(source)
    with open(profile['intrinsics_path']) as source:
        tracker = FloorTracker(profile, json.load(source))
    timings, results = [], []
    for repeat in range(repeats):
        for index, pair in enumerate(pairs):
            cv2.setRNGSeed(1234 + index)
            started = time.perf_counter()
            try:
                r, t, q = tracker.motion(*pair)
                outcome = dict(rotation=r.tolist(), translation=t.tolist(), **q)
            except RuntimeError as exc:
                outcome = dict(error=str(exc))
            timings.append(1000 * (time.perf_counter() - started))
            if repeat == 0:
                results.append(outcome)
    return dict(threads=threads, pairs=len(pairs), repeats=repeats,
                median_ms=float(np.median(timings)), p95_ms=float(np.percentile(timings, 95)),
                max_ms=max(timings), rejected=sum('error' in r for r in results), results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+')
    parser.add_argument('--threads', nargs='+', type=int, default=[1, 2, 4])
    parser.add_argument('--limit', type=int, default=80)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not 1 <= args.limit <= 500 or not 1 <= args.repeats <= 20 or any(t not in (1, 2, 4) for t in args.threads):
        parser.error('Bounded benchmark: limit 1–500, repeats 1–20, threads 1/2/4')
    pairs = load_pairs(args.directories, args.limit)
    default_threads = cv2.getNumThreads()
    reports = [measure(pairs, threads, args.repeats) for threads in args.threads]
    output = dict(opencv=cv2.__version__, original_threads=default_threads, reports=reports)
    with open(args.output, 'w') as target:
        json.dump(output, target, indent=2)
    print(json.dumps(dict(original_threads=default_threads,
                         reports=[{k: v for k, v in r.items() if k != 'results'} for r in reports]), indent=2))


if __name__ == '__main__':
    main()
