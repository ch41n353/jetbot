#!/usr/bin/env python3
"""Offline feature-count comparison on fixed recorded frames; no hardware."""
import argparse
import json
from unittest.mock import patch
import cv2
import numpy as np
from benchmark_tracker import load_pairs, measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    pairs = load_pairs(args.directories, limit=160)
    original = cv2.goodFeaturesToTrack
    reports = []
    for budget in (250, 125, 80, 250):
        def detect(image, max_corners, *fields, **kwargs):
            return original(image, budget, *fields, **kwargs)
        with patch('cv2.goodFeaturesToTrack', side_effect=detect):
            report = measure(pairs, threads=4, repeats=2)
        report['feature_budget'] = budget
        reports.append(report)
        print(json.dumps({k: v for k, v in report.items() if k != 'results'}), flush=True)
    with open(args.output, 'w') as target:
        json.dump(dict(reports=reports), target, indent=2)


if __name__ == '__main__':
    main()
