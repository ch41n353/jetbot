#!/usr/bin/env python3
"""Compare a braking hypothesis on held-out simulated plants; no hardware."""
import argparse
import json
import os
from unittest.mock import patch
import numpy as np
from braking import braking_distance
from simulate_route import run_case


def candidate_braking_distance(speed, age, interval):
    braking_distance(speed, age, interval)  # Preserve the existing input validation.
    # Hypothesis: 80 ms coast instead of 40 ms, bounded by a 220 ms horizon.
    # No physical calibration or automatic deployment is implied.
    return min(3., max(0., speed) * min(.22, age + .5 * interval + .08))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=int, default=16)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not 1 <= args.cases <= 100:
        parser.error('cases must be 1–100')
    rng = np.random.RandomState(14092026)
    reports = []
    for seed in range(args.cases):
        parameters = dict(speed=float(rng.uniform(8, 16)), coast=float(rng.uniform(.035, .12)),
                          camera_hz=float(rng.uniform(10, 15)), latency=float(rng.uniform(.005, .035)),
                          compute=.045, seed=1001 + seed, noise=float(rng.uniform(0, 2)))
        original = run_case(parameters, True)
        with patch('route_executor.braking_distance', candidate_braking_distance):
            candidate = run_case(parameters, True)
        reports.append(dict(original=original, candidate=candidate))
        print(json.dumps(dict(case=seed, original_error=original['true_error_cm'],
                              candidate_error=candidate['true_error_cm'],
                              original_outcome=original['result']['outcome'],
                              candidate_outcome=candidate['result']['outcome'])), flush=True)
        with open(args.output + '.tmp', 'w') as source:
            json.dump(dict(complete=seed == args.cases - 1, hypothesis='80 ms coast, 220 ms horizon, 3 cm cap',
                           held_out_seed=14092026, reports=reports), source, indent=2)
        os.replace(args.output + '.tmp', args.output)


if __name__ == '__main__':
    main()
