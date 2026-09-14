#!/usr/bin/env python3
"""Deterministic held-out stress scenarios for real controller code; offline only."""
import argparse
import json
from simulate_route import run_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    scenarios = [
        ('normal_5cm', {}, 5),
        ('longer_10cm', {}, 10),
        ('longer_15cm', {}, 15),
        ('slow_15cm_deadline', {'speed': 6.}, 15),
        ('coarse_frames', {'camera_hz': 8., 'latency': .04, 'coast': .12}, 5),
        ('slow_processing', {'compute': .14, 'latency': .03}, 5),
        ('exposure_jump', {'exposure_jump': 40.}, 5),
        ('read_noise', {'noise': 8.}, 5),
        ('low_texture', {'low_texture': True}, 5),
        ('stall', {'stall': True}, 5),
        ('sensor_failure', {'frame_fault_after': .25}, 5),
        ('camera_freeze', {'blocked_frame_after': .25}, 5),
        ('external_stop', {'cancel_after': .25}, 5),
        ('imu_drift', {'imu_yaw_bias_dps': 5.}, 5),
        ('height_error_15_percent', {'height_scale': 1.15}, 5),
    ]
    reports = []
    for name, parameters, target in scenarios:
        for budget in (250, 125):
            result = run_case(dict(seed=404, **parameters), True, budget, target)
            result['scenario'] = name
            reports.append(result)
            print(json.dumps(dict(scenario=name, feature_budget=budget,
                                  outcome=result['result']['outcome'],
                                  reason=result['result'].get('reason'), true_error_cm=result['true_error_cm'])), flush=True)
    with open(args.output, 'w') as out:
        json.dump(dict(complete=True, reports=reports), out, indent=2)


if __name__ == '__main__':
    main()
