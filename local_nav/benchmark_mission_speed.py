"""Offline mission timing benchmark; never connects to robot hardware.

Times are simulated command intervals, not measured hardware timing. Watchdog
faults make commanded powered duration unreliable and are reported explicitly.
"""
import argparse
import hashlib
import json
import os
import time
import cv2
import object_mission
from simulate_object_mission import run_case


def summarize(run):
    result = run['result']
    commands = run['commands']
    start = commands[0]['time']
    end = start + result['elapsed_seconds']
    intervals = []
    mode = 'zero'
    since = start
    transitions = 0
    for command in commands:
        action, fields = command['action'], command['fields']
        if action not in ('motors_hold', 'stop'):
            continue
        left, right = (0., 0.) if action == 'stop' else (fields['left'], fields['right'])
        next_mode = 'zero' if not (left or right) else 'turn' if left * right < 0 else 'drive'
        if next_mode == mode:
            continue
        if mode != 'zero' and next_mode == 'zero':
            transitions += 1
        intervals.append(dict(mode=mode, start_seconds=since-start,
                              duration_seconds=command['time']-since))
        mode, since = next_mode, command['time']
    intervals.append(dict(mode=mode, start_seconds=since-start, duration_seconds=max(0., end-since)))
    totals = {label: sum(row['duration_seconds'] for row in intervals if row['mode'] == label)
              for label in ('zero', 'drive', 'turn')}
    recovery_start = None
    recovery_seconds = 0.
    for event in result['events']:
        if event['event'] == 'local_recovery':
            recovery_start = event['elapsed_seconds']
        elif recovery_start is not None and event['event'] in ('target_reacquired', 'remote_planner_required'):
            recovery_seconds += event['elapsed_seconds'] - recovery_start
            recovery_start = None
    if recovery_start is not None:
        recovery_seconds += result['elapsed_seconds'] - recovery_start
    return dict(outcome=result['outcome'], reason=result.get('reason'),
                elapsed_seconds=result['elapsed_seconds'],
                commanded_powered_seconds=totals['drive'] + totals['turn'],
                commanded_zero_seconds=totals['zero'], drive_seconds=totals['drive'],
                turn_seconds=totals['turn'], powered_to_zero_transitions=transitions,
                longest_powered_interval_seconds=max([row['duration_seconds'] for row in intervals
                                                     if row['mode'] != 'zero'] or [0.]),
                recovery_seconds=recovery_seconds, local_recoveries=result['local_recoveries'],
                local_searches=result['local_searches'], events=result['events'], intervals=intervals,
                true_pose=run['true_pose'], true_target_range_cm=run['true_target_range_cm'],
                standoff_error_cm=run['true_target_range_cm']-20.,
                clearance_violation_count=len(run['clearance_violations']),
                watchdog_stops=run['watchdog_stops'], final_motor_output=run['final_motor_output'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--perception-ms', type=float, default=45.)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    source_path = object_mission.__file__.replace('.pyc', '.py')
    with open(source_path, 'rb') as source:
        controller_source = source.read()
    report = dict(controller_sha256=hashlib.sha256(controller_source).hexdigest(),
                  controller_source=controller_source.decode(),
                  perception_compute_seconds=args.perception_ms/1000.,
                  timing_basis='simulated commanded output; excludes real Python search/render wall time',
                  cases=[], complete=False)
    report['component_sha256']={}
    for name in ('point_controller.py','target_tracker.py','spatial_planner.py'):
        with open(os.path.join(os.path.dirname(source_path),name),'rb') as source:
            report['component_sha256'][name]=hashlib.sha256(source.read()).hexdigest()
    cases = [('angled_seed1', (-17, 60)), ('aligned_seed1', (0, 100))]
    for name, target in cases:
        started = time.monotonic()
        parameters = dict(speed=11, coast=.05, seed=1, pivot_cm=6)
        run = run_case(parameters, target=target, perception_compute=args.perception_ms/1000.)
        report['cases'].append(dict(name=name, parameters=parameters, target_cm=target,
                                    benchmark_wall_seconds=time.monotonic()-started,
                                    **summarize(run)))
        with open(args.output, 'w') as out:
            json.dump(report, out, indent=2)
        print(json.dumps({k:v for k,v in report['cases'][-1].items() if k not in ('events', 'intervals')}), flush=True)
    report['complete'] = True
    with open(args.output, 'w') as out:
        json.dump(report, out, indent=2)


if __name__ == '__main__':
    main()
