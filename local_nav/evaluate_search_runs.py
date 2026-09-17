#!/usr/bin/env python3
"""Score recorded search runs: recognizer behaviour and controller behaviour.

Reads only what live runs already wrote (event journals and turn logs), so it
costs no battery and no API calls. It reports what actually happened, including
failures; it does not decide whether a recognition was correct, because the
journals hold no ground truth for that.
"""
import argparse
import collections
import glob
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_events(path):
    out = []
    with open(path) as source:
        for line in source:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def session_of(path):
    return os.path.basename(path).split('-route-')[0]


def score_runs(patterns):
    runs = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            events = load_events(path)
            if not events:
                continue
            calls = [e for e in events if e.get('event') == 'sol_result']
            actions = [e for e in events if e.get('event') == 'local_action_complete']
            stops = [e for e in events if e.get('event') in
                     ('search_stopped', 'localization_required')]
            visible = [e for e in calls
                       if (e.get('response', {}).get('answer') or {}).get('target_visible')]
            latency = []
            starts = {}
            for e in events:
                if e.get('event') == 'sol_prefetched':
                    starts[len(latency)] = e.get('elapsed_seconds')
                elif e.get('event') == 'sol_result':
                    begin = starts.get(len(latency))
                    if begin is not None:
                        latency.append(e.get('elapsed_seconds', 0) - begin)
            runs.append(dict(
                path=path, session=session_of(path),
                sol_calls=len(calls), visible=len(visible), actions=len(actions),
                latency=latency,
                rotation=sum(abs(a['pose'][2] - p['pose'][2]) if p else abs(a['pose'][2])
                             for p, a in zip([None] + actions, actions)),
                final_pose=actions[-1]['pose'] if actions else None,
                elapsed=events[-1].get('elapsed_seconds', 0.),
                stop=(stops[0].get('reason') or stops[0].get('event')) if stops else None,
                obstacles=sum(len((e.get('response', {}).get('answer') or {}).get('obstacles') or [])
                              for e in calls)))
    return runs


def score_turns(patterns):
    turns = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            try:
                with open(path) as source:
                    data = json.load(source)
            except (ValueError, IOError):
                continue
            if 'target_degrees' not in data:
                continue
            samples = data.get('samples') or []
            settling = [s for s in (data.get('settling_samples') or [])
                        if isinstance(s, dict) and 'yaw_rate_degrees_s' in s]
            peak = max([abs(s['yaw_rate_degrees_s']) for s in samples + settling] or [0.])
            power = data.get('power_start') or {}
            turns.append(dict(
                path=path, session=session_of(path),
                requested=data.get('target_degrees'),
                final=data.get('final_angle_degrees'),
                outcome=data.get('outcome'), reason=data.get('reason'),
                peak_rate=peak, pack=power.get('pack_voltage_v'),
                powered=data.get('powered_seconds')))
    return turns


def report(runs, turns):
    print('=' * 72)
    print('RECOGNIZER (GPT-5.6 Sol)')
    print('=' * 72)
    calls = sum(r['sol_calls'] for r in runs)
    visible = sum(r['visible'] for r in runs)
    latency = [v for r in runs for v in r['latency'] if v is not None]
    print('runs with events        : %d' % len(runs))
    print('recognition calls       : %d' % calls)
    print('reported target visible : %d (%.0f%% of calls)'
          % (visible, 100. * visible / calls if calls else 0.))
    print('obstacle boxes returned : %d' % sum(r['obstacles'] for r in runs))
    if latency:
        latency.sort()
        print('latency seconds         : median %.2f  min %.2f  max %.2f'
              % (latency[len(latency) // 2], latency[0], latency[-1]))
    print('NOTE: journals carry no ground truth, so "visible" counts what the')
    print('      model claimed, not whether it was right.')

    print()
    print('=' * 72)
    print('SCAN TURN CONTROLLER')
    print('=' * 72)
    done = [t for t in turns if t['outcome'] == 'turn_reached_imu_estimate']
    failed = [t for t in turns if t['outcome'] != 'turn_reached_imu_estimate']
    print('turns attempted         : %d' % len(turns))
    print('turns completed         : %d (%.0f%%)'
          % (len(done), 100. * len(done) / len(turns) if turns else 0.))
    if done:
        err = [t['final'] - t['requested'] for t in done
               if t['final'] is not None and t['requested']]
        if err:
            print('angle error degrees     : mean %+.2f  min %+.2f  max %+.2f'
                  % (sum(err) / len(err), min(err), max(err)))
            print('                          (negative = short of the request)')
        rate = [t['peak_rate'] for t in done if t['peak_rate']]
        if rate:
            print('peak yaw rate deg/s     : median %.1f  max %.1f'
                  % (sorted(rate)[len(rate) // 2], max(rate)))
    if failed:
        print('failure reasons:')
        for reason, count in collections.Counter(
                t['reason'] or t['outcome'] for t in failed).most_common():
            print('   %3d  %s' % (count, reason))
    packs = [t['pack'] for t in turns if t['pack']]
    if packs:
        print('pack volts at turn start: max %.2f  min %.2f' % (max(packs), min(packs)))

    print()
    print('=' * 72)
    print('RUN OUTCOMES')
    print('=' * 72)
    for reason, count in collections.Counter(
            r['stop'] or 'no recorded stop' for r in runs).most_common():
        print('   %3d  %s' % (count, reason))
    best = max(runs, key=lambda r: r['actions']) if runs else None
    if best:
        print()
        print('deepest run: %s' % os.path.basename(best['path']))
        print('   %d local actions, %d recognition calls, %.1f s, final pose %s'
              % (best['actions'], best['sol_calls'], best['elapsed'], best['final_pose']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--goals', default=os.path.join(ROOT, 'local_nav/goals'))
    parser.add_argument('--since', help='only sessions at or after this stamp, '
                                        'e.g. 20260915-230000')
    args = parser.parse_args()
    events = [os.path.join(args.goals, '*.events.jsonl'),
              os.path.join(args.goals, 'dashboard', '*.events.jsonl')]
    turn_logs = [os.path.join(args.goals, '*.turn.json'),
                 os.path.join(args.goals, 'dashboard', '*.turn.json')]
    runs, turns = score_runs(events), score_turns(turn_logs)
    if args.since:
        keep = lambda row: row['session'].replace('session-', '') >= args.since
        runs, turns = [r for r in runs if keep(r)], [t for t in turns if keep(t)]
        print('filtered to sessions from %s: %d runs, %d turns\n'
              % (args.since, len(runs), len(turns)))
    report(runs, turns)


if __name__ == '__main__':
    main()
