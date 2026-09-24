#!/usr/bin/env python3
"""Failure taxonomy for the chained local-executor trials.

Groups outcomes by the executor's own phase and error text, so a run answers
"what breaks and how often" rather than "did it pass". Reads the JSONL written
by local_reach_loop.py in the current USB run root.
"""
import collections
import json
import os
import re
import sys


def normalise(error):
    """Collapse the numbers out of an error so identical faults group."""
    if not error:
        return None
    text = re.sub(r'-?\d+\.?\d*', 'N', str(error))
    return text[:90]


def load(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def main():
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        with open('/mnt/robotlogs/current-search.json') as handle:
            root = json.load(handle)['root']
        paths = [os.path.join(root, 'local-reach-chain.jsonl')]

    rows = [r for path in paths for r in load(path)]
    if not rows:
        print('no records')
        return

    print('%d cycle(s)' % len(rows))
    print()
    phases = collections.Counter(r.get('phase') for r in rows)
    width = max(len(str(p)) for p in phases)
    for phase, count in phases.most_common():
        print('  %-*s %3d  (%.0f%%)' % (width, phase, count,
                                        100. * count / len(rows)))

    print()
    print('by requested shape:')
    shapes = collections.defaultdict(collections.Counter)
    for r in rows:
        shapes[r.get('shape')][r.get('phase')] += 1
    for shape in sorted(shapes):
        inner = shapes[shape]
        total = sum(inner.values())
        ok = inner.get('completed', 0) + inner.get('paused', 0)
        print('  %-14s %2d run(s), %2d finished  %s'
              % (shape, total, ok,
                 ', '.join('%s x%d' % (p, n) for p, n in inner.most_common()
                           if p not in ('completed', 'paused')) or '-'))

    faults = collections.Counter(normalise(r.get('error')) for r in rows
                                 if r.get('error'))
    if faults:
        print()
        print('distinct faults:')
        for text, count in faults.most_common():
            print('  x%-3d %s' % (count, text))

    driven = [r.get('travelled_cm') or 0. for r in rows]
    asked = []
    for r in rows:
        points = r['trajectory']['waypoints_cm']
        total, last = 0., [0., 0.]
        for point in points:
            total += ((point[0] - last[0]) ** 2 + (point[1] - last[1]) ** 2) ** .5
            last = point
        asked.append(total)
    print()
    print('driven %.0f cm total over %d cycle(s); asked %.0f cm'
          % (sum(driven), len(rows), sum(asked)))
    rejections = [r.get('tracking_rejections') or 0 for r in rows]
    print('tracking rejections: %d total, worst single run %d'
          % (sum(rejections), max(rejections) if rejections else 0))
    blocked = sum(1 for r in rows if r.get('blocked_after'))
    print('cycles ending blocked (next goal sent behind): %d' % blocked)
    packs = [r['pack_v'] for r in rows if r.get('pack_v')]
    if packs:
        print('pack %.2f -> %.2f V' % (packs[0], packs[-1]))


if __name__ == '__main__':
    main()
