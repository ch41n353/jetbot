#!/usr/bin/env python3
"""Score the search layer on stored room photographs. No motors, no service.

Two questions, because they fail differently.

The sweep question is a detector, and a detector is judged by what it says about
pictures whose answer is known. `looks` asks it about a labelled list of frames
and counts the four outcomes -- with a false "sure" weighted heavily in the
reading, because that is the one that sends the reach controller off after
something that is not there.

The search loop is a policy, and a policy is judged by what it does with a room.
`replay` takes a real rotation -- the frames of one turn on the spot, in the
order they were taken -- and runs the actual sweep, refinement and planning code
against them, with a stand-in for the wheels that turns the picture instead of
the robot. The heading of each frame is assumed to be evenly spaced around the
rotation, which is the one thing here that is not measured.

Usage:
  evaluate_explore.py looks  --manifest FILE [--repeat N]
  evaluate_explore.py replay --sweep FILE [--repeat N]

Manifest for `looks`:   [{"image": ..., "target": ..., "present": true}, ...]
Sweep file for `replay`: {"target": ..., "span_degrees": 360,
                          "frames": ["...jpg", ...]}

Needs OPENAI_API_KEY.
"""

import argparse
import json
import math
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explore
import fetch
from evaluate_gpt_routes import Lens


class Turntable(Lens):
    """The wheels, replaced by a stack of photographs taken while turning.

    Search only ever asks the robot to turn and to hand over a frame, so a
    rotation that has already happened can be replayed exactly: the heading is
    bookkeeping and the picture is whichever frame was taken nearest that way.
    Everything above it -- the cheap question, the refinement rule, the journal,
    the plan and its validation -- is the shipping code, unmodified.
    """

    def __init__(self, frames, span=360.):
        Lens.__init__(self)
        self.frames = frames
        self.span = float(span)
        self.heading = 0.
        self.asked = []

    def turn(self, degrees):
        self.heading = explore.wrap(self.heading + degrees)
        self.asked.append(degrees)
        return degrees

    def index(self):
        """The frame taken nearest the way the robot is now facing.

        `span` is the whole angle the stored rotation covered, first frame to
        last, and the frames are assumed evenly spaced across it. That
        assumption is the one thing in this harness that is not measured: the
        recorded per-turn angles disagree with the pictures, so the span is
        fitted by eye from where a fixed object sits in the first and last
        frame. It is good to a few degrees, which is enough to choose which
        photograph a heading belongs to and not enough to score a bearing
        against.
        """
        per = self.span / max(1, len(self.frames) - 1)
        best, gap = 0, None
        for index in range(len(self.frames)):
            apart = abs(explore.wrap(index * per - self.heading))
            if gap is None or apart < gap:
                best, gap = index, apart
        return best

    def frame(self):
        path = self.frames[self.index()]
        image = cv2.imread(path)
        if image is None:
            raise fetch.Stop('could not read %s' % path)
        return image, dict(path=path)


def confusion(rows):
    """The four outcomes, and the two that matter, as plain counts."""
    out = dict(sure_hit=0, unsure_hit=0, miss=0, sure_false=0, unsure_false=0,
               clean=0)
    for row in rows:
        sure = row['confidence'] == explore.SURE
        unsure = row['confidence'] == explore.UNSURE
        if row['present']:
            out['sure_hit' if sure else 'unsure_hit' if unsure else 'miss'] += 1
        else:
            out['sure_false' if sure else 'unsure_false' if unsure
                else 'clean'] += 1
    return out


def looks(manifest, repeat, out_dir=None):
    """Ask the cheap question about every labelled frame and count what it said."""
    lens = Lens()
    rows = []
    for item in manifest:
        image = cv2.imread(item['image'])
        if image is None:
            print('%-28s could not be read' % os.path.basename(item['image']))
            continue
        for attempt in range(repeat):
            try:
                answer = explore.look(image, item['target'])
            except (fetch.Stop, ValueError, OSError) as exc:
                print('%-28s error: %s' % (os.path.basename(item['image']), exc))
                continue
            pixel = answer.get('target_pixel')
            angle, reach = None, None
            if isinstance(pixel, dict):
                angle = explore.bearing(lens, pixel['x'], pixel['y'])
                try:
                    reach = round(math.hypot(*lens.ground(pixel['x'], pixel['y'])), 1)
                except (fetch.Stop, TypeError, ValueError):
                    reach = None      # above the horizon: a bearing but no range,
                                      # which is the case this layer is built for
            row = dict(scene=os.path.basename(item['image']),
                       pixel=None if not isinstance(pixel, dict) else
                       [pixel['x'], pixel['y']], range_cm=reach,
                       target=item['target'], present=bool(item.get('present')),
                       confidence=answer.get('confidence'),
                       visible=bool(answer.get('visible')),
                       bearing=None if angle is None else round(angle, 1),
                       edge=explore.edge_sighting(pixel),
                       closer=bool(answer.get('worth_a_closer_look')),
                       blocked=explore.blocked_ahead(answer),
                       open_cm=open_range(lens, answer),
                       scene_words=str(answer.get('scene', ''))[:44])
            rows.append(row)
            print('%-34s %-5s %-6s %-7s %-8s %-6s %s' % (
                row['scene'][-34:], 'here' if row['present'] else 'absent',
                row['confidence'],
                '' if row['bearing'] is None else '%+.0f deg' % row['bearing'],
                '' if row['range_cm'] is None else '%.0f cm' % row['range_cm'],
                'look' if row['closer'] else '', row['scene_words']))
    tally = confusion(rows)
    print('\nof %d looks: %d sure and right, %d unsure and right, %d missed it, '
          '%d sure and wrong, %d unsure and wrong, %d correctly said nothing'
          % (len(rows), tally['sure_hit'], tally['unsure_hit'], tally['miss'],
             tally['sure_false'], tally['unsure_false'], tally['clean']))
    if out_dir:
        with open(os.path.join(out_dir, 'looks.json'), 'w') as handle:
            json.dump(dict(rows=rows, tally=tally), handle, indent=1)
    return rows


def open_range(lens, answer):
    """How far the open patch it chose actually is, in centimetres."""
    point = answer.get('open_pixel')
    if not isinstance(point, dict):
        return None
    try:
        spot = lens.ground(float(point['x']), float(point['y']))
    except (fetch.Stop, KeyError, TypeError, ValueError):
        return None
    return round(math.hypot(*spot), 1)


def replay(spec, repeat, out_dir=None):
    """Run the real sweep and planner against a rotation that already happened."""
    for attempt in range(repeat):
        wheels = Turntable(spec['frames'], spec.get('span_degrees', 360.))
        events = []

        def record(event, **fields):
            events.append(dict(event=event, **fields))
            print(json.dumps(dict(event=event, **fields), default=str))

        hunt = explore.Search(wheels, spec['target'], record=record,
                              step_deg=spec.get('step_degrees',
                                                explore.SWEEP_STEP_DEG))
        found = hunt.sweep()
        print('\n--- journal ---\n%s\n' % hunt.journal.render())
        if found is not None:
            print('found at %+.0f, aiming %+.0f: %s'
                  % (found['heading'], found['candidate_bearing'] or 0.,
                     found['scene']))
        else:
            steps, note = explore.propose(hunt.journal, record=record)
            print('plan: %s' % note)
            for step in steps:
                print('  %s' % json.dumps(step, default=str))
        if out_dir:
            with open(os.path.join(out_dir, 'replay-%d.json' % (attempt + 1)),
                      'w') as handle:
                json.dump(dict(events=events, journal=hunt.journal.render()),
                          handle, indent=1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('looks', 'replay'))
    parser.add_argument('--manifest')
    parser.add_argument('--sweep')
    parser.add_argument('--repeat', type=int, default=1,
                        help='the model is not deterministic and one answer is '
                             'not a measurement')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    if not os.environ.get('OPENAI_API_KEY'):
        raise SystemExit('OPENAI_API_KEY is not set')
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    if args.mode == 'looks':
        if not args.manifest:
            parser.error('looks wants --manifest')
        with open(args.manifest) as handle:
            looks(json.load(handle), args.repeat, args.out)
        return 0
    if not args.sweep:
        parser.error('replay wants --sweep')
    with open(args.sweep) as handle:
        replay(json.load(handle), args.repeat, args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
