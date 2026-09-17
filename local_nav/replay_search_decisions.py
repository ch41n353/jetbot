#!/usr/bin/env python3
"""Replay every recorded capture through the live decision path, motor-free.

Each recorded run left a frame, its sensor metadata and the recognizer answer
that was made from it. This drives the same code a powered run would -- map
certification, candidate selection, aiming, approach preparation and the route
preview -- and reports what each one would do now. It moves no motors, opens no
service and makes no API call, so it can hunt remaining stop conditions while
the robot is on charge.

It is a decision replay, not a simulation of the plant: controllers, tracking
and arrival are not exercised, so a clean report here does not mean a run will
succeed. It only proves these particular frames no longer hit a refusal.
"""
import argparse
import collections
import glob
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'local_nav'))

import cv2

from point_controller import FloorTracker
from route_executor import load_json
from sol_search import SearchState, local_map
from spatial_planner import SpatialPlanner, sweep


def recorded_cases(goals):
    """Every step frame paired with the recognizer answer taken from it."""
    cases = []
    for events in sorted(glob.glob(os.path.join(goals, '**', '*.events.jsonl'),
                                   recursive=True)):
        prefix = events[:-len('.events.jsonl')]
        answers, order = [], 0
        for line in open(events):
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get('event') == 'local_action_complete':
                order += 1
            elif event.get('event') == 'sol_result':
                answer = (event.get('response') or {}).get('answer')
                if isinstance(answer, dict):
                    answers.append((order, answer))
        for index, answer in answers:
            frame = '%s.step-%02d.jpg' % (prefix, index)
            meta = '%s.step-%02d.capture.json' % (prefix, index)
            if os.path.isfile(frame) and os.path.isfile(meta):
                cases.append((frame, meta, answer))
    return cases


def replay(frame, meta, answer, tracker, plan):
    """What the search would decide from this frame, or why it would refuse."""
    state = SearchState(dict(plan))
    state.certify(tracker)
    state.retain_obstacles(tracker, answer)
    choice = state.select(answer, state.candidates())
    if choice.get('kind') != 'approach':
        return choice.get('kind', 'hold'), None

    box = answer['target_box']
    coords = [float(box[k]) for k in ('x0', 'y0', 'x1', 'y1')]
    pixel = answer.get('contact_pixel') or {}
    if not (coords[0] <= pixel.get('x', -1) <= coords[2]
            and abs(pixel.get('y', -1) - coords[3]) <= 8):
        return 'refused', 'Sol floor contact inconsistent with target box'

    ground = tracker.ground([[(coords[0] + coords[2]) / 2, coords[3]]])[0]
    bearing = math.degrees(math.atan2(ground[0], ground[1]))
    if abs(bearing) > 12.:
        planner = SpatialPlanner(state.mapped(goal_cm=state.pose[:2],
                                              measured_map_drive=True,
                                              precise_turn_sweep=True))
        turn = max(-30., min(30., bearing))
        if planner.clear(sweep(state.pose, ('turn', turn), precise_turn=True)):
            return 'aim', None
        return 'refused', 'cannot turn to face a target %.0f degrees off axis' % bearing

    capture = json.load(open(meta))
    capture.update(image_path=frame, metadata_path=meta,
                   captured_monotonic=capture['time'])
    import object_mission
    from spatial_preview import write_preview
    real = time.monotonic
    time.monotonic = lambda: capture['time'] + 1.
    try:
        corridor = local_map(dict(plan), state.pose,
                             state.mapped()['inspected_free_rectangles_cm'])
        request = dict(corridor, image_path=frame, target_box=coords,
                       target_label=plan.get('target_label', 'target'),
                       standoff_cm=plan.get('standoff_cm', 25),
                       target_radius_cm=plan.get('target_radius_cm', 7),
                       projective_contact=True, obstacle_image_boxes=[])
        mission, preview = object_mission.prepare(capture, request)
        scratch = os.path.join(os.path.dirname(frame), '.replay-preview.html')
        write_preview(preview, scratch)
        os.remove(scratch)
        return 'approach', None
    except Exception as exc:
        return 'refused', str(exc)
    finally:
        time.monotonic = real


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--goals', default=os.path.join(ROOT, 'local_nav/goals'))
    parser.add_argument('--show', type=int, default=6,
                        help='example frames to name per refusal reason')
    args = parser.parse_args()

    profile = load_json(os.path.join(ROOT, 'calibration/floor_geometry.json'))
    tracker = FloorTracker(profile, load_json(profile['intrinsics_path']), 125)
    plan = dict(inspected_free_rectangle_cm=[-35, -35, 35, 45],
                obstacle_rectangles_cm=[], search_stations_cm=[],
                scan_step_degrees=10, scan_direction=1,
                target_label='Advil bottle', standoff_cm=25, target_radius_cm=7)

    cases = recorded_cases(args.goals)
    outcomes = collections.Counter()
    refusals = collections.defaultdict(list)
    for frame, meta, answer in cases:
        try:
            kind, reason = replay(frame, meta, answer, tracker, plan)
        except Exception as exc:
            kind, reason = 'error', '%s: %s' % (type(exc).__name__, exc)
        outcomes[kind] += 1
        if reason:
            refusals[reason].append(os.path.basename(frame))

    print('replayed %d recorded frames through the live decision path' % len(cases))
    for kind, count in outcomes.most_common():
        print('  %-9s %d' % (kind, count))
    if refusals:
        print('\nrefusals still reachable:')
        for reason, frames in sorted(refusals.items(), key=lambda kv: -len(kv[1])):
            print('  %3d  %s' % (len(frames), reason[:100]))
            for name in frames[:args.show]:
                print('       %s' % name)
    else:
        print('\nno refusal on any recorded frame')
    return 1 if refusals else 0


if __name__ == '__main__':
    sys.exit(main())
