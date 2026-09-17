#!/usr/bin/env python3
"""Score the recognizer's trajectories on stored room photographs.

The reach controller asks the model for a route and then makes the wheels
follow it. Whether that route is any good has until now only been visible by
driving it, one run at a time, on a robot that has to be put back afterwards.
This replays saved frames instead: no motors, no service, no robot.

Two kinds of check, because they answer different questions.

The arithmetic ones need no ground truth. A route is the model's own claim and
its obstacle list is the model's own claim, so a route that drives through its
own reported obstacle is wrong on its own terms, and can be said so without
anyone labelling the picture. Same for a route that starts behind the robot,
runs backwards, or leaves the floor the calibration can place.

The one that needs eyes is whether the obstacle list is complete -- a cable the
model never mentioned is invisible to every arithmetic check here. So each
scene is also rendered, route and obstacles drawn on the photograph beside a
plan view, for a person to look at.

Usage:
  evaluate_gpt_routes.py --scene IMAGE:TARGET [--scene ...] [--out DIR]
  evaluate_gpt_routes.py --manifest FILE [--out DIR] [--repeat N]

Needs OPENAI_API_KEY. Writes <out>/<name>.overlay.jpg and <out>/report.json.
"""

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A route point is judged against the same corridor the wheels will use, so
# that "the planner would refuse this" means exactly what it means in fetch.
HALF_WIDTH_CM = fetch.CORRIDOR_HALF_CM
OBSTACLE_RADIUS_CM = fetch.OBSTACLE_RADIUS_CM
START_NEAR_CM = 35.        # a first waypoint further out than this skips floor
                           # the robot cannot see past, and is not a start
INDIRECT_RATIO = 1.8       # a detour costs distance; this much over the direct
                           # line is a way round, more than this is lost

PLAN_PIXELS = 420
PLAN_RANGE_CM = 160.


class Lens(object):
    """Just enough of Robot to place a pixel on the floor, with no hardware.

    fetch.Robot opens the control socket in its constructor because everything
    else it does needs the wheels. Projection does not, and holding a socket
    open to score a photograph would mean this could not run while the robot is
    doing something else.
    """

    ground = fetch.Robot.ground

    def __init__(self):
        with open(os.path.join(ROOT, 'calibration/floor_geometry.json')) as handle:
            profile = json.load(handle)
        with open(profile['intrinsics_path']) as handle:
            intrinsics = json.load(handle)
        self.K = np.asarray(intrinsics['K'], dtype=float)
        self.D = np.asarray(intrinsics['D'], dtype=float)
        self.height = float(profile['camera_height_cm'])
        self.pitch = math.radians(profile['pitch_degrees'])


def segment_clearance(start, end, point):
    """Shortest distance from `point` to the travelled segment, in cm."""
    sx, sz = start
    ex, ez = end
    dx, dz = ex - sx, ez - sz
    span = dx * dx + dz * dz
    if span < 1e-9:
        return math.hypot(point[0] - sx, point[1] - sz)
    along = ((point[0] - sx) * dx + (point[1] - sz) * dz) / span
    along = max(0., min(1., along))
    return math.hypot(point[0] - (sx + along * dx), point[1] - (sz + along * dz))


def score(route, obstacles, goal):
    """What is wrong with this route, judged only against what the model said.

    Returns (findings, closest) where findings is a list of plain sentences and
    closest is the tightest clearance to any reported obstacle, in cm.
    """
    findings = []
    if not route:
        return ['no route returned'], None

    first = math.hypot(*route[0])
    if first > START_NEAR_CM:
        findings.append('starts %.0f cm out, not in front of the robot' % first)
    if route[0][1] < 0.:
        findings.append('starts behind the robot')

    for index in range(1, len(route)):
        back = route[index - 1]
        here = route[index]
        if math.hypot(here[0] - back[0], here[1] - back[1]) < 1e-6:
            findings.append('waypoint %d repeats the one before' % (index + 1))

    # Indirectness is judged over the whole route, not per waypoint. Going
    # sideways is what avoiding something looks like, so a step that increases
    # the range to the target is not by itself a fault -- an implausibly long
    # way round is.
    if goal is not None:
        legs = [(0., 0.)] + list(route)
        travelled = sum(math.hypot(legs[i + 1][0] - legs[i][0],
                                   legs[i + 1][1] - legs[i][1])
                        for i in range(len(legs) - 1))
        direct = math.hypot(*goal)
        if direct > 1. and travelled > INDIRECT_RATIO * direct:
            findings.append('route is %.1fx longer than driving straight at it'
                            % (travelled / direct))
        miss = math.hypot(goal[0] - route[-1][0], goal[1] - route[-1][1])
        if miss > fetch.STEP_MAX_CM:
            findings.append('route ends %.0f cm from the target it was asked for'
                            % miss)

    closest = None
    blocked = HALF_WIDTH_CM + OBSTACLE_RADIUS_CM
    legs = [(0., 0.)] + list(route)
    for label, spot in obstacles:
        gap = min(segment_clearance(legs[i], legs[i + 1], spot)
                  for i in range(len(legs) - 1))
        if closest is None or gap < closest:
            closest = gap
        if gap < blocked:
            findings.append('drives within %.0f cm of its own "%s" '
                            '(needs %.0f cm)' % (gap, label, blocked))
    return findings, closest


def would_execute(route, obstacles):
    """How far fetch.follow would actually get before the corridor stops it.

    The model's route is only worth what the wheels can do with it, and the
    executor refuses legs that run into a reported obstacle. Driving it dry
    gives the honest answer without touching a motor.
    """
    events = []

    class Dry(object):
        dry_run = True
        speed = fetch.SPEED_CM_S

        def turn(self, degrees):
            return degrees

    pose = fetch.follow(Dry(), None, route, lambda k, **f: events.append((k, f)),
                        obstacles)
    reached = sum(1 for k, f in events if k == 'waypoint' and f.get('reached'))
    stopped = [f for k, f in events if k == 'leg_blocked']
    return dict(waypoints=len(route), reached=reached,
                blocked_by=stopped[0]['blocked_by'] if stopped else None,
                ended_at=[round(v, 1) for v in pose])


def plan_view(route, obstacles, goal, repaired=()):
    """Bird's-eye of the same answer: overhead is where clearance is legible."""
    canvas = np.full((PLAN_PIXELS, PLAN_PIXELS, 3), 32, np.uint8)
    scale = PLAN_PIXELS / (2. * PLAN_RANGE_CM)

    def place(right, forward):
        return (int(PLAN_PIXELS / 2. + right * scale),
                int(PLAN_PIXELS - forward * scale))

    for ring in range(25, int(PLAN_RANGE_CM) + 1, 25):
        cv2.circle(canvas, place(0., 0.), int(ring * scale), (60, 60, 60), 1)
        cv2.putText(canvas, '%d' % ring, (PLAN_PIXELS // 2 + 3,
                    place(0., ring)[1]), cv2.FONT_HERSHEY_PLAIN, .7, (90, 90, 90), 1)
    for label, spot in obstacles:
        centre = place(*spot)
        cv2.circle(canvas, centre, max(3, int(OBSTACLE_RADIUS_CM * scale)),
                   (60, 60, 220), -1)
        cv2.circle(canvas, centre,
                   max(4, int((OBSTACLE_RADIUS_CM + HALF_WIDTH_CM) * scale)),
                   (60, 60, 220), 1)
        cv2.putText(canvas, label[:14], (centre[0] + 6, centre[1] - 6),
                    cv2.FONT_HERSHEY_PLAIN, .8, (120, 150, 255), 1)
    track = [place(0., 0.)] + [place(*p) for p in route]
    for index in range(1, len(track)):
        cv2.line(canvas, track[index - 1], track[index], (90, 230, 90), 2)
    for index, point in enumerate(track[1:]):
        cv2.circle(canvas, point, 4, (90, 230, 90), -1)
        cv2.putText(canvas, str(index + 1), (point[0] + 5, point[1] + 4),
                    cv2.FONT_HERSHEY_PLAIN, .8, (160, 255, 160), 1)
    mended = [place(0., 0.)] + [place(*p) for p in repaired]
    for index in range(1, len(mended)):
        cv2.line(canvas, mended[index - 1], mended[index], (230, 200, 90), 2)
    if goal is not None:
        cv2.drawMarker(canvas, place(*goal), (60, 220, 220), cv2.MARKER_STAR, 14, 2)
    cv2.drawMarker(canvas, place(0., 0.), (255, 255, 255), cv2.MARKER_TRIANGLE_UP,
                   12, 2)
    return canvas


def overlay(image, answer, route, obstacles, goal, findings, repaired=()):
    """The photograph with the answer drawn on it, beside the plan view."""
    shot = image.copy()
    pixels = [(float(p['x']), float(p['y']))
              for p in answer.get('route_pixels') or []
              if isinstance(p, dict) and 'x' in p and 'y' in p]
    for index in range(1, len(pixels)):
        cv2.line(shot, tuple(int(v) for v in pixels[index - 1]),
                 tuple(int(v) for v in pixels[index]), (90, 230, 90), 2)
    for index, point in enumerate(pixels):
        cv2.circle(shot, tuple(int(v) for v in point), 5, (90, 230, 90), -1)
        cv2.putText(shot, str(index + 1),
                    (int(point[0]) + 6, int(point[1]) - 6),
                    cv2.FONT_HERSHEY_PLAIN, 1., (160, 255, 160), 1)
    for item in answer.get('obstacles') or []:
        point = item.get('contact_pixel') or {}
        if 'x' not in point or 'y' not in point:
            continue
        spot = (int(point['x']), int(point['y']))
        cv2.drawMarker(shot, spot, (60, 60, 220), cv2.MARKER_TILTED_CROSS, 14, 2)
        cv2.putText(shot, str(item.get('label', '?'))[:18], (spot[0] + 8, spot[1]),
                    cv2.FONT_HERSHEY_PLAIN, .9, (120, 150, 255), 1)
    contact = answer.get('contact_pixel')
    if isinstance(contact, dict) and 'x' in contact:
        cv2.drawMarker(shot, (int(contact['x']), int(contact['y'])),
                       (60, 220, 220), cv2.MARKER_STAR, 18, 2)

    plan = plan_view(route, obstacles, goal, repaired)
    height = max(shot.shape[0], plan.shape[0]) + 22 * (len(findings) + 1)
    board = np.full((height, shot.shape[1] + plan.shape[1], 3), 24, np.uint8)
    board[:shot.shape[0], :shot.shape[1]] = shot
    board[:plan.shape[0], shot.shape[1]:] = plan
    row = max(shot.shape[0], plan.shape[0]) + 16
    cv2.putText(board, 'clean' if not findings else '%d finding(s)' % len(findings),
                (8, row), cv2.FONT_HERSHEY_PLAIN, 1.,
                (120, 255, 120) if not findings else (120, 180, 255), 1)
    for line in findings:
        row += 22
        cv2.putText(board, '- ' + line, (8, row), cv2.FONT_HERSHEY_PLAIN, 1.,
                    (120, 180, 255), 1)
    return board


def examine(lens, path, target, out_dir, attempt=None):
    image = cv2.imread(path)
    if image is None:
        return dict(scene=os.path.basename(path), target=target,
                    error='could not read the image')
    name = os.path.splitext(os.path.basename(path))[0]
    if attempt is not None:
        name = '%s.try%d' % (name, attempt)

    try:
        answer = fetch.recognize(image, target)
    except fetch.Stop as exc:
        return dict(scene=name, target=target, error=str(exc))

    goal = None
    contact = answer.get('contact_pixel')
    if answer.get('visible') and isinstance(contact, dict):
        try:
            goal = lens.ground(float(contact['x']), float(contact['y']))
        except (fetch.Stop, KeyError, TypeError, ValueError):
            goal = None

    route, dropped = [], 0
    for point in answer.get('route_pixels') or []:
        if not isinstance(point, dict) or 'x' not in point or 'y' not in point:
            dropped += 1
            continue
        try:
            route.append(lens.ground(float(point['x']), float(point['y'])))
        except (fetch.Stop, TypeError, ValueError):
            dropped += 1      # above the horizon: not a place on the floor
    obstacles = fetch.project_obstacles(lens, answer)

    findings, closest = score(route, obstacles, goal)
    # What the local planner can do about it: the model has no scale, so push
    # the route out to the clearance the wheels actually need and score again.
    repaired, nudged = fetch.avoid(route, obstacles, goal)
    mended, after = score(repaired, obstacles, goal)
    if dropped:
        findings.append('%d route point(s) were not on placeable floor' % dropped)
    if answer.get('visible') and goal is None:
        findings.append('says the target is visible but its contact pixel '
                        'is not on the floor')

    cv2.imwrite(os.path.join(out_dir, name + '.overlay.jpg'),
                overlay(image, answer, route, obstacles, goal, findings,
                        repaired))
    return dict(
        scene=name, target=target, visible=bool(answer.get('visible')),
        note=str(answer.get('note', ''))[:90],
        range_cm=round(math.hypot(*goal), 1) if goal else None,
        route_points=len(route),
        obstacles=[[label, round(spot[0], 1), round(spot[1], 1)]
                   for label, spot in obstacles],
        closest_obstacle_cm=round(closest, 1) if closest is not None else None,
        findings=findings,
        route_cm=[[round(v, 1) for v in p] for p in route],
        repaired_cm=[[round(v, 1) for v in p] for p in repaired],
        waypoints_nudged=nudged,
        repaired_closest_cm=round(after, 1) if after is not None else None,
        repaired_findings=mended,
        execution=would_execute(route, obstacles) if route else None,
        repaired_execution=would_execute(repaired, obstacles) if repaired else None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', action='append', default=[],
                        metavar='IMAGE:TARGET',
                        help='a photograph and the object to plan a route to')
    parser.add_argument('--manifest',
                        help='JSON list of {"image": ..., "target": ...}')
    parser.add_argument('--out', default='gpt-route-review')
    parser.add_argument('--repeat', type=int, default=1,
                        help='ask this many times per scene; the model is not '
                             'deterministic and one answer is not a measurement')
    args = parser.parse_args(argv)

    scenes = []
    for entry in args.scene:
        image, _, target = entry.rpartition(':')
        if not image or not target:
            parser.error('--scene wants IMAGE:TARGET, got %r' % entry)
        scenes.append((image, target))
    if args.manifest:
        with open(args.manifest) as handle:
            for item in json.load(handle):
                scenes.append((item['image'], item['target']))
    if not scenes:
        parser.error('nothing to examine: pass --scene or --manifest')
    if not os.environ.get('OPENAI_API_KEY'):
        raise SystemExit('OPENAI_API_KEY is not set')

    os.makedirs(args.out, exist_ok=True)
    lens = Lens()
    results = []
    for image, target in scenes:
        for attempt in range(args.repeat):
            label = None if args.repeat == 1 else attempt + 1
            found = examine(lens, image, target, args.out, label)
            results.append(found)
            print('%-26s %-16s %s' % (
                found['scene'][:26], target[:16],
                found.get('error') or
                ('not visible' if not found['visible'] else
                 '%d pts, nearest obstacle %s cm, %s' % (
                     found['route_points'],
                     found['closest_obstacle_cm'],
                     ('clean' if not found['findings']
                  else '%d finding(s)' % len(found['findings'])) +
                 (' -> repaired clean (%d nudged)' % found['waypoints_nudged']
                  if found['findings'] and not found['repaired_findings']
                  else ' -> %d left after repair' % len(found['repaired_findings'])
                  if found['findings'] else '')))))
            for line in found.get('findings', []):
                print('%28s - %s' % ('', line))

    with open(os.path.join(args.out, 'report.json'), 'w') as handle:
        json.dump(results, handle, indent=1, sort_keys=True)
    clean = sum(1 for r in results if not r.get('error') and not r['findings'])
    print('\n%d of %d answers clean; overlays and report in %s/'
          % (clean, len(results), args.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
