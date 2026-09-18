#!/usr/bin/env python3
"""Find an object that is not in view, then hand the reach controller the job.

    sweep -> if it is in sight, aim at it and hand over
          -> if not, write down what each direction looked like
          -> ask the model for a few steps of exploration, check them, run them
          -> revise as soon as a step turns up something the plan did not expect

The layer below this one (planner_demo.mission, fetch.fetch) drives to a named
object it can already see, and stops the moment it cannot. This layer only has
to produce a direction for it to start from.

That is why the whole thing is built out of rotation. A turn lands within about
a degree of what was asked; a drive lands wherever the carpet allows, and the
visual odometer reads it anywhere from a fifth of the time to all of it. So the
thing this file measures and remembers is bearing, never position: the model
reports the target's pixel, the lens turns that pixel into an angle, and the
robot turns by it. A pixel becomes an angle at any height in the frame,
including well above the horizon where it cannot be turned into a distance --
which is the whole trick, because the model reliably reports seeing things at
two metres and the floor projection is only trusted to 140 cm.

What that costs is honesty about coverage. Stations are labels with bearings
between them, not coordinates; the robot cannot return to one it has left, and
nothing here proves the object is absent. A search that ends without a sighting
means the budget ran out, and that is what it says.

Geometry: bearings in degrees, positive to the right, folded into (-180, 180].
Requires a running local_nav/service.py and OPENAI_API_KEY in the environment.
"""
import argparse
import base64
import json
import math
import os
import sys
import time
import urllib.request

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch

WIDTH, HEIGHT = 640, 480

SWEEP_STEP_DEG = 60.       # Measured off the lens: at the height in the frame
                           # where distant things sit, the picture spans about
                           # +-63 degrees, and the middle +-45 of that is where
                           # a small object is still several pixels across. Six
                           # looks therefore close a circle with the cells
                           # overlapping by half a frame, which is what keeps an
                           # object from living permanently on a frame edge.
SWEEP_MIN_STEP_DEG = 20.
SWEEP_REFINE_DEG = 20.     # a flanking look, a third of a cell to each side
REFINE_BUDGET = 4          # closer looks allowed per sweep; more than this and
                           # the sweep costs as much as another station
CENTRED_DEG = 12.          # a candidate already this near the middle of the
                           # frame gains nothing from being looked at again
EDGE_MARGIN_PX = 70.       # a sighting inside this much of the frame edge is on
                           # the distorted rim, where the fisheye smears a small
                           # object across very few pixels: treat it as a lead,
                           # not as a sighting
NEAR_THIRD_Y = 320.        # bottom third of the picture: an obstacle whose
                           # contact pixel is this low is too close to drive past
FAR_PLACEABLE_Y = 200.     # above this a floor pixel projects past 140 cm, where
                           # the ground plane is no longer trusted

LOOK_BUDGET = 40           # model calls a search may spend on looking
TRAVEL_BUDGET_CM = 600.    # how far it may drive in total
STATION_BUDGET = 5         # how many places it may sweep from
PLAN_MAX_STEPS = 4         # a longer plan is stale before it finishes
PLAN_ATTEMPTS = 3          # rejected plans re-asked before falling back
GO_MIN_CM = 20.            # shorter than this is not a change of viewpoint
GO_MAX_CM = 150.           # one hop. Further than this is not refused because
                           # the wheels cannot do it but because nothing checks
                           # the floor past the end of the last look, and the
                           # robot has no position with which to notice.
GO_LEG_CM = 40.            # a hop is driven in slices this long, looking between
GO_STALL_CM = 3.           # a slice that buys less than this is not progress
SURE = 'sure'
UNSURE = 'unsure'
ABSENT = 'no'

SWEEP_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['visible', 'target_pixel', 'confidence', 'worth_a_closer_look',
                 'open_pixel', 'obstacles', 'scene'],
    'properties': {
        'visible': {'type': 'boolean'},
        # Where the target meets the floor, if it is in this picture. It is used
        # for a bearing, not a range, so a pixel far above the floor line is
        # still worth having -- the angle survives where the distance does not.
        'target_pixel': {'anyOf': [fetch.POINT, {'type': 'null'}]},
        'confidence': {'type': 'string', 'enum': [SURE, UNSURE, ABSENT]},
        'worth_a_closer_look': {'type': 'boolean'},
        # The far end of the clear carpet in this direction. This is what a hop
        # drives at, so it is a pixel like everything else and the calibration
        # supplies the centimetres.
        'open_pixel': {'anyOf': [fetch.POINT, {'type': 'null'}]},
        'obstacles': {'type': 'array', 'items': {
            'type': 'object', 'additionalProperties': False,
            'required': ['label', 'contact_pixel'],
            'properties': {'label': {'type': 'string'},
                           'contact_pixel': fetch.POINT}}},
        'scene': {'type': 'string'},
    },
}

SWEEP_PROMPT = """You are the eyes of a small floor robot that is looking for one
object. It is turning on the spot and photographing each direction in turn, and
this is one of those photographs. Answer about THIS picture only.

The camera sits 9 cm above the carpet, so the bottom of the picture is the floor
at the robot's wheels and the top is the far side of the room. The picture is
640 wide and 480 tall.

visible - whether you can actually see the object here. Being far away is fine:
across the room still counts. Half a metre outside the frame does not.

confidence - "sure" if you can see it and could point at it, "unsure" if
something here could be it but you would not bet on it, "no" if it is not here.
Use "unsure" freely. It is cheap: the robot turns a little and photographs that
direction again from the middle of the frame, where you get a much better look.
A wrong "sure" is expensive - the robot drives off after the wrong thing.

target_pixel - where the object meets the floor, at its horizontal centre, or
null. Give this whenever you say "sure" or "unsure", even for something small
and far off and even if it is near the top of the picture. The robot only takes
a direction from it, so a pixel high in the frame is still useful.

worth_a_closer_look - true when the object could be in this direction AND this
photograph cannot settle it: an opening or doorway you cannot see through, the
shadow under or behind furniture, a heap it could be inside, something
interesting cut off by the edge of the frame.

False whenever you can see the whole of the floor in this direction, however
much is standing on it. A wall lined with boxes you can see all of is answered,
not hidden. Empty carpet is answered. Say false unless you can name the place
the object would be hiding, and name that place in scene.

The robot is turning the whole way round and can only afford two or three
closer looks in a circuit, so they have to be worth the turn.

open_pixel - the furthest patch of clear carpet the robot could drive to in this
direction, or null if there is none. Rules you can apply from this picture:

  * Put it on carpet with nothing on it, not on or behind an object.
  * Keep it roughly below the halfway line - y greater than 200. Higher than
    that is floor the robot cannot measure, and it will be ignored.
  * Keep it within about a sixth of the width of the middle - x between 210 and
    430. The robot is already facing up the picture and drives forward; it turns
    first if it wants to go elsewhere.

obstacles - things on the floor in this direction that the robot would hit, at
most four, each with the pixel where it meets the carpet. Do not describe
walls beyond the floor you can see.

scene - at most ten words on what is in this direction. This is the only thing
the robot remembers about a direction it has turned away from, so make it
describe the place, not the picture: "open carpet to a doorway" or "desk legs
and cables against the wall".

Judge closeness in pixels of this picture, never in centimetres - you cannot see
scale. An obstacle whose contact pixel is in the bottom third, below y=320, is
close enough that the robot cannot drive past it at all."""

PLAN_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['steps', 'note'],
    'properties': {
        'steps': {'type': 'array', 'items': {
            'type': 'object', 'additionalProperties': False,
            'required': ['action', 'heading_degrees', 'arc_degrees',
                         'distance_cm', 'why'],
            'properties': {
                'action': {'type': 'string',
                           'enum': ['sweep', 'go', 'give_up']},
                'heading_degrees': {'anyOf': [{'type': 'number'},
                                              {'type': 'null'}]},
                'arc_degrees': {'anyOf': [{'type': 'number'},
                                          {'type': 'null'}]},
                'distance_cm': {'anyOf': [{'type': 'number'},
                                          {'type': 'null'}]},
                'why': {'type': 'string'},
            }}},
        'note': {'type': 'string'},
    },
}

PLAN_PROMPT = """You are directing a small floor robot that is hunting for one
object it has not found yet. You are given its journal: every direction it has
looked in so far, from every place it has stood, and what it reported seeing.
Propose the next few steps of the search.

The robot is 12 cm wide and knee-high to a skirting board. It turns on the spot
accurately, to about a degree. It does NOT know where it is: it has no map and
no reliable odometry, so it cannot go back to somewhere it has been, and a
distance you give it is roughly what it will drive, not exactly.

Headings are in degrees, negative to the left, positive to the right, and they
are always measured from the way the robot is facing at the START of the step -
which for the first step is the direction it faced when it arrived where it is
now, marked "facing 0" in the journal. After a "go" the robot is facing the way
it drove, so the next step's 0 is that direction.

The actions you may use, and nothing else:

  sweep    - turn on the spot and photograph the way round. Give
             heading_degrees for the middle of the arc you want covered and
             arc_degrees for how much of the circle. 360 looks everywhere.
             distance_cm must be null.

  go       - turn to heading_degrees and drive about distance_cm forward, to
             see the room from somewhere else. Between %(go_min).0f and
             %(go_max).0f cm. arc_degrees must be null. The robot looks where it
             is going and will stop early if something is in the way.

  give_up  - there is nothing left worth trying. Use it rather than proposing a
             step you do not believe in. heading, arc and distance must be null.

Rules that will get your plan rejected, so check them:

  * At most %(max_steps)d steps, and give_up may only be the last one.
  * A "go" that starts from where the robot is NOW must be along a heading the
    journal says it has already looked at, and must not be one it reported as
    blocked. The robot will not drive into floor nobody has photographed.
  * The total of your distances must fit the travel budget in the journal.

How to think about it. Looking is cheap and driving is expensive: a sweep costs
a few seconds, a hop costs a metre of a budget that does not come back, and the
robot cannot undo a hop. So prefer another look from here over a move, and move
only towards something the journal makes a case for - an unsure sighting, a
doorway, a cluttered corner it could not see into, a direction it has not
covered.

Small objects on the floor are hidden by very little. A direction reported as
open carpet with nothing on it has been searched; a direction reported as
"boxes against the wall" or "under the desk" has not, however hard the robot
looked at it from here.

note - one sentence: what you think is going on and what you are trying.""" % (
    dict(go_min=GO_MIN_CM, go_max=GO_MAX_CM, max_steps=PLAN_MAX_STEPS))


def wrap(degrees):
    """Fold an angle into [-180, 180), so one heading has one name."""
    return fetch.Robot.unwrap(degrees)


def bearing(lens, x, y):
    """Which way a pixel lies, in degrees right of straight ahead.

    The companion of Robot.ground, and the reason a search can work further out
    than a drive can. ground() needs the ray to strike the floor, so it refuses
    anything at or above the horizon and is only trusted to about 140 cm; the
    angle needs no such thing. The model reports seeing objects across a room,
    and this turns that report into a turn the robot can make.
    """
    xy = cv2.fisheye.undistortPoints(np.array([[[float(x), float(y)]]]),
                                     lens.K, lens.D).reshape(2)
    ray = np.array([xy[0], xy[1], 1.])
    down = np.array([0., math.cos(lens.pitch), math.sin(lens.pitch)])
    forward = np.array([0., 0., 1.]) - down * down[2]
    forward /= np.linalg.norm(forward)
    right = np.cross(down, forward)
    return math.degrees(math.atan2(float(ray.dot(right)), float(ray.dot(forward))))


def sweep_headings(centre=0., arc=360., step=SWEEP_STEP_DEG):
    """The headings one sweep looks along, in the order they are turned to.

    A full circle closes on itself, so its cells start at the centre heading and
    run the whole way round. A partial arc is covered by cells centred inside
    it, which is why the first look is half a step in from the edge rather than
    on it: the edge of an arc is the middle of a photograph's field, not the
    middle of a cell.
    """
    step = max(SWEEP_MIN_STEP_DEG, min(180., abs(float(step))))
    arc = max(step, min(360., abs(float(arc))))
    count = max(1, int(round(arc / step)))
    if arc > 359.9:
        return [wrap(centre + index * step) for index in range(count)]
    first = centre - arc / 2. + step / 2.
    return [wrap(first + index * step) for index in range(count)]


def refine_headings(looks, step=SWEEP_STEP_DEG, budget=REFINE_BUDGET):
    """Where a coarse sweep should look again, most promising first.

    Two different reasons to look twice, and they want different answers. An
    unsure sighting has a pixel, and a pixel is a bearing, so the second look is
    aimed straight at the candidate: it lands in the middle of the frame where
    the lens is sharp and the object is biggest. A direction that merely could
    hide something has no pixel to aim at, so the second look goes to the two
    flanks of the cell instead, which is where a coarse sweep is weakest.

    `looks` are dicts as the journal stores them. Pure arithmetic on purpose --
    this is the part of the sweep worth testing without a robot.
    """
    aimed, sides = [], []
    for look in looks:
        if look.get('refined'):
            continue                 # this heading is already a second look
        heading = float(look.get('heading', 0.))
        candidate = look.get('candidate_bearing')
        # Anything with a pixel that was not good enough to act on. That is an
        # unsure sighting, and also a confident one on the rim of the frame,
        # which is a claim about a dozen smeared pixels.
        lead = (look.get('confidence') == UNSURE
                or (look.get('confidence') == SURE and look.get('edge')))
        if lead and candidate is not None:
            if abs(float(candidate)) < CENTRED_DEG:
                continue             # already centred: a second look sees the
                                     # same pixels and answers the same way
            aimed.append(wrap(heading + float(candidate)))
        elif look.get('worth_a_closer_look'):
            sides.append((wrap(heading - step / 3.), wrap(heading + step / 3.)))
    # One flank from each flagged direction before the second flank of any of
    # them. Measured over 52 stored frames, the flag comes back true on more
    # than half of a cluttered room's directions, so taking them in order spent
    # the whole refinement budget on the first two cells and left the rest of
    # the circle at coarse resolution. Interleaving spreads it.
    flanks = [side for pair in zip(*sides) for side in pair] if sides else []
    wanted, seen = [], []
    for heading in aimed + flanks:
        if any(abs(wrap(heading - done)) < SWEEP_MIN_STEP_DEG / 2. for done in seen):
            continue                 # two leads a few degrees apart are one look
        seen.append(heading)
        wanted.append(heading)
        if len(wanted) >= budget:
            break
    return wanted


def edge_sighting(pixel):
    """Whether a sighting sits on the distorted rim of the frame.

    The fisheye puts more than 120 degrees across 640 pixels, so an object at
    the left or right edge is squeezed into very few of them and its apparent
    position moves fast with the robot's heading. A sighting there is worth
    turning towards and photographing again, not worth driving at.

    Only the sides count. High in the frame is far away, not distorted sideways,
    and the bearing taken from it is as good as any other -- that is exactly the
    case this whole layer exists to use.
    """
    if not isinstance(pixel, dict):
        return False
    try:
        x = float(pixel['x'])
    except (KeyError, TypeError, ValueError):
        return False
    return x < EDGE_MARGIN_PX or x > WIDTH - EDGE_MARGIN_PX


def blocked_ahead(answer):
    """Whether this view says the robot cannot drive off this way at all.

    The criterion is the model's own, in its own pixels: something it listed as
    on the floor, in the bottom third and near the middle. The clearance
    arithmetic in fetch says the same thing in centimetres later, but this is
    what the journal can carry about a direction the robot has turned away from.
    """
    for obstacle in answer.get('obstacles') or []:
        point = obstacle.get('contact_pixel')
        if not isinstance(point, dict):
            continue
        try:
            x, y = float(point['x']), float(point['y'])
        except (KeyError, TypeError, ValueError):
            continue
        if y >= NEAR_THIRD_Y and WIDTH / 4. <= x <= 3. * WIDTH / 4.:
            return True
    return False


class Journal(object):
    """What has been looked at, from where, and what was there.

    The only record of the search, and deliberately topological: a station is a
    name and the bearing and rough distance that reached it, never a position.
    Anything else would be a claim the odometry cannot support, and a model
    handed coordinates will reason about them as though they were true.
    """

    def __init__(self, target, look_budget=LOOK_BUDGET,
                 travel_budget_cm=TRAVEL_BUDGET_CM):
        self.target = target
        self.stations = []
        self.looks_left = look_budget
        self.travel_left = float(travel_budget_cm)
        self.arrive('start')

    def arrive(self, name, heading=None, distance=None):
        """Open a new station. `heading` and `distance` are how it was reached."""
        self.stations.append(dict(name=name, heading=heading, distance=distance,
                                  looks=[]))
        return self.stations[-1]

    @property
    def here(self):
        return self.stations[-1]

    def note(self, heading, answer, candidate_bearing=None, refined=False):
        """Record one look. Returns the entry, which the caller may read back."""
        entry = dict(heading=wrap(float(heading)),
                     confidence=str(answer.get('confidence') or ABSENT),
                     visible=bool(answer.get('visible')),
                     worth_a_closer_look=bool(answer.get('worth_a_closer_look')),
                     blocked=blocked_ahead(answer),
                     edge=edge_sighting(answer.get('target_pixel')),
                     candidate_bearing=(None if candidate_bearing is None
                                        else round(float(candidate_bearing), 1)),
                     scene=str(answer.get('scene') or '')[:60],
                     refined=bool(refined))
        self.here['looks'].append(entry)
        self.looks_left -= 1
        return entry

    def looked(self, station=None):
        """Headings already photographed from a station, nearest cell first."""
        station = station or self.here
        return [look['heading'] for look in station['looks']]

    def covers(self, heading, tolerance=None):
        """Whether some look from here faced near enough to `heading`."""
        tolerance = SWEEP_STEP_DEG / 2. if tolerance is None else tolerance
        return any(abs(wrap(heading - done)) <= tolerance for done in self.looked())

    def blocked(self, heading, tolerance=None):
        """Whether the nearest look that way reported something in the corridor."""
        tolerance = SWEEP_STEP_DEG / 2. if tolerance is None else tolerance
        nearest = None
        for look in self.here['looks']:
            gap = abs(wrap(heading - look['heading']))
            if gap <= tolerance and (nearest is None or gap < nearest[0]):
                nearest = (gap, look)
        return bool(nearest and nearest[1]['blocked'])

    def render(self):
        """The journal as the model sees it: compact, ordered, and honest.

        Kept as text rather than as the raw dicts because it is read once and
        never parsed: a table of a dozen short lines costs a fraction of the
        tokens and is far harder to misread than nested JSON.
        """
        lines = ['looking for: %s' % self.target,
                 'the robot has no map and no position. Headings are exact to '
                 'about a degree; distances are what the wheels were asked for.']
        for index, station in enumerate(self.stations):
            if station['heading'] is None:
                where = 'station %s (where the search started)' % station['name']
            else:
                where = ('station %s (from %s: turned %+.0f, drove about %.0f cm)'
                         % (station['name'], self.stations[index - 1]['name'],
                            station['heading'], station['distance'] or 0.))
            if index == len(self.stations) - 1:
                where += '   <- the robot is here, facing 0'
            lines.append('')
            lines.append(where)
            if not station['looks']:
                lines.append('  (nothing looked at yet)')
            for look in station['looks']:
                marks = []
                if look['confidence'] == SURE:
                    marks.append('SEEN')
                elif look['confidence'] == UNSURE:
                    marks.append('might be it')
                if look['worth_a_closer_look']:
                    marks.append('could hide it')
                if look['blocked']:
                    marks.append('blocked')
                if look['refined']:
                    marks.append('close look')
                lines.append('  %+04.0f  %-38s %s'
                             % (look['heading'], look['scene'][:38],
                                ', '.join(marks)))
        lines.append('')
        lines.append('budget left: %d looks, %.0f cm of driving, %d more place(s) '
                     'to stand' % (max(0, self.looks_left),
                                   max(0., self.travel_left),
                                   max(0, STATION_BUDGET - len(self.stations))))
        return '\n'.join(lines)


def validate_plan(answer, journal, max_steps=PLAN_MAX_STEPS):
    """Check a proposed plan against what the robot can actually do.

    Returns (steps, problems). A plan with any problem is not repaired and not
    partly run: the problems are handed back to the model and it is asked again.
    Repairing it quietly would mean driving somewhere nobody chose, and the one
    thing this layer must not do is invent a reason to move.

    The floor checks only bind the first hop. Steps after a "go" start from a
    place the robot has never stood, so there is nothing to check them against
    and pretending otherwise would reject perfectly sensible plans.
    """
    problems = []
    steps = answer.get('steps')
    if not isinstance(steps, list) or not steps:
        return [], ['the plan has no steps']
    if len(steps) > max_steps:
        problems.append('%d steps is more than the %d allowed'
                        % (len(steps), max_steps))
    checked, budget, moved = [], journal.travel_left, False
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            problems.append('step %d is not a step' % (index + 1))
            continue
        action = step.get('action')
        heading = step.get('heading_degrees')
        arc = step.get('arc_degrees')
        distance = step.get('distance_cm')

        def number(value, name):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                problems.append('step %d: %s is missing' % (index + 1, name))
                return None
            if not math.isfinite(value):
                problems.append('step %d: %s is not a number' % (index + 1, name))
                return None
            return float(value)

        if action == 'give_up':
            if index != len(steps) - 1:
                problems.append('step %d: give_up has to be the last step'
                                % (index + 1))
            checked.append(dict(action='give_up', why=str(step.get('why', ''))))
            continue
        if action == 'sweep':
            centre = 0. if heading is None else number(heading, 'heading_degrees')
            span = 360. if arc is None else number(arc, 'arc_degrees')
            if centre is None or span is None:
                continue
            if abs(centre) > 180.:
                problems.append('step %d: heading %.0f is not within +-180'
                                % (index + 1, centre))
                continue
            if not SWEEP_MIN_STEP_DEG <= span <= 360.:
                problems.append('step %d: an arc of %.0f degrees is not a sweep '
                                '(ask for %.0f to 360)'
                                % (index + 1, span, SWEEP_MIN_STEP_DEG))
                continue
            checked.append(dict(action='sweep', heading=wrap(centre), arc=span,
                                why=str(step.get('why', ''))))
            continue
        if action == 'go':
            where = number(heading, 'heading_degrees')
            far = number(distance, 'distance_cm')
            if where is None or far is None:
                continue
            if abs(where) > 180.:
                problems.append('step %d: heading %.0f is not within +-180'
                                % (index + 1, where))
                continue
            if not GO_MIN_CM <= far <= GO_MAX_CM:
                problems.append('step %d: %.0f cm is outside the %.0f to %.0f cm '
                                'a hop may be' % (index + 1, far, GO_MIN_CM,
                                                  GO_MAX_CM))
                continue
            if not moved:
                # Only the first hop starts from floor the journal describes.
                if not journal.covers(where):
                    problems.append('step %d: the robot has not looked along '
                                    '%+.0f from where it stands'
                                    % (index + 1, where))
                    continue
                if journal.blocked(where):
                    problems.append('step %d: %+.0f is blocked by something on '
                                    'the floor' % (index + 1, where))
                    continue
            budget -= far
            if budget < 0.:
                problems.append('step %d: that is %.0f cm more driving than the '
                                'budget has left' % (index + 1, -budget))
                continue
            moved = True
            checked.append(dict(action='go', heading=wrap(where), distance=far,
                                why=str(step.get('why', ''))))
            continue
        problems.append('step %d: "%s" is not one of sweep, go, give_up'
                        % (index + 1, action))
    if not checked and not problems:
        problems.append('the plan has no steps the robot can run')
    return checked, problems


def fallback_plan(journal):
    """What to do when the model will not produce a plan the robot can run.

    Deliberately dull: look everywhere that has not been looked at, and if the
    circle is closed, hop towards the most promising direction that is not
    blocked. It exists so that a model outage degrades the search instead of
    ending it.
    """
    if len(journal.looked()) < len(sweep_headings()):
        return [dict(action='sweep', heading=0., arc=360.,
                     why='nothing has been looked at from here yet')]
    leads = [look for look in journal.here['looks']
             if not look['blocked'] and (look['confidence'] == UNSURE
                                         or look['worth_a_closer_look'])]
    if leads and journal.travel_left >= GO_MIN_CM:
        best = leads[0]
        return [dict(action='go', heading=best['heading'],
                     distance=min(GO_MAX_CM, journal.travel_left),
                     why='the only direction here that could hide it'),
                dict(action='sweep', heading=0., arc=360.,
                     why='look around from there')]
    return [dict(action='give_up', why='every direction from here has been '
                                       'looked at and none of them leads on')]


def _ask(instructions, schema, name, payload, image=None, max_tokens=400,
         timeout=30., effort='none'):
    """One structured question to the model, with or without a photograph.

    The same plumbing as fetch.recognize, kept here rather than shared out of
    it: fetch is the controller that drives the robot and is worth leaving
    alone, and the two questions asked here want different schemas, different
    token bounds and, for the planner, no image at all.
    """
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        raise fetch.Stop('OPENAI_API_KEY is not set')
    content = [dict(type='input_text', text=json.dumps(payload))]
    if image is not None:
        ok, encoded = cv2.imencode('.jpg', image)
        if not ok:
            raise fetch.Stop('could not encode the camera frame')
        content.append(dict(type='input_image', detail='high',
                            image_url='data:image/jpeg;base64,'
                                      + base64.b64encode(encoded).decode('ascii')))
    body = dict(model=fetch.MODEL, reasoning=dict(effort=effort), store=False,
                max_output_tokens=max_tokens, instructions=instructions,
                input=[dict(role='user', content=content)],
                text=dict(format=dict(type='json_schema', name=name, strict=True,
                                      schema=schema)))
    request = urllib.request.Request(
        'https://api.openai.com/v1/responses', data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json',
                 'Authorization': 'Bearer ' + key})
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        raw = json.load(reply)
    if raw.get('status') != 'completed':
        raise fetch.Stop('model returned %s' % raw.get('status'))
    text = ''.join(part.get('text', '')
                   for item in raw.get('output', []) if item.get('type') == 'message'
                   for part in item.get('content', []) if part.get('type') == 'output_text')
    return json.loads(text)


def look(image, target, timeout=20.):
    """Ask the cheap question: is it here, roughly where, and is it worth more.

    A tenth of the route prompt and a third of its output. The expensive
    question -- how to drive there without hitting anything -- is not asked at
    all until the object has been found, because until then there is nowhere to
    drive to.
    """
    answer = _ask(SWEEP_PROMPT, SWEEP_SCHEMA, 'sweep_look',
                  dict(object=target), image=image, max_tokens=300,
                  timeout=timeout)
    pixel = answer.get('target_pixel')
    if isinstance(pixel, dict):
        try:
            x, y = float(pixel['x']), float(pixel['y'])
        except (KeyError, TypeError, ValueError):
            answer['target_pixel'] = None
        else:
            if not (0 <= x <= WIDTH and 0 <= y <= HEIGHT):
                # Outside the frame is not a sighting of this photograph, and a
                # bearing taken from it would point at nothing.
                answer['target_pixel'] = None
                answer['visible'] = False
                answer['confidence'] = ABSENT
    if answer.get('confidence') == SURE and not answer.get('target_pixel'):
        answer['confidence'] = UNSURE     # sure of what, exactly? There is no
                                          # direction in it, so it is a lead
    return answer


def propose(journal, timeout=40., attempts=PLAN_ATTEMPTS, record=None):
    """Ask the model for the next few steps, and check them before running any.

    A rejected plan is re-asked with the reasons attached rather than repaired,
    up to `attempts` times, and then the dull fallback runs instead. The point
    of stating the rejection is that the second answer is usually right: the
    model is not guessing at the constraints, it just did not apply one.
    """
    problems = []
    for attempt in range(attempts):
        payload = dict(journal=journal.render())
        if problems:
            payload['rejected'] = problems
            payload['note'] = ('your previous plan was refused for these '
                               'reasons; give one that is not')
        try:
            answer = _ask(PLAN_PROMPT, PLAN_SCHEMA, 'search_plan', payload,
                          max_tokens=600, timeout=timeout)
        except (fetch.Stop, ValueError, OSError) as exc:
            if record:
                record('plan_failed', attempt=attempt + 1, error=str(exc)[:80])
            break
        steps, problems = validate_plan(answer, journal)
        if record:
            record('plan', attempt=attempt + 1, steps=steps,
                   note=str(answer.get('note', ''))[:100], rejected=problems)
        if not problems:
            return steps, str(answer.get('note', ''))[:100]
    return fallback_plan(journal), 'fallback: the model did not give a usable plan'


class Search(object):
    """One hunt: the journal, the budgets, and the wheels.

    Held together in a class only because a sweep, a hop and a re-plan all need
    the same four things and threading them through as arguments made every
    signature longer than the function.
    """

    def __init__(self, robot, target, odometer=None, record=None,
                 look_budget=LOOK_BUDGET, travel_budget_cm=TRAVEL_BUDGET_CM,
                 stations=STATION_BUDGET, step_deg=SWEEP_STEP_DEG, watch=None):
        self.robot = robot
        self.target = target
        self.odometer = odometer
        self.journal = Journal(target, look_budget, travel_budget_cm)
        self.record = record or (lambda event, **fields: None)
        self.stations = stations
        self.step_deg = step_deg
        self.heading = 0.     # where the robot points, relative to this station
        # Handed every photograph as it is taken. A sweep spends most of a
        # minute turning, and without this the bench's picture stays on whatever
        # the robot was looking at when it started -- which is the one direction
        # the search has already finished with.
        self.watch = watch

    def face(self, heading):
        """Turn to a heading of the current station, by the shorter way round."""
        wanted = wrap(heading - self.heading)
        if abs(wanted) < 1.:
            return
        turned = 0.
        # Chunked like every other turn in this stack, so a big rotation is a
        # sequence of measured ones rather than one open-loop commitment.
        while abs(wanted - turned) > fetch.TURN_TOLERANCE_DEG:
            piece = max(-fetch.TURN_MAX_PER_LEG_DEG,
                        min(fetch.TURN_MAX_PER_LEG_DEG, wanted - turned))
            got = self.robot.turn(piece)
            turned += got
            if abs(got) < .5:
                break            # the wheels are not moving; stop asking
        self.heading = wrap(self.heading + turned)

    def one_look(self, heading, refined=False):
        """Face a heading, photograph it, ask the cheap question, write it down."""
        self.face(heading)
        image, _ = self.robot.frame()
        if self.watch:
            self.watch(image)
        answer = look(image, self.target)
        candidate = None
        pixel = answer.get('target_pixel')
        if isinstance(pixel, dict):
            try:
                candidate = bearing(self.robot, float(pixel['x']), float(pixel['y']))
            except (KeyError, TypeError, ValueError, fetch.Stop):
                candidate = None
        entry = self.journal.note(self.heading, answer, candidate, refined)
        self.record('look', heading=round(self.heading, 1),
                    confidence=entry['confidence'],
                    bearing=entry['candidate_bearing'],
                    closer=entry['worth_a_closer_look'], refined=refined,
                    scene=entry['scene'])
        return answer, entry

    def sighting(self, answer, entry):
        """Whether this look is good enough to stop searching and hand over.

        A sighting on the rim of the frame is not, however confident the model
        is: the object is a few pixels wide there and the bearing moves fast. It
        becomes a refinement instead, and the second look decides.
        """
        return (answer.get('confidence') == SURE and entry['candidate_bearing']
                is not None and not entry['edge'])

    def sweep(self, centre=0., arc=360.):
        """Coarse rotation, then a closer look wherever the coarse pass flagged.

        Returns the entry that found the target, or None. The coarse pass stops
        the moment it is sure, because everything after it is a look spent on a
        question that has been answered.
        """
        self.record('sweep', centre=round(centre, 1), arc=round(arc, 1),
                    step=self.step_deg)
        fresh = []
        for heading in sweep_headings(centre, arc, self.step_deg):
            if self.journal.looks_left <= 0:
                return None
            answer, entry = self.one_look(heading)
            fresh.append(entry)
            if self.sighting(answer, entry):
                return entry
        for heading in refine_headings(fresh, self.step_deg):
            if self.journal.looks_left <= 0:
                return None
            answer, entry = self.one_look(heading, refined=True)
            if self.sighting(answer, entry):
                return entry
        return None

    def hop(self, heading, distance):
        """Drive roughly `distance` along `heading`, looking as it goes.

        Built out of the same look-drive-look cadence as the reach controller,
        for the same reason: the picture is the only thing that says what is in
        front of the wheels, and it stops being true as soon as they turn. Each
        slice is also a search look, so travel is never time the robot spends
        not looking -- which is how the target is usually found, the plan having
        only said where to stand.

        Returns the entry that found the target, or None.
        """
        self.face(heading)
        self.record('hop', heading=round(self.heading, 1), distance_cm=distance)
        gone = 0.
        while gone < distance - fetch.ROUTE_MIN_LEG_CM:
            if self.journal.looks_left <= 0 or self.journal.travel_left <= 0.:
                break
            answer, entry = self.one_look(self.heading)
            if self.sighting(answer, entry):
                return entry
            route, obstacles = self.floor(answer, min(GO_LEG_CM, distance - gone))
            if not route:
                self.record('hop_blocked', gone_cm=round(gone, 1),
                            of_cm=distance, why='no clear floor ahead')
                break
            pose = fetch.follow(self.robot, self.odometer, route, self.relay,
                                obstacles, complete=False)
            # follow turns to face its waypoint, so the robot is not necessarily
            # pointing where the hop started. Carrying that into self.heading is
            # what keeps the journal's bearings meaning one thing.
            self.heading = wrap(self.heading + pose[2])
            step = math.hypot(pose[0], pose[1])
            gone += step
            self.journal.travel_left -= step
            self.record('hopped', step_cm=round(step, 1), gone_cm=round(gone, 1),
                        of_cm=distance)
            if step < GO_STALL_CM:
                self.record('hop_blocked', gone_cm=round(gone, 1), of_cm=distance,
                            why='the wheels bought no ground')
                break
        return None

    def floor(self, answer, wanted):
        """The one short leg this view supports, and what is on the floor here.

        The model's open patch is a pixel; the calibration turns it into a
        distance and the clearance arithmetic decides whether it may be driven.
        A leg is never longer than the model's own open pixel allows, because
        past that point the picture says nothing.
        """
        obstacles = fetch.project_obstacles(self.robot, answer)
        point = answer.get('open_pixel')
        if not isinstance(point, dict):
            return [], obstacles
        try:
            spot = self.robot.ground(float(point['x']), float(point['y']))
        except (fetch.Stop, KeyError, TypeError, ValueError):
            return [], obstacles
        reach = math.hypot(*spot)
        if reach < fetch.ROUTE_MIN_LEG_CM or reach > fetch.ROUTE_RANGE_CM:
            return [], obstacles
        scale = min(1., wanted / reach)
        route = [(spot[0] * scale, spot[1] * scale)]
        route, _ = fetch.avoid(route, obstacles)
        return route, obstacles

    def relay(self, event, **fields):
        """fetch.follow's log, forwarded under one name so a run reads as one."""
        self.record('drive_' + event, **fields)

    def run(self, handoff=None):
        """Search until the target is found or the budget is spent.

        The plan is proposed a few steps at a time and abandoned as soon as a
        step tells it something it did not know -- a hop that was stopped early,
        a sweep that turned up a lead. Carrying on through the rest of a plan
        made before that would be spending the budget on a question that has
        changed.
        """
        self.record('start', target=self.target, looks=self.journal.looks_left,
                    travel_cm=self.journal.travel_left)
        found = self.sweep()
        while found is None:
            if self.journal.looks_left <= 0:
                return self.give_up('out of looks')
            if len(self.journal.stations) >= self.stations:
                return self.give_up('no more places to stand')
            steps, note = propose(self.journal, record=self.record)
            self.record('planned', note=note, steps=len(steps))
            for index, step in enumerate(steps):
                if step['action'] == 'give_up':
                    return self.give_up('the plan says there is nowhere left: %s'
                                        % step.get('why', ''))
                if step['action'] == 'sweep':
                    before = len(self.journal.here['looks'])
                    found = self.sweep(step['heading'], step['arc'])
                    if found is not None:
                        break
                    if self.surprised(before):
                        self.record('replanning', why='the sweep turned up '
                                                      'something to follow')
                        break
                    continue
                if len(self.journal.stations) >= self.stations:
                    self.record('replanning', why='no budget for another place '
                                                  'to stand')
                    break
                # The turn belongs to the station being left; the station being
                # reached is named for the direction it was reached along, and
                # its own headings start from there.
                asked = min(step['distance'], self.journal.travel_left)
                self.face(step['heading'])
                self.journal.arrive(chr(ord('A') + len(self.journal.stations) - 1),
                                    step['heading'], asked)
                self.heading = 0.
                found = self.hop(0., asked)
                if found is not None:
                    break
                if index == len(steps) - 1:
                    # Arrived somewhere new with no instruction about what to do
                    # there. Looking around is the only thing a hop was for.
                    found = self.sweep()
                    if found is not None:
                        break
        return self.arrive_at(found, handoff)

    def surprised(self, before):
        """Whether the looks since `before` are a reason to think again."""
        for look in self.journal.here['looks'][before:]:
            if look['confidence'] in (SURE, UNSURE):
                return True
        return False

    def arrive_at(self, entry, handoff):
        """Point the robot at what it found, then let the reach controller drive."""
        self.face(wrap(entry['heading'] + (entry['candidate_bearing'] or 0.)))
        self.record('found', target=self.target,
                    station=self.journal.here['name'],
                    heading=round(self.heading, 1), scene=entry['scene'])
        if handoff is None:
            return dict(outcome='found', station=self.journal.here['name'],
                        heading=round(self.heading, 1),
                        looks_used=LOOK_BUDGET - self.journal.looks_left)
        return dict(outcome='handed_over', station=self.journal.here['name'],
                    heading=round(self.heading, 1), result=handoff())

    def give_up(self, why):
        """End the search saying what ran out, never that the object is absent.

        There is no coverage guarantee anywhere in this file, so there is no
        honest way to report one. The robot looked in the directions listed in
        the journal and did not see it.
        """
        self.record('gave_up', why=why,
                    looks_used=LOOK_BUDGET - self.journal.looks_left,
                    stations=len(self.journal.stations))
        return dict(outcome='not_found', why=why,
                    stations=len(self.journal.stations),
                    looks_used=LOOK_BUDGET - self.journal.looks_left,
                    journal=self.journal.render())


def search(target, dry_run=False, log=None, look_budget=LOOK_BUDGET,
           travel_budget_cm=TRAVEL_BUDGET_CM, stations=STATION_BUDGET,
           step_deg=SWEEP_STEP_DEG, handoff=None):
    """Find `target` from a standing start and hand over when it is in sight."""
    robot = fetch.Robot(dry_run=dry_run)
    odometer = fetch.Odometer(robot)
    events = []
    started = time.monotonic()

    def record(event, **fields):
        row = dict(event=event, t=round(time.monotonic() - started, 2), **fields)
        events.append(row)
        print(json.dumps(row, default=str), flush=True)
        if log:
            with open(log, 'w') as handle:
                json.dump(dict(target=target, events=events), handle, indent=2)
        return row

    hunt = Search(robot, target, odometer, record, look_budget,
                  travel_budget_cm, stations, step_deg)
    try:
        return hunt.run(handoff)
    except fetch.Stop as exc:
        robot.halt()
        record('stopped', reason=str(exc))
        return dict(outcome='stopped', reason=str(exc))
    except KeyboardInterrupt:
        robot.halt()
        record('stopped', reason='interrupted')
        return dict(outcome='stopped', reason='interrupted')
    finally:
        try:
            robot.hold(0., 0.)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target')
    parser.add_argument('--looks', type=int, default=LOOK_BUDGET)
    parser.add_argument('--travel-cm', type=float, default=TRAVEL_BUDGET_CM)
    parser.add_argument('--stations', type=int, default=STATION_BUDGET)
    parser.add_argument('--step-deg', type=float, default=SWEEP_STEP_DEG)
    parser.add_argument('--log')
    parser.add_argument('--dry-run', action='store_true',
                        help='look and plan, but never command the motors')
    args = parser.parse_args()
    result = search(args.target, args.dry_run, args.log, args.looks,
                    args.travel_cm, args.stations, args.step_deg)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get('outcome') in ('found', 'handed_over') else 1


if __name__ == '__main__':
    sys.exit(main())
