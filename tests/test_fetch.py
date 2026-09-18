import json
import math
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, 'local_nav'))
import fetch


STATUS = dict(session_id='s', control_epoch=1, healthy=True, motion_enabled=True,
              motor={'output': [0, 0]},
              power=dict(motion_allowed=True, stale=False, stop_latched=False,
                         pack_voltage_v=12.1),
              imu=dict(acceleration=[0., 0., 9.80665], gyro=[0., 0., 0.],
                       gyro_units='deg/s', time=1.))


def robot(**overrides):
    with patch.object(fetch, 'call', return_value=dict(STATUS, **overrides)):
        return fetch.Robot()


class GeometryTests(unittest.TestCase):
    """Every centimetre comes from calibration, never from the recognizer."""

    def test_floor_projection_matches_the_recorded_sighting(self):
        # The live run of 2026-09-16 put the bottle's contact at this pixel and
        # drove to it; the projection must still agree.
        right, forward = robot().ground(457., 216.)
        self.assertAlmostEqual(right, 36.5, delta=1.)
        self.assertAlmostEqual(forward, 77.4, delta=1.)

    def test_range_falls_as_a_pixel_moves_down_the_frame(self):
        near = robot().ground(320., 440.)[1]
        far = robot().ground(320., 240.)[1]
        self.assertLess(near, far)

    def test_a_pixel_at_the_horizon_is_refused_not_guessed(self):
        with self.assertRaisesRegex(fetch.Stop, 'horizon'):
            robot().ground(320., 20.)


class GuardTests(unittest.TestCase):
    """The few conditions that must end a run, and nothing else."""

    def test_power_cancellation_and_health_stop_the_run(self):
        for bad, expected in (
                (dict(power=dict(STATUS['power'], motion_allowed=False)), 'power guard'),
                (dict(power=dict(STATUS['power'], stop_latched=True)), 'power guard'),
                (dict(power=dict(STATUS['power'], stale=True)), 'stale'),
                (dict(healthy=False), 'unhealthy'),
                (dict(control_epoch=9), 'control generation'),
                (dict(session_id='other'), 'control generation')):
            machine = robot()
            with self.assertRaises(fetch.Stop, msg=str(bad)) as caught:
                machine.check(dict(STATUS, **bad))
            self.assertIn(expected, str(caught.exception))

    def test_a_healthy_status_passes(self):
        machine = robot()
        self.assertEqual(machine.check(STATUS)['session_id'], 's')

    def test_tilt_is_not_measured_while_accelerating(self):
        machine = robot()
        # Stationary and level: measurable, and level.
        self.assertLess(machine.tilt(STATUS['imu']), 1.)
        # Accelerating: the vector is not gravity, so refuse to call it tilt
        # rather than reporting a tilt the robot does not have.
        moving = dict(STATUS['imu'], acceleration=[4., 0., 9.8])
        self.assertIsNone(machine.tilt(moving))

    def test_yaw_rate_is_signed_about_gravity(self):
        machine = robot()
        right = dict(STATUS['imu'], gyro=[0., 0., -40.])
        left = dict(STATUS['imu'], gyro=[0., 0., 40.])
        self.assertGreater(machine.yaw_rate(right), 0.)
        self.assertLess(machine.yaw_rate(left), 0.)


class RecognizerContractTests(unittest.TestCase):
    def test_schema_is_strict_and_minimal(self):
        self.assertFalse(fetch.SCHEMA['additionalProperties'])
        self.assertEqual(set(fetch.SCHEMA['required']), set(fetch.SCHEMA['properties']))
        # Perception only: there is no field in which a motion could be returned.
        for name in fetch.SCHEMA['properties']:
            self.assertNotIn(name, ('action', 'drive', 'turn', 'route', 'speed'))

    def test_a_contact_outside_the_frame_is_not_a_sighting(self):
        answer = dict(visible=True, contact_pixel=dict(x=900., y=300.), note='')
        reply = dict(status='completed', output=[dict(type='message', content=[
            dict(type='output_text', text=json.dumps(answer))])])

        class Reply(object):
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return json.dumps(reply).encode()
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test'}), \
                patch('json.load', return_value=reply), \
                patch('urllib.request.urlopen', return_value=Reply()), \
                patch('cv2.imencode', return_value=(True, np.zeros(4, np.uint8))):
            self.assertFalse(fetch.recognize(np.zeros((480, 640, 3), np.uint8), 'x')['visible'])


class ApproachTests(unittest.TestCase):
    """Closing in is what corrects an early, badly-conditioned measurement."""

    def test_step_shrinks_to_the_remaining_gap(self):
        standoff = 20.
        for distance, expected in ((80., 15.), (30., 10.), (24., 4.)):
            step = max(fetch.STEP_MIN_CM,
                       min(fetch.STEP_MAX_CM, distance - standoff))
            self.assertAlmostEqual(step, expected)

    def test_arrival_is_a_band_around_the_standoff(self):
        self.assertLessEqual(17., 20. + fetch.ARRIVAL_CM)
        self.assertGreater(30., 20. + fetch.ARRIVAL_CM)



class ObstacleTests(unittest.TestCase):
    """Obstacles must cut a leg short rather than be discovered by the wheels."""

    def test_something_dead_ahead_stops_the_leg_short_of_it(self):
        room, blame = fetch.clear_distance([('bin', (0., 40.))], (0., 0.), 0., 30.)
        self.assertAlmostEqual(room, 40. - fetch.OBSTACLE_RADIUS_CM - fetch.KEEP_BACK_CM)
        self.assertEqual(blame, 'bin')

    def test_something_beside_the_corridor_does_not(self):
        wide = fetch.CORRIDOR_HALF_CM + fetch.OBSTACLE_RADIUS_CM + 2.
        room, blame = fetch.clear_distance([('block', (wide, 30.))], (0., 0.), 0., 25.)
        self.assertEqual(room, 25.)
        self.assertIsNone(blame)

    def test_something_behind_is_not_in_the_way(self):
        room, _ = fetch.clear_distance([('cable', (0., -20.))], (0., 0.), 0., 25.)
        self.assertEqual(room, 25.)

    def test_the_corridor_turns_with_the_robot(self):
        # An obstacle due north blocks a north leg, but not an east one.
        north = fetch.clear_distance([('bin', (0., 40.))], (0., 0.), 0., 30.)[0]
        east = fetch.clear_distance([('bin', (0., 40.))], (0., 0.), 90., 30.)[0]
        self.assertLess(north, 30.)
        self.assertEqual(east, 30.)

    def test_an_obstacle_on_top_of_the_robot_yields_no_room(self):
        room, _ = fetch.clear_distance([('x', (0., 5.))], (0., 0.), 0., 25.)
        self.assertEqual(room, 0.)

class FollowTests(unittest.TestCase):
    """A waypoint is a destination, not a direction to nudge once toward."""

    class Dry(object):
        dry_run = True
        speed = 7.5

        def __init__(self, cap=None):
            self.cap = cap
            self.turns = []

        def turn(self, degrees):
            if self.cap is not None:      # a robot that cannot finish a turn
                degrees = max(-self.cap, min(self.cap, degrees))
            self.turns.append(degrees)
            return degrees

    def drive(self, route, robot=None):
        robot = robot or self.Dry()
        events = []
        pose = fetch.follow(robot, None, route, lambda k, **f: events.append((k, f)))
        return pose, events, robot

    def test_the_run_the_user_reported_reaches_both_waypoints(self):
        # 20 cm forward and 20 cm left, then 40 cm forward and 40 cm right.
        # The second needs a 105-degree turn and a 58 cm leg: past both the
        # per-turn and per-leg chunk sizes, which used to end the waypoint.
        route = [(-20., 20.), (40., 40.)]
        pose, events, _ = self.drive(route)
        for kind, fields in events:
            if kind == 'waypoint':
                self.assertTrue(fields['reached'], fields)
        self.assertLess(math.hypot(pose[0] - 40., pose[1] - 40.), fetch.ROUTE_MIN_LEG_CM)

    def test_a_bearing_past_the_chunk_is_finished_by_further_turns(self):
        _, events, robot = self.drive([(40., 0.)])   # 90 degrees: past the 60 cap
        faced = [f for k, f in events if k == 'faced']
        self.assertGreater(len(faced), 1)             # chunked, not abandoned
        for fields in faced:
            self.assertLessEqual(abs(fields['asked_deg']),
                                 fetch.TURN_MAX_PER_LEG_DEG + 1e-9)
        self.assertLess(abs(faced[-1]['remaining_deg']), fetch.HEADING_TOLERANCE_DEG)

    def test_a_leg_past_the_chunk_is_finished_by_further_legs(self):
        _, events, _ = self.drive([(0., 60.)])
        legs = [f for k, f in events if k == 'leg']
        self.assertGreater(len(legs), 1)
        self.assertLessEqual(max(f['wanted_cm'] for f in legs), fetch.STEP_MAX_CM)
        self.assertAlmostEqual(sum(f['moved_cm'] for f in legs), 60., delta=2.)

    def test_aiming_allows_for_the_pivot_sliding_the_robot(self):
        # The robot turns about a point behind the lens, so the naive bearing is
        # stale once the turn finishes. Applying the solved turn must leave the
        # target dead ahead.
        pose, target = [0., 0., 0.], (30., 30.)
        turn = fetch.aim_turn(pose, target)
        shift = fetch.pivot_shift(turn)
        left = math.degrees(math.atan2(target[0] - shift[0],
                                       target[1] - shift[1])) - turn
        self.assertLess(abs(left), .1)

    def test_a_pivot_that_cannot_be_read_falls_back_to_spinning_in_place(self):
        saved = fetch._PIVOT
        try:
            fetch._PIVOT = (0., 0.)
            self.assertEqual(fetch.pivot_shift(90.), (0., 0.))
        finally:
            fetch._PIVOT = saved

    def test_the_search_path_still_advances_one_chunk_then_looks_again(self):
        # fetch() re-measures the range from wherever it ends up, so committing
        # more open-loop travel would spend the accuracy the approach buys.
        # complete=False must keep that cadence: one chunk, no 'unreached' noise.
        events = []
        robot = self.Dry()
        fetch.follow(robot, None, [(0., 60.), (0., 120.)],
                     lambda k, **f: events.append((k, f)), complete=False)
        legs = [f for k, f in events if k == 'leg']
        self.assertEqual(len(legs), 2)                    # one per waypoint
        self.assertFalse(any(k == 'unreached' for k, _ in events))

    def test_a_turn_that_never_closes_ends_the_waypoint_instead_of_looping(self):
        # Wheels that can only manage 1 degree must not spin the loop forever.
        pose, events, robot = self.drive([(40., 0.)], robot=self.Dry(cap=1.))
        self.assertTrue(any(k == 'cannot_face' for k, _ in events))
        self.assertLessEqual(len(robot.turns), fetch.FACE_ATTEMPTS)
        self.assertTrue(any(k == 'waypoint' and not f['reached'] for k, f in events))

    def test_a_waypoint_inside_the_pivot_circle_is_not_chased_in_circles(self):
        # Turning swings the lens ~7.4 cm about the pivot. A waypoint nearer
        # than that cannot be faced -- each turn carries the lens further than
        # the gap -- and the loop used to orbit it for 1440 degrees.
        _, events, robot = self.drive([(6., -1.)])
        self.assertTrue(any(k == 'too_close_to_face' for k, _ in events))
        self.assertLess(sum(abs(t) for t in robot.turns), 90.)

    def test_no_reachable_waypoint_makes_the_robot_spin(self):
        # Sweep the floor the recognizer can propose routes on. Before the
        # orbit and face-attempt guards, 81 of these spun a full 1440 degrees.
        worst = 0.
        for x in range(-120, 121, 15):
            for z in range(-45, 136, 15):
                if math.hypot(x, z) < fetch.ROUTE_MIN_LEG_CM:
                    continue
                robot = self.Dry()
                fetch.follow(robot, None, [(float(x), float(z))],
                             lambda k, **f: None)
                worst = max(worst, sum(abs(t) for t in robot.turns))
        self.assertLess(worst, 540.)

    def test_overshooting_past_a_waypoint_does_not_loop_on_it(self):
        # Short legs do overshoot -- 8.7 cm asked, 11.5 cm driven, measured on
        # carpet. A gross overshoot sails past the waypoint and the correction
        # sails back, so the loop needs a guard that is not about turning.
        events = []
        with patch.object(fetch, 'drive_leg',
                          side_effect=lambda robot, odo, cm, rec, stop=None: cm * 3.):
            fetch.follow(self.Dry(), None, [(0., 30.)],
                         lambda k, **f: events.append((k, f)))
        self.assertTrue(any(k == 'no_progress' for k, _ in events))
        self.assertLess(len(events), fetch.WAYPOINT_MAX_LEGS)

    def test_wheels_that_do_not_move_the_floor_stop_the_run(self):
        class Stuck(FollowTests.Dry):
            pass
        stuck = Stuck()
        events = []
        with patch.object(fetch, 'drive_leg', return_value=0.):
            fetch.follow(stuck, None, [(0., 40.)], lambda k, **f: events.append((k, f)))
        self.assertTrue(any(k == 'stalled' for k, _ in events))


class AvoidTests(unittest.TestCase):
    """The model has no scale, so the clearance has to be imposed locally."""

    CLEAR = fetch.OBSTACLE_RADIUS_CM + fetch.CORRIDOR_HALF_CM

    def nearest(self, route, obstacles):
        legs = [(0., 0.)] + list(route)
        best = None
        for _, spot in obstacles:
            for i in range(len(legs) - 1):
                ax, az = legs[i]
                bx, bz = legs[i + 1]
                dx, dz = bx - ax, bz - az
                span = dx * dx + dz * dz
                t = 0. if span < 1e-9 else max(0., min(1., (
                    (spot[0] - ax) * dx + (spot[1] - az) * dz) / span))
                gap = math.hypot(spot[0] - (ax + t * dx), spot[1] - (az + t * dz))
                best = gap if best is None else min(best, gap)
        return best

    def test_a_segment_that_shaves_an_obstacle_is_widened(self):
        # Both endpoints are clear; only the line between them is not. A first
        # version of this checked waypoints alone and so fixed nothing here.
        obstacles = [('box', (0., 40.))]
        route = [(-20., 20.), (-20., 60.), (0., 80.)]
        route[1] = (0., 40. + 2.)          # drag the middle onto the box
        widened, nudged = fetch.avoid(route, obstacles, (0., 80.))
        self.assertGreater(nudged, 0)
        self.assertGreaterEqual(self.nearest(widened, obstacles), self.CLEAR - .1)

    def test_clearance_improves_and_the_target_is_kept(self):
        obstacles = [('cable', (5., 30.))]
        goal = (0., 70.)
        route = [(0., 20.), (4., 32.), (0., 50.), goal]
        widened, _ = fetch.avoid(route, obstacles, goal)
        self.assertGreater(self.nearest(widened, obstacles),
                           self.nearest(route, obstacles))
        self.assertEqual(widened[-1], goal)

    def test_obstacles_beside_the_target_are_left_alone(self):
        # A bottle among scattered toys cannot be reached if the toys are
        # routed away from; the last stretch is allowed to come close.
        goal = (0., 60.)
        obstacles = [('toys', (4., 56.))]
        route = [(0., 20.), (0., 40.), goal]
        widened, nudged = fetch.avoid(route, obstacles, goal)
        self.assertEqual(nudged, 0)
        self.assertEqual(widened, route)

    def test_nothing_is_pushed_behind_the_robot(self):
        # Shoving a waypoint clear of something close and to one side used to
        # put it behind the wheels, which no turn can then face.
        obstacles = [('cable', (-6., 8.)), ('strip', (-6., 16.))]
        goal = (20., 26.)
        route = [(0., 6.), (2., 9.), (5., 12.), (9., 16.), goal]
        widened, _ = fetch.avoid(route, obstacles, goal)
        self.assertTrue(all(spot[1] >= 0. for spot in widened), widened)
        for i in range(1, len(widened)):
            self.assertGreaterEqual(
                math.hypot(widened[i][0] - widened[i - 1][0],
                           widened[i][1] - widened[i - 1][1]),
                fetch.ROUTE_MIN_LEG_CM - 1e-6, widened)

    def test_a_clear_route_is_returned_unchanged(self):
        route = [(0., 20.), (0., 40.)]
        widened, nudged = fetch.avoid(route, [('far', (80., 80.))], (0., 40.))
        self.assertEqual(nudged, 0)
        self.assertEqual(widened, route)


class MemoryTests(unittest.TestCase):
    """Each look used to start blind. Carrying the last answer forward fixes
    the one thing a single photograph cannot supply: what is beside the robot."""

    def lens(self):
        machine = robot()
        return machine

    def test_a_pixel_survives_a_round_trip_through_the_floor(self):
        machine = self.lens()
        for spot in ((320., 400.), (457., 216.), (120., 300.)):
            back = machine.pixel(*machine.ground(*spot))
            self.assertAlmostEqual(back[0], spot[0], delta=.01)
            self.assertAlmostEqual(back[1], spot[1], delta=.01)

    def test_floor_behind_the_camera_has_no_pixel(self):
        self.assertIsNone(self.lens().pixel(0., -30.))

    def test_rebase_moves_the_room_into_the_robots_new_frame(self):
        # Drive 40 cm forward and turn 90 right: a point that was 100 cm dead
        # ahead is now 60 cm off to the left and level with the robot.
        moved = fetch.rebase([(0., 100.)], [0., 40., 90.])[0]
        self.assertAlmostEqual(moved[0], -60., delta=.01)
        self.assertAlmostEqual(moved[1], 0., delta=.01)

    def test_rebase_of_no_motion_changes_nothing(self):
        self.assertEqual(fetch.rebase([(5., 20.)], [0., 0., 0.]), [(5., 20.)])

    def test_recall_redraws_the_last_answer_in_the_new_picture(self):
        machine = self.lens()
        memory = dict(route=[(0., 40.), (0., 80.)],
                      obstacles=[('cable', (0., 60.))])
        prior = fetch.recall(machine, memory, [0., 20., 0.])
        self.assertEqual(len(prior['route_pixels']), 2)
        self.assertEqual(prior['obstacles'][0]['label'], 'cable')
        # Everything is 20 cm nearer than it was, so it sits lower in the frame.
        was = machine.pixel(0., 40.)
        self.assertGreater(prior['route_pixels'][0]['y'], was[1])

    def test_what_has_left_the_picture_is_named_not_dropped(self):
        # An obstacle the robot has driven past is the whole reason for doing
        # this: it is still on the floor, and the next photograph cannot say so.
        machine = self.lens()
        memory = dict(route=[], obstacles=[('cable', (0., 20.))])
        prior = fetch.recall(machine, memory, [0., 60., 0.])
        self.assertEqual(prior['obstacles'], [])
        self.assertEqual(prior['out_of_frame'], ['cable'])

    def test_nothing_remembered_sends_no_prior(self):
        self.assertIsNone(fetch.recall(self.lens(),
                                       dict(route=[], obstacles=[]), [0., 0., 0.]))

    def test_the_prior_rides_in_the_request_beside_the_target(self):
        sent = {}

        class Reply(object):
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return b'{}'

        def capture(request, timeout=None):
            sent['body'] = json.loads(request.data.decode())
            return Reply()

        answer = dict(visible=False, contact_pixel=None, route_pixels=[],
                      obstacles=[], note='')
        reply = dict(status='completed', output=[dict(type='message', content=[
            dict(type='output_text', text=json.dumps(answer))])])
        prior = dict(route_pixels=[dict(x=1., y=2.)], obstacles=[],
                     out_of_frame=['cable'], since_then='drove 10 cm')
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test'}), \
                patch('json.load', return_value=reply), \
                patch('urllib.request.urlopen', side_effect=capture), \
                patch('cv2.imencode', return_value=(True, np.zeros(4, np.uint8))):
            fetch.recognize(np.zeros((480, 640, 3), np.uint8), 'bottle', prior)
        text = sent['body']['input'][0]['content'][0]['text']
        self.assertEqual(json.loads(text)['last_time'], prior)
        # And without a prior the field is absent rather than null.
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test'}), \
                patch('json.load', return_value=reply), \
                patch('urllib.request.urlopen', side_effect=capture), \
                patch('cv2.imencode', return_value=(True, np.zeros(4, np.uint8))):
            fetch.recognize(np.zeros((480, 640, 3), np.uint8), 'bottle')
        self.assertNotIn('last_time',
                         json.loads(sent['body']['input'][0]['content'][0]['text']))


class TurnRequestTests(unittest.TestCase):
    """The model may ask for one motion: a rotation.

    It exists because of geometry it cannot see. The corridor is 15 cm wide, so
    an obstacle 10 cm away blocks every heading until it is nearly behind the
    robot -- there is no route to draw, and a route drawn anyway is refused
    while the robot stands still. Rotating is the only move that helps, and the
    only one safe enough to hand over: it shifts the corridor without carrying
    the robot into anything.
    """

    def test_the_schema_offers_a_turn_and_stays_strict(self):
        self.assertIn('turn_degrees', fetch.SCHEMA['properties'])
        # A strict json_schema requires every property to be listed as required.
        self.assertEqual(set(fetch.SCHEMA['required']),
                         set(fetch.SCHEMA['properties']))
        self.assertFalse(fetch.SCHEMA['additionalProperties'])

    def test_a_turn_may_be_null_or_a_number(self):
        kinds = fetch.SCHEMA['properties']['turn_degrees']['anyOf']
        self.assertIn({'type': 'number'}, kinds)
        self.assertIn({'type': 'null'}, kinds)

    def test_rotation_is_still_the_only_motion_on_offer(self):
        # Everything else stays a question about pixels; nothing in the schema
        # can carry a distance to drive.
        for name in fetch.SCHEMA['properties']:
            self.assertNotIn(name, ('drive', 'drive_cm', 'speed', 'action',
                                    'forward_cm'))

    def test_the_prompt_says_how_to_tell_that_driving_is_hopeless(self):
        # The model cannot judge centimetres, so the rule has to be in pixels;
        # phrasing it as a distance measured as no change at all.
        self.assertIn('BOTTOM THIRD', fetch.PROMPT)


class ApproachShapeTests(unittest.TestCase):
    """Two things the operator saw go wrong on the robot."""

    def test_a_route_stops_short_of_the_target_it_is_sent_to(self):
        # The model is asked to end on the object's contact point, which is the
        # right answer to the question asked and the wrong thing to drive. A
        # planner that checks the range only at look time then drives a fixed
        # slice blind turns a 25 cm sighting into a collision.
        goal = (0., 60.)
        cut = fetch.stop_short([(0., 20.), (0., 40.), (0., 55.), goal], goal, 20.)
        self.assertTrue(cut)
        self.assertAlmostEqual(
            math.hypot(cut[-1][0] - goal[0], cut[-1][1] - goal[1]), 20., delta=.6)
        for spot in cut:
            self.assertGreaterEqual(
                math.hypot(spot[0] - goal[0], spot[1] - goal[1]), 19.4)

    def test_already_inside_the_standoff_leaves_nothing_to_drive(self):
        self.assertEqual(fetch.stop_short([(0., 8.)], (0., 10.), 20.), [])

    def test_no_target_means_no_trimming(self):
        route = [(0., 20.), (0., 40.)]
        self.assertEqual(fetch.stop_short(route, None, 20.), route)

    def test_widening_does_not_leave_a_first_waypoint_that_cannot_be_faced(self):
        # Pushing a near waypoint clear of something close moves it sideways:
        # (-0.1, 6.3) came back as (7.8, 2.5) on a real frame, a 72 degree pivot
        # to reach a point 8 cm away -- which follow then skips as unfaceable.
        obstacles = [('cable', (-6., 8.))]
        route = [(-0.1, 6.3), (2., 14.), (6., 30.), (10., 50.)]
        widened, _ = fetch.avoid(route, obstacles, (10., 50.))
        self.assertTrue(widened)
        orbit = fetch.pivot_radius() + fetch.ROUTE_MIN_LEG_CM
        self.assertGreaterEqual(math.hypot(widened[0][0], widened[0][1]), orbit)

    def test_a_clear_route_still_starts_straight_ahead(self):
        route = [(0., 20.), (0., 40.), (0., 60.)]
        widened, _ = fetch.avoid(route, [('far', (90., 90.))], (0., 60.))
        turn = abs(math.degrees(math.atan2(widened[0][0], widened[0][1])))
        self.assertLess(turn, 15., widened)


if __name__ == '__main__':
    unittest.main()
