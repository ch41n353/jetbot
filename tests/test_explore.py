import json
import math
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, 'local_nav'))
import explore
import fetch


STATUS = dict(session_id='s', control_epoch=1, healthy=True, motion_enabled=True,
              motor={'output': [0, 0]},
              power=dict(motion_allowed=True, stale=False, stop_latched=False,
                         pack_voltage_v=12.1),
              imu=dict(acceleration=[0., 0., 9.80665], gyro=[0., 0., 0.],
                       gyro_units='deg/s', time=1.))


def robot(**overrides):
    with patch.object(fetch, 'call', return_value=dict(STATUS, **overrides)):
        return fetch.Robot(dry_run=True)


def answer(**fields):
    """A sweep answer with every field the schema requires."""
    return dict(dict(visible=False, target_pixel=None, confidence=explore.ABSENT,
                     worth_a_closer_look=False, open_pixel=None, obstacles=[],
                     scene=''), **fields)


def journal(looks=(), target='the Advil bottle', travel=explore.TRAVEL_BUDGET_CM):
    """A journal at one station with `looks` already written into it."""
    book = explore.Journal(target, travel_budget_cm=travel)
    for heading, reply in looks:
        book.note(heading, reply, explore.bearing(
            robot(), reply['target_pixel']['x'], reply['target_pixel']['y'])
            if reply.get('target_pixel') else None)
    return book


class BearingTests(unittest.TestCase):
    """A pixel is a direction long after it has stopped being a distance."""

    def test_bearing_agrees_with_the_floor_projection_where_both_work(self):
        # The recorded sighting of 2026-09-16: fetch places it at 36.5 cm right
        # and 77.4 forward, so the angle to it is fixed by that.
        machine = robot()
        right, forward = machine.ground(457., 216.)
        self.assertAlmostEqual(explore.bearing(machine, 457., 216.),
                               math.degrees(math.atan2(right, forward)), delta=.1)

    def test_a_pixel_above_the_horizon_still_has_a_bearing(self):
        # This is the whole enabler: the model reports seeing things far beyond
        # where the floor can be placed, and a search only needs the direction.
        machine = robot()
        with self.assertRaises(fetch.Stop):
            machine.ground(560., 20.)
        self.assertGreater(explore.bearing(machine, 560., 20.), 20.)

    def test_the_frame_edge_is_beyond_a_quadrant(self):
        # Sizes the sweep step: one photograph covers about 126 degrees, so 60
        # degree cells overlap by half a frame instead of leaving gaps.
        machine = robot()
        self.assertGreater(explore.bearing(machine, 640., 240.), 55.)
        self.assertLess(explore.bearing(machine, 0., 240.), -55.)
        self.assertLess(abs(explore.bearing(machine, 320., 240.)), 2.)


class SchemaTests(unittest.TestCase):
    """Both questions are closed: the model picks from what it is offered."""

    def test_sweep_schema_is_strict_and_complete(self):
        self.assertFalse(explore.SWEEP_SCHEMA['additionalProperties'])
        self.assertEqual(set(explore.SWEEP_SCHEMA['required']),
                         set(explore.SWEEP_SCHEMA['properties']))
        self.assertEqual(explore.SWEEP_SCHEMA['properties']['confidence']['enum'],
                         [explore.SURE, explore.UNSURE, explore.ABSENT])
        # Perception only. A look may not ask for a motion; the plan does that,
        # and only from the action set below.
        for name in explore.SWEEP_SCHEMA['properties']:
            self.assertNotIn(name, ('turn_degrees', 'drive_cm', 'action'))

    def test_plan_schema_offers_three_actions_and_nothing_else(self):
        step = explore.PLAN_SCHEMA['properties']['steps']['items']
        self.assertFalse(step['additionalProperties'])
        self.assertEqual(set(step['required']), set(step['properties']))
        self.assertEqual(step['properties']['action']['enum'],
                         ['sweep', 'go', 'give_up'])
        # Every distance the model may name is in centimetres of travel, not a
        # position: there is no field in which it could hand back a coordinate.
        for name in step['properties']:
            self.assertNotIn(name, ('x', 'y', 'position', 'waypoint', 'station'))


class LookTests(unittest.TestCase):
    def reply(self, payload):
        raw = dict(status='completed', output=[dict(type='message', content=[
            dict(type='output_text', text=json.dumps(payload))])])

        class Reply(object):
            def __enter__(self): return self
            def __exit__(self, *a): return False
        return patch.dict(os.environ, {'OPENAI_API_KEY': 'test'}), \
            patch('json.load', return_value=raw), \
            patch('urllib.request.urlopen', return_value=Reply()), \
            patch('cv2.imencode', return_value=(True, np.zeros(4, np.uint8)))

    def ask(self, payload):
        env, load, url, encode = self.reply(payload)
        with env, load, url, encode:
            return explore.look(np.zeros((480, 640, 3), np.uint8), 'x')

    def test_a_pixel_outside_the_frame_is_not_a_sighting(self):
        seen = self.ask(answer(visible=True, confidence=explore.SURE,
                               target_pixel=dict(x=900., y=300.)))
        self.assertFalse(seen['visible'])
        self.assertEqual(seen['confidence'], explore.ABSENT)
        self.assertIsNone(seen['target_pixel'])

    def test_sure_without_a_pixel_is_demoted_to_a_lead(self):
        # "Sure" with nowhere to point is not something the robot can act on,
        # and acting on it would mean turning to a heading nobody chose.
        seen = self.ask(answer(visible=True, confidence=explore.SURE))
        self.assertEqual(seen['confidence'], explore.UNSURE)

    def test_a_pixel_in_the_frame_survives(self):
        seen = self.ask(answer(visible=True, confidence=explore.SURE,
                               target_pixel=dict(x=200., y=260.)))
        self.assertEqual(seen['confidence'], explore.SURE)
        self.assertEqual(seen['target_pixel']['x'], 200.)


class SweepShapeTests(unittest.TestCase):
    def test_a_full_circle_is_closed_by_evenly_spaced_cells(self):
        headings = explore.sweep_headings(0., 360., 60.)
        self.assertEqual(len(headings), 6)
        for index in range(len(headings)):
            gap = explore.wrap(headings[(index + 1) % 6] - headings[index])
            self.assertAlmostEqual(gap, 60., delta=.01)

    def test_a_partial_arc_puts_its_cells_inside_the_arc(self):
        headings = explore.sweep_headings(90., 120., 60.)
        self.assertEqual(headings, [60., 120.])

    def test_a_silly_step_is_brought_back_to_something_a_sweep_can_do(self):
        self.assertEqual(len(explore.sweep_headings(0., 360., 1.)),
                         int(round(360. / explore.SWEEP_MIN_STEP_DEG)))
        self.assertEqual(len(explore.sweep_headings(0., 10., 60.)), 1)


class RefinementTests(unittest.TestCase):
    """A coarse sweep is cheap; the second look is where the answer comes from."""

    def test_an_unsure_sighting_is_looked_at_again_head_on(self):
        looks = [dict(heading=60., confidence=explore.UNSURE,
                      candidate_bearing=-40., worth_a_closer_look=False)]
        self.assertEqual(explore.refine_headings(looks), [20.])

    def test_a_candidate_already_in_the_middle_is_not_re_photographed(self):
        looks = [dict(heading=60., confidence=explore.UNSURE,
                      candidate_bearing=3., worth_a_closer_look=False)]
        self.assertEqual(explore.refine_headings(looks), [])

    def test_a_direction_that_could_hide_it_gets_its_flanks(self):
        looks = [dict(heading=0., confidence=explore.ABSENT,
                      candidate_bearing=None, worth_a_closer_look=True)]
        self.assertEqual(explore.refine_headings(looks, 60.), [-20., 20.])

    def test_sightings_are_followed_before_hunches_and_the_budget_holds(self):
        looks = [dict(heading=0., confidence=explore.ABSENT,
                      candidate_bearing=None, worth_a_closer_look=True),
                 dict(heading=180., confidence=explore.UNSURE,
                      candidate_bearing=30., worth_a_closer_look=False)]
        wanted = explore.refine_headings(looks, 60., budget=2)
        self.assertEqual(wanted[0], -150.)      # the sighting, centred
        self.assertEqual(len(wanted), 2)

    def test_the_budget_is_spread_round_the_room_not_spent_on_one_corner(self):
        # Measured over 52 stored frames: in a cluttered room more than half the
        # directions come back flagged, so taking them in order refines the
        # first two cells twice and never reaches the others.
        looks = [dict(heading=heading, confidence=explore.ABSENT,
                      candidate_bearing=None, worth_a_closer_look=True)
                 for heading in (0., 60., 120., 180.)]
        self.assertEqual(explore.refine_headings(looks, 60., budget=4),
                         [-20., 40., 100., 160.])

    def test_a_close_look_is_not_itself_refined(self):
        looks = [dict(heading=0., confidence=explore.UNSURE,
                      candidate_bearing=40., worth_a_closer_look=True,
                      refined=True)]
        self.assertEqual(explore.refine_headings(looks), [])


class ReadingTests(unittest.TestCase):
    def test_a_sighting_on_the_rim_of_the_frame_is_only_a_lead(self):
        self.assertTrue(explore.edge_sighting(dict(x=20., y=300.)))
        self.assertTrue(explore.edge_sighting(dict(x=620., y=300.)))
        self.assertFalse(explore.edge_sighting(dict(x=320., y=300.)))
        self.assertFalse(explore.edge_sighting(None))

    def test_blocked_is_the_models_own_pixel_rule_not_a_distance(self):
        near = answer(obstacles=[dict(label='box', contact_pixel=dict(x=320., y=400.))])
        aside = answer(obstacles=[dict(label='box', contact_pixel=dict(x=40., y=400.))])
        far = answer(obstacles=[dict(label='box', contact_pixel=dict(x=320., y=210.))])
        self.assertTrue(explore.blocked_ahead(near))
        self.assertFalse(explore.blocked_ahead(aside))
        self.assertFalse(explore.blocked_ahead(far))


class JournalTests(unittest.TestCase):
    def test_it_reads_as_a_table_of_directions_with_no_positions_in_it(self):
        book = journal([(0., answer(scene='open carpet to a doorway',
                                    worth_a_closer_look=True)),
                        (60., answer(scene='desk legs and cables',
                                     obstacles=[dict(label='cable',
                                                     contact_pixel=dict(x=330., y=430.))]))])
        text = book.render()
        self.assertIn('looking for: the Advil bottle', text)
        self.assertIn('+000', text)
        self.assertIn('could hide it', text)
        self.assertIn('blocked', text)
        self.assertIn('the robot is here, facing 0', text)
        self.assertIn('budget left', text)
        # Nothing in the journal may look like a coordinate: the robot has no
        # position, and a model given numbers will use them.
        self.assertNotIn('cm,', text)
        self.assertNotIn('x=', text)

    def test_a_second_station_records_how_it_was_reached_not_where_it_is(self):
        book = journal()
        book.arrive('A', heading=-90., distance=100.)
        text = book.render()
        self.assertIn('station A (from start: turned -90, drove about 100 cm)',
                      text)

    def test_coverage_and_blockage_are_answered_per_direction(self):
        book = journal([(0., answer(obstacles=[dict(label='bin',
                                                    contact_pixel=dict(x=300., y=450.))])),
                        (60., answer())])
        self.assertTrue(book.covers(20.))
        self.assertFalse(book.covers(150.))
        self.assertTrue(book.blocked(10.))
        self.assertFalse(book.blocked(55.))

    def test_looks_are_counted_against_the_budget(self):
        book = journal([(0., answer()), (60., answer())])
        self.assertEqual(book.looks_left, explore.LOOK_BUDGET - 2)


class PlanValidationTests(unittest.TestCase):
    """The model proposes; this decides. Nonsense is refused, never patched."""

    def book(self):
        return journal([(0., answer(scene='open carpet', worth_a_closer_look=True)),
                        (60., answer(scene='boxes against the wall',
                                     obstacles=[dict(label='box',
                                                     contact_pixel=dict(x=320., y=420.))])),
                        (-60., answer(scene='bare wall'))])

    def plan(self, *steps):
        return dict(steps=[dict(dict(action='sweep', heading_degrees=None,
                                     arc_degrees=None, distance_cm=None, why=''),
                                **step) for step in steps], note='')

    def test_a_sensible_plan_survives_unchanged(self):
        steps, problems = explore.validate_plan(
            self.plan(dict(action='go', heading_degrees=0., distance_cm=100.),
                      dict(action='sweep', heading_degrees=0., arc_degrees=360.)),
            self.book())
        self.assertEqual(problems, [])
        self.assertEqual(steps[0], dict(action='go', heading=0., distance=100.,
                                        why=''))
        self.assertEqual(steps[1]['arc'], 360.)

    def test_an_action_the_robot_does_not_have_is_refused(self):
        steps, problems = explore.validate_plan(
            self.plan(dict(action='map_the_room')), self.book())
        self.assertEqual(steps, [])
        self.assertIn('not one of sweep, go, give_up', problems[0])

    def test_a_hop_into_floor_nobody_has_photographed_is_refused(self):
        _, problems = explore.validate_plan(
            self.plan(dict(action='go', heading_degrees=150., distance_cm=80.)),
            self.book())
        self.assertIn('has not looked along +150', problems[0])

    def test_a_hop_into_something_it_reported_on_the_floor_is_refused(self):
        _, problems = explore.validate_plan(
            self.plan(dict(action='go', heading_degrees=60., distance_cm=80.)),
            self.book())
        self.assertIn('blocked', problems[0])

    def test_a_hop_the_robot_cannot_measure_is_refused(self):
        for distance in (5., 400.):
            _, problems = explore.validate_plan(
                self.plan(dict(action='go', heading_degrees=0.,
                               distance_cm=distance)), self.book())
            self.assertIn('outside the', problems[0])

    def test_a_missing_distance_is_refused_rather_than_guessed(self):
        _, problems = explore.validate_plan(
            self.plan(dict(action='go', heading_degrees=0.)), self.book())
        self.assertIn('distance_cm is missing', problems[0])

    def test_driving_past_the_budget_is_refused(self):
        book = journal([(0., answer())], travel=120.)
        _, problems = explore.validate_plan(
            self.plan(dict(action='go', heading_degrees=0., distance_cm=100.),
                      dict(action='go', heading_degrees=0., distance_cm=100.)),
            book)
        self.assertIn('more driving than the budget', problems[0])

    def test_steps_after_the_first_hop_are_not_checked_against_this_room(self):
        # The robot will be standing somewhere it has never stood, so there is
        # nothing to check them against and rejecting them would be a guess.
        steps, problems = explore.validate_plan(
            self.plan(dict(action='go', heading_degrees=0., distance_cm=60.),
                      dict(action='go', heading_degrees=170., distance_cm=60.)),
            self.book())
        self.assertEqual(problems, [])
        self.assertEqual(len(steps), 2)

    def test_giving_up_has_to_come_last(self):
        _, problems = explore.validate_plan(
            self.plan(dict(action='give_up'),
                      dict(action='sweep', heading_degrees=0., arc_degrees=360.)),
            self.book())
        self.assertIn('give_up has to be the last step', problems[0])

    def test_an_empty_or_malformed_plan_is_refused(self):
        for bad in (dict(steps=[], note=''), dict(note=''),
                    dict(steps='sweep everywhere', note='')):
            steps, problems = explore.validate_plan(bad, self.book())
            self.assertEqual(steps, [])
            self.assertEqual(problems, ['the plan has no steps'])

    def test_a_plan_longer_than_the_robot_can_believe_in_is_refused(self):
        _, problems = explore.validate_plan(
            self.plan(*[dict(action='sweep', heading_degrees=0., arc_degrees=360.)]
                      * (explore.PLAN_MAX_STEPS + 1)), self.book())
        self.assertIn('more than the', problems[0])

    def test_a_heading_off_the_compass_is_refused(self):
        _, problems = explore.validate_plan(
            self.plan(dict(action='sweep', heading_degrees=400.,
                           arc_degrees=90.)), self.book())
        self.assertIn('not within', problems[0])

    def test_the_fallback_looks_before_it_drives_and_says_so_when_done(self):
        empty = explore.Journal('x')
        self.assertEqual(explore.fallback_plan(empty)[0]['action'], 'sweep')
        swept = journal([(heading, answer())
                         for heading in explore.sweep_headings()])
        self.assertEqual(explore.fallback_plan(swept)[0]['action'], 'give_up')
        lead = journal([(heading, answer(worth_a_closer_look=heading == 120.))
                        for heading in explore.sweep_headings()])
        self.assertEqual(explore.fallback_plan(lead)[0],
                         dict(action='go', heading=120.,
                              distance=explore.GO_MAX_CM,
                              why='the only direction here that could hide it'))


class Wheels(object):
    """A robot that turns perfectly and photographs nothing, for the sweep loop."""

    def __init__(self, replies):
        machine = robot()
        self.K, self.D, self.pitch = machine.K, machine.D, machine.pitch
        self.replies = replies
        self.turned = []

    def turn(self, degrees):
        self.turned.append(degrees)
        return degrees

    def frame(self):
        return np.zeros((480, 640, 3), np.uint8), {}


class SweepLoopTests(unittest.TestCase):
    """The loop that spends the budget, with the model and the motors stubbed."""

    def hunt(self, replies):
        wheels = Wheels(list(replies))
        hunt = explore.Search(wheels, 'the Advil bottle')
        asked = []

        def answering(image, target, timeout=20.):
            asked.append(target)
            return wheels.replies.pop(0) if wheels.replies else answer()
        return wheels, hunt, asked, patch.object(explore, 'look', answering)

    def test_the_coarse_pass_stops_the_moment_it_is_sure(self):
        replies = [answer(), answer(),
                   answer(visible=True, confidence=explore.SURE,
                          target_pixel=dict(x=200., y=300.))]
        wheels, hunt, asked, stub = self.hunt(replies)
        with stub:
            found = hunt.sweep()
        self.assertIsNotNone(found)
        self.assertEqual(len(asked), 3)          # not the whole circle
        self.assertEqual(found['heading'], 120.)

    def test_a_flagged_direction_is_photographed_again_more_finely(self):
        replies = [answer(worth_a_closer_look=True)] + [answer()] * 5
        wheels, hunt, asked, stub = self.hunt(replies)
        with stub:
            hunt.sweep()
        # Six coarse looks and the two flanks of the flagged one.
        self.assertEqual(len(asked), 8)
        self.assertEqual([look['heading'] for look in hunt.journal.here['looks']],
                         [0., 60., 120., -180., -120., -60., -20., 20.])

    def test_a_sighting_on_the_rim_is_chased_rather_than_driven_at(self):
        replies = [answer(visible=True, confidence=explore.SURE,
                          target_pixel=dict(x=628., y=240.))] + [answer()] * 5
        wheels, hunt, asked, stub = self.hunt(replies)
        with stub:
            found = hunt.sweep()
        self.assertIsNone(found)                 # the rim is not a handover
        self.assertTrue(hunt.journal.here['looks'][0]['edge'])
        self.assertGreater(len(asked), 6)        # it went back for a closer look

    def test_a_search_that_finds_nothing_says_what_ran_out(self):
        wheels, hunt, asked, stub = self.hunt([])
        hunt.journal.looks_left = 6
        with stub, patch.object(explore, 'propose',
                                return_value=([], 'none')):
            result = hunt.run()
        self.assertEqual(result['outcome'], 'not_found')
        self.assertIn('out of looks', result['why'])
        self.assertIn('looking for', result['journal'])

    def test_the_handover_turns_to_put_the_target_ahead(self):
        machine = robot()
        pixel = dict(x=200., y=300.)
        replies = [answer(visible=True, confidence=explore.SURE, target_pixel=pixel)]
        wheels, hunt, asked, stub = self.hunt(replies)
        with stub:
            result = hunt.run(handoff=lambda: 'drove')
        self.assertEqual(result['outcome'], 'handed_over')
        self.assertEqual(result['result'], 'drove')
        # It ends facing the object, which is a bearing taken from its pixel.
        self.assertAlmostEqual(result['heading'],
                               explore.bearing(machine, 200., 300.), delta=1.)


if __name__ == '__main__':
    unittest.main()
