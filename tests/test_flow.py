"""Checks on the pipelined planner's bookkeeping.

Flowing means several looks are in the air at once so the wheels never stop.
Two things then have to be right, and neither is visible from watching the
robot: answers come back out of order because latency is not constant (2.18 to
4.56 s measured), and every answer describes the frame of the picture it was
planned from rather than where the robot is when it arrives.
"""

import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, 'local_nav'))
import planner_demo
import fetch


class Thread(object):
    def __init__(self, alive):
        self.alive = alive

    def is_alive(self):
        return self.alive


def job(seq, finished, answered=True):
    return dict(seq=seq, thread=Thread(not finished),
                holder=({'answer': {}} if answered else {'error': 'boom'}),
                shot=[0., 0., 0.])


class OrderingTests(unittest.TestCase):
    """An answer older than the plan in use must be discarded, not applied."""

    def test_a_newer_answer_is_taken(self):
        chosen, keep, dropped = planner_demo.Bench.freshest([job(6, True)], 5)
        self.assertEqual(chosen['seq'], 6)
        self.assertEqual(keep, [])
        self.assertEqual(dropped, 0)

    def test_an_answer_older_than_the_plan_in_use_is_dropped(self):
        # The robot already steers on look 5; look 3 arriving late describes a
        # staler picture, so using it would be worse than not having asked.
        chosen, _, dropped = planner_demo.Bench.freshest([job(3, True)], 5)
        self.assertIsNone(chosen)
        self.assertEqual(dropped, 1)

    def test_when_several_land_together_the_newest_wins(self):
        chosen, _, dropped = planner_demo.Bench.freshest(
            [job(3, True), job(7, True), job(4, True)], 5)
        self.assertEqual(chosen['seq'], 7)
        self.assertEqual(dropped, 2)

    def test_calls_still_in_flight_are_kept(self):
        chosen, keep, dropped = planner_demo.Bench.freshest(
            [job(6, False), job(7, False)], 5)
        self.assertIsNone(chosen)
        self.assertEqual([j['seq'] for j in keep], [6, 7])
        self.assertEqual(dropped, 0)

    def test_older_calls_still_in_flight_are_abandoned(self):
        # Look 6 landed, so look 4 is already overtaken; waiting for it would
        # keep a slot busy for an answer that could never be applied.
        chosen, keep, _ = planner_demo.Bench.freshest(
            [job(4, False), job(6, True)], 5)
        self.assertEqual(chosen['seq'], 6)
        self.assertEqual(keep, [])


class RebaseTests(unittest.TestCase):
    """A route arrives in the frame of the picture it was planned from."""

    def bench(self):
        return planner_demo.Bench.__new__(planner_demo.Bench)

    def test_composing_motions_tracks_the_robot(self):
        machine = self.bench()
        pose = machine.compose([0., 0., 0.], [0., 20., 0.])
        pose = machine.compose(pose, [0., 0., 90.])
        pose = machine.compose(pose, [0., 10., 0.])
        self.assertAlmostEqual(pose[0], 10., delta=.01)
        self.assertAlmostEqual(pose[1], 20., delta=.01)
        self.assertAlmostEqual(pose[2], 90., delta=.01)

    def test_motion_since_a_shot_is_given_in_that_shots_frame(self):
        machine = self.bench()
        shot = machine.compose([0., 0., 0.], [0., 20., 0.])
        shot = machine.compose(shot, [0., 0., 90.])
        now = machine.compose(shot, [0., 10., 0.])
        drift = machine.since_pose(shot, now)
        self.assertAlmostEqual(drift[0], 0., delta=.01)
        self.assertAlmostEqual(drift[1], 10., delta=.01)
        self.assertAlmostEqual(drift[2], 0., delta=.01)

    def test_a_route_rebased_by_the_drive_lands_where_the_floor_is(self):
        # Planned 60 cm ahead; by the time it arrives the robot has driven 25.
        drift = [0., 25., 0.]
        moved = fetch.rebase([(0., 60.)], drift)[0]
        self.assertAlmostEqual(moved[1], 35., delta=.01)

    def test_no_motion_during_the_call_leaves_the_route_alone(self):
        route = [(5., 30.), (10., 50.)]
        self.assertEqual(fetch.rebase(route, [0., 0., 0.]), route)


class InterruptTests(unittest.TestCase):
    """A landed plan should take over at once, not at the end of the leg."""

    class Dry(object):
        dry_run = True
        speed = fetch.SPEED_CM_S

        def turn(self, degrees):
            return degrees

    def test_an_interrupt_hands_control_back_early(self):
        rings = [0]

        def ring():
            rings[0] += 1
            return rings[0] > 2

        pose = fetch.follow(self.Dry(), None, [(0., 20.), (0., 40.), (0., 60.)],
                            lambda k, **f: None, interrupt=ring)
        self.assertLess(math.hypot(pose[0], pose[1]), 60.)

    def test_without_one_the_route_is_driven_out(self):
        pose = fetch.follow(self.Dry(), None, [(0., 20.), (0., 40.)],
                            lambda k, **f: None, interrupt=lambda: False)
        self.assertAlmostEqual(pose[1], 40., delta=1.)


if __name__ == '__main__':
    unittest.main()
