"""The trial harness must not be able to report a pass it did not observe.

`/api/local-execution` always answers with the most recent job. A submission
the planner refused therefore leaves the *previous* job's terminal result
sitting there, and a poll for "any terminal phase" reads it as success.

Measured: sixteen randomised turns reported 16/16 completed in 0.0-1.0 s each
with 0 cm driven, while only four of them had moved the robot at all. A test
harness whose only possible outcome is success is worse than no harness.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'scripts'))
import local_reach_loop as loop
import local_reach_trial as trial


class StalePollTests(unittest.TestCase):
    def setUp(self):
        self.original = (trial.get, trial.post)

    def tearDown(self):
        trial.get, trial.post = self.original

    def test_a_stale_terminal_result_is_not_mistaken_for_this_run(self):
        replies = [
            {'execution_id': 'OLD', 'phase': 'completed'},   # the previous job
            {'execution_id': 'OLD', 'phase': 'completed'},
            {'execution_id': 'NEW', 'phase': 'driving'},
            {'execution_id': 'NEW', 'phase': 'failed', 'error': 'no progress'},
        ]
        trial.get = lambda path, timeout=15: replies.pop(0)
        result = trial.wait_for_terminal('NEW', limit=5.)
        self.assertEqual(result['execution_id'], 'NEW')
        self.assertEqual(result['phase'], 'failed')

    def test_waiting_for_a_run_that_never_starts_times_out_rather_than_passing(self):
        trial.get = lambda path, timeout=15: {'execution_id': 'OLD',
                                              'phase': 'completed'}
        result = trial.wait_for_terminal('NEW', limit=1.)
        self.assertEqual(result['phase'], 'harness-timeout')
        self.assertNotIn(result['phase'], trial.TERMINAL)

    def test_a_refused_submission_is_reported_as_rejected(self):
        # The planner answers a refusal with bench state, which carries no
        # execution_id. That must end the trial, not fall through to a poll.
        trial.get = lambda path, timeout=15: {'running': False}
        trial.post = lambda path, body, timeout=30: (
            {'frame_token': 3} if path == '/api/capture'
            else {'log': 'a mission is running; press STOP first'})
        result = trial.execute(dict(waypoints_cm=[[0., 20.]], initial_turn_deg=0.))
        self.assertEqual(result['phase'], 'rejected')
        self.assertIn('did not start', result['error'])


class BlockedClassificationTests(unittest.TestCase):
    """A dead lease says nothing about the floor.

    Counting it as blockage sent twenty consecutive goals backwards while the
    real problem was that no goal in any direction could start at all.
    """

    def test_a_lease_fault_is_not_blockage(self):
        result = {'phase': 'failed',
                  'error': 'control generation changed (cancelled elsewhere)',
                  'travelled_cm': 0.}
        self.assertTrue(loop.lease_fault(result))
        self.assertFalse(loop.blocked_by(result))

    def test_the_other_half_of_the_lease_message_is_recognised_too(self):
        self.assertTrue(loop.lease_fault(
            {'error': 'Control cancelled or service restarted; replan'}))

    def test_a_stall_is_blockage(self):
        for error in ('no progress toward next waypoint', 'waypoint timeout',
                      'wheels are not turning the robot (stalled)'):
            self.assertTrue(loop.blocked_by(
                {'phase': 'failed', 'error': error, 'travelled_cm': 0.}), error)

    def test_a_run_that_went_nowhere_is_blockage_whatever_it_said(self):
        self.assertTrue(loop.blocked_by(
            {'phase': 'failed', 'error': 'something new', 'travelled_cm': 1.}))

    def test_a_finished_run_is_never_blockage(self):
        for phase in ('completed', 'paused'):
            self.assertFalse(loop.blocked_by(
                {'phase': phase, 'error': None, 'travelled_cm': 40.}), phase)


class GoalBoundsTests(unittest.TestCase):
    """Goals stay inside the radius the local planner actually owns."""

    def test_forward_goals_respect_the_scanned_range(self):
        import random
        rng = random.Random(11)
        room = [(0, 40.), (15, 100.), (-30, 55.)]
        for _ in range(200):
            goal = loop.forward_goal(rng, room)
            self.assertLessEqual(math_hypot(goal), loop.RADIUS_CM + 1e-6)

    def test_backward_goals_are_behind_and_short(self):
        import random
        rng = random.Random(12)
        for _ in range(200):
            goal = loop.backward_goal(rng)
            self.assertLess(goal[1], 0., 'backward goal was not behind')
            self.assertLessEqual(math_hypot(goal), 46.)

    def test_a_blocked_robot_is_sent_backwards_even_with_room_reported(self):
        import random
        shape, traj, _ = loop.build(random.Random(3), [(0, 100.)], True)
        self.assertEqual(shape, 'backward')
        self.assertLess(traj['waypoints_cm'][0][1], 0.)


def math_hypot(point):
    import math
    return math.hypot(point[0], point[1])


if __name__ == '__main__':
    unittest.main()
