"""A failed run must report why it failed, not what failed afterwards.

`Execution.run` stops the motors in a `finally`. That stop used to overwrite
`self.error` unconditionally, and once a control lease is invalidated *every*
later command fails the same way -- including the stop. So the run reported
"stop verification failed: Control cancelled or service restarted", which is
the one thing guaranteed to be true and says nothing about the cause.

Measured: a 273 cm trajectory died at 29.5 cm and reported only that. With the
first error preserved, the same trajectory named its real cause immediately --
an AttributeError from projecting an empty feature set.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'local_nav'))
import fetch
import trajectory_executor as ex


class Bench:
    def __init__(self):
        self.said = []

    def say(self, line):
        self.said.append(line)

    class _Abort:
        @staticmethod
        def is_set():
            return False

    abort = _Abort()


class Robot:
    """Fails the way an invalidated lease does: the stop fails too."""

    def __init__(self, turn_error=None, stop_error=None, fail_stop_from=0):
        self.turn_error = turn_error
        self.stop_error = stop_error
        self.fail_stop_from = fail_stop_from
        self.holds = 0

    def check(self):
        return {}

    def hold(self, left, right):
        self.holds += 1
        if self.stop_error and self.holds > self.fail_stop_from:
            raise fetch.Stop(self.stop_error)

    def frame(self):
        import numpy as np
        return np.zeros((4, 4, 3), np.uint8), {'imu_samples': [{'yaw': 0.}]}

    def heading(self, sample):
        return 0.

    def turn_continuous(self, degrees):
        if self.turn_error:
            raise fetch.Stop(self.turn_error)
        return degrees

    turn = turn_continuous


class ErrorPreservationTests(unittest.TestCase):
    def run_execution(self, robot):
        job = ex.Execution.__new__(ex.Execution)
        job.bench = Bench()
        job.robot = robot
        job.command = dict(waypoints_cm=[], initial_turn_deg=45.)
        job.route = []
        job.pose = [0., 0., 0.]
        job.trace = [[0., 0., 0.]]
        job.completed = 0
        job.travelled = 0.
        job.stop_after = None
        job.stop_waypoints = None
        job.deadline = None
        job.drive_started = None
        job.drive_start_distance = 0.
        job.last_control = None
        job.control_intervals = []
        job.tracking_rejections = 0
        job.tracking_rejection_streak = 0
        job.tracking_gap_started = None
        job.pack_v = None
        job.phase = 'ready'
        job.error = None
        job.camera = None
        job.check = lambda: None
        job.publish = lambda *a, **k: None
        job.result = lambda: {}
        with tempfile.TemporaryDirectory() as room:
            job.root = Path(room)
            job.run()
        return job

    def test_the_first_error_survives_a_failing_stop(self):
        job = self.run_execution(Robot(
            turn_error='control generation changed (cancelled elsewhere)',
            stop_error='Control cancelled or service restarted; replan'))
        self.assertEqual(job.phase, 'failed')
        self.assertIn('control generation changed', job.error)
        # The stop failure is still reported -- it is real -- but as a rider.
        self.assertIn('stop verification also failed', job.error)

    def test_a_failing_stop_alone_is_still_reported(self):
        # The run itself is clean and only the final stop fails, so the stop is
        # the whole story and must be named as the cause, not as a rider. The
        # route is empty, so hold() is called once on the way out of the try
        # body and once in the finally -- only the second one fails here.
        job = self.run_execution(Robot(stop_error='motors would not confirm zero',
                                       fail_stop_from=1))
        self.assertEqual(job.phase, 'failed')
        self.assertEqual(job.error, 'stop verification failed: motors would not confirm zero')

    def test_a_stop_that_fails_everywhere_still_names_the_cause_once(self):
        job = self.run_execution(Robot(stop_error='motors would not confirm zero'))
        self.assertEqual(job.phase, 'failed')
        self.assertIn('motors would not confirm zero', job.error)
        self.assertIn('stop verification also failed', job.error)

    def test_a_clean_run_keeps_no_error(self):
        job = self.run_execution(Robot())
        self.assertEqual(job.phase, 'completed')
        self.assertIsNone(job.error)

    def test_the_operator_line_carries_the_cause(self):
        # The bench log is what the dashboard shows and what an operator reads
        # first; it must not be the one place the cause is missing.
        job = self.run_execution(Robot(
            turn_error='floor tracking unavailable after 6 consecutive rejected frames',
            stop_error='Control cancelled or service restarted; replan'))
        self.assertTrue(any('floor tracking unavailable' in line
                            for line in job.bench.said), job.bench.said)


if __name__ == '__main__':
    unittest.main()
