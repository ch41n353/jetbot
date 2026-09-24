"""A new instruction re-syncs the control generation instead of inheriting a dead one.

The service invalidates the control generation on any request it rejects, and
Robot.halt() used to be the only code that ever re-read it. So one transient
rejection left the bench holding a dead lease and every later local execution
died instantly with 'control generation changed (cancelled elsewhere)'. Measured
live: twenty consecutive trajectories refused in 0.2 s each, 0 cm driven, until
an operator pressed stop.

launch() now re-syncs, which is safe precisely there: bench.running() has just
been checked, so no controller is mid-run to be resumed.
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'local_nav'))
import fetch
import trajectory_executor


class FakeRobot:
    def __init__(self, epoch):
        self.token = {'session_id': 'S', 'control_epoch': epoch}
        self.resynced = 0

    def resync(self):
        self.resynced += 1
        self.token['control_epoch'] = 41
        return 41


class FakeBench:
    def __init__(self, epoch=7):
        self.robot = FakeRobot(epoch)
        self.frame = object()
        self.frame_token = 3
        self.local_execution = None
        self.flight = None

        class Abort:
            def clear(self):
                pass
        self.abort = Abort()

    def running(self):
        return False


class LeaseResyncTests(unittest.TestCase):
    def body(self):
        return dict(trajectory=dict(waypoints_cm=[[0., 20.]], initial_turn_deg=0.),
                    frame_token=3)

    def test_launch_resyncs_before_starting(self):
        bench = FakeBench(epoch=7)
        started = {}

        class Job:
            def __init__(self, bench_, body_):
                started['ok'] = True

            def result(self):
                return {'phase': 'ready'}

            def run(self):
                pass

        with patch.object(trajectory_executor, 'Execution', Job):
            trajectory_executor.launch(bench, self.body())
        self.assertEqual(bench.robot.resynced, 1)
        self.assertEqual(bench.robot.token['control_epoch'], 41)
        self.assertTrue(started.get('ok'))

    def test_busy_controller_is_refused_without_resync(self):
        # Re-syncing while something is driving is the one case the generation
        # exists to prevent, so the busy check must come first.
        bench = FakeBench()
        bench.running = lambda: True
        with self.assertRaises(ValueError):
            trajectory_executor.launch(bench, self.body())
        self.assertEqual(bench.robot.resynced, 0)

    def test_stale_frame_token_is_refused_without_resync(self):
        bench = FakeBench()
        body = self.body()
        body['frame_token'] = 99
        with self.assertRaises(ValueError):
            trajectory_executor.launch(bench, body)
        self.assertEqual(bench.robot.resynced, 0)

    def test_robot_resync_reads_the_service_generation(self):
        robot = fetch.Robot.__new__(fetch.Robot)
        robot.token = {'session_id': 'S', 'control_epoch': 2}
        with patch.object(fetch, 'call',
                          return_value={'session_id': 'S', 'control_epoch': 9}):
            self.assertEqual(robot.resync(), 9)
        self.assertEqual(robot.token['control_epoch'], 9)
        # The session id is deliberately untouched: a genuinely restarted
        # service is a different robot, and Robot.check() must still catch it.
        self.assertEqual(robot.token['session_id'], 'S')


if __name__ == '__main__':
    unittest.main()
