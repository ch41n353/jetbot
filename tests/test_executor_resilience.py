"""Three guards that ended healthy runs, and the shape of their fixes.

All three were found by driving randomised trajectories rather than by reading
the code, and all three share a failure mode: a bound that was correct in one
regime applied flatly across every regime.

1. Turn convergence was a flat 8 degrees -- 80% of a 10 degree request and 5%
   of a 160 degree one. A -160.1 degree escape pivot off a wall was declared a
   failure, aborting the trajectory the pivot had just made possible.
2. The yaw-rate runaway guard fired on a single sample. One 315 deg/s reading,
   a knock or a gyro glitch, ended a leg.
3. Drive duty was pinned at the 20% measured on a full pack, with no account of
   the pack sagging to 10.8 V over a session.
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'local_nav'))
import time
import unittest.mock

import numpy as np

import fetch
import trajectory_executor as ex


class TurnToleranceTests(unittest.TestCase):
    def test_small_turns_keep_the_absolute_floor(self):
        # A proportional-only rule would give a 10 degree turn 0.8 degrees,
        # which no surface delivers.
        self.assertEqual(ex.turn_tolerance(10.), 8.)
        self.assertEqual(ex.turn_tolerance(-5.), 8.)

    def test_large_turns_scale(self):
        self.assertAlmostEqual(ex.turn_tolerance(160.), 12.8)
        self.assertAlmostEqual(ex.turn_tolerance(-180.), 14.4)

    def test_the_measured_escape_pivot_would_now_pass(self):
        # The live case: -160.1 measured against a request near -170.
        self.assertLessEqual(abs(-170. - -160.1), ex.turn_tolerance(-170.))


class Spy:
    """A robot whose turns fall short by a fixed fraction."""

    def __init__(self, delivers=1.0, cap=None):
        self.delivers = delivers
        self.cap = cap
        self.asked = []

    def check(self):
        return {}

    def frame(self):
        import numpy as np
        return np.zeros((4, 4, 3), np.uint8), {'imu_samples': [{'yaw': 0.}]}

    def turn_continuous(self, degrees):
        self.asked.append(degrees)
        moved = degrees * self.delivers
        if self.cap is not None and abs(moved) > self.cap:
            moved = math.copysign(self.cap, moved)
        return moved

    # The executor selects the sweep with
    # getattr(robot, 'turn_continuous', robot.turn), and Python evaluates that
    # default eagerly, so both names have to exist even when only one is used.
    turn = turn_continuous


class TurnRetryTests(unittest.TestCase):
    def execution(self, robot):
        job = ex.Execution.__new__(ex.Execution)
        job.robot = robot
        job.pose = [0., 0., 0.]
        job.trace = [[0., 0., 0.]]
        job.camera = None
        job.phase = 'ready'
        job.check = lambda: None
        job.publish = lambda *a, **k: None
        return job

    def test_a_short_sweep_is_retried_until_it_converges(self):
        robot = Spy(delivers=.9)
        job = self.execution(robot)
        swept = job.turn(100.)
        self.assertGreater(len(robot.asked), 1, 'residual was never retried')
        self.assertLessEqual(abs(100. - swept), ex.turn_tolerance(100.))
        # The pose must carry what was actually turned, not what was asked.
        self.assertAlmostEqual(job.pose[2], swept, places=6)

    def test_one_good_sweep_does_not_retry(self):
        robot = Spy(delivers=1.0)
        job = self.execution(robot)
        job.turn(90.)
        self.assertEqual(len(robot.asked), 1)

    def test_a_turn_that_cannot_move_still_fails(self):
        # Retrying must not mask a robot that is going nowhere: a sweep that
        # delivers nothing ends the attempt instead of burning the budget.
        robot = Spy(delivers=0.)
        job = self.execution(robot)
        with self.assertRaises(fetch.Stop) as caught:
            job.turn(90.)
        self.assertIn('did not converge', str(caught.exception))
        self.assertEqual(len(robot.asked), 1)

    def test_a_persistently_short_turn_eventually_fails(self):
        robot = Spy(delivers=.2)
        job = self.execution(robot)
        with self.assertRaises(fetch.Stop):
            job.turn(120.)
        self.assertEqual(len(robot.asked), ex.TURN_ATTEMPTS)


class YawRateGuardTests(unittest.TestCase):
    def test_a_single_spike_is_not_a_runaway(self):
        self.assertGreater(fetch.YAW_RATE_PERSIST, 1)

    def test_the_guard_still_trips_quickly(self):
        # It must stay a safety guard: a runaway has to be caught inside a
        # fraction of a turn, not after one.
        samples_per_second = 30.
        seconds = fetch.YAW_RATE_PERSIST / samples_per_second
        self.assertLess(seconds * fetch.YAW_RATE_LIMIT, 45.,
                        'guard would allow more than 45 degrees of runaway')


class DriveDutyTests(unittest.TestCase):
    def test_unknown_voltage_keeps_the_measured_carpet_duty(self):
        self.assertEqual(ex.drive_duty(50), .20)
        self.assertEqual(ex.drive_duty(50, None), .20)
        self.assertEqual(ex.drive_duty(50, 0), .20)

    def test_a_sagging_pack_gets_more_duty(self):
        full = ex.drive_duty(50, ex.DRIVE_NOMINAL_PACK_V)
        low = ex.drive_duty(50, 10.9)
        self.assertGreater(low, full)
        self.assertAlmostEqual(low, .20 * ex.DRIVE_NOMINAL_PACK_V / 10.9, places=6)

    def test_a_full_pack_is_never_driven_below_the_stiction_wall(self):
        self.assertEqual(ex.drive_duty(50, 13.5), .20)

    def test_the_ceiling_leaves_room_for_the_steering_trim(self):
        # The service rejects any |speed| over 0.3 outright, and the trim adds
        # up to 0.025 to one wheel. A breach there is not a slow leg, it is an
        # invalid command that invalidates the whole control generation.
        worst = ex.drive_duty(50, 1.0) + .025
        self.assertLessEqual(worst, .3)
        self.assertEqual(ex.drive_duty(50, 1.0), ex.DRIVE_DUTY_CEILING)


class YawRateLoopTests(unittest.TestCase):
    """The persistence counter must actually reset, not merely exist.

    Testing the constants alone would still pass if the `else: over = 0` branch
    were dropped, and a counter that never resets turns the guard back into a
    one-sample trip the moment two spikes land anywhere in the same turn.
    """

    def sweeper(self, rates):
        robot = fetch.Robot.__new__(fetch.Robot)
        robot.dry_run = False
        robot.coast_s = fetch.COAST_SECONDS
        robot.turn_gain = fetch.TURN_GAIN_DEG_S_PER_DUTY
        robot.duty = fetch.TURN_MIN_DUTY
        robot.hold = lambda left, right: None
        robot.note_motion = lambda *a: None
        robot.floor_duty = lambda d: d
        heading = [0.]
        supply = list(rates)

        def check():
            heading[0] += 5.          # keep swept advancing past the stall guard
            return {'imu': {'euler': [heading[0], 0., 0.]}}

        robot.check = check
        robot.yaw_rate = lambda imu: supply.pop(0) if supply else 10.
        return robot

    def test_isolated_spikes_do_not_stop_the_sweep(self):
        # Alternating: never YAW_RATE_PERSIST in a row, so never a runaway.
        spikes = [fetch.YAW_RATE_LIMIT + 55., 10.] * 12
        robot = self.sweeper(spikes)
        robot._sweep(1., 100., 30., continuous=True)      # must not raise

    def test_sustained_over_rate_still_stops_the_sweep(self):
        robot = self.sweeper([fetch.YAW_RATE_LIMIT + 55.] * 10)
        with self.assertRaises(fetch.Stop) as caught:
            robot._sweep(1., 100., 30., continuous=True)
        self.assertIn('yaw rate', str(caught.exception))

    def test_it_stops_on_exactly_the_persistence_threshold(self):
        # One short of the threshold, then a good sample, then a full run of
        # them: the guard must wait for the run, not remember the earlier ones.
        rates = ([fetch.YAW_RATE_LIMIT + 55.] * (fetch.YAW_RATE_PERSIST - 1)
                 + [10.]
                 + [fetch.YAW_RATE_LIMIT + 55.] * fetch.YAW_RATE_PERSIST)
        robot = self.sweeper(rates)
        with self.assertRaises(fetch.Stop):
            robot._sweep(1., 500., 30., continuous=True)


class DutyPlumbingTests(unittest.TestCase):
    """The compensation is only real if the measured voltage reaches it.

    drive_duty() being correct in isolation proves nothing if check() never
    captures the pack or the drive loop never passes it.
    """

    def job(self, status):
        job = ex.Execution.__new__(ex.Execution)
        job.bench = type('B', (), {'abort': type('A', (), {
            'is_set': staticmethod(lambda: False)})()})()
        job.robot = type('R', (), {'check': staticmethod(lambda: status)})()
        job.started = time.time()
        job.render_error = None
        job.deadline = None
        job.pack_v = None
        return job

    def test_check_captures_the_pack_voltage(self):
        job = self.job({'power': {'pack_voltage_v': 10.9}})
        job.check()
        self.assertAlmostEqual(job.pack_v, 10.9)
        self.assertGreater(ex.drive_duty(30., job.pack_v), ex.DRIVE_MIN_DUTY)

    def test_a_status_without_power_leaves_the_duty_at_the_measured_floor(self):
        for status in ({}, None, {'power': None}):
            job = self.job(status)
            job.check()
            self.assertIsNone(job.pack_v)
            self.assertEqual(ex.drive_duty(30., job.pack_v), ex.DRIVE_MIN_DUTY)


class Odometer:
    def __init__(self, step_cm):
        self.tracker = self
        self.step = step_cm

    def motion(self, before, after):
        # Identity rotation, translation that reads as `step` forward.
        return np.eye(2), np.array([0., -self.step]), {}


class TurningRobot:
    """Turns slowly enough to outlast the no-progress budget."""

    def __init__(self, clock, turn_seconds):
        self.clock = clock
        self.turn_seconds = turn_seconds
        self.turns = []

    def check(self):
        return {}

    def hold(self, left, right):
        pass

    def frame(self):
        return np.zeros((4, 4, 3), np.uint8), {'imu_samples': [{'yaw': 0.}]}

    def heading(self, sample):
        return 0.

    def turn_continuous(self, degrees):
        self.turns.append(degrees)
        self.clock.advance(self.turn_seconds)
        return degrees

    turn = turn_continuous


class Clock:
    def __init__(self):
        self.now = 1000.

    def advance(self, seconds):
        self.now += seconds

    def __call__(self):
        self.now += .01           # ordinary loop iterations cost a little
        return self.now


class ProgressTimerTests(unittest.TestCase):
    """Aiming must not be charged to the no-progress timer.

    A pivot barely changes range to the waypoint -- only by the 7.28 cm pivot
    offset -- so a turn that takes longer than the 5 s budget used to make the
    robot give up for having done exactly what the bearing asked. The turn
    retry added in this change makes long turns more likely, so the two have to
    be correct together.
    """

    def execution(self, clock, turn_seconds):
        job = ex.Execution.__new__(ex.Execution)
        robot = TurningRobot(clock, turn_seconds)
        job.bench = type('B', (), {
            'abort': type('A', (), {'is_set': staticmethod(lambda: False)})(),
            'odometer': Odometer(6.),
            'say': staticmethod(lambda line: None)})()
        job.robot = robot
        job.command = dict(waypoints_cm=[[0., 24.]], initial_turn_deg=0.)
        job.route = [[0., 24.]]
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
        return job, robot

    def drive(self, turn_seconds, bearing_first):
        clock = Clock()
        job, robot = self.execution(clock, turn_seconds)
        if bearing_first:
            # Start facing 90 degrees off, so the first loop iteration turns.
            job.pose = [0., 0., 90.]
        import tempfile
        from pathlib import Path
        # Pin the pivot shift to zero. A real turn slides the camera 8-12 cm,
        # which changes range to the waypoint by more than the 0.5 cm the
        # progress guard counts as improvement -- so the timer gets reset as a
        # side effect and the test passes whether or not the fix is present.
        # Verified by mutation: with the shift left in, removing the reset did
        # not fail this test.
        with unittest.mock.patch.object(ex.fetch, 'pivot_shift',
                                        lambda degrees: (0., 0.)):
            with unittest.mock.patch.object(ex.time, 'monotonic', clock):
                with tempfile.TemporaryDirectory() as room:
                    job.root = Path(room)
                    job.run()
        return job, robot

    def test_a_slow_turn_does_not_count_as_no_progress(self):
        job, robot = self.drive(turn_seconds=8., bearing_first=True)
        self.assertTrue(robot.turns, 'the case never turned at all')
        self.assertNotEqual(job.error, 'no progress toward next waypoint')
        self.assertEqual(job.phase, 'completed')

    def test_a_genuinely_stuck_drive_still_gives_up(self):
        # No turn to excuse it: the odometer reports nothing moving, so the
        # guard must still fire rather than waiting for the 25 s timeout.
        clock = Clock()
        job, robot = self.execution(clock, 0.)
        job.bench.odometer = Odometer(0.)
        import tempfile
        from pathlib import Path
        with unittest.mock.patch.object(ex.time, 'monotonic', clock):
            with tempfile.TemporaryDirectory() as room:
                job.root = Path(room)
                job.run()
        self.assertEqual(job.phase, 'failed')
        self.assertIn('no progress', job.error)


if __name__ == '__main__':
    unittest.main()
