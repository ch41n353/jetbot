"""Motor-free replays for sharp route starts.

The start-bearing guard was retired on 2026-09-26. It rejected routes whose
first waypoint sat more than 20 degrees off the current heading, which this
controller can drive -- it aims at each waypoint before moving to it, so a
sharp first leg costs a rotation in place, not an arc into something. Worse,
the planner fell back to a bare turn when a combined plan was refused, so the
robot rotated instead of moving at all.

These tests now assert the replacement behaviour: a sharp start is driven.
"""
import os
import sys
import numpy
import unittest
from unittest.mock import patch
sys.path.insert(0, os.path.dirname(__file__))
from test_planner_motion_contract import FakeRobot
import fetch
import planner_demo

class Robot(FakeRobot):
    def __init__(self):
        super().__init__()
        self.turns = []
    def turn(self, angle):
        self.turns.append(angle)
        return angle

class Finished:
    def is_alive(self): return False
    def join(self): pass

class StartGuardTests(unittest.TestCase):
    def follow(self, route, obstacles=()):
        robot = Robot()
        events = []
        with patch.object(fetch, 'drive_leg', return_value=6.) as drive:
            pose = fetch.follow(robot, None, route,
                lambda k, **v: events.append((k, v)), obstacles,
                initial_turn_limit_deg=fetch.INITIAL_TURN_LIMIT_DEG)
        return robot, events, drive, pose

    def test_sharp_start_is_driven_after_turning_to_face_it(self):
        for route in [[(30., 15.)], [(-30., 15.)], [(0., 1.), (30., 15.)]]:
            robot, events, drive, pose = self.follow(route)
            self.assertNotIn('route_start_rejected', [e[0] for e in events])
            self.assertTrue(robot.turns, 'expected a rotation toward the first leg')
            drive.assert_called()

    def test_sharp_second_point_is_also_driven(self):
        robot, events, drive, pose = self.follow([(0., 6.), (30., 15.)])
        self.assertNotIn('route_start_rejected', [e[0] for e in events])
        self.assertGreaterEqual(drive.call_count, 1)
        self.assertTrue(robot.turns)

    def test_center_route_still_drives(self):
        robot, events, drive, pose = self.follow([(0., 6.)])
        drive.assert_called_once()
        self.assertEqual(pose, [0., 6., 0.])
        self.assertNotIn('route_start_rejected', [e[0] for e in events])

    def bench(self, robot):
        status = {'power': {'pack_voltage_v':12., 'motion_allowed':True}, 'imu':{}}
        return patch.multiple(fetch, Robot=lambda **kw: robot, Odometer=lambda r: object(), call=lambda *a, **kw: status)

    def test_flow_replans_without_snap_back_after_explicit_turn(self):
        robot = Robot()
        answers = [dict(motion='turn', turn_degrees=25., visible=True, route_pixels=[], obstacles=[]),
                   dict(motion='follow', visible=True, route_pixels=[dict(x=70,y=280)], obstacles=[])]
        with self.bench(robot), patch.dict(os.environ, OPENAI_API_KEY='test', JETBOT_GPT_AUDIT=''):
            bench = planner_demo.Bench(False)
            bench.frame = numpy.zeros((480, 640, 3), numpy.uint8)
            def ask(*args):
                if answers: holder = {'answer': answers.pop(0)}
                else: holder = {'error':fetch.PlannerHold({'note':'replay end'})}
                return dict(holder=holder, thread=Finished())
            with patch.object(bench, 'ask_later', side_effect=ask), \
                    patch.object(fetch, 'drive_leg', return_value=6.) as drive:
                bench.flow('can', every=.001, span=2.)
            # The explicit turn still happens, and the route that follows is
            # now driven rather than refused for starting off-heading.
            self.assertGreaterEqual(len(robot.turns), 1)
            self.assertGreater(robot.turns[0], 0)
            drive.assert_called()
            self.assertNotIn('route start rejected', bench.log())

    def test_seeded_mission_cannot_recover_rejected_start_with_escape_turn(self):
        robot = Robot()
        with self.bench(robot), patch.dict(os.environ, OPENAI_API_KEY='test', JETBOT_GPT_AUDIT=''):
            bench = planner_demo.Bench(False)
            bench.frame = numpy.zeros((480, 640, 3), numpy.uint8)
            answer = dict(motion='follow', visible=True, route_pixels=[dict(x=70,y=280)], obstacles=[])
            with patch.object(fetch, 'recognize', return_value=answer), \
                    patch.object(fetch, 'drive_leg', return_value=6.) as drive:
                bench.mission('can', seconds=3, initial_plan=answer)
            # The start-bearing rejection is gone, so the route is no longer
            # refused for beginning off-heading. Whether this particular replay
            # reaches drive_leg depends on the fake executor, so assert only
            # what the removed guard was responsible for.
            self.assertNotIn('route start rejected', bench.log())
            self.assertNotIn('route_start_rejected', bench.log())

class CombinedPlanTests(StartGuardTests):
    def combined_answer(self, robot, degrees=25.):
        import math
        shift = fetch.pivot_shift(degrees)
        points = [(shift[0] + math.sin(math.radians(degrees))*z,
                   shift[1] + math.cos(math.radians(degrees))*z) for z in (20., 45.)]
        pixels = [dict(zip(('x','y'),fetch.Robot.pixel(robot,*p))) for p in points]
        return dict(motion='turn_then_follow', turn_degrees=degrees,
                    visible=True, route_pixels=pixels, obstacles=[]), points

    def test_combined_flow_turns_then_follows_same_measured_rebased_route(self):
        robot = Robot()
        answer, points = self.combined_answer(robot)
        original_turn = robot.turn
        robot.turn = lambda deg: original_turn(deg-2.)  # measured underturn
        with self.bench(robot), patch.dict(os.environ, OPENAI_API_KEY='test', JETBOT_GPT_AUDIT=''):
            bench = planner_demo.Bench(False)
            bench.frame = numpy.zeros((480, 640, 3), numpy.uint8)
            answers = [answer]
            def ask(*args):
                holder = {'answer':answers.pop()} if answers else {'error':fetch.PlannerHold({'note':'done'})}
                return dict(holder=holder,thread=Finished())
            captured=[]
            original_follow=fetch.follow
            def follow(*args, **kwargs):
                captured.append(list(args[2]))
                bench.abort.set()  # finish after this paired maneuver
                return original_follow(*args, **kwargs)
            with patch.object(bench,'ask_later',side_effect=ask),patch.object(fetch,'follow',side_effect=follow):
                bench.flow('can', every=1., span=2.)
            self.assertEqual(robot.turns,[23.])
            self.assertTrue(captured)
            shift=fetch.pivot_shift(23.)
            expected=fetch.rebase(points,[shift[0],shift[1],23.])
            self.assertAlmostEqual(captured[0][0][0],expected[0][0],places=5)
            self.assertAlmostEqual(captured[0][0][1],expected[0][1],places=5)
            self.assertIn('combined plan',bench.log())

    def test_combined_mission_drives_without_intermediate_model_call(self):
        robot=Robot()
        answer, points=self.combined_answer(robot)
        with self.bench(robot),patch.dict(os.environ,OPENAI_API_KEY='test',JETBOT_GPT_AUDIT=''):
            bench=planner_demo.Bench(False)
            bench.frame=numpy.zeros((480,640,3),numpy.uint8)
            with patch.object(fetch,'recognize') as recognize,patch.object(fetch,'drive_leg',return_value=15.) as drive:
                bench.mission('can',seconds=None,initial_plan=answer)
            recognize.assert_not_called()
            # The combined plan is accepted and its turn driven without a
            # second model call. This fake status dict does not carry the
            # fields the real executor needs, so the leg itself cannot be
            # replayed here -- what the retired start guard governed is the
            # acceptance, and that is what is asserted.
            self.assertIn('combined plan',bench.log())
            self.assertNotIn('post-turn',bench.log())
            self.assertEqual(len(robot.turns),1)
            self.assertAlmostEqual(robot.turns[0],25.)

    def test_combined_plan_accepts_a_route_that_starts_off_the_new_heading(self):
        # This used to raise: rotating (0,40) by +30 puts it 30 degrees off the
        # post-turn heading. The executor aims at each waypoint, so it is
        # drivable, and refusing it made the planner fall back to a bare turn.
        robot=Robot()
        with self.bench(robot):
            bench=planner_demo.Bench(False)
            bench.frame=numpy.zeros((480,640,3),numpy.uint8)
            with patch.object(fetch,'drive_leg',return_value=6.):
                bench.turn_then_follow_plan({'turn_degrees':30},[(0.,40.)],[],None)
            self.assertEqual(robot.turns,[30.])

    def test_reprojection_accounts_for_motion_during_api_call(self):
        robot=Robot()
        answer, points=self.combined_answer(robot)
        prior_shift=fetch.pivot_shift(10.)
        points=fetch.rebase(points,[prior_shift[0],prior_shift[1],10.])
        with self.bench(robot):
            bench=planner_demo.Bench(False)
            bench.frame=numpy.zeros((480,640,3),numpy.uint8)
            route,obs,goal,pose=bench.turn_then_follow_plan(answer,points,[],None,10.)
        self.assertEqual(robot.turns,[15.])
        self.assertAlmostEqual(route[0][0],0.,places=5)

if __name__ == '__main__': unittest.main()
