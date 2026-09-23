"""Motor-free controller and dashboard replays for sharp route starts."""
import os
import sys
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

    def test_sharp_start_stops_before_turn_or_drive(self):
        for route in [[(30., 15.)], [(-30., 15.)], [(0., -20.)], [(0., 1.), (30., 15.)]]:
            robot, events, drive, pose = self.follow(route)
            self.assertFalse(robot.turns)
            drive.assert_not_called()
            self.assertEqual(pose, [0., 0., 0.])
            self.assertIn('route_start_rejected', [e[0] for e in events])

    def test_short_center_point_does_not_license_sharp_second_point(self):
        robot, events, drive, pose = self.follow([(0., 6.), (30., 15.)])
        self.assertEqual(drive.call_count, 1)
        self.assertFalse(robot.turns)
        self.assertIn('route_start_rejected', [e[0] for e in events])

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
            def ask(*args):
                if answers: holder = {'answer': answers.pop(0)}
                else: holder = {'error':fetch.PlannerHold({'note':'replay end'})}
                return dict(holder=holder, thread=Finished())
            with patch.object(bench, 'ask_later', side_effect=ask), patch.object(fetch, 'drive_leg') as drive:
                bench.flow('can', every=.001, span=2.)
            self.assertEqual(len(robot.turns), 1)
            self.assertGreater(robot.turns[0], 0)
            drive.assert_not_called()
            self.assertIn('route start rejected', bench.log())
            self.assertIn('PLANNER HOLD', bench.log())

    def test_seeded_mission_cannot_recover_rejected_start_with_escape_turn(self):
        robot = Robot()
        with self.bench(robot), patch.dict(os.environ, OPENAI_API_KEY='test', JETBOT_GPT_AUDIT=''):
            bench = planner_demo.Bench(False)
            answer = dict(motion='follow', visible=True, route_pixels=[dict(x=70,y=280)], obstacles=[])
            with patch.object(fetch, 'recognize', return_value=answer), patch.object(fetch, 'drive_leg') as drive:
                bench.mission('can', seconds=3, initial_plan=answer)
            self.assertFalse(robot.turns)
            drive.assert_not_called()
            self.assertIn('PLANNER HOLD', bench.log())
            self.assertNotIn('mission complete', bench.log())

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
            with patch.object(fetch,'recognize') as recognize,patch.object(fetch,'drive_leg',return_value=15.) as drive:
                bench.mission('can',seconds=None,initial_plan=answer)
            recognize.assert_not_called()
            self.assertTrue(drive.called)
            self.assertEqual(len(robot.turns),1)
            self.assertAlmostEqual(robot.turns[0],25.)

    def test_combined_plan_cannot_turn_back_toward_old_heading(self):
        robot=Robot()
        with self.bench(robot):
            bench=planner_demo.Bench(False)
            with self.assertRaisesRegex(fetch.Stop,'post-turn'):
                bench.turn_then_follow_plan({'turn_degrees':30},[(0.,40.)],[],None)
            self.assertFalse(robot.turns)

    def test_reprojection_accounts_for_motion_during_api_call(self):
        robot=Robot()
        answer, points=self.combined_answer(robot)
        prior_shift=fetch.pivot_shift(10.)
        points=fetch.rebase(points,[prior_shift[0],prior_shift[1],10.])
        with self.bench(robot):
            bench=planner_demo.Bench(False)
            route,obs,goal,pose=bench.turn_then_follow_plan(answer,points,[],None,10.)
        self.assertEqual(robot.turns,[15.])
        self.assertAlmostEqual(route[0][0],0.,places=5)

if __name__ == '__main__': unittest.main()
