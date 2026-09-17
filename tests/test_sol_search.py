import os
import json
import sys
import unittest
import tempfile
import time
import base64
from unittest.mock import patch
import cv2
import numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','local_nav'))
from sol_search import SearchState,local_map,perimeter_stations
import json as _json
with open(os.path.join(os.path.dirname(__file__),"..","calibration","turn_pivot.json")) as _p:
    _pivot=_json.load(_p)
PIVOT_CM=(_pivot["pivot_x_cm"],_pivot["pivot_z_cm"])
from spatial_planner import transform,bounds
from route_geometry import contains
from maneuver_session import route_command


class RetainedMapTests(unittest.TestCase):
    """The seed rectangle is the floor the camera cannot see, not the map."""
    def plan(self):
        return dict(inspected_free_rectangle_cm=[-35,-35,35,45],
                    obstacle_rectangles_cm=[],search_stations_cm=[])

    def test_seed_alone_is_the_map_until_something_is_certified(self):
        state=SearchState(self.plan())
        self.assertEqual(state.mapped()['inspected_free_rectangles_cm'],[[-35,-35,35,45]])

    def test_certified_floor_extends_free_space_beyond_the_seed(self):
        state=SearchState(self.plan())
        state.observed_free=[[-20.,45.,20.,90.]]
        free=state.mapped()['inspected_free_rectangles_cm']
        self.assertGreater(max(r[3] for r in free),45.)

    def test_union_supports_a_corridor_the_seed_alone_refuses(self):
        # A corridor reaching 70 cm ahead leaves the 45 cm seed, so it is
        # refused until floor further out has actually been certified.
        shallow=dict(self.plan(),inspected_free_rectangle_cm=[-35,-35,35,15])
        with self.assertRaisesRegex(ValueError,'No inspected local approach corridor'):
            local_map(shallow,[0.,0.,0.])
        # Certifying the floor beyond the seed makes the same corridor available,
        # and it must be proven against the union, not any single rectangle.
        union=[[-35.,-35.,35.,15.],[-35.,15.,35.,90.]]
        corridor=local_map(shallow,[0.,0.,0.],union)
        self.assertGreaterEqual(corridor['inspected_free_rectangle_cm'][3],20.)

    def test_certification_failure_adds_nothing_rather_than_guessing(self):
        state=SearchState(self.plan())
        class Broken(object):
            def ground(self,*a,**k): raise RuntimeError('calibration missing')
        self.assertEqual(state.certify(Broken()),0)
        self.assertEqual(state.observed_free,[])
        self.assertEqual(state.mapped()['inspected_free_rectangles_cm'],[[-35,-35,35,45]])

    def test_recognized_obstacles_are_advisory_unless_asked_for(self):
        """One protection is off by default; the rest of the map still applies."""
        class Flat(object):
            """Fake floor projection: pixels map to a patch well clear of the robot."""
            def ground(self,points,attitude=None):
                pixels=np.asarray(points,dtype=float).reshape(-1,2)
                return np.column_stack((pixels[:,0]/10.-32.,pixels[:,1]/10.))
        answer={'obstacles':[{'label':'box','box':{'x0':300,'y0':300,'x1':360,'y1':400}}]}
        state=SearchState(self.plan())
        self.assertEqual(state.retain_obstacles(Flat(),answer),0)
        self.assertEqual(state.observed_obstacles,[])
        # The seed map and everything certified from vision still constrain it.
        self.assertEqual(state.mapped()['inspected_free_rectangles_cm'],[[-35,-35,35,45]])
        opted=SearchState(dict(self.plan(),retain_recognized_obstacles=True))
        self.assertEqual(opted.retain_obstacles(Flat(),answer),1)
        self.assertEqual(len(opted.observed_obstacles),1)

    def test_unprojectable_obstacle_boxes_are_dropped_not_guessed(self):
        state=SearchState(self.plan())
        class Broken(object):
            def ground(self,*a,**k): raise ValueError('beyond trusted range')
        added=state.retain_obstacles(Broken(),{'obstacles':[
            {'label':'far thing','box':{'x0':1,'y0':1,'x1':2,'y1':2}}]})
        self.assertEqual(added,0)
        self.assertEqual(state.observed_obstacles,[])


class SearchTests(unittest.TestCase):
    def plan(self):
        return dict(inspected_free_rectangle_cm=[-150,-150,150,150],
                    obstacle_rectangles_cm=[],search_stations_cm=[[30,0]])

    def test_absent_searches_every_heading_without_looping(self):
        state=SearchState(self.plan())
        for heading in range(0,360,30):
            state.pose=[0,0,heading];state.observed()
            state.rotation['start']=heading
            candidates=state.candidates()
            if heading<330:
                choice=state.select(dict(target_visible=False,mission_action='search',candidate_id='scan_right'),candidates)
                self.assertEqual(choice['degrees'],30.)
        self.assertEqual(len(state.visited['start']),12)
        self.assertEqual(state.candidates()[0]['kind'],'turn')
        state.rotation['start']=360
        self.assertTrue(all(c['kind']=='relocate' for c in state.candidates()))

    def test_no_candidates_does_not_claim_object_absent_or_arrival(self):
        plan=self.plan();plan['search_stations_cm']=[]
        state=SearchState(plan);state.visited['start']=list(range(0,360,30))
        state.rotation['start']=360
        self.assertEqual(state.candidates(),[])
        self.assertEqual(state.select(dict(target_visible=False),[])['kind'],'hold')

    def test_nearby_obstacle_blocks_search_turn(self):
        plan=self.plan();plan['search_stations_cm']=[]
        plan['obstacle_rectangles_cm']=[[8,-10,15,0]]
        state=SearchState(plan);state.observed()
        self.assertEqual(state.candidates(),[])

    def test_model_cannot_override_search_or_approach_missing_target(self):
        state=SearchState(self.plan());state.observed()
        for action in ('hold','approach','drive_through_wall'):
            choice=state.select(dict(mission_action=action,target_visible=False),state.candidates())
            self.assertEqual(choice['id'],'scan_right')
        with self.assertRaises(ValueError):state.select(dict(target_visible=True),state.candidates())

    def test_perimeter_is_inset_and_clockwise(self):
        points=perimeter_stations([-100,-100,100,100])
        self.assertEqual(points[0],[-65,-65])
        self.assertTrue(points[1][1]>points[0][1])
        self.assertTrue(all(max(abs(x),abs(z))==65 for x,z in points))
        self.assertEqual(len(points),len(set(tuple(p) for p in points)))

    def test_full_scan_required_before_wall_following(self):
        state=SearchState(self.plan());state.observed()
        state.rotation['start']=355
        self.assertEqual(state.candidates()[0]['degrees'],5)
        state.rotation['start']=359
        self.assertEqual(state.candidates()[0]['kind'],'relocate')

    def test_scan_step_can_be_reduced_for_tight_turn_clearance(self):
        plan=self.plan();plan['scan_step_degrees']=5
        state=SearchState(plan)
        self.assertEqual(state.candidates()[0]['degrees'],5)
        state.rotation['start']=357
        self.assertEqual(state.candidates()[0]['degrees'],3)

    def test_scan_step_rejects_unbounded_values(self):
        for value in (0,31):
            plan=self.plan();plan['scan_step_degrees']=value
            with self.assertRaises(ValueError):SearchState(plan)

    def test_scan_direction_can_be_left(self):
        plan=self.plan();plan['scan_step_degrees']=5;plan['scan_direction']=-1
        self.assertEqual(SearchState(plan).candidates()[0]['degrees'],-5)

    def test_visible_target_transitions_to_approach(self):
        state=SearchState(self.plan());state.observed()
        answer=dict(target_visible=True,target_box=dict(x0=300,y0=200,x1=340,y1=240),contact_pixel=dict(x=320,y=240))
        self.assertEqual(state.select(answer,state.candidates())['kind'],'approach')

    def test_rotated_free_map_is_inner_bound_obstacle_is_outer_bound(self):
        plan=self.plan();plan['obstacle_rectangles_cm']=[[35,50,45,60]]
        pose=[10,5,45]
        local=local_map(plan,pose)
        x0,z0,x1,z1=local['inspected_free_rectangle_cm']
        world=bounds([transform(pose,x,z) for x in (x0,x1) for z in (z0,z1)])
        self.assertTrue(contains(plan['inspected_free_rectangle_cm'],world))
        self.assertGreater(local['obstacle_rectangles_cm'][0][2]-local['obstacle_rectangles_cm'][0][0],10)

    def test_recorded_dashboard_advil_geometry_has_previewable_route(self):
        """Regression for the dashboard's 2026-09-15 no_checked_route failure."""
        from spatial_preview import write_preview
        image=np.zeros((480,640,3),np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            image_path=os.path.join(directory,'frame.jpg')
            preview_path=os.path.join(directory,'preview.html')
            cv2.imwrite(image_path,image)
            retained=dict(inspected_free_rectangle_cm=[-35,-35,35,45],
                          obstacle_rectangles_cm=[[-29.4,47.,-15.4,55.],
                                                  [-23.1,50.5,4.7,60.]])
            local=local_map(retained,[0.,0.,0.])
            self.assertEqual(local['inspected_free_rectangle_cm'],[-30,-32,30,45])
            plan=dict(local,image_path=image_path,goals_cm=[[-12.3846,28.1973]],
                      measured_map_drive=True,precise_turn_sweep=True,
                      goal_heading_turns=True,coalesce_drives=True)
            route=write_preview(plan,preview_path)
            self.assertEqual(route['outcome'],'route_found')
            self.assertTrue(route['actions'])
            self.assertTrue(os.path.isfile(preview_path))

    def test_supervisor_dispatch_and_mutual_exclusion(self):
        self.assertTrue(any(v.endswith('sol_search.py') for v in route_command('plan','log',search=True)))
        with self.assertRaises(ValueError):route_command('plan','log',search=True,mission=True)

    def run_harness(self,cancel=False,search_only=False,uncertain=False,found=True,
                    recognition_every_actions=1,max_search_actions=72,
                    scan_turn_controller='pulse',approach_result=None):
        from sol_search import execute
        with tempfile.TemporaryDirectory() as directory:
            image=np.zeros((480,640,3),np.uint8)
            path=os.path.join(directory,'anchor.jpg');cv2.imwrite(path,image)
            encoded=base64.b64encode(cv2.imencode('.jpg',image)[1]).decode()
            plan=dict(self.plan(),session_id='test',control_epoch=0,image_path=path,
                      captured_monotonic=time.monotonic(),target_label='duct tape',
                      preview_path=os.path.join(directory,'preview.html'),search_only=search_only,
                      recognition_every_actions=recognition_every_actions,
                      max_search_actions=max_search_actions,
                      scan_turn_controller=scan_turn_controller)
            epoch=[0];actions=[]
            def call(command):
                if command=='stop':actions.append('stop');epoch[0]+=1;return {}
                status=dict(session_id='test',control_epoch=epoch[0],healthy=True,
                            motion_enabled=True,motor={'output':[0,0]})
                if command=='observation':status.update(time=time.monotonic(),jpeg_base64=encoded)
                return status
            def turn(*args,**kwargs):
                actions.append('turn');epoch[0]+=1
                return dict(outcome='turn_reached_estimate',final_position_cm=[0,0],
                    settling_samples=[dict(yaw_degrees=30,position_sigma_cm=.1,yaw_sigma_degrees=20. if uncertain else .1)])
            def continuous(*args,**kwargs):
                actions.append('continuous_turn');epoch[0]+=1
                self.assertEqual(args[0]['tolerance_degrees'],3.)
                self.assertEqual(args[0]['brake_margin_degrees'],1.)
                return dict(outcome='turn_reached_imu_estimate',final_angle_degrees=30.)
            calls=[]
            def recognize(*args):
                calls.append(args)
                if cancel:epoch[0]+=1
                return dict(answer=dict(target_visible=len(calls)>1 and found,
                    target_box=dict(x0=300,y0=200,x1=340,y1=240),contact_pixel=dict(x=320,y=240)))
            with patch.dict(os.environ,{'OPENAI_API_KEY':'test-only'}), \
                 patch('point_controller.call',side_effect=call), \
                 patch('point_controller.frame',return_value=(image,time.monotonic(),{})), \
                 patch('point_controller.FloorTracker') as tracker, \
                 patch('state_estimator.AttitudeTimeline'), \
                 patch('async_scene.request_scene',side_effect=recognize), \
                 patch('spatial_turn.execute_turn',side_effect=turn), \
                 patch('continuous_turn.execute',side_effect=continuous), \
                 patch('turn_sweep_preview.write_turn_preview'), \
                 patch('spatial_preview.write_preview'), \
                 patch('object_mission.prepare',return_value=({},{})), \
                 patch('object_mission.execute',return_value=(approach_result or dict(
                       outcome='object_reached_estimate',requires_planner=False))) as approach:
                calls_to_motion=[0]
                def measured_motion(*args,**kwargs):
                    calls_to_motion[0]+=1
                    angle=30 if scan_turn_controller=='continuous' and calls_to_motion[0]==3 else 0
                    a=np.radians(angle)
                    # The robot is differential drive and rotates about its wheel
                    # axis, not about the lens, so a turn does translate the
                    # camera. Zero translation here modelled a pivot the robot
                    # does not have, and the turn envelope is now sized for the
                    # measured one (calibration/turn_pivot.json).
                    px,pz=PIVOT_CM
                    shift=np.array([(1-np.cos(a))*px-np.sin(a)*pz,
                                    np.sin(a)*px+(1-np.cos(a))*pz])
                    rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
                    # PlanarState recovers the camera's displacement as
                    # -rotation.T @ translation, so hand back the inverse of the
                    # displacement the pivot produces, not the displacement.
                    return (rot,-rot.dot(shift),
                            dict(residual_cm=.05,yaw_variance=np.radians(.2)**2))
                tracker.return_value.motion.side_effect=measured_motion
                result=execute(plan,os.path.join(directory,'result.json'))
                with open(result['checkpoint_path']) as source:
                    result['_test_checkpoint']=json.load(source)
                return result,actions,len(calls),approach.call_count

    def test_motor_free_harness_searches_then_approaches_without_astra(self):
        result,actions,calls,approaches=self.run_harness()
        self.assertEqual(result['outcome'],'object_reached_estimate')
        self.assertEqual(actions,['turn','stop'])
        self.assertEqual((calls,approaches),(2,1))

    def test_approach_failure_reason_is_propagated_to_dashboard_result(self):
        result,_,_,_=self.run_harness(approach_result=dict(
            outcome='stopped',requires_planner=True,
            reason='Target recovery exhausted',travel_cm=12.))
        self.assertEqual(result['outcome'],'stopped')
        self.assertEqual(result['reason'],'Target recovery exhausted')
        self.assertEqual(result['travel_cm'],12.)

    def test_cancellation_during_inference_cannot_start_action(self):
        result,actions,calls,approaches=self.run_harness(cancel=True)
        self.assertEqual(result['outcome'],'stopped')
        self.assertIn('cancelled',result['reason'])
        self.assertEqual(actions,['stop'])
        self.assertEqual(approaches,0)

    def test_search_only_recognizes_after_motion_uncertainty_limit(self):
        result,actions,calls,approaches=self.run_harness(search_only=True,uncertain=True)
        self.assertEqual(result['outcome'],'object_found')
        self.assertFalse(result['requires_astra'])
        self.assertFalse(result['found_pose_valid'])
        self.assertIsNone(result['found_pose_cm_degrees'])
        self.assertEqual((calls,approaches),(2,0))
        self.assertEqual(actions,['turn','stop'])
        # Earlier event history must not silently change after later turns.
        self.assertEqual(result['events'][0]['context']['scanned_rotation_degrees']['start'],0)

    def test_uncertain_pose_blocks_approach_after_recognition(self):
        result,actions,calls,approaches=self.run_harness(uncertain=True)
        self.assertEqual(result['outcome'],'localization_required')
        self.assertTrue(result['target_visible'])
        self.assertEqual((calls,approaches),(2,0))
        self.assertEqual(actions,['turn','stop'])

    def test_uncertain_absent_target_does_not_start_more_turns(self):
        result,actions,calls,approaches=self.run_harness(uncertain=True,found=False)
        self.assertEqual(result['outcome'],'localization_required')
        self.assertFalse(result['target_visible'])
        self.assertEqual(calls,2)
        self.assertEqual(actions,['turn','stop'])

    def test_recognition_cadence_skips_semantics_but_keeps_host_policy(self):
        result,actions,calls,approaches=self.run_harness(
            found=False,recognition_every_actions=2,max_search_actions=2)
        self.assertEqual(result['outcome'],'search_budget_exhausted')
        self.assertEqual(result['handoff'],'intervention')
        self.assertEqual((calls,approaches),(2,0))
        self.assertEqual(actions,['turn','turn','stop'])
        self.assertEqual(len([e for e in result['events'] if e['event']=='recognition_skipped']),1)

    def test_checkpoint_is_final_atomic_local_state(self):
        result,actions,_,_=self.run_harness()
        checkpoint=result['_test_checkpoint']
        self.assertEqual(checkpoint['phase'],'finished')
        self.assertEqual(checkpoint['outcome'],'object_reached_estimate')
        self.assertEqual(checkpoint['handoff'],'final')
        self.assertEqual(checkpoint['state']['pose'],[0.,0.,30.])
        self.assertFalse(checkpoint['resume_automatically'])
        self.assertEqual(actions,['turn','stop'])

    def test_scan_defaults_to_one_continuous_motor_owner_with_post_turn_vio(self):
        result,actions,calls,approaches=self.run_harness(scan_turn_controller='continuous')
        self.assertEqual(result['outcome'],'object_reached_estimate')
        self.assertEqual(actions,['continuous_turn','stop'])
        self.assertEqual((calls,approaches),(2,1))
        completed=[e for e in result['events'] if e['event']=='local_action_complete']
        self.assertAlmostEqual(completed[0]['pose'][2],30.)

    def test_checkpoint_restore_continues_deterministic_policy(self):
        original=SearchState(self.plan());original.pose=[1.,2.,30.]
        original.rotation['start']=30.;original.observed();original.travel=3.5
        restored=SearchState(self.plan());restored.restore(original.snapshot())
        candidate=restored.candidates()[0]
        self.assertEqual(candidate['kind'],'turn')
        self.assertEqual(candidate['degrees'],30.)
        self.assertEqual(restored.select(dict(target_visible=False),[candidate]),candidate)


if __name__=='__main__':unittest.main()
