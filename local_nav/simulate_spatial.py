"""Exercise production spatial planning, turns, drives and VIO without hardware.

Carpet rendering does not render obstacles: only static map checking is tested.
IMU attitudes are simulated directly; this does not validate physical IMU fusion.
"""
import json
import math
import os
import tempfile
from unittest.mock import patch
import cv2

from simulate_route import Plant
from point_controller import ROOT, FloorTracker
from route_executor import load_json
from spatial_executor import execute_navigation


def run_spatial_case(parameters, goal=(30,30), obstacles=None, cancel_between=False, goals=None, free=None,
                     measured_map_drive=False, initial_pose=None):
    plant = Plant(**parameters)
    if initial_pose is not None:
        plant.x,plant.z=initial_pose[:2]
        plant.yaw=math.radians(initial_pose[2])
    violations=[]
    def audit(x,z,yaw):
        from spatial_planner import bounds,transform
        from route_geometry import contains,overlap
        body=bounds([transform((x,z,math.degrees(yaw)),u,v) for u in (-6,6) for v in (-15,0)],5.)
        if not contains(free or [-120,-120,120,160],body) or any(overlap(body,b) for b in (obstacles or [])):
            if not violations:
                violations.append(dict(time=plant.time,pose=[x,z,math.degrees(yaw)],body=body))
    plant.pose_audit=audit
    profile = load_json(os.path.join(ROOT,'calibration/floor_geometry.json'))
    tracker = FloorTracker(profile,load_json(profile['intrinsics_path']),max_features=125)
    original = tracker.motion
    initializations = []
    cancelled = [False]

    def motion(*args):
        try:
            return original(*args)
        finally:
            plant.advance(plant.compute)

    def call(action,**fields):
        if cancel_between and action=='status' and plant.token['control_epoch']>0 and not cancelled[0]:
            plant.token['control_epoch'] += 1
            cancelled[0] = True
        return plant.call(action,**fields)

    class Timeline:
        def __init__(self,mount):
            self.up = plant.initial_up.copy()
            initializations.append(plant.time)

    with tempfile.TemporaryDirectory(prefix='jetbot-spatial-sim-') as directory:
        path = os.path.join(directory,'anchor.png')
        cv2.imwrite(path,plant.render())
        plan = dict(goal_cm=list(goal),inspected_free_rectangle_cm=free or [-120,-120,120,160],
                    measured_map_drive=measured_map_drive,
                    obstacle_rectangles_cm=obstacles or [],image_path=path,
                    captured_monotonic=plant.time,**plant.token)
        if goals is not None:
            plan['goals_cm'] = goals
        if initial_pose is not None:
            plan['initial_pose_cm_degrees']=initial_pose
        with patch('point_controller.call',side_effect=call), \
                patch('point_controller.frame',side_effect=plant.frame), \
                patch('point_controller.FloorTracker',return_value=tracker), \
                patch.object(tracker,'motion',side_effect=motion), \
                patch('state_estimator.AttitudeTimeline',Timeline), \
                patch('route_executor.time.monotonic',side_effect=lambda:plant.time), \
                patch('route_executor.time.sleep',side_effect=plant.advance):
            result = execute_navigation(plan,os.path.join(directory,'result.json'))
        for action in result['actions']:
            with open(action.pop('result_path')) as source:
                action['result'] = json.load(source)
        plant.advance(1.)
    final_goal=goals[-1] if goals else goal
    return dict(parameters=parameters,goal_cm=goal,goals_cm=goals,result=result,
                true_final_pose=[plant.x,plant.z,math.degrees(plant.yaw)],
                true_goal_error_cm=math.hypot(plant.x-final_goal[0],plant.z-final_goal[1]),
                motor_commands=plant.commands,timeline_initializations=initializations,
                final_motor_output=[plant.left,plant.right],watchdog_stops=plant.watchdog_stops)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    case = run_spatial_case(dict(speed=12,coast=.05,seed=3,pivot_cm=7.5))
    with open(args.output,'w') as out:
        json.dump(case,out,indent=2)
    print(json.dumps({k:v for k,v in case.items() if k not in ('result','motor_commands')}))
    print(json.dumps({k:v for k,v in case['result'].items() if k!='actions'}))
