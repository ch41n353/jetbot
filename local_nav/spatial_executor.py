#!/usr/bin/env python3
"""Execute a target in an inspected static map with local measured replanning.

Offline search is the default. Experimental powered mode retains primitive
limits, cancellation generations, pose uncertainty and fresh-image anchors.
No semantic recognition or new-obstacle detector is implemented here.
"""
import argparse
import json
import math
import os
import time

from spatial_planner import SpatialPlanner, sweep, transform, normalize, predict, bounds
from route_geometry import StraightRoute, finite, contains


def execute_navigation(plan, log):
    from point_controller import call, ROOT
    from route_executor import execute, load_json
    from spatial_turn import execute_turn
    from state_estimator import AttitudeTimeline
    result = dict(outcome='stopped', actions=[], local_planning_seconds=0.,
                  intermediate_model_calls=0, requires_planner=True)
    started = time.monotonic()
    try:
        goals = plan.get('goals_cm', [plan.get('goal_cm')])
        if not isinstance(goals,list) or not 1<=len(goals)<=4:
            raise ValueError('Expected one to four local targets')
        planners = [SpatialPlanner(dict(plan,goal_cm=goal)) for goal in goals]
        goal_index = 0
        planner = planners[0]
        result['reached_targets'] = []
        timeline = AttitudeTimeline(load_json(os.path.join(ROOT,'calibration/imu_mount.json')))
        capture = {k:plan[k] for k in ('session_id','control_epoch','image_path','captured_monotonic')}
        pose = tuple(finite(v) for v in plan.get('initial_pose_cm_degrees',[0.,0.,0.]))
        if len(pose)!=3:
            raise ValueError('Initial map pose must be x,z,yaw')
        position_variance=finite(plan.get('initial_position_variance_cm2',0.))
        yaw_variance=finite(plan.get('initial_yaw_variance_deg2',0.))
        if not 0<=position_variance<=4 or not 0<=yaw_variance<=16:
            raise ValueError('Initial map uncertainty exceeds limits')
        travel = 0.
        pending = []
        result['local_searches'] = 0
        for index in range(13):
            if time.monotonic()-started > 60:
                raise RuntimeError('Navigation deadline exceeded')
            status = call('status')
            if any(status[k]!=capture[k] for k in ('session_id','control_epoch')):
                raise RuntimeError('Navigation cancelled between actions')
            if not status['healthy'] or not status['motion_enabled'] or status['motor']['output'] != [0,0]:
                raise RuntimeError('Navigation sensors unhealthy or motors not stopped')
            while planner.arrived(pose):
                result['reached_targets'].append(dict(goal_cm=list(planner.goal),measured_pose=list(pose),
                                                     elapsed_seconds=time.monotonic()-started))
                goal_index += 1
                if goal_index == len(planners):
                    break
                planner = planners[goal_index]
                pending = []
            if goal_index == len(planners):
                result.update(outcome='target_reached_estimate',requires_planner=False,final_pose=pose)
                break
            # Keep a useful route suffix to avoid search quantization causing
            # forward/reverse oscillation. Recheck ALL its sweeps from measured
            # pose; discard it on conflict or material endpoint drift.
            projected = pose
            valid = bool(pending)
            for candidate in pending:
                primitive = (candidate['kind'],candidate['value'])
                if not planner.clear(planner.envelope(projected,primitive)):
                    valid = False
                    break
                projected = predict(projected,primitive)
            if math.hypot(projected[0]-planner.goal[0],projected[1]-planner.goal[1]) > planner.tolerance+2:
                valid = False
            if not valid:
                route = planner.search(pose)
                result['local_searches'] += 1
                result['local_planning_seconds'] += route.get('elapsed_seconds',0.)
                if route['outcome'] != 'route_found':
                    raise RuntimeError('Local planning: '+route['outcome'])
                pending = route['actions']
            if index == 12:
                raise RuntimeError('Navigation action budget exceeded')
            action = dict(pending.pop(0))
            primitive = (action['kind'],action['value'])
            action.update(predicted_start_pose=list(pose),predicted_end_pose=list(predict(pose,primitive)),
                          swept_bounds_cm=planner.envelope(pose,primitive))
            local = dict(capture)
            action_log = log+'.action-%02d.json' % (index+1)
            if action['kind'] == 'drive':
                distance = action['value']
                local.update(waypoints_cm=[abs(distance)],
                             travel_direction='forward' if distance>0 else 'reverse',
                             inspected_free_rectangle_cm=sweep((0,0,0),('drive',distance)),
                             obstacle_rectangles_cm=[])
                # Local free rectangle is backed by the world-map sweep proof.
                if planner.measured_map_drive:
                    from mapped_drive import MappedDriveRoute
                    drive_route=MappedDriveRoute(planner,pose,distance)
                else:
                    drive_route=StraightRoute(local)
                step = execute(local,drive_route,action_log,predictive_braking=True,
                               feature_budget=125,attitude_timeline=timeline)
                expected_stops = 2
                accepted = ('goal_reached','overshot_goal','stopped_short')
            else:
                def world_guard(x,z,heading):
                    gx,gz=transform(pose,x,z)
                    body=bounds([transform((gx,gz,pose[2]+heading),u,v)
                                 for u in (-6.,6.) for v in (-15.,0.)],7.)
                    if not contains(action['swept_bounds_cm'],body) or not planner.clear(body):
                        raise RuntimeError('Measured turn chassis left the checked world-map sweep')
                step = execute_turn(local,action['value'],sweep((0,0,0),('turn',action['value'])),
                                    action_log,timeline,world_guard=world_guard)
                expected_stops = 1
                accepted = ('turn_reached_estimate',)
            result['actions'].append(dict(plan=action,result_path=action_log,outcome=step['outcome'],
                                          reason=step.get('reason'),target_index=goal_index))
            if step['outcome'] not in accepted or 'stop_error' in step or 'final_anchor' not in step:
                raise RuntimeError('Action failed or final pose unverified: '+step.get('reason',step['outcome']))
            final = step['settling_samples'][-1]
            position_variance += final['position_sigma_cm']**2
            yaw_variance += final['yaw_sigma_degrees']**2
            if position_variance>4 or yaw_variance>16:
                raise RuntimeError('Accumulated navigation uncertainty exceeded limits')
            dx,dz = step['final_position_cm']
            x,z = transform(pose,dx,dz)
            pose = (x,z,normalize(pose[2]+final['yaw_degrees']))
            travel += math.hypot(dx,dz)
            result.update(last_verified_pose=pose,travel_cm=travel,
                          position_variance_cm2=position_variance,yaw_variance_deg2=yaw_variance,
                          goal_error_cm=math.hypot(x-planner.goal[0],z-planner.goal[1]))
            if travel>90:
                raise RuntimeError('Navigation travel budget exceeded')
            # Never adopt a newer epoch supplied by the service: extra stops cancel.
            capture = dict(step['final_anchor'],session_id=capture['session_id'],
                           control_epoch=capture['control_epoch']+expected_stops)
        else:
            raise RuntimeError('Navigation action budget exceeded')
    except Exception as exc:
        result.update(outcome='stopped',reason=str(exc),requires_planner=True)
    finally:
        try:
            call('stop')
        except Exception as exc:
            result.update(outcome='stopped',stop_error=str(exc),requires_planner=True)
        result['elapsed_seconds'] = time.monotonic()-started
        with open(log,'w') as output:
            json.dump(result,output,indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--log')
    args = parser.parse_args()
    with open(args.plan) as source:
        plan = json.load(source)
    if args.execute:
        if not args.log:
            parser.error('--execute requires --log')
        result = execute_navigation(plan,args.log)
    else:
        result = SpatialPlanner(plan).search()
    print(json.dumps(result,indent=2))
    return 0 if result['outcome'] in ('route_found','target_reached_estimate') else 1


if __name__ == '__main__':
    raise SystemExit(main())
