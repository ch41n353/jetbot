"""Experimental local object mission with continuous tracking and motor renewal.

One remote-supplied target identity and inspected static map. No semantic object
recognition or general new-obstacle detector. Default CLI validates, never moves.
"""
import argparse
import json
import math
import os
import time
import cv2
import numpy as np
from target_tracker import TargetTracker,TargetLost
from spatial_planner import SpatialPlanner,transform,bounds,normalize,sweep
from route_geometry import finite,rectangle,swept_pose_bounds


def validate(plan):
    rectangle(plan['inspected_free_rectangle_cm'])
    for box in plan['obstacle_rectangles_cm']:rectangle(box)
    standoff=finite(plan.get('standoff_cm',20))
    if not 18<=standoff<=30:raise ValueError('Mission standoff must be18–30cm')
    image=cv2.imread(plan['image_path'])
    if image is None:raise ValueError('Missing target image')
    TargetTracker(image,plan['target_box'])
    return image,standoff


def drive_clear(planner,pose,left,right):
    # Conservative short lookahead over nominal speed/yaw uncertainty, plus the
    # existing four-cm coast, two-cm pose allowance and five-cm body clearance.
    for speed in (8.,18.):
        rate=70*(left-right)/.28
        for yaw_rate in (rate-10,rate+10):
            x,z,yaw=pose
            for _ in range(6):
                yaw+=yaw_rate*.05
                x+=math.sin(math.radians(yaw))*speed*.05
                z+=math.cos(math.radians(yaw))*speed*.05
                if not planner.clear(swept_pose_bounds(x,z,yaw,1,10000)):
                    return False
    return True


def execute(plan,log):
    from point_controller import call,frame,FloorTracker,ROOT
    from state_estimator import AttitudeTimeline,PlanarState
    from route_executor import load_json
    result=dict(outcome='stopped',requires_planner=True,intermediate_model_calls=0,
                events=[],samples=[],local_recoveries=0,local_searches=0)
    started=time.monotonic()
    def event(name,**fields):
        value=json.loads(json.dumps(dict(event=name,elapsed_seconds=time.monotonic()-started,**fields)))
        result['events'].append(value)
        with open(log+'.events.jsonl','a') as out:out.write(json.dumps(value)+'\n')
    def publish(**fields):
        fields['time']=time.monotonic()
        with open(log+'.status.json.tmp','w') as out:json.dump(fields,out)
        os.replace(log+'.status.json.tmp',log+'.status.json')
    try:
        anchor,standoff=validate(plan)
        status=call('status')
        result['power_start']=status.get('power')
        token={k:plan[k] for k in ('session_id','control_epoch')}
        if any(status[k]!=v for k,v in token.items()):raise RuntimeError('Stale mission generation')
        if not status['healthy'] or not status['motion_enabled'] or status['motor']['output']!=[0,0]:
            raise RuntimeError('Mission requires healthy sensors, power and stopped motors')
        if not 0<=time.monotonic()-plan['captured_monotonic']<=120:raise RuntimeError('Mission image expired')
        footprint=load_json(os.path.join(ROOT,'calibration/robot_footprint.json'))
        if (footprint['width_cm'],footprint['length_cm'],footprint['camera_location'])!=(12,15,'front_center'):
            raise RuntimeError('Footprint differs from mission geometry')
        profile=load_json(os.path.join(ROOT,'calibration/floor_geometry.json'))
        floor=FloorTracker(profile,load_json(profile['intrinsics_path']),max_features=125)
        timeline=AttitudeTimeline(load_json(os.path.join(ROOT,'calibration/imu_mount.json')))
        old,t0,a0=frame(timeline,settled=False)
        rotation,translation,_=floor.motion(anchor,old)
        if np.linalg.norm(translation)>.3 or abs(math.atan2(rotation[1,0],rotation[0,0]))>math.radians(1):
            raise RuntimeError('Robot moved since mission preview')
        target=TargetTracker(anchor,plan['target_box'])
        observation=target.update(old)
        target_world=floor.ground([observation['base_pixel']],a0)[0]
        if not np.isfinite(target_world).all() or not 20<=np.linalg.norm(target_world)<=120:
            raise RuntimeError('Target floor projection outside mission range')
        if 'target_ground_cm' in plan:
            expected=np.asarray(plan['target_ground_cm'],dtype=float)
            if expected.shape!=(2,) or not np.isfinite(expected).all() or np.linalg.norm(expected-target_world)>max(5.,.1*np.linalg.norm(target_world)):
                raise RuntimeError('Measured target projection disagrees with preview map')
        # Identity is fixed; a moving target does not drag the inspected map.
        target_radius=finite(plan.get('target_radius_cm',5.))
        if not 5<=target_radius<=15:raise ValueError('Invalid target radius')
        obstacles=list(plan['obstacle_rectangles_cm'])+[[target_world[0]-target_radius,target_world[1]-3,
                                                       target_world[0]+target_radius,target_world[1]+3]]
        goal=target_world*(1-standoff/np.linalg.norm(target_world))
        planner=SpatialPlanner(dict(plan,goal_cm=goal.tolist(),obstacle_rectangles_cm=obstacles,
                                    measured_map_drive=True,precise_turn_sweep=True,goal_heading_turns=True))
        state=PlanarState()
        up=timeline.up.copy()
        path=[]
        mode='planning'
        last_command=None
        recovery_started=None
        recovery_good=0
        arrival=[]
        travel=0.
        progress=[]
        turn_start=None
        turn_braking=False
        turn_wait=None
        searches=0
        search_refresh=False
        planning_quiet_since=None
        planning_wait_started=None
        clearance_rejections=0
        clearance_wait=None
        next_publish=0.
        event('mission_started',target_label=plan.get('target_label','selected object'))
        while time.monotonic()-started<180:
            current,t1,a1=frame(timeline,settled=False)
            if t1<=t0:continue
            dt=t1-t0
            after_search=search_refresh
            search_refresh=False
            if dt>(.6 if after_search else .18):raise RuntimeError('Mission camera gap exceeded180ms' if not after_search else 'Stopped search refresh exceeded600ms')
            if last_command is not None and time.monotonic()-last_command>.18:
                raise RuntimeError('Mission renewal deadline exceeded')
            floor.excluded_box=target.box
            r,t,q=floor.motion(old,current,a0,a1)
            pose=state.update(r,t,dt,q)
            if after_search and (np.linalg.norm(t)>.1 or abs(math.degrees(math.atan2(r[1,0],r[0,0])))>.5):
                raise RuntimeError('Robot moved during stopped search')
            x,z=pose['position_cm'];yaw=pose['yaw_degrees']
            if abs(math.degrees(math.atan2(r[1,0],r[0,0]))/dt)>90:
                raise RuntimeError('Mission rotation rate exceeded90deg/s')
            travel+=float(np.linalg.norm(t))
            if travel>120:raise RuntimeError('Mission travel limit120cm')
            if timeline.up.dot(up)<math.cos(math.radians(5)):raise RuntimeError('Mission tilt guard')
            body=swept_pose_bounds(x,z,yaw,1,10000)
            if not planner.clear(body):raise RuntimeError('Mission chassis/braking space left inspected map')
            result['last_measured_pose']=[x,z,yaw]
            result['travel_cm']=travel
            if time.monotonic()>=next_publish:
                publish(state=mode,pose_cm_degrees=[x,z,yaw],travel_cm=travel,
                        elapsed_seconds=time.monotonic()-started,remote_planner_required=False)
                next_publish=time.monotonic()+.5
            old,t0,a0=current,t1,a1
            try:
                observation=target.reacquire(current) if mode=='recovering' else target.update(current)
                local_target=floor.ground([observation['base_pixel']],a1)[0]
                observed_world=np.array(transform((x,z,yaw),*local_target))
                if not np.isfinite(observed_world).all() or np.linalg.norm(observed_world-target_world)>max(5.,.1*np.linalg.norm(local_target)):
                    raise TargetLost('Target moved or projection inconsistent with retained map')
            except TargetLost as exc:
                call('motors_hold',left=0.,right=0.,**token)
                last_command=None
                recovery_good=0
                if mode!='recovering':
                    mode='recovering';recovery_started=time.monotonic();recovery_good=0
                    planning_quiet_since=None;planning_wait_started=None
                    result['local_recoveries']+=1
                    event('local_recovery',reason=str(exc))
                if time.monotonic()-recovery_started>2 or result['local_recoveries']>2:
                    raise RuntimeError('Target recovery exhausted: '+str(exc))
                continue
            if mode=='recovering':
                if time.monotonic()-recovery_started>2:
                    raise RuntimeError('Target recovery deadline exceeded')
                recovery_good+=1
                call('motors_hold',left=0.,right=0.,**token)
                if recovery_good<3 or np.linalg.norm(pose['velocity_cm_s'])>.5:continue
                mode='planning';path=[];progress=[]
                turn_start=None;turn_wait=None
                event('target_reacquired')
            distance=float(np.linalg.norm(local_target))
            result['samples'].append(dict(time=t1,mode=mode,target_range_cm=distance,
                                          target_confidence=observation['confidence'],**pose))
            if clearance_wait is not None:
                call('motors_hold',left=0.,right=0.,**token);last_command=None
                if np.linalg.norm(pose['velocity_cm_s'])>.5:
                    if time.monotonic()-clearance_wait>1:raise RuntimeError('Clearance stop not verified')
                    continue
                clearance_wait=None
            within_stopped_tolerance=(clearance_rejections>0 and distance<=standoff+3
                                      and np.linalg.norm(pose['velocity_cm_s'])<=.5)
            if distance<=standoff+1.5 or within_stopped_tolerance or mode=='settling':
                call('motors_hold',left=0.,right=0.,**token);last_command=None
                if mode!='settling':
                    mode='settling';arrival=[];settle_started=time.monotonic()
                    event('arrival_braking')
                arrival.append((t1,x,z,yaw,distance))
                arrival=[v for v in arrival if t1-v[0]<=.4]
                if len(arrival)>=3 and t1-arrival[0][0]>=.18:
                    stable=all(max(v[k] for v in arrival)-min(v[k] for v in arrival)<limit
                               for k,limit in ((1,.1),(2,.1),(3,.5)))
                    if stable:
                        if not standoff-3<=distance<=standoff+3:raise RuntimeError('Final object standoff outside tolerance')
                        result.update(outcome='object_reached_estimate',requires_planner=False,final_range_cm=distance)
                        event('mission_complete',remote_vlm_needed=False)
                        break
                if time.monotonic()-settle_started>1:raise RuntimeError('Arrival settling not verified')
                continue
            final_approach=distance<standoff+12
            if mode=='planning' and final_approach:
                mode='following';path=[observed_world.tolist()];progress=[]
            if mode=='planning':
                # Search runs with motors zero. Its CPU latency cannot mask stale
                # tracking or sustain a motor lease. No remote request is involved.
                call('motors_hold',left=0.,right=0.,**token);last_command=None
                if planning_wait_started is None:planning_wait_started=time.monotonic()
                if time.monotonic()-planning_wait_started>1:raise RuntimeError('Planning stop not verified')
                angular_rate=math.degrees(math.atan2(r[1,0],r[0,0]))/dt
                if np.linalg.norm(pose['velocity_cm_s'])>.5 or abs(angular_rate)>2:
                    planning_quiet_since=None
                    continue
                if planning_quiet_since is None:planning_quiet_since=t1
                if t1-planning_quiet_since<.2:continue
                route=planner.search((x,z,yaw),timeout_seconds=.4)
                planning_quiet_since=None
                planning_wait_started=None
                search_refresh=True
                searches+=1;result['local_searches']=searches
                if searches>6 or route['outcome']!='route_found':raise RuntimeError('No local checked route: '+route['outcome'])
                if any(a['kind']=='drive' and a['value']<0 for a in route['actions']):
                    raise RuntimeError('Mission requires reverse recovery; remote review needed')
                path=[a['predicted_end_pose'][:2] for a in route['actions'] if a['kind']=='drive']
                if not path:path=[goal.tolist()]
                mode='following';progress=[]
                event('local_route',waypoints_cm=path)
                # Refresh after search rather than renewing from a stale image.
                continue
            while len(path)>1 and math.hypot(path[0][0]-x,path[0][1]-z)<6:
                path.pop(0)
            # Close to the object, use its measured bearing rather than asking
            # the coarse five-cm search lattice to correct a sub-cell endpoint.
            waypoint=observed_world if final_approach else path[0]
            bearing=normalize(math.degrees(math.atan2(waypoint[0]-x,waypoint[1]-z))-yaw)
            if abs(bearing)>12 or turn_start is not None:
                if turn_start is None:
                    call('motors_hold',left=0.,right=0.,**token);last_command=None
                    if turn_wait is None:turn_wait=time.monotonic()
                    angular_rate=math.degrees(math.atan2(r[1,0],r[0,0]))/dt
                    if time.monotonic()-turn_wait<.2 or np.linalg.norm(pose['velocity_cm_s'])>.5 or abs(angular_rate)>2:
                        if time.monotonic()-turn_wait>1:raise RuntimeError('Drive-to-turn stop not verified')
                        continue
                    direction=1 if bearing>0 else -1
                    if not planner.clear(sweep((x,z,yaw),('turn',direction*30),precise_turn=True)):
                        mode='planning';call('motors_hold',left=0.,right=0.,**token);last_command=None;continue
                    turn_start=yaw
                    turn_direction=direction
                    turn_begun=time.monotonic()
                    turn_wait=None
                    turn_braking=False
                    progress=[]
                # Brake using measured angular rate, never reverse to chase overshoot.
                angular_rate=math.degrees(math.atan2(r[1,0],r[0,0]))/dt
                angle=normalize(yaw-turn_start)
                if turn_direction*angle < -3:raise RuntimeError('Mission turn moved in wrong direction')
                if abs(angle)>35:raise RuntimeError('Mission turn exceeded heading limit')
                if time.monotonic()-turn_begun>2:raise RuntimeError('Mission turn exceeded two-second limit')
                if turn_braking or abs(bearing)<=max(2.,abs(angular_rate)*.12) or abs(normalize(yaw-turn_start))>=28:
                    turn_braking=True
                    call('motors_hold',left=0.,right=0.,**token);last_command=None
                    if abs(angular_rate)<2:
                        turn_start=None;progress=[]
                    continue
                left=.14*turn_direction;right=-left
                metric=abs(normalize(yaw-turn_start));minimum=.5
            else:
                steer=float(np.clip(math.radians(bearing)*.3,-.025,.025))
                left,right=.16+steer,.16-steer
                if not drive_clear(planner,(x,z,yaw),left,right):
                    clearance_rejections+=1
                    if clearance_rejections>=3:raise RuntimeError('Forward clearance recovery exhausted')
                    clearance_wait=time.monotonic()
                    mode='planning';call('motors_hold',left=0.,right=0.,**token);last_command=None;continue
                clearance_rejections=0
                metric=travel;minimum=.3
            progress.append((t1,metric))
            previous=[v for v in progress if t1-v[0]>=.4]
            if previous and metric-previous[-1][1]<minimum:raise RuntimeError('Mission motion stalled')
            progress=progress[-50:]
            if time.monotonic()-t1>.18:raise RuntimeError('Mission perception exceeded180ms')
            call('motors_hold',left=left,right=right,**token)
            last_command=time.monotonic()
        else:raise RuntimeError('Mission deadline180seconds')
    except Exception as exc:
        result.update(outcome='stopped',requires_planner=True,reason=str(exc))
        event('remote_planner_required',reason=str(exc))
    finally:
        try:call('stop')
        except Exception as exc:
            result.update(outcome='stopped',requires_planner=True,stop_error=str(exc))
            event('remote_planner_required',reason='Final stop request failed',stop_error=str(exc))
        try:result['power_end']=call('status').get('power')
        except Exception as exc:result['power_end_error']=str(exc)
        if 'current' in locals():
            observation_path=log+'.last-observed.jpg'
            if cv2.imwrite(observation_path,current):
                result['last_observation']=dict(image_path=observation_path,time=t1,
                    settled=result['outcome']=='object_reached_estimate')
        result['elapsed_seconds']=time.monotonic()-started
        with open(log,'w') as out:json.dump(result,out,indent=2)
        publish(state=result['outcome'],remote_planner_required=result['requires_planner'],
                reason=result.get('reason'),elapsed_seconds=result['elapsed_seconds'])
    return result


def prepare(capture,request):
    from point_controller import FloorTracker,ROOT
    from route_executor import load_json
    if request.get('image_path')!=capture['image_path']:raise ValueError('Mission must use latest capture')
    if not 0<=time.monotonic()-capture['captured_monotonic']<=120:raise ValueError('Mission capture expired')
    plan={k:capture[k] for k in ('image_path','captured_monotonic','session_id','control_epoch')}
    for key in ('target_box','target_label','inspected_free_rectangle_cm','obstacle_rectangles_cm'):
        plan[key]=request[key]
    plan['standoff_cm']=request.get('standoff_cm',20)
    plan['target_radius_cm']=request.get('target_radius_cm',5)
    image,standoff=validate(plan)
    profile=load_json(os.path.join(ROOT,'calibration/floor_geometry.json'))
    tracker=FloorTracker(profile,load_json(profile['intrinsics_path']))
    x,y,r,b=plan['target_box']
    target=tracker.ground([[(x+r)/2,b]])[0]
    distance=float(np.linalg.norm(target))
    if not np.isfinite(target).all() or not standoff+5<=distance<=120:
        raise ValueError('Target not in mission approach range')
    plan['target_ground_cm']=target.tolist()
    radius=finite(plan['target_radius_cm'])
    if not 5<=radius<=15:raise ValueError('Invalid target radius')
    obstacles=list(plan['obstacle_rectangles_cm'])+[[target[0]-radius,target[1]-3,target[0]+radius,target[1]+3]]
    preview=dict(plan,goals_cm=[(target*(1-standoff/distance)).tolist()],
                 obstacle_rectangles_cm=obstacles,measured_map_drive=True,precise_turn_sweep=True,
                 coalesce_drives=True,goal_heading_turns=True)
    return plan,preview


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan');parser.add_argument('--execute',action='store_true');parser.add_argument('--log')
    args=parser.parse_args()
    with open(args.plan) as source:plan=json.load(source)
    if not args.execute:
        validate(plan);print('Mission input validated; motors not accessed');return
    if not args.log:parser.error('--log required')
    result=execute(plan,args.log)
    print(json.dumps({k:v for k,v in result.items() if k!='samples'}))
    raise SystemExit(0 if result['outcome']=='object_reached_estimate' else 1)


if __name__=='__main__':main()
