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

#: Closest the chassis is asked to park. The camera stops seeing the floor about
#: 2.7cm ahead of the lens and the body reaches 15cm behind it, so a standoff
#: under this cannot be confirmed by observation.
MINIMUM_STANDOFF_CM=12.
#: Past this the floor projection is not independently validated: one pixel of
#: box-bottom error is worth several centimetres. Approaches aim no further and
#: re-observe as they close instead.
TRUSTED_APPROACH_CM=120.


def floor_alignment(plan):
    """Optional local floor alignment; never a replacement for IMU tilt guards.

    A fixed camera-frame correction cannot distinguish mounting error from a
    sloped patch. Its use is therefore restricted to ten degrees of mission yaw.
    Calibration on disk and the IMU's reference gravity remain unchanged.
    """
    value=plan.get('floor_alignment_rotation')
    if value is None:return None
    matrix=np.asarray(value,dtype=float)
    if (matrix.shape!=(3,3) or not np.isfinite(matrix).all()
            or not np.allclose(matrix@matrix.T,np.eye(3),atol=1e-6)
            or abs(np.linalg.det(matrix)-1)>1e-6):
        raise ValueError('Floor alignment must be a proper rotation')
    angle=math.acos(float(np.clip((np.trace(matrix)-1)/2,-1,1)))
    if angle>math.radians(8):raise ValueError('Floor alignment exceeds eight degrees')
    return matrix


def approach_goal(target,standoff,aligned=False):
    target=np.asarray(target,dtype=float)
    if not aligned:return target*(1-standoff/np.linalg.norm(target))
    # Arrival is a standoff region, not a requirement to face the object's
    # centre. Reserve three degrees inside the local alignment's yaw guard.
    heading=np.clip(math.atan2(target[0],target[1]),-math.radians(7),math.radians(7))
    direction=np.array([math.sin(heading),math.cos(heading)])
    along=float(target@direction)
    lateral2=max(0.,float(target@target)-along*along)
    if lateral2>=standoff*standoff:
        raise ValueError('Target standoff cannot be reached inside local alignment heading scope')
    return direction*(along-math.sqrt(standoff*standoff-lateral2))


def target_projection_limit(local_target,target_radius):
    """Bound contact-point ambiguity without accepting an arbitrary target move.

    A tracked bottle on its side can change its visible bottom contact across its
    footprint as viewpoint changes. The allowance remains capped at 12 cm.
    """
    # Measured on 2026-09-16: an approach to a bottle 70-100 cm out disagreed by
    # a steady 9.2-10.7 cm against a 9.0 cm allowance and was abandoned as a
    # moved target. A steady offset at that range is projection error, not
    # motion -- one pixel of contact error is worth about 2.5% of range out
    # there -- so the range term carries more weight and the cap is higher. A
    # target that genuinely moves still breaks the retained-map check, and the
    # approach re-observes continuously as it closes and the error shrinks.
    return max(6.,min(20.,finite(target_radius)+4.),
               .18*float(np.linalg.norm(local_target)))


def validate(plan):
    floor_alignment(plan)
    if type(plan.get('sol_shadow',False)) is not bool:
        raise ValueError('sol_shadow must be boolean')
    if type(plan.get('projective_contact',False)) is not bool:
        raise ValueError('projective_contact must be boolean')
    rectangle(plan['inspected_free_rectangle_cm'])
    for box in plan['obstacle_rectangles_cm']:rectangle(box)
    standoff=finite(plan.get('standoff_cm',20))
    if not MINIMUM_STANDOFF_CM<=standoff<=30:
        raise ValueError('Mission standoff must be%g–30cm'%MINIMUM_STANDOFF_CM)
    image=cv2.imread(plan['image_path'])
    if image is None:raise ValueError('Missing target image')
    TargetTracker(image,plan['target_box'],projective_contact=plan.get('projective_contact',False))
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


FLOOR_RECOVERY_ERRORS = ('Camera tilt/height', 'Floor motion', 'Insufficient carpet',
                         'Optical flow', 'Reverse optical', 'Implausible floor')


def recover_floor(floor,timeline,old,t0,a0,current,t1,a1,up,state,planner,token,frame,call):
    """Motor-off bridge across transient floor failure; never reset the estimator.

    Same-timeline attitudes remain authoritative. Quiet gravity is used only as
    a stationary check, never substituted into one side of a floor-motion pair.
    A persistent height/scale change still fails the original fit checks.
    """
    from state_estimator import rotation2
    began=time.monotonic()
    call('motors_hold',left=0.,right=0.,**token)
    # Already stopped: collect evidence even when a hypothetical +/-20-degree
    # turn would not fit. Before accepting a bridge, verify the entire measured
    # coast yaw interval plus uncertainty and a full four-cm translation disk.
    start_pose=(*state.position.tolist(),math.degrees(state.yaw))

    def measured_coast_angles():
        history=list(timeline.states)
        times=np.asarray([sample['time'] for sample in history],dtype=float)
        if len(times)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
            raise RuntimeError('Incomplete IMU history for floor recovery')
        before=np.flatnonzero(times<=t0)
        after=np.flatnonzero(times>=t1)
        if not len(before) or not len(after):
            raise RuntimeError('Incomplete IMU history for floor recovery endpoints')
        first,last=int(before[-1]),int(after[0])
        if last<=first or np.max(np.diff(times[first:last+1]))>.08+1e-9:
            raise RuntimeError('Incomplete IMU history for floor recovery: sample gap')
        angles=np.degrees(np.asarray([sample['yaw'] for sample in history[first:last+1]]
                                     +[a0['yaw'],a1['yaw']],dtype=float)-a0['yaw'])
        if not np.isfinite(angles).all():
            raise RuntimeError('Invalid IMU yaw history for floor recovery')
        if np.max(np.abs(angles))>20.:
            raise RuntimeError('Floor recovery exceeded twenty-degree coast allowance')
        return min(0.,float(angles.min())-2.),max(0.,float(angles.max())+2.)
    previous=None
    quiet_count=0
    last_error='No quiet floor observations'
    while True:
        if time.monotonic()-began>1.:
            raise RuntimeError('Floor recovery deadline exceeded: '+last_error)
        call('motors_hold',left=0.,right=0.,**token)
        if timeline.up.dot(up)<math.cos(math.radians(5)):
            raise RuntimeError('Mission tilt guard during floor recovery')
        rotation_samples=[sample['yaw'] for sample in timeline.states if t0<=sample['time']<=t1]
        if any(abs(math.degrees(angle-a0['yaw']))>20. for angle in rotation_samples+[a1['yaw']]):
            raise RuntimeError('Floor recovery exceeded twenty-degree coast allowance')
        samples=[sample for sample in timeline.states if t1-.18<=sample['time']<=t1]
        quiet=False
        if len(samples)>=6 and samples[-1]['time']-samples[0]['time']>=.12:
            acceleration=np.array([sample['acceleration'] for sample in samples])
            gyro=np.array([sample['gyro'] for sample in samples])
            quiet=(np.isfinite(acceleration).all() and np.isfinite(gyro).all()
                   and np.max(np.linalg.norm(gyro,axis=1))<math.radians(1.5)
                   and np.max(acceleration.std(axis=0))<.12
                   and np.max(np.abs(np.linalg.norm(acceleration,axis=1)-9.80665))<.35)
        if quiet:
            if previous is None:
                quiet_count=1
            else:
                before,previous_time,attitude=previous
                try:
                    qr,qt,_=floor.motion(before,current,attitude,a1)
                    quiet=(np.linalg.norm(qt)<.1 and
                           abs(math.degrees(math.atan2(qr[1,0],qr[0,0])))<.5)
                except RuntimeError as exc:
                    if not str(exc).startswith(FLOOR_RECOVERY_ERRORS):raise
                    quiet=False
                    last_error=str(exc)
                quiet_count=quiet_count+1 if quiet else 0
            previous=(current,t1,a1) if quiet else None
        else:
            quiet_count=0;previous=None
        if quiet_count>=3:
            try:
                r,t,q=floor.motion(old,current,a0,a1)
            except RuntimeError as exc:
                if not str(exc).startswith(FLOOR_RECOVERY_ERRORS):raise
                last_error=str(exc)
            else:
                if time.monotonic()-began>1.:
                    raise RuntimeError('Floor recovery deadline exceeded during bridge')
                if np.linalg.norm(t)>4.:
                    raise RuntimeError('Floor recovery exceeded four-cm braking allowance')
                angles=measured_coast_angles()
                envelopes=[sweep(start_pose,('turn',angle),precise_turn=True) for angle in angles]
                # Expanding x/z by four cm contains the full translation disk;
                # this deliberately does not assume that coast followed a chord.
                coast_envelope=[min(box[0] for box in envelopes)-4.,min(box[1] for box in envelopes)-4.,
                                max(box[2] for box in envelopes)+4.,max(box[3] for box in envelopes)+4.]
                if not planner.clear(coast_envelope):
                    raise RuntimeError('Floor recovery coast corridor left inspected map')
                delta=rotation2(-state.yaw)@(-r.T@t)
                angle=math.atan2(r[1,0],r[0,0])
                if not angles[0]<=math.degrees(angle)<=angles[1]:
                    raise RuntimeError('Floor recovery bridge disagrees with measured coast yaw')
                for fraction in np.linspace(0.,1.,21):
                    x,z=state.position+fraction*delta
                    yaw=math.degrees(state.yaw+fraction*angle)
                    if not planner.clear(swept_pose_bounds(x,z,yaw,1,10000)):
                        raise RuntimeError('Floor recovery coast corridor left inspected map')
                # Charge uncertainty for the untracked interval in addition to
                # the unchanged fit residual; never clear accumulated variance.
                gap=t1-t0
                q=dict(q,residual_cm=math.hypot(q['residual_cm'],.5*gap),
                       yaw_variance=q.get('yaw_variance',math.radians(.8)**2)+math.radians(gap)**2)
                return current,t1,a1,r,t,q
        while True:
            next_image,next_time,next_attitude=frame(timeline,settled=False)
            if time.monotonic()-began>1.:raise RuntimeError('Floor recovery deadline exceeded waiting for frame')
            if timeline.up.dot(up)<math.cos(math.radians(5)):
                raise RuntimeError('Mission tilt guard during floor recovery')
            if next_time>t1:break
        if next_time-t1>.18:
            raise RuntimeError('Floor recovery camera gap exceeded180ms')
        current,t1,a1=next_image,next_time,next_attitude


def execute(plan,log):
    from point_controller import call,frame,FloorTracker,ROOT
    from state_estimator import AttitudeTimeline,PlanarState
    from route_executor import load_json
    result=dict(outcome='stopped',requires_planner=True,intermediate_model_calls=0,
                events=[],samples=[],local_recoveries=0,local_searches=0,floor_recoveries=0)
    started=time.monotonic()
    shadow=None
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
        if plan.get('sol_shadow',False):
            from async_scene import AsyncScene,assess
            if not os.environ.get('OPENAI_API_KEY'):
                raise ValueError('sol_shadow requires OPENAI_API_KEY')
            shadow=AsyncScene()
            result['sol_shadow']=True
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
        alignment=floor_alignment(plan)
        if alignment is not None:timeline.matrix=alignment@timeline.matrix
        timeline.keep_last_observation=True
        old,t0,a0=frame(timeline,settled=False)
        anchor_attitude=plan.get('capture_attitude')
        rotation,translation,_=floor.motion(anchor,old,anchor_attitude,a0) if anchor_attitude is not None else floor.motion(anchor,old)
        if np.linalg.norm(translation)>.3 or abs(math.atan2(rotation[1,0],rotation[0,0]))>math.radians(1):
            raise RuntimeError('Robot moved since mission preview')
        target=TargetTracker(anchor,plan['target_box'],projective_contact=plan.get('projective_contact',False))
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
        goal=approach_goal(target_world,standoff,alignment is not None)
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
        turn_rest_until=0.
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
            if timeline.up.dot(up)<math.cos(math.radians(5)):raise RuntimeError('Mission tilt guard')
            try:
                r,t,q=floor.motion(old,current,a0,a1)
            except RuntimeError as exc:
                if not str(exc).startswith(FLOOR_RECOVERY_ERRORS):raise
                call('motors_hold',left=0.,right=0.,**token);last_command=None
                result['floor_recoveries']+=1
                if result['floor_recoveries']>3:raise RuntimeError('Floor recovery entry limit exceeded')
                event('floor_recovery_started',reason=str(exc))
                publish(state='floor_recovery',remote_planner_required=False)
                current,t1,a1,r,t,q=recover_floor(floor,timeline,old,t0,a0,current,t1,a1,
                                                 up,state,planner,token,frame,call)
                dt=t1-t0
                # Every route and turn decision must be rebuilt from the bridged
                # pose. Target identity still has to survive its normal checks.
                mode='planning';path=[];progress=[]
                turn_start=None;turn_wait=None;turn_braking=False
                planning_quiet_since=None;planning_wait_started=None
                event('floor_recovered',bridge_seconds=dt,bridge_cm=float(np.linalg.norm(t)))
            pose=state.update(r,t,dt,q)
            if after_search and (np.linalg.norm(t)>.1 or abs(math.degrees(math.atan2(r[1,0],r[0,0])))>.5):
                raise RuntimeError('Robot moved during stopped search')
            x,z=pose['position_cm'];yaw=pose['yaw_degrees']
            if alignment is not None and abs(yaw)>10:
                raise RuntimeError('Local floor alignment exceeded ten-degree heading scope')
            if abs(math.degrees(math.atan2(r[1,0],r[0,0]))/dt)>90:
                raise RuntimeError('Mission rotation rate exceeded90deg/s')
            travel+=float(np.linalg.norm(t))
            if travel>120:raise RuntimeError('Mission travel limit120cm')
            if timeline.up.dot(up)<math.cos(math.radians(5)):raise RuntimeError('Mission tilt guard')
            body=swept_pose_bounds(x,z,yaw,1,10000)
            if not planner.clear(body):raise RuntimeError('Mission chassis/braking space left inspected map')
            result['last_measured_pose']=[x,z,yaw]
            result['travel_cm']=travel
            if shadow is not None:
                reply=shadow.poll()
                if reply is not None:
                    if 'response' in reply:
                        assessment=assess(reply['snapshot'],reply['response'],time.monotonic(),
                                          [x,z,yaw],token,floor.ground)
                        if 'target_world_cm' in assessment:
                            assessment['target_disagreement_cm']=float(np.linalg.norm(
                                np.asarray(assessment['target_world_cm'])-target_world))
                        event('sol_shadow_response',assessment=assessment,
                              response=reply['response'],capture_time=reply['snapshot']['time'])
                    else:
                        event('sol_shadow_error',error=reply['error'])
                if shadow.submit(current,t1,[x,z,yaw],a1,token,plan.get('target_label','Advil bottle')):
                    result['intermediate_model_calls']+=1
                    event('sol_request_started',model='gpt-5.6-sol',reasoning='none',
                          capture_time=t1,pose_cm_degrees=[x,z,yaw],shadow_only=True)
            if time.monotonic()>=next_publish:
                publish(state=mode,pose_cm_degrees=[x,z,yaw],travel_cm=travel,
                        elapsed_seconds=time.monotonic()-started,remote_planner_required=False)
                next_publish=time.monotonic()+.5
            old,t0,a0=current,t1,a1
            try:
                observation=target.reacquire(current) if mode=='recovering' else target.update(current)
                local_target=floor.ground([observation['base_pixel']],a1)[0]
                observed_world=np.array(transform((x,z,yaw),*local_target))
                projection_limit=target_projection_limit(local_target,target_radius)
                if (not np.isfinite(observed_world).all() or
                        np.linalg.norm(observed_world-target_world)>projection_limit):
                    # Retain the actual failing measurement, not only the last
                    # accepted sample, so identity and projection faults can be
                    # distinguished without another powered experiment.
                    diagnostic=dict(frame_time=float(t1),target_box=observation['box'],
                        base_pixel=observation['base_pixel'],confidence=float(observation['confidence']),
                        local_target_cm=local_target.tolist(),observed_world_cm=observed_world.tolist(),
                        target_world_cm=target_world.tolist(),pose_cm_degrees=[float(x),float(z),float(yaw)],
                        disagreement_cm=float(np.linalg.norm(observed_world-target_world)),
                        limit_cm=float(projection_limit),
                        frame_attitude=dict(down_camera=np.asarray(a1['down_camera']).tolist(),
                                            yaw=float(a1.get('yaw',0.)),
                                            variance=float(a1.get('variance',0.))))
                    result['last_target_projection_fault']=diagnostic
                    event('target_projection_rejected',**diagnostic)
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
                if time.monotonic()-recovery_started>6 or result["local_recoveries"]>6:
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
                if alignment is not None:
                    from mapped_drive import nominal_bounds
                    direct_heading=math.degrees(math.atan2(goal[0]-x,goal[1]-z))
                    direct_length=float(np.linalg.norm(goal-np.array([x,z])))
                    if (not planner.clear(nominal_bounds((x,z,direct_heading),direct_length))
                            or not planner.clear(sweep((x,z,yaw),('turn',normalize(direct_heading-yaw)),precise_turn=True))):
                        raise RuntimeError('Local floor alignment requires a clear direct approach corridor')
                    path=[goal.tolist()]
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
            if alignment is not None:
                bearing=float(np.clip(normalize(yaw+bearing),-7.,7.))-yaw
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
                if time.monotonic()-turn_begun>12:raise RuntimeError('Mission turn exceeded twelve-second limit')
                if turn_braking or abs(bearing)<=max(2.,abs(angular_rate)*.12) or abs(normalize(yaw-turn_start))>=28:
                    turn_braking=True
                    call('motors_hold',left=0.,right=0.,**token);last_command=None
                    if abs(angular_rate)<2:
                        turn_start=None;progress=[]
                    continue
                left=.14*turn_direction;right=-left
                metric=abs(normalize(yaw-turn_start));minimum=.5
            else:
                damping=.03*math.atan2(r[1,0],r[0,0])/dt if alignment is not None else 0.
                steer=float(np.clip(math.radians(bearing)*.3-damping,-.025,.025))
                left,right=.16+steer,.16-steer
                if not drive_clear(planner,(x,z,yaw),left,right):
                    clearance_rejections+=1
                    if clearance_rejections>=10:raise RuntimeError("Forward clearance recovery exhausted")
                    clearance_wait=time.monotonic()
                    mode='planning';call('motors_hold',left=0.,right=0.,**token);last_command=None;continue
                clearance_rejections=0
                metric=travel;minimum=.3
            progress.append((t1,metric))
            previous=[v for v in progress if t1-v[0]>=.4]
            if previous and metric-previous[-1][1]<minimum:raise RuntimeError('Mission motion stalled')
            progress=progress[-50:]
            if time.monotonic()-t1>.18:raise RuntimeError('Mission perception exceeded180ms')
            if turn_start is not None:
                if time.monotonic()<turn_rest_until:continue
                call('motors_hold',left=left,right=right,**token)
                time.sleep(.06)
                call('motors_hold',left=0.,right=0.,**token)
                last_command=None
                turn_rest_until=time.monotonic()+.18
                continue
            call('motors_hold',left=left,right=right,**token)
            last_command=time.monotonic()
        else:raise RuntimeError('Mission deadline180seconds')
    except Exception as exc:
        result.update(outcome='stopped',requires_planner=True,reason=str(exc))
        event('remote_planner_required',reason=str(exc))
    finally:
        if shadow is not None:
            shadow.close()
            result['sol_requests_sent']=shadow.sent
            result['sol_response_pending_at_stop']=shadow.pending
        try:call('stop')
        except Exception as exc:
            result.update(outcome='stopped',requires_planner=True,stop_error=str(exc))
            event('remote_planner_required',reason='Final stop request failed',stop_error=str(exc))
        try:result['power_end']=call('status').get('power')
        except Exception as exc:result['power_end_error']=str(exc)
        if result['requires_planner'] and 'timeline' in locals() and hasattr(timeline,'last_sensor_observation'):
            evidence=log+'.fault-observation.json'
            with open(evidence,'w') as out:json.dump(timeline.last_sensor_observation,out)
            result['fault_observation_path']=evidence
            if cv2.imwrite(log+'.fault-observation.jpg',timeline.last_observation_image):
                result['fault_image_path']=log+'.fault-observation.jpg'
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


def capture_attitude(capture):
    """Freeze calibrated camera gravity at this capture using execution's startup path."""
    from point_controller import ROOT
    from route_executor import load_json
    from state_estimator import AttitudeTimeline
    metadata=load_json(capture['metadata_path'])
    if abs(finite(metadata['time'])-finite(capture['captured_monotonic']))>1e-6:
        raise ValueError('Capture attitude metadata timestamp mismatch')
    for key in ('session_id','control_epoch'):
        if metadata[key]!=capture[key]:
            raise ValueError('Capture attitude metadata generation mismatch')
    timeline=AttitudeTimeline(load_json(os.path.join(ROOT,'calibration/imu_mount.json')))
    timeline.initialize_stationary(metadata['imu_samples'],metadata['time'])
    attitude=timeline.at(metadata['time'])
    return {key:value.tolist() if isinstance(value,np.ndarray) else value
            for key,value in attitude.items()}


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
    plan['projective_contact']=request.get('projective_contact',False)
    plan['sol_shadow']=request.get('sol_shadow',False)
    if 'floor_alignment_rotation' in request:
        plan['floor_alignment_rotation']=request['floor_alignment_rotation']
        plan['floor_alignment_evidence']=request.get('floor_alignment_evidence')
    image,standoff=validate(plan)
    profile=load_json(os.path.join(ROOT,'calibration/floor_geometry.json'))
    tracker=FloorTracker(profile,load_json(profile['intrinsics_path']))
    plan['capture_attitude']=capture_attitude(capture)
    alignment=floor_alignment(plan)
    if alignment is not None:
        plan['capture_attitude']=dict(plan['capture_attitude'],down_camera=(
            alignment@np.asarray(plan['capture_attitude']['down_camera'])).tolist())
    x,y,r,b=plan['target_box']
    target=tracker.ground([[(x+r)/2,b]],plan['capture_attitude'])[0]
    distance=float(np.linalg.norm(target))
    if not np.isfinite(target).all() or distance<=0:
        raise ValueError('Target does not project onto the floor ahead')
    # Distance is not a reason to refuse. A target the robot can see is one it
    # can work toward, so both extremes adjust the standoff instead of failing:
    #   far  - aim at the edge of the trusted projection range and re-observe
    #          while closing, since the tracker keeps updating the target's
    #          ground position during the drive;
    #   near - shorten the standoff so the goal stays ahead of the chassis.
    # The old standoff+5 floor made the last few centimetres unreachable: every
    # successful approach left the robot inside the next approach's minimum.
    if distance>TRUSTED_APPROACH_CM:
        standoff=max(standoff,distance-TRUSTED_APPROACH_CM)
    standoff=min(standoff,max(MINIMUM_STANDOFF_CM,distance-3.))
    if distance<=MINIMUM_STANDOFF_CM:
        raise ValueError('Target is %.1f cm away, already inside the %.0f cm '
                         'minimum standoff; it does not need an approach'
                         % (distance,MINIMUM_STANDOFF_CM))
    plan['standoff_cm']=standoff
    plan['target_ground_cm']=target.tolist()
    projected_obstacles=[]
    image_obstacles=request.get('obstacle_image_boxes',[])
    if not isinstance(image_obstacles,list) or len(image_obstacles)>8:
        raise ValueError('Invalid image obstacle list')
    for obstacle in image_obstacles:
        box=obstacle.get('box') if isinstance(obstacle,dict) else None
        if not isinstance(box,dict):raise ValueError('Invalid image obstacle box')
        values=[finite(box.get(k)) for k in ('x0','y0','x1','y1')]
        ox0,oy0,ox1,oy1=values
        if not (0<=ox0<ox1<=640 and 0<=oy0<oy1<=480):
            raise ValueError('Image obstacle box outside frame')
        ground=tracker.ground([[ox0,oy1],[(ox0+ox1)/2,oy1],[ox1,oy1]],
                              plan['capture_attitude'])
        if not np.isfinite(ground).all():raise ValueError('Invalid obstacle floor projection')
        usable=[point for point in ground if 10<=np.linalg.norm(point)<=150]
        if len(usable)!=3:raise ValueError('Obstacle floor projection outside range')
        projected_obstacles.append(bounds(usable,2.))
    plan['obstacle_rectangles_cm']=list(plan['obstacle_rectangles_cm'])+projected_obstacles
    radius=finite(plan['target_radius_cm'])
    if not 5<=radius<=15:raise ValueError('Invalid target radius')
    obstacles=list(plan['obstacle_rectangles_cm'])+[[target[0]-radius,target[1]-3,target[0]+radius,target[1]+3]]
    preview=dict(plan,goals_cm=[approach_goal(target,standoff,alignment is not None).tolist()],
                 obstacle_rectangles_cm=obstacles,measured_map_drive=True,precise_turn_sweep=True,
                 coalesce_drives=True,goal_heading_turns=True,direct_approach=alignment is not None)
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
