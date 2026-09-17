"""Deterministic 360-degree/perimeter search, local A*, and Sol recognition.

Default is motor-free validation. --execute requires an existing guarded service
and an exact fresh capture anchor. No automatic free-space expansion or SLAM.
"""
import argparse
import base64
import copy
import json
import math
import os
import tempfile
import time

from spatial_planner import SpatialPlanner,transform,bounds,normalize,sweep,predict
from route_geometry import contains,rectangle
from route_executor import load_json
from free_space import free_rectangles,merge as merge_free
from vision_map import fan_rectangles,obstacle_footprint,world_rectangle


#: Step-and-look approach. Each step is bounded so the next recognition happens
#: before the robot has travelled far enough for the old measurement to go stale,
#: and arrival is decided by a fresh projected range rather than dead reckoning.
APPROACH_STEP_MAX_CM=15.
APPROACH_STEP_MIN_CM=4.
ARRIVAL_TOLERANCE_CM=6.


def turn_pivot():
    """Measured rotation pivot in body coordinates, lens at the origin.

    Fitted from recorded turns (calibration/turn_pivot.json) and independently
    confirmed by the operator. Falls back to the lens itself, which predicts no
    translation at all, rather than inventing a pivot.
    """
    try:
        from point_controller import ROOT
        with open(os.path.join(ROOT,'calibration/turn_pivot.json')) as source:
            data=json.load(source)
        return float(data['pivot_x_cm']),float(data['pivot_z_cm'])
    except (OSError,ValueError,KeyError,TypeError,ImportError):
        return 0.,0.


def local_point(pose,point):
    x,z=point[0]-pose[0],point[1]-pose[1]
    a=math.radians(pose[2])
    return [math.cos(a)*x-math.sin(a)*z,math.sin(a)*x+math.cos(a)*z]


def perimeter_stations(room,clearance=35.,spacing=40.):
    """Clockwise observation stations inside a measured rectangular room.

Room bounds are supplied geometry, never inferred from the current camera FOV.
Every resulting station and route still needs the inspected-free-space check.
"""
    x0,z0,x1,z1=rectangle(room)
    if not (math.isfinite(clearance) and clearance>=35 and math.isfinite(spacing) and spacing>0):
        raise ValueError('Invalid perimeter spacing or clearance')
    x0+=clearance;x1-=clearance;z0+=clearance;z1-=clearance
    if x0>=x1 or z0>=z1:raise ValueError('Room too small for full-turn perimeter clearance')
    corners=[(x0,z0),(x0,z1),(x1,z1),(x1,z0),(x0,z0)]
    points=[]
    for a,b in zip(corners,corners[1:]):
        steps=int(math.ceil(math.hypot(b[0]-a[0],b[1]-a[1])/spacing))
        if steps>100:raise ValueError('Room extent exceeds bounded search')
        points.extend([a[0]+(b[0]-a[0])*i/steps,a[1]+(b[1]-a[1])*i/steps] for i in range(steps))
    return points


def _covered(box,rects):
    """True when the union of `rects` covers every part of `box`."""
    from spatial_planner import uncovered
    return not uncovered(list(box),rects)


def local_map(plan,pose,free_rects=None):
    """Inner free rectangle, enclosing obstacle rectangles: never invent space.

    Downstream controllers still take a single camera-local rectangle, so the
    retained map is reduced to the largest corridor provably inside it. With a
    union of rectangles a corridor counts as available when every corner lies
    within the union, which is what lets floor certified across several scan
    headings support one approach.
    """
    free=rectangle(plan['inspected_free_rectangle_cm'])
    union=[rectangle(r) for r in (free_rects or [])] or [free]
    inside=lambda box: any(contains(r,box) for r in union) or _covered(box,union)
    available=None
    # Prefer enough lateral room for the conservative measured turn envelope.
    # Fall back to the legacy narrow corridor only when the retained map cannot
    # contain it. The chosen rectangle is always proven inside the retained map.
    for half_width in range(30,19,-1):
        candidate_at_width=None
        for depth in range(20,121):
            candidate=[-half_width,-32,half_width,depth]
            world=bounds([transform(pose,x,z)
                          for x in (-half_width,half_width) for z in (-32,depth)])
            if not inside(world):break
            candidate_at_width=candidate
        if candidate_at_width is not None:
            available=candidate_at_width
            break
    if available is None:raise ValueError('No inspected local approach corridor')
    obstacles=[]
    for obstacle in plan['obstacle_rectangles_cm']:
        x0,z0,x1,z1=rectangle(obstacle)
        obstacles.append(bounds([local_point(pose,[x,z]) for x in (x0,x1) for z in (z0,z1)]))
    return dict(inspected_free_rectangle_cm=available,obstacle_rectangles_cm=obstacles)


class SearchState:
    def __init__(self,plan):
        self.plan=plan
        self.scan_step_degrees=float(plan.get('scan_step_degrees',30.))
        if not 1. <= self.scan_step_degrees <= 30.:
            raise ValueError('scan_step_degrees must be between 1 and 30')
        self.scan_direction=int(plan.get('scan_direction',1))
        if self.scan_direction not in (-1,1):
            raise ValueError('scan_direction must be -1 or 1')
        self.pose=[0.,0.,0.]
        self.station='start'
        self.visited={'start':[]}
        self.rotation={'start':0.}
        self.stations=plan.get('search_stations_cm')
        if self.stations is None:
            self.stations=perimeter_stations(plan['room_bounds_cm']) if 'room_bounds_cm' in plan else []
        self.position_variance=0.
        self.yaw_variance=0.
        self.travel=0.
        # Floor certified by the camera as the scan turns, plus obstacle
        # footprints projected from recognizer boxes. The supplied rectangle is
        # the seed, not the whole map: it covers the floor under the chassis,
        # which the camera can never see, and vision extends it outward.
        self.observed_free=[]
        self.observed_obstacles=[]

    def mapped(self,**extra):
        """Plan dict whose free space is the seed plus everything since seen."""
        free=[rectangle(self.plan['inspected_free_rectangle_cm'])]+list(self.observed_free)
        obstacles=list(self.plan['obstacle_rectangles_cm'])+list(self.observed_obstacles)
        return dict(self.plan,inspected_free_rectangles_cm=merge_free(free),
                    obstacle_rectangles_cm=obstacles,**extra)

    def certify(self,tracker,attitude=None):
        """Add the floor visible from the current pose to the retained map.

        Fails closed: any projection or calibration problem simply certifies
        nothing, leaving the map as it was. Refusing to extend free space can
        only stop the robot, never drive it somewhere unverified.
        """
        try:
            bands=fan_rectangles(tracker,attitude)
            added=free_rectangles([(list(self.pose),band) for band in bands])
        except Exception:
            return 0
        self.observed_free=merge_free(list(self.observed_free)+added)
        return len(added)

    def retain_obstacles(self,tracker,answer,attitude=None):
        """Project recognizer obstacle boxes into retained map geometry.

        Off by default: the user's robot is allowed to bump into things, and a
        bounding box around clutter is a poor floor footprint -- an axis-aligned
        box over a diagonal cable is mostly empty carpet, and treating it as
        occupied walled the robot in where it plainly had room. Set
        `retain_recognized_obstacles` to make them block motion instead.

        Projection, logging and the overlay are unaffected either way; this only
        decides whether recognizer boxes are allowed to refuse a route. Physical
        guards -- stall, power, tilt, collision, clearance of the inspected map
        -- are untouched and still stop the robot.

        Boxes beyond the trusted projection range are dropped, never guessed.
        """
        if not self.plan.get('retain_recognized_obstacles',False):
            return 0
        added=0
        for obstacle in (answer.get('obstacles') or []):
            box=obstacle.get('box')
            if not isinstance(box,dict):
                continue
            try:
                local=obstacle_footprint(tracker,box,attitude)
            except Exception:
                continue
            # A footprint covering the robot's own body contradicts the fact
            # that the robot is standing there. A bounding box around a long
            # diagonal object projects its near contact almost under the lens,
            # so this happens without the recognizer being wrong; retaining it
            # would wall the robot in where it demonstrably fits.
            if local[0]<11. and local[2]>-11. and local[1]<2.:
                continue
            self.observed_obstacles.append(world_rectangle(self.pose,local))
            added+=1
        return added

    def snapshot(self):
        return dict(pose=list(self.pose),station=self.station,
                    visited=copy.deepcopy(self.visited),rotation=copy.deepcopy(self.rotation),
                    position_variance=self.position_variance,
                    yaw_variance=self.yaw_variance,travel=self.travel)

    def restore(self,snapshot):
        required=('pose','station','visited','rotation','position_variance','yaw_variance','travel')
        if any(key not in snapshot for key in required):
            raise ValueError('Incomplete search checkpoint state')
        pose=[float(v) for v in snapshot['pose']]
        if len(pose)!=3 or not all(math.isfinite(v) for v in pose):
            raise ValueError('Invalid checkpoint pose')
        self.pose=pose;self.station=str(snapshot['station'])
        self.visited=copy.deepcopy(snapshot['visited']);self.rotation=copy.deepcopy(snapshot['rotation'])
        self.position_variance=float(snapshot['position_variance'])
        self.yaw_variance=float(snapshot['yaw_variance']);self.travel=float(snapshot['travel'])
        if (self.station not in self.visited or self.station not in self.rotation or
                min(self.position_variance,self.yaw_variance,self.travel)<0):
            raise ValueError('Invalid checkpoint search state')

    def observed(self):
        headings=self.visited.setdefault(self.station,[])
        if all(abs(normalize(self.pose[2]-h))>10 for h in headings):
            headings.append(self.pose[2])

    def candidates(self):
        planner=SpatialPlanner(self.mapped(goal_cm=self.pose[:2],measured_map_drive=True,precise_turn_sweep=True))
        if self.rotation[self.station]<358.5:
            step=min(self.scan_step_degrees,360.-self.rotation[self.station])
            # Try the configured direction first, then the other one. Clutter is
            # not symmetric: a scan can be walled in on one side while the
            # opposite sweep is clear, and refusing to look the other way ends
            # the search with most of the circle unexamined. Coverage counts
            # measured absolute rotation, so either direction makes progress.
            for direction in (self.scan_direction,-self.scan_direction):
                angle=direction*step
                if planner.clear(sweep(self.pose,('turn',angle),precise_turn=True)):
                    return [dict(id='scan_right',kind='turn',degrees=angle)]
            # Parked too close to something to rotate at all. A differential
            # robot turns inside its own radius, so the way out is to make that
            # radius fit: reverse a little, then scan from there. The reverse is
            # checked against the same retained map, and the rear is covered by
            # the supplied starting clearance rather than by anything the camera
            # can see, so it stays short and stops at the first clear option.
            for back in (-5.,-10.):
                if planner.clear(sweep(self.pose,('drive',back),precise_turn=True)):
                    return [dict(id='scan_backoff',kind='drive',cm=back,
                                 goal_cm=list(predict(tuple(self.pose),('drive',back))[:2]))]
            return []
        found=[]
        indices=list(range(len(self.stations)))
        if self.station!='start':
            start=int(self.station.split('_')[1])+1
            indices=indices[start:]+indices[:start]
        for i in indices:
            point=self.stations[i]
            key='station_%d'%i
            if key in self.visited:continue
            route=SpatialPlanner(self.mapped(goal_cm=point,measured_map_drive=True,precise_turn_sweep=True)).search(tuple(self.pose))
            if route['outcome']=='route_found':
                cost=sum(abs(a['value']) if a['kind']=='drive' else abs(a['value'])*.2 for a in route['actions'])
                found.append(dict(id=key,kind='relocate',goal_cm=point,path_cost=cost))
        # Choose the least-cost entry station, then follow the perimeter order.
        return sorted(found,key=lambda c:(c['path_cost'],c['id'])) if self.station=='start' else found

    def context(self,candidates):
        return dict(task='search_and_navigate',policy='360_then_perimeter_astar',pose_cm_degrees=self.pose,searched_headings=self.visited,
                    scanned_rotation_degrees=self.rotation,
                    station=self.station,candidates=candidates,travel_cm=self.travel,
                    completion_rule='Only measured controller arrival; target absent means continue search')

    def select(self,answer,candidates):
        if answer.get('target_visible') is True:
            if answer.get('target_box') is None or answer.get('contact_pixel') is None:
                raise ValueError('Approach missing target grounding')
            box=answer['target_box'];pixel=answer['contact_pixel']
            x0,y0,x1,y1=[float(box[k]) for k in ('x0','y0','x1','y1')]
            u,v=float(pixel['x']),float(pixel['y'])
            if not (0<=x0<x1<=640 and 0<=y0<y1<=480 and x0<=u<=x1 and
                    y0<=v<=y1 and abs(v-y1)<=8):
                raise ValueError('Invalid target geometry')
            return dict(kind='approach')
        if answer.get('target_visible') is not False:raise ValueError('Recognizer omitted visibility')
        # Ignore any model-authored motion choices. Candidates are deterministically
        # ordered by coverage phase and local A* path cost/perimeter order.
        return candidates[0] if candidates else dict(kind='hold')


def atomic_json(path,payload):
    """Durably replace a mission checkpoint without exposing partial JSON."""
    directory=os.path.dirname(os.path.abspath(path))
    fd,temporary=tempfile.mkstemp(prefix='.sol-search-',dir=directory)
    try:
        with os.fdopen(fd,'w') as output:
            json.dump(payload,output,indent=2,allow_nan=False)
            output.flush();os.fsync(output.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def execute(plan,log):
    import cv2
    import numpy as np
    from point_controller import call,ROOT
    from object_mission import capture_attitude,prepare,execute as approach
    from spatial_turn import execute_turn
    from continuous_turn import execute as execute_continuous_turn
    from spatial_executor import execute_navigation
    from state_estimator import AttitudeTimeline,PlanarState
    from async_scene import AsyncScene,request_scene
    from spatial_preview import write_preview
    from turn_sweep_preview import write_turn_preview
    state=SearchState(plan)
    result=dict(outcome='stopped',events=[],sol_calls=0,requires_astra=False,
                handoff='intervention')
    started=time.monotonic()
    expected={k:plan[k] for k in ('session_id','control_epoch')}
    checkpoint_path=os.path.abspath(plan.get('checkpoint_path',log+'.checkpoint.json'))
    recognition_every_actions=int(plan.get('recognition_every_actions',1))
    if not 1 <= recognition_every_actions <= 12:
        raise ValueError('recognition_every_actions must be between 1 and 12')
    recognition_timeout=float(plan.get('recognition_timeout_seconds',15.))
    if not 3 <= recognition_timeout <= 30 or not math.isfinite(recognition_timeout):
        raise ValueError('recognition_timeout_seconds must be between 3 and 30')
    max_search_actions=int(plan.get('max_search_actions',72))
    if not 1 <= max_search_actions <= 144:
        raise ValueError('max_search_actions must be between 1 and 144')
    actions_since_recognition=recognition_every_actions
    action_count=0
    recognizer=None
    def event(name,**data):
        row=copy.deepcopy(dict(event=name,elapsed_seconds=time.monotonic()-started,**data))
        result['events'].append(row)
        with open(log+'.events.jsonl','a') as output:output.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    def checkpoint(phase,**extra):
        payload=dict(version=1,mission='sol_search',phase=phase,
            target_label=plan['target_label'],session_id=expected['session_id'],
            control_epoch=expected['control_epoch'],state=state.snapshot(),
            action_count=action_count,actions_since_recognition=actions_since_recognition,
            sol_calls=result['sol_calls'],updated_monotonic=time.monotonic(),
            resume_automatically=False,
            resume_note='Revalidate pose, map, sensors, power, and control generation before reuse')
        payload.update(extra)
        atomic_json(checkpoint_path,payload)
        result['checkpoint_path']=checkpoint_path
    def stopped_status(reason):
        status=call('status')
        if any(status.get(k)!=v for k,v in expected.items()):
            raise ValueError('Search cancelled '+reason)
        if (not status.get('healthy') or not status.get('motion_enabled') or
                status.get('motor',{}).get('output')!=[0,0]):
            raise ValueError('Search requires healthy stopped robot')
        power=status.get('power',{})
        if power and not power.get('motion_allowed',False):
            raise ValueError('Power guard blocks search motion')
        return status
    def wait_recognition(deadline):
        while time.monotonic()<deadline:
            reply=recognizer.poll()
            if reply is not None:
                if 'error' in reply:raise ValueError('Sol recognition failed: '+reply['error'])
                return reply['response']
            stopped_status('during inference')
            time.sleep(.02)
        raise ValueError('Sol recognition timed out')
    try:
        if not os.environ.get('OPENAI_API_KEY'):raise ValueError('OPENAI_API_KEY required')
        resume=plan.get('resume_checkpoint_path')
        anchor_path=plan['image_path'];anchor_time=plan['captured_monotonic']
        if resume:
            with open(os.path.abspath(resume)) as source:saved=json.load(source)
            if (saved.get('version')!=1 or saved.get('mission')!='sol_search' or
                    saved.get('phase')!='action_complete' or not saved.get('resume_eligible') or
                    saved.get('target_label')!=plan['target_label']):
                raise ValueError('Checkpoint is not an eligible search continuation')
            expected={k:saved[k] for k in ('session_id','control_epoch')}
            state.restore(saved['state'])
            action_count=int(saved['action_count'])
            actions_since_recognition=int(saved['actions_since_recognition'])
            result['sol_calls']=int(saved['sol_calls'])
            anchor_path=saved['resume_anchor']['image_path']
            anchor_time=saved['resume_anchor']['captured_monotonic']
        # Never start at a new, unregistered pose relative to the inspected map.
        from point_controller import FloorTracker,TURN_FLOW_WINDOW,frame
        profile=load_json(os.path.join(ROOT,'calibration/floor_geometry.json'))
        mount=load_json(os.path.join(ROOT,'calibration/imu_mount.json'))
        timeline=AttitudeTimeline(mount)
        image,ts,attitude=frame(timeline,settled=False)
        anchor=cv2.imread(anchor_path)
        if anchor is None or not 0<=time.monotonic()-anchor_time<=120:
            raise ValueError('Missing or expired search anchor')
        # Scan turns sweep carpet features sideways, so the drive-tuned landing
        # window is too tight. The crop stays on: at a 10 degree step the moved
        # features remain inside it, and full-frame flow is slow enough here to
        # open a gap in the IMU history before the turn starts.
        floor=FloorTracker(profile,load_json(profile["intrinsics_path"]),250,
                           flow_window=TURN_FLOW_WINDOW)
        r,t,_=floor.motion(anchor,image)
        if np.linalg.norm(t)>.3 or abs(math.atan2(r[1,0],r[0,0]))>math.radians(1):
            raise ValueError('Search pose changed since map inspection')
        max_recognition_calls=int(plan.get('max_recognition_calls',20))
        if not 1 <= max_recognition_calls <= 72:
            raise ValueError('max_recognition_calls must be between 1 and 72')
        recognizer=AsyncScene(request=request_scene,min_interval_seconds=0.,
                              max_requests=max_recognition_calls)
        checkpoint('resumed' if resume else 'started',resume_source=resume)
        for index in range(max_search_actions+1):
            if time.monotonic()-started>300 or state.travel>120:
                raise ValueError('Search time or travel budget exceeded')
            stopped_status('or generation changed')
            motion_blocked=state.position_variance>64 or state.yaw_variance>225
            prefix=log+'.step-%02d'%index
            observation=call('observation')
            with open(prefix+'.jpg','wb') as out:out.write(base64.b64decode(observation.pop('jpeg_base64')))
            with open(prefix+'.capture.json','w') as out:json.dump(observation,out)
            capture=dict(expected,image_path=prefix+'.jpg',metadata_path=prefix+'.capture.json',captured_monotonic=observation['time'])
            recognition_due=(actions_since_recognition>=recognition_every_actions or motion_blocked)
            context=state.context([])
            if recognition_due:
                if result['sol_calls']>=max_recognition_calls:
                    result.update(outcome='recognition_budget_exhausted',handoff='intervention',
                                  reason='Configured Sol recognition-call ceiling reached')
                    event('intervention_required',reason=result['reason'])
                    checkpoint('intervention',last_observation=capture['image_path'])
                    break
                image_for_sol=cv2.imread(capture['image_path'])
                submitted=recognizer.submit(image_for_sol,observation['time'],state.pose,attitude,
                    [expected['session_id'],expected['control_epoch']],plan['target_label'],context)
                if not submitted:raise ValueError('Sol recognition scheduler rejected a due request')
                event('sol_prefetched',target=plan['target_label'],context=context)
            # Local geometry search overlaps the network request and never uses
            # model output to authorize or select motion.
            candidates=[] if motion_blocked else state.candidates()
            if motion_blocked:
                event('motion_paused_for_localization',position_sigma_cm=math.sqrt(state.position_variance),
                      yaw_sigma_degrees=math.sqrt(state.yaw_variance),recognition_continues=True)
            response=None
            if recognition_due:
                response=wait_recognition(time.monotonic()+recognition_timeout)
                result['sol_calls']+=1
                actions_since_recognition=0
                if not motion_blocked:state.observed()
                # Retain what this stopped view showed before acting on it: the
                # floor the camera certified, and the obstacles it named. Both
                # are evidence from this pose, so they are recorded whether or
                # not the target was recognised.
                bands=state.certify(floor)
                blocked=state.retain_obstacles(floor,response['answer'])
                event('map_extended',free_bands=bands,obstacles_added=blocked,
                      free_rectangles=len(state.observed_free),
                      obstacle_rectangles=len(state.observed_obstacles))
                choice=state.select(response['answer'],candidates)
                event('sol_result',target=plan['target_label'],response=response)
            else:
                choice=candidates[0] if candidates else dict(kind='hold')
                event('recognition_skipped',cadence_actions=recognition_every_actions,
                      actions_since_recognition=actions_since_recognition)
            event('search_decision',policy='deterministic',choice=choice,
                  recognition_used=recognition_due)
            stopped_status('during inference')
            checkpoint('decision',last_observation=capture['image_path'],choice=choice)
            if choice['kind']=='approach' and plan.get('search_only',False):
                result.update(outcome='object_found',requires_astra=False,
                              handoff='final',
                              found_image=capture['image_path'],recognition=response,
                              found_pose_cm_degrees=None if motion_blocked else list(state.pose),
                              found_pose_valid=not motion_blocked)
                event('object_found',image_path=capture['image_path'],target=plan['target_label'],
                      pose_valid=not motion_blocked)
                break
            if motion_blocked:
                result.update(outcome='localization_required',requires_astra=True,
                              handoff='intervention',
                              target_visible=response['answer']['target_visible'],
                              last_observation=capture['image_path'],
                              reason='Accumulated search pose uncertainty exceeded; stopped recognition completed')
                event('localization_required',target_visible=result['target_visible'])
                break
            if choice['kind']=='hold':
                result.update(outcome='search_blocked',requires_astra=True,
                              handoff='intervention');break
            if choice['kind']!='approach' and action_count>=max_search_actions:
                result.update(outcome='search_budget_exhausted',requires_astra=True,
                              handoff='intervention',
                              reason='Configured local-action ceiling reached')
                event('intervention_required',reason=result['reason'])
                break
            aim=None
            if choice["kind"]=="approach":
                box=response['answer']['target_box'];pixel=response['answer']['contact_pixel']
                coords=[float(box[k]) for k in ('x0','y0','x1','y1')]
                if not (coords[0]<=pixel['x']<=coords[2] and abs(pixel['y']-coords[3])<=8):
                    raise ValueError('Sol floor contact inconsistent with target box')
                # The mission needs standoff+5 cm of room to run. Successive
                # approaches leave the robot closer, so a fixed standoff makes
                # the last few centimetres unreachable: the target ends up too
                # near to start. Close in, pull the standoff down to the range
                # the controller still accepts rather than refusing to move.
                standoff=float(plan.get('standoff_cm',25))
                try:
                    ground=floor.ground([[(coords[0]+coords[2])/2,coords[3]]],
                                        capture_attitude(capture))[0]
                    range_cm=math.hypot(*ground)
                    bearing=math.degrees(math.atan2(ground[0],ground[1]))
                except Exception:
                    range_cm=float('inf')   # leave the standoff to the controller
                    bearing=0.
                # The approach corridor is centred on the robot's heading, so a
                # target well off to one side puts its own goal outside the
                # corridor and no route exists. Face it first: a turn in place
                # is cheap, centres the target for a better projection, and is
                # what a person would do before walking towards something.
                aim=None
                if abs(bearing)>12.:
                    turn=max(-30.,min(30.,bearing))
                    aiming=SpatialPlanner(state.mapped(goal_cm=state.pose[:2],
                        measured_map_drive=True,precise_turn_sweep=True))
                    if aiming.clear(sweep(state.pose,('turn',turn),precise_turn=True)):
                        aim=dict(id='aim_at_target',kind='turn',degrees=turn)
                        event('aiming_at_target',bearing_degrees=bearing,turn_degrees=turn,
                              range_cm=None if range_cm==float('inf') else range_cm)
            if aim is not None:
                choice=aim
            elif choice['kind']=='approach':
                # Step and look, rather than committing to one route from the
                # worst vantage point the approach will ever have. One bounded
                # drive, then the next pass re-recognises and re-measures from
                # closer in, where a pixel of contact error is worth far less
                # (about 0.7% of range near the robot against 2.5% out at arm's
                # length). Nothing is frozen, so nothing has to be recovered.
                if range_cm<=standoff+ARRIVAL_TOLERANCE_CM:
                    result.update(outcome='object_reached_estimate',handoff='final',
                                  requires_astra=False,final_range_cm=range_cm,
                                  last_measured_pose=list(state.pose),
                                  travel_cm=state.travel)
                    event('object_reached_estimate',range_cm=range_cm,
                          standoff_cm=standoff,pose=list(state.pose))
                    checkpoint('finished',outcome=result['outcome'])
                    break
                step=max(APPROACH_STEP_MIN_CM,
                         min(APPROACH_STEP_MAX_CM,range_cm-standoff))
                goal=list(predict(tuple(state.pose),('drive',step))[:2])
                stepping=SpatialPlanner(state.mapped(goal_cm=goal,
                    measured_map_drive=True,precise_turn_sweep=True))
                if not stepping.clear(sweep(state.pose,('drive',step),precise_turn=True)):
                    raise ValueError('Approach step of %.0f cm is not clear'%step)
                choice=dict(id='approach_step',kind='drive',cm=step,goal_cm=goal)
                event('approach_step',range_cm=range_cm,standoff_cm=standoff,
                      step_cm=step,bearing_degrees=bearing)
            if choice['kind']=='turn':
                degrees=choice['degrees']
                write_turn_preview(capture['image_path'],degrees,plan['preview_path'])
                event('trajectory_preview',path=plan['preview_path'],kind='search_turn')
                planner=SpatialPlanner(dict(plan,goal_cm=state.pose[:2],measured_map_drive=True,precise_turn_sweep=True))
                envelope=sweep(state.pose,('turn',degrees),precise_turn=True)
                if not planner.clear(envelope):raise ValueError('Search turn no longer clear')
                def guard(x,z,yaw):
                    gx,gz=transform(state.pose,x,z)
                    body=bounds([transform((gx,gz,state.pose[2]+yaw),u,v) for u in (-6,6) for v in (-15,0)],7.)
                    if not contains(envelope,body) or not planner.clear(body):raise RuntimeError('Search turn left inspected space')
                turn_mode=plan.get('scan_turn_controller','continuous')
                if turn_mode not in ('continuous','pulse'):
                    raise ValueError('scan_turn_controller must be continuous or pulse')
                if turn_mode=='pulse':
                    outcome=execute_turn(capture,degrees,sweep((0,0,0),('turn',degrees)),prefix+'.turn.json',
                                         AttitudeTimeline(mount),world_guard=guard,power=plan.get('search_turn_power',.14))
                    accepted='turn_reached_estimate'
                else:
                    # The continuous controller owns only the motor lease and IMU
                    # loop.  Search still verifies the pre-turn anchor, retained
                    # sweep, and post-turn camera translation before adopting pose.
                    turn_timeline=AttitudeTimeline(mount)
                    before,bt,ba=frame(turn_timeline,settled=False)
                    anchor=cv2.imread(capture['image_path'])
                    r0,t0,_=floor.motion(anchor,before)
                    if np.linalg.norm(t0)>.3 or abs(math.atan2(r0[1,0],r0[0,0]))>math.radians(1):
                        raise ValueError('Robot moved before continuous search turn')
                    turn_plan=dict(capture,degrees=degrees,power=plan.get('search_turn_power',.14),
                                   timeout_seconds=plan.get('search_turn_timeout_seconds',8.),
                                   tolerance_degrees=plan.get('search_turn_tolerance_degrees',3.),
                                   brake_margin_degrees=plan.get('search_turn_brake_margin_degrees',1.))
                    outcome=execute_continuous_turn(turn_plan,prefix+'.turn.json',timeline=turn_timeline)
                    accepted='turn_reached_imu_estimate'
                    if outcome.get('outcome')==accepted and 'stop_error' not in outcome:
                        after,at,aa=frame(turn_timeline,settled=False)
                        # Preserve the frame pair before fitting so a tracking
                        # failure can be replayed offline instead of re-run.
                        settled_path=prefix+'.turn.settled.jpg'
                        if not cv2.imwrite(settled_path,after):
                            raise ValueError('Could not save continuous-turn anchor')
                        cv2.imwrite(prefix+'.turn.before.jpg',before)
                        imu_angle=float(outcome['final_angle_degrees'])
                        try:
                            r,t,q=floor.motion(before,after,ba,aa)
                            measured=PlanarState().update(r,t,max(.001,at-bt),q)
                            if abs(normalize(measured['yaw_degrees']-imu_angle))>3:
                                raise ValueError('Post-turn camera/IMU yaw disagreement')
                            measured['yaw_sigma_degrees']=max(.5,measured['yaw_sigma_degrees'])
                        except (RuntimeError,ValueError) as exc:
                            # The turn happened; only the picture of it failed.
                            # On plain carpet the floor fit can lose its inliers
                            # or its scale, and abandoning a whole search for
                            # that throws away a completed, IMU-verified turn.
                            # The chassis rotates about a measured pivot, so the
                            # camera's small translation is predictable from the
                            # angle alone -- and this estimate is believed less
                            # than a measured one, so its uncertainty is larger.
                            angle=math.radians(imu_angle)
                            px,pz=turn_pivot()
                            measured=dict(
                                position_cm=[(1-math.cos(angle))*px-math.sin(angle)*pz,
                                             math.sin(angle)*px+(1-math.cos(angle))*pz],
                                yaw_degrees=imu_angle,position_sigma_cm=2.,
                                yaw_sigma_degrees=1.5)
                            event('turn_translation_predicted',reason=str(exc),
                                  imu_degrees=imu_angle,
                                  position_cm=measured['position_cm'])
                        measured['yaw_degrees']=imu_angle
                        dx,dz=measured['position_cm']
                        guard(dx,dz,imu_angle)
                        outcome.update(final_position_cm=[dx,dz],settling_samples=[measured],
                            final_anchor=dict(image_path=settled_path,captured_monotonic=at))
                if outcome['outcome']!=accepted or 'stop_error' in outcome:
                    raise ValueError('Search turn failed: '+outcome.get('reason',outcome['outcome']))
                final=outcome['settling_samples'][-1]
                dx,dz=outcome['final_position_cm']
                gx,gz=transform(state.pose,dx,dz)
                state.pose=[gx,gz,normalize(state.pose[2]+final['yaw_degrees'])]
                state.position_variance+=final['position_sigma_cm']**2
                state.yaw_variance+=final['yaw_sigma_degrees']**2
                state.travel+=math.hypot(dx,dz)
                state.rotation[state.station]+=abs(final['yaw_degrees'])
                expected['control_epoch']+=1
            else:
                navigation=dict(plan,**capture)
                navigation.update(goals_cm=[choice['goal_cm']],initial_pose_cm_degrees=state.pose,
                    initial_position_variance_cm2=state.position_variance,initial_yaw_variance_deg2=state.yaw_variance,
                    measured_map_drive=True,precise_turn_sweep=True,coalesce_drives=True)
                # Navigation preview assumes an initial-origin image; use a fresh local map.
                preview=dict(capture,**local_map(plan,state.pose,state.mapped()["inspected_free_rectangles_cm"]))
                preview.update(goals_cm=[local_point(state.pose,choice['goal_cm'])],measured_map_drive=True,precise_turn_sweep=True)
                write_preview(preview,plan['preview_path'])
                event('trajectory_preview',path=plan['preview_path'],kind='relocate')
                outcome=execute_navigation(navigation,prefix+'.navigation.json')
                if outcome['outcome']!='target_reached_estimate':raise ValueError('Search relocation failed: '+outcome.get('reason','unknown'))
                state.pose=list(outcome['final_pose'])
                state.position_variance=outcome['position_variance_cm2']
                state.yaw_variance=outcome['yaw_variance_deg2']
                state.travel+=outcome['travel_cm']
                expected['control_epoch']+=1+sum(2 if a['plan']['kind']=='drive' else 1 for a in outcome['actions'])
                if choice['kind']=='relocate':
                    state.station=choice['id'];state.visited[state.station]=[];state.rotation[state.station]=0.
                # A back-off is not a new observation station: it shuffles the
                # robot to make turning room, so the coverage already scanned
                # here must survive it rather than restarting the sweep.
            action_count+=1
            actions_since_recognition+=1
            event('local_action_complete',pose=state.pose,travel_cm=state.travel)
            # A crash-safe continuation anchor is captured only after the local
            # controller has verified its stop and we have adopted measured pose.
            resume_observation=call('observation')
            if (any(resume_observation.get(k)!=v for k,v in expected.items()) or
                    resume_observation.get('motor',{}).get('output') not in (None,[0,0])):
                raise ValueError('Could not bind checkpoint to stopped control generation')
            resume_path=log+'.checkpoint-%03d.jpg'%action_count
            with open(resume_path,'wb') as output:
                output.write(base64.b64decode(resume_observation.pop('jpeg_base64')))
            checkpoint('action_complete',resume_eligible=True,
                resume_anchor=dict(image_path=resume_path,
                                   captured_monotonic=resume_observation['time']))
        else:result.update(outcome='search_budget_exhausted',requires_astra=True,
                           handoff='intervention')
    except Exception as exc:
        result.update(outcome='stopped',reason=str(exc),requires_astra=True,
                      handoff='intervention')
        event('search_stopped',reason=str(exc))
    finally:
        if recognizer is not None:recognizer.close()
        try:call('stop')
        except Exception as exc:result['stop_error']=str(exc)
        result.update(elapsed_seconds=time.monotonic()-started,searched_headings=state.visited)
        try:checkpoint('finished',outcome=result['outcome'],handoff=result['handoff'])
        except Exception as exc:result['checkpoint_error']=str(exc)
        atomic_json(log,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('plan');p.add_argument('--execute',action='store_true');p.add_argument('--log')
    args=p.parse_args()
    plan=load_json(args.plan)
    state=SearchState(plan);state.observed()
    if not args.execute:print(json.dumps(state.context(state.candidates()),indent=2));return
    if not args.log:p.error('--log required')
    result=execute(plan,args.log)
    raise SystemExit(0 if result['outcome'] in ('object_reached_estimate','object_found') else 1)


if __name__=='__main__':main()
