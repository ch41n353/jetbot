"""Token-cancelled bounded turn with measured camera translation and settling.

Uses the existing 60 ms pulse / 180 ms rest policy, not a new motor tuning.
This experimental adapter is needed before turns can preserve a local map.
"""
import json
import math
import os
import time

from spatial_planner import bounds, transform
from route_geometry import contains
from turn_controller import turn_command


def execute_turn(plan, degrees, envelope, log, timeline, world_guard=None):
    import cv2
    import numpy as np
    from point_controller import ROOT, call, frame, FloorTracker
    from state_estimator import PlanarState
    from route_executor import load_json
    result = dict(outcome='stopped', samples=[], settling_samples=[])
    started = time.monotonic()
    token = {k: plan[k] for k in ('session_id', 'control_epoch')}
    try:
        turn_command(degrees, 0, .14)
        status = call('status')
        result['power_start']=status.get('power')
        if any(status[k] != v for k,v in token.items()):
            raise RuntimeError('Turn cancelled or stale control generation')
        if not status['healthy'] or not status['motion_enabled'] or status['motor']['output'] != [0,0]:
            raise RuntimeError('Turn requires healthy sensors and stopped motors')
        age = time.monotonic()-plan['captured_monotonic']
        if not 0 <= age <= 120:
            raise RuntimeError('Turn anchor expired')
        profile = load_json(os.path.join(ROOT, 'calibration/floor_geometry.json'))
        tracker = FloorTracker(profile, load_json(profile['intrinsics_path']), max_features=125)
        timeline.record_directory = log+'.observations'
        os.makedirs(timeline.record_directory, exist_ok=True)
        old, t0, a0 = frame(timeline, settled=False)
        anchor = cv2.imread(plan['image_path'])
        if anchor is None or anchor.shape != old.shape:
            raise RuntimeError('Missing turn anchor')
        r,t,_ = tracker.motion(anchor, old)
        if np.linalg.norm(t) > .3 or abs(math.atan2(r[1,0],r[0,0])) > math.radians(1):
            raise RuntimeError('Robot moved since turn anchor')
        if getattr(timeline,'route_reference_up',None) is None:
            timeline.route_reference_up = timeline.up.copy()
        state = PlanarState()
        resting_until = started
        settled = []
        reached_at = None
        while time.monotonic()-started < 12:
            current,t1,a1 = frame(timeline,settled=False)
            if t1 <= t0:
                time.sleep(.005)
                continue
            if not 0 < t1-t0 <= .35:
                raise RuntimeError('Turn tracking gap exceeded 350 ms')
            r,t,q = tracker.motion(old,current,a0,a1)
            pose = state.update(r,t,t1-t0,q)
            x,z = pose['position_cm']
            body = bounds([transform((x,z,pose['yaw_degrees']),u,v)
                           for u in (-6.,6.) for v in (-15.,0.)],7.)
            if not contains(envelope,body):
                raise RuntimeError('Measured turn chassis left inspected sweep')
            if world_guard is not None:
                world_guard(x,z,pose['yaw_degrees'])
            if timeline.up.dot(timeline.route_reference_up) < math.cos(math.radians(5)):
                raise RuntimeError('Turn tilt guard')
            result['samples'].append(dict(time=t1,**pose))
            old,t0,a0 = current,t1,a1
            command = turn_command(degrees,pose['yaw_degrees'],.14)
            now = time.monotonic()
            if command is None or reached_at is not None:
                if reached_at is None:
                    reached_at = now
                    call('motors_hold',left=0.,right=0.,**token)
                if t1 >= reached_at:
                    settled.append(dict(time=t1,**pose))
                    settled = [s for s in settled if t1-s['time'] <= .4]
                if len(settled)>=3 and t1-settled[0]['time']>=.18:
                    stable = all(max(s['position_cm'][i] for s in settled)-min(s['position_cm'][i] for s in settled)<=.1 for i in (0,1))
                    stable = stable and max(s['yaw_degrees'] for s in settled)-min(s['yaw_degrees'] for s in settled)<=.5
                    if stable:
                        status = call('status')
                        if any(status[k]!=v for k,v in token.items()) or status['motor']['output'] != [0,0]:
                            raise RuntimeError('Turn cancelled or motors not stopped')
                        path = log+'.settled.jpg'
                        if not cv2.imwrite(path,current):
                            raise RuntimeError('Could not save turn anchor')
                        result.update(outcome='turn_reached_estimate',final_position_cm=[x,z],
                                      settling_samples=settled,final_anchor=dict(image_path=path,captured_monotonic=t1))
                        break
                if now-reached_at>1.:
                    raise RuntimeError('Turn final position not verified')
                continue
            if now-started>3 and abs(pose['yaw_degrees'])<1:
                raise RuntimeError('Turn stalled')
            history=[s for s in result['samples'] if t1-s['time']>=.8]
            if history and abs(pose['yaw_degrees']-history[-1]['yaw_degrees'])<.5:
                raise RuntimeError('Turn stopped making progress')
            if now >= resting_until:
                if time.monotonic()-t1>.18:
                    raise RuntimeError('Turn image processing exceeded 180 ms')
                call('motors_hold',**dict(command,**token))
                time.sleep(.06)
                call('motors_hold',left=0.,right=0.,**token)
                resting_until = time.monotonic()+.18
        else:
            raise RuntimeError('Turn deadline exceeded')
    except Exception as exc:
        result.update(outcome='stopped',reason=str(exc))
    finally:
        try:
            call('stop')
        except Exception as exc:
            result.update(outcome='stopped',stop_error=str(exc))
        try:result['power_end']=call('status').get('power')
        except Exception as exc:result['power_end_error']=str(exc)
        result['elapsed_seconds'] = time.monotonic()-started
        with open(log,'w') as output:
            json.dump(result,output,indent=2)
    return result
