#!/usr/bin/env python3
"""One-second continuous-drive probe on visually inspected clear carpet.

No obstacle avoidance. Renew a motor lease only after a valid visual/IMU update.
Use gyro-propagated attitude throughout, never stationary gravity corrections.
"""
import argparse,json,math,os,time
import cv2
import numpy as np
from point_controller import ROOT,FloorTracker,call,frame
from state_estimator import AttitudeTimeline

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--log',required=True)
    parser.add_argument('--distance-cm',type=float,default=5)
    args=parser.parse_args()
    if not 1<=args.distance_cm<=15:parser.error("Distance must be 1 to 15 cm")
    profile=json.load(open(os.path.join(ROOT,'calibration/floor_geometry.json')))
    tracker=FloorTracker(profile,json.load(open(profile['intrinsics_path'])))
    timeline=AttitudeTimeline(json.load(open(os.path.join(ROOT,'calibration/imu_mount.json'))))
    timeline.record_directory=args.log+'.observations'
    os.makedirs(timeline.record_directory,exist_ok=True)
    result=dict(mode='continuous' if args.execute else 'stationary',samples=[])
    drive_start=None;last_command=None;distance=0.;yaw=0.
    try:
        call('stop')
        old,t0,_=frame(timeline);a0=timeline.at(t0)
        cv2.imwrite(args.log+'.before.jpg',old)
        start=time.monotonic()
        while time.monotonic()-start<3:
            current,t1,_=frame(timeline);a1=timeline.at(t1)
            if t1<=t0:
                time.sleep(.005)
                continue
            dt=t1-t0
            if dt>.18:raise RuntimeError('Frame interval exceeds continuous-drive limit')
            if last_command is not None and time.monotonic()-last_command>.18:
                raise RuntimeError('Control update too slow to maintain continuous lease')
            r,t,q=tracker.motion(old,current,a0,a1)
            distance+=float(np.linalg.norm(t))
            yaw+=math.atan2(r[1,0],r[0,0])
            result['samples'].append(dict(time=t1,dt=dt,travel_cm=distance,yaw_degrees=math.degrees(yaw),**q))
            if abs(yaw)>math.radians(5):raise RuntimeError('Heading deviation exceeds five degrees')
            if distance>=args.distance_cm or (drive_start is not None and time.monotonic()-drive_start>=2):
                result['outcome']='bounded_probe_completed'
                break
            if args.execute:
                steer=float(np.clip(-.3*yaw,-.025,.025))
                if last_command is not None and time.monotonic()-last_command>.18:
                    raise RuntimeError('Processing exceeded continuous lease budget')
                call('motors_hold',left=.16+steer,right=.16-steer)
                last_command=time.monotonic()
                if drive_start is None:drive_start=last_command
            elif time.monotonic()-start>=1:
                if distance>.5:raise RuntimeError('Stationary tracking drift')
                result['outcome']='stationary_preflight_passed'
                break
            old,t0,a0=current,t1,a1
        else:raise RuntimeError('Probe timeout')
    except Exception as exc:result.update(outcome='stopped',reason=str(exc))
    finally:
        try:call('stop')
        except Exception as exc:result.update(outcome='stopped',stop_error=str(exc))
        result['travel_cm']=distance
        result['drive_elapsed']=None if drive_start is None else time.monotonic()-drive_start
        with open(args.log,'w') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2))

if __name__=='__main__':main()
