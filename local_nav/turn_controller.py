#!/usr/bin/env python3
"""Short IMU-measured pivot on an inspected clear robot footprint.

Positive angles turn right. Camera health and expiring motor leases are required.
This does not detect obstacles; inspect the scene between bounded turns.
"""
import argparse
import json
import math
import os
import time
import cv2
import numpy as np
from point_controller import ROOT,call,frame
from state_estimator import AttitudeTimeline


def turn_command(target,angle,power=.12):
    if not math.isfinite(power) or not .12<=power<=.16:
        raise ValueError('Turn power must be between 0.12 and 0.16')
    if not all(math.isfinite(v) for v in (target,angle)) or not 0<abs(target)<=30:
        raise ValueError('Turn target must be finite and within 30 degrees')
    direction=1 if target>0 else -1
    if direction*angle < -3:
        raise RuntimeError('Turn moved in the wrong direction')
    if abs(angle)>abs(target)+5:
        raise RuntimeError('Turn exceeded heading limit')
    remaining=direction*(target-angle)
    if remaining<=1.5:
        return None
    return dict(left=power*direction,right=-power*direction)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--degrees',type=float,required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--power',type=float,default=.12)
    p.add_argument('--log',required=True)
    args=p.parse_args()
    turn_command(args.degrees,0,args.power)
    timeline=AttitudeTimeline(json.load(open(os.path.join(ROOT,'calibration','imu_mount.json'))))
    timeline.record_directory=args.log+'.observations'
    os.makedirs(timeline.record_directory,exist_ok=True)
    result=dict(target_degrees=args.degrees,power=args.power,samples=[],mode='execute' if args.execute else 'preflight')
    start=time.monotonic()
    try:
        call('stop')
        status=call('status')
        if not status['healthy']:raise RuntimeError('Sensors unhealthy')
        if args.execute and not status['motion_enabled']:raise RuntimeError('Motors disabled')
        im,t0,a0=frame(timeline)
        cv2.imwrite(args.log+'.before.jpg',im)
        while time.monotonic()-start<12:
            im,ts,a=frame(timeline)
            if ts-t0>.65:raise RuntimeError('Turn observation gap exceeds 650 ms')
            angle=math.degrees(a['yaw']-a0['yaw'])
            if np.dot(a['down_camera'],a0['down_camera'])<math.cos(math.radians(5)):
                raise RuntimeError('Excessive tilt during turn')
            command=turn_command(args.degrees,angle,args.power)
            result['samples'].append(dict(time=ts,angle_degrees=angle,command=command))
            if command is None:
                result['outcome']='turn_reached_imu_estimate'
                break
            if not args.execute:
                if time.monotonic()-start>1:
                    if abs(angle)>1:raise RuntimeError('Stationary heading unstable')
                    result['outcome']='stationary_preflight_passed'
                    break
                time.sleep(.1)
            else:
                if time.monotonic()-start>3 and abs(angle)<1:
                    raise RuntimeError('Turn made no measured progress')
                call('motors',**command)
                time.sleep(.06)
                call('stop')
                time.sleep(.18)
            t0=ts
        else:raise RuntimeError('Turn timeout')
    except Exception as exc:
        result.update(outcome='stopped',reason=str(exc))
    finally:
        try:call('stop')
        except Exception as exc:result.update(outcome='stopped',stop_error=str(exc))
        result['elapsed']=time.monotonic()-start
        with open(args.log,'w') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2))


if __name__=='__main__':main()
