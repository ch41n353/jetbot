#!/usr/bin/env python3
"""Identify sensor axes from level and nose-up holds; never moves motors."""
import argparse,json,math,os,time
import numpy as np
from point_controller import call,ROOT

PATH=os.path.join(ROOT,'calibration','imu_mount.json')


def unit(v):
    v=np.asarray(v,dtype=float)
    if v.shape!=(3,) or not np.isfinite(v).all() or np.linalg.norm(v)<1e-8:
        raise ValueError('Invalid axis')
    return v/np.linalg.norm(v)


def solve(level,lifted,pitch_degrees):
    up=unit(level)
    nose=unit(lifted)
    angle=math.degrees(math.acos(float(np.clip(up.dot(nose),-1,1))))
    if not 8<=angle<=35:raise ValueError('Lift the nose by 8–35 degrees, with no sideways tilt')
    forward=unit(nose-up*up.dot(nose))
    right=unit(np.cross(forward,up))
    p=math.radians(pitch_degrees)
    matrix=np.array([right,-math.sin(p)*forward-math.cos(p)*up,
                     math.cos(p)*forward-math.sin(p)*up])
    if not np.allclose(matrix.dot(matrix.T),np.eye(3),atol=1e-6) or np.linalg.det(matrix)<.99:
        raise ValueError('Invalid axis solution')
    return dict(imu_to_camera=matrix.tolist(),up_imu=up.tolist(),forward_imu=forward.tolist(),nose_up_angle_degrees=angle)


def hold():
    call('stop')
    samples=[]
    last=None
    for _ in range(70):
        s=call('status')
        if not s['healthy'] or s['motion_enabled']:raise ValueError('Calibration requires healthy sensors and a disarmed service')
        i=s['imu']
        if i['time']!=last:
            gyro=np.asarray(i['gyro'])
            if i['gyro_units']=='rad/s':gyro=np.degrees(gyro)
            samples.append((np.asarray(i['acceleration']),gyro))
            last=i['time']
        time.sleep(.03)
    a=np.array([v[0] for v in samples]);g=np.array([v[1] for v in samples])
    if len(a)<30 or np.max(np.std(a,axis=0))>.12 or np.max(np.linalg.norm(g,axis=1))>1.5:
        raise ValueError('Hold the robot still during capture')
    avg=np.median(a,axis=0)
    if not 9<np.linalg.norm(avg)<10.5:raise ValueError('Gravity reading implausible')
    return avg.tolist(),np.median(g,axis=0).tolist()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['level','nose-up','verify-level'])
    args=parser.parse_args()
    acceleration,bias=hold()
    floor=json.load(open(os.path.join(ROOT,'calibration','floor_geometry.json')))
    if args.phase=='level':
        profile=dict(version=1,verified=False,level_acceleration=acceleration,gyro_bias_deg_s=bias,
                     pitch_degrees=floor['pitch_degrees'],camera_height_cm=floor['camera_height_cm'],
                     note='Requires nose-up hold without roll and return-level validation; fixed IMU/camera mount assumed')
    else:
        profile=json.load(open(PATH))
        if args.phase=='nose-up':
            profile.update(solve(profile['level_acceleration'],acceleration,profile['pitch_degrees']))
            profile['verified']=False
        else:
            if 'imu_to_camera' not in profile:raise ValueError('Capture nose-up pose first')
            error=math.degrees(math.acos(float(np.clip(unit(acceleration).dot(unit(profile['level_acceleration'])),-1,1))))
            if error>2:raise ValueError('Robot has not returned to the original level pose')
            profile.update(verified=True,return_level_error_degrees=error,gyro_bias_deg_s=bias)
    with open(PATH+'.tmp','w') as f:json.dump(profile,f,indent=2)
    os.replace(PATH+'.tmp',PATH)
    print(json.dumps(profile,indent=2))

if __name__=='__main__':main()
