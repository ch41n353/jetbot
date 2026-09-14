"""Bounded steady-power 15 degree turn experiment; clear footprint required."""
import json,math,os,time
import numpy as np
from point_controller import call,frame,ROOT
from state_estimator import AttitudeTimeline

def main():
    timeline=AttitudeTimeline(json.load(open(os.path.join(ROOT,'calibration/imu_mount.json'))))
    result={'target_degrees':15,'power':.16,'samples':[]}
    start=None;last_command=None
    try:
        call('stop')
        status=call('status')
        if not status['healthy'] or not status['motion_enabled']:raise RuntimeError('Service not ready for motion')
        _,ts,_=frame(timeline)
        origin=timeline.yaw;up=timeline.up.copy()
        start=time.monotonic()
        call('motors_hold',left=.16,right=-.16);last_command=time.monotonic()
        while True:
            now=time.monotonic()
            if now-start>2:raise RuntimeError('Two-second turn limit')
            if now-last_command>.15:raise RuntimeError('Motor renewal delayed')
            status=call('status')
            if not status['healthy']:raise RuntimeError('Sensor health lost')
            sample=status['imu']
            if sample['time']<=timeline.last:
                time.sleep(.005);continue
            timeline.feed([sample])
            angle=math.degrees(timeline.yaw-origin)
            rate=-math.degrees(float(timeline.gyro.dot(timeline.up)))
            result['samples'].append(dict(time=sample['time'],angle=angle,rate=rate,imu=sample))
            if angle < -2 or angle>20:raise RuntimeError('Turn direction or angle guard')
            if timeline.up.dot(up)<math.cos(math.radians(5)):raise RuntimeError('Tilt guard')
            if now-start>.4 and angle<1:raise RuntimeError('No progress after 400 ms')
            if len(result['samples'])>12 and now-start>.5:
                old=[s for s in result['samples'] if sample['time']-s['time']>=.3]
                if old and angle-old[-1]['angle']<.5:raise RuntimeError('Turn stalled')
            if 15-angle<=max(1.5,max(0,rate)*.05):
                result['outcome']='braked_near_target';break
            call('motors_hold',left=.16,right=-.16);last_command=time.monotonic()
        call('stop')
        result['powered_seconds']=time.monotonic()-start
        time.sleep(.25)
        observation=call('observation',since=timeline.last)
        timeline.feed(observation['imu_samples'])
        result['final_angle_degrees']=math.degrees(timeline.yaw-origin)
    except Exception as exc:result.update(outcome='stopped',reason=str(exc))
    finally:
        try:call('stop')
        except Exception as exc:result['stop_error']=str(exc)
        with open(os.path.join(ROOT,'local_nav/goals/smooth-turn-016.json'),'w') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2))

if __name__=='__main__':main()
