#!/usr/bin/env python3
"""One-process, IMU-closed-loop turn using renewable motor leases.

Positive angles turn right.  This controller does not inspect the turn sweep;
the caller must supply a plan made from an inspected, clear footprint.
"""
import argparse
import json
import math
import os
import time


class TurnPolicy:
    """Hardware-independent turn/brake/settle state machine."""
    # Coast measured on carpet at 0.14 power over two powered turns:
    #   57.8 deg/s -> 4.63 deg   (k=0.0801 s)
    #   95.5 deg/s -> 7.59 deg   (k=0.0795 s)
    # Coast is linear in rate; a two-term fit returns a negative quadratic
    # coefficient, so the plant stops abruptly once the lease actually drops and
    # the dominant term is command latency. The old 0.055 s / 700 deg/s^2 pair
    # over-predicted coast by 0.94 deg at 58 deg/s and 4.17 deg at 95 deg/s,
    # braking early enough to settle outside tolerance and abort the search.
    def __init__(self, degrees, power=.16, timeout=8., tolerance=2.,
                 brake_latency=.080, brake_deceleration=4000.,brake_margin=1.):
        values=(degrees,power,timeout,tolerance,brake_latency,brake_deceleration,brake_margin)
        if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in values):
            raise ValueError('Continuous-turn settings must be finite numbers')
        if not 0 < abs(degrees) <= 180:
            raise ValueError('Turn must be nonzero and at most 180 degrees')
        if not .14 <= power <= .16:
            raise ValueError('Turn power must be between 0.14 and 0.16')
        if not 1 <= timeout <= 12 or not .5 <= tolerance <= 3:
            raise ValueError('Invalid timeout or angle tolerance')
        if not .02 <= brake_latency <= .15 or not 100 <= brake_deceleration <= 8000:
            raise ValueError('Invalid braking model')
        if not .25 <= brake_margin <= 2:
            raise ValueError('Invalid braking margin')
        self.degrees=float(degrees); self.direction=1 if degrees>0 else -1
        self.target=abs(float(degrees)); self.power=float(power)
        self.timeout=float(timeout); self.tolerance=float(tolerance)
        self.brake_latency=float(brake_latency); self.brake_deceleration=float(brake_deceleration)
        self.brake_margin=float(brake_margin)
        self.history=[]; self.braking=False; self.settling=[]

    def braking_degrees(self, directed_rate):
        rate=max(0.,directed_rate)
        # Command/lease latency plus a bounded constant-deceleration coast model.
        return min(12.,rate*self.brake_latency+rate*rate/(2*self.brake_deceleration))

    def update(self, elapsed, sample_time, angle, yaw_rate, tilt_degrees):
        values=(elapsed,sample_time,angle,yaw_rate,tilt_degrees)
        if not all(math.isfinite(v) for v in values):
            raise RuntimeError('Nonfinite continuous-turn state')
        if elapsed > self.timeout:
            raise RuntimeError('Continuous-turn deadline exceeded')
        if tilt_degrees > 5:
            raise RuntimeError('Turn tilt guard')
        # Measured chassis yaw, projected onto gravity-up. A 0.14-power scan turn
        # was recorded at 65-80 deg/s, so 90 left no margin for surface variation.
        if abs(yaw_rate) > 120:
            raise RuntimeError('Turn angular-rate guard')
        if self.history and sample_time <= self.history[-1]['time']:
            raise RuntimeError('IMU samples must have increasing timestamps')
        progress=self.direction*angle; rate=self.direction*yaw_rate
        if progress < -2:
            raise RuntimeError('Turn moved in the wrong direction')
        if progress > self.target+5:
            raise RuntimeError('Turn exceeded heading limit')
        sample=dict(time=sample_time,elapsed=elapsed,angle_degrees=angle,
                    yaw_rate_degrees_s=yaw_rate,progress_degrees=progress)
        self.history.append(sample)
        if self.braking:
            self.settling.append(sample)
            self.settling=[s for s in self.settling if sample_time-s['time'] <= .4]
            if (len(self.settling)>=6 and sample_time-self.settling[0]['time']>=.20
                    and max(s['progress_degrees'] for s in self.settling)-min(s['progress_degrees'] for s in self.settling)<=.5
                    and max(abs(s['yaw_rate_degrees_s']) for s in self.settling)<=1.5):
                error=progress-self.target
                return 'settled',dict(final_angle_degrees=angle,
                    final_error_degrees=self.direction*error,
                    # Signed in progress terms for both directions: negative is
                    # short of the request, positive is past it.
                    progress_error_degrees=error,
                    requested_degrees=self.degrees,
                    settled_window_seconds=sample_time-self.settling[0]['time'],
                    settling_sample_count=len(self.settling))
            return 'brake',None
        if elapsed>.4 and progress<1:
            raise RuntimeError('Turn made no measured progress after 400 ms')
        old=[s for s in self.history if sample_time-s['time']>=.30]
        if elapsed>.55 and old and progress-old[-1]['progress_degrees']<.5:
            raise RuntimeError('Turn stalled')
        remaining=self.target-progress
        sample['predicted_braking_degrees']=self.braking_degrees(rate)
        if remaining <= self.brake_margin+sample['predicted_braking_degrees']:
            self.braking=True
            return 'brake',None
        return 'drive',dict(left=self.power*self.direction,right=-self.power*self.direction)


def validate_status(status, token, require_motion=True):
    if any(status.get(k)!=v for k,v in token.items()):
        raise RuntimeError('Turn cancelled or service restarted')
    if not status.get('healthy'):
        raise RuntimeError('Sensor health lost')
    power=status.get('power')
    if (not isinstance(power,dict) or power.get('motion_allowed') is not True
            or power.get('service_stop_latched') or power.get('stale') is True):
        raise RuntimeError('Power guard blocked motion')
    if require_motion and not status.get('motion_enabled'):
        raise RuntimeError('Motion disabled')


def execute(plan, log, call_fn=None, monotonic=None, sleep=None, timeline=None):
    """Execute a validated plan. Dependencies are injectable for motor-free tests."""
    if call_fn is None:
        from point_controller import call as call_fn
    if monotonic is None: monotonic=time.monotonic
    if sleep is None: sleep=time.sleep
    if timeline is None:
        from point_controller import ROOT
        from state_estimator import AttitudeTimeline
        with open(os.path.join(ROOT,'calibration','imu_mount.json')) as source:
            timeline=AttitudeTimeline(json.load(source))
    policy=TurnPolicy(plan['degrees'],plan.get('power',.16),plan.get('timeout_seconds',8.),
                      plan.get('tolerance_degrees',2.),plan.get('brake_latency_seconds',.055),
                      plan.get('brake_deceleration_degrees_s2',700.),
                      plan.get('brake_margin_degrees',1.))
    result=dict(outcome='stopped',target_degrees=policy.degrees,power=policy.power,
                timeout_seconds=policy.timeout,tolerance_degrees=policy.tolerance,
                braking_model=dict(latency_seconds=policy.brake_latency,
                    deceleration_degrees_s2=policy.brake_deceleration,
                    margin_degrees=policy.brake_margin),
                samples=[],settling_samples=[])
    started=monotonic(); powered_at=None; stopped_at=None; last_command=None
    try:
        status=call_fn('status')
        result['power_start']=status.get('power')
        token={k:plan[k] for k in ('session_id','control_epoch') if k in plan}
        if len(token) not in (0,2): raise ValueError('Plan control token is incomplete')
        if not token: token={k:status[k] for k in ('session_id','control_epoch')}
        validate_status(status,token)
        if status.get('motor',{}).get('output') not in ([0,0],(0,0)):
            raise RuntimeError('Turn requires stopped motors')
        timeline.feed([status['imu']]); origin=timeline.yaw; reference_up=timeline.up.copy()
        command=dict(left=policy.power*policy.direction,right=-policy.power*policy.direction)
        call_fn('motors_hold',**dict(command,**token)); powered_at=last_command=monotonic()
        while True:
            now=monotonic()
            if stopped_at is None and now-last_command>.15:
                raise RuntimeError('Motor lease renewal delayed')
            status=call_fn('status'); validate_status(status,token,stopped_at is None)
            now=monotonic()
            if stopped_at is None and now-last_command>.15:
                raise RuntimeError('Motor lease renewal delayed')
            sample=status['imu']
            if sample['time']<=timeline.last:
                sleep(.005); continue
            timeline.feed([sample])
            angle=math.degrees(timeline.yaw-origin)
            rate=-math.degrees(float(timeline.gyro.dot(timeline.up)))
            tilt=math.degrees(math.acos(max(-1.,min(1.,float(timeline.up.dot(reference_up))))))
            phase,evidence=policy.update(now-powered_at,sample['time'],angle,rate,tilt)
            record=dict(policy.history[-1],tilt_degrees=tilt,phase=phase)
            (result['samples'] if stopped_at is None else result['settling_samples']).append(record)
            if phase=='drive':
                call_fn('motors_hold',**dict(evidence,**token)); last_command=monotonic()
            elif stopped_at is None:
                call_fn('motors_hold',left=0.,right=0.,**token); stopped_at=monotonic()
            elif phase=='settled':
                status=call_fn('status'); validate_status(status,token,False)
                if status.get('motor',{}).get('output') not in ([0,0],(0,0)):
                    raise RuntimeError('Motor output nonzero after turn')
                # Asymmetric on purpose. Overshoot can carry the chassis past the
                # swept corridor the caller checked, so it stays a hard failure.
                # Undershoot cannot: it leaves the robot inside that same checked
                # sweep, having simply rotated less. Callers accumulate measured
                # rotation rather than the request, so a short turn is a usable
                # result and only costs one more turn to close the scan.
                shortfall=-evidence['progress_error_degrees']
                if evidence['progress_error_degrees']>policy.tolerance:
                    raise RuntimeError('Settled past turn tolerance by %.2f degrees'
                                       % evidence['progress_error_degrees'])
                if shortfall>max(policy.tolerance,.5*policy.target):
                    raise RuntimeError('Turn settled %.2f degrees short of %.1f; '
                                       'too little rotation to trust as progress'
                                       % (shortfall,policy.target))
                result.update(evidence,outcome='turn_reached_imu_estimate',
                              shortfall_degrees=max(0.,shortfall))
                break
            if stopped_at is not None and monotonic()-stopped_at>1.:
                raise RuntimeError('Final turn angle did not settle')
            sleep(.005)
    except Exception as exc:
        result.update(outcome='stopped',reason=str(exc))
    finally:
        try: call_fn('stop')
        except Exception as exc: result['stop_error']=str(exc)
        try: result['power_end']=call_fn('status').get('power')
        except Exception as exc: result['power_end_error']=str(exc)
        result['powered_seconds']=None if powered_at is None else (stopped_at or monotonic())-powered_at
        result['elapsed_seconds']=monotonic()-started
        with open(log,'w') as output: json.dump(result,output,indent=2)
    return result


def load_plan(args):
    plan={}
    if args.plan:
        with open(args.plan) as source: plan=json.load(source)
    for key,value in (('degrees',args.degrees),('power',args.power),
                      ('timeout_seconds',args.timeout_seconds)):
        if value is not None: plan[key]=value
    if 'degrees' not in plan: raise ValueError('Turn degrees are required')
    TurnPolicy(plan['degrees'],plan.get('power',.16),plan.get('timeout_seconds',8.),
               plan.get('tolerance_degrees',2.),plan.get('brake_latency_seconds',.055),
               plan.get('brake_deceleration_degrees_s2',700.))
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan',nargs='?',help='JSON plan; CLI values override it')
    parser.add_argument('--degrees',type=float)
    parser.add_argument('--power',type=float)
    parser.add_argument('--timeout-seconds',type=float)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--log')
    args=parser.parse_args()
    try: plan=load_plan(args)
    except (ValueError,OSError,TypeError,KeyError) as exc: parser.error(str(exc))
    if not args.execute:
        print(json.dumps(dict(outcome='continuous_turn_preflight_passed',plan=plan,
                              requires_inspected_clear_turn_sweep=True),indent=2)); return 0
    if not args.log: parser.error('--execute requires --log')
    result=execute(plan,args.log)
    print(json.dumps({k:v for k,v in result.items() if k not in ('samples','settling_samples')},indent=2))
    return 0 if result['outcome']=='turn_reached_imu_estimate' else 1


if __name__=='__main__': raise SystemExit(main())
