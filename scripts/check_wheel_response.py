#!/usr/bin/env python3
"""Bounded powered readiness check: prove wheel torque produces measured yaw."""
import argparse,json,math,sys,time
sys.path.insert(0,'local_nav')
from fetch import call

def wrap(value): return (value+180.)%360.-180.

def pulse(left,right,seconds,token):
    deadline=time.monotonic()+seconds
    observed=[]
    try:
        while time.monotonic()<deadline:
            status=call('status')
            if not status.get('healthy') or not status.get('motion_enabled'):
                raise RuntimeError('service became unhealthy or disarmed')
            observed.append(status.get('motor',{}).get('output'))
            call('motors_hold',left=left,right=right,**token)
            time.sleep(.04)
    finally:
        call('motors_hold',left=0.,right=0.,**token)
        time.sleep(.25)
    return observed

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--duty',type=float,default=.22)
    parser.add_argument('--seconds',type=float,default=.28)
    parser.add_argument('--minimum-degrees',type=float,default=3.)
    args=parser.parse_args()
    if not .10<=args.duty<=.28 or not .1<=args.seconds<=.4:
        raise ValueError('probe bounds exceeded')
    first=call('status');token={k:first[k] for k in ('session_id','control_epoch')}
    if (not first.get('healthy') or not first.get('motion_enabled') or
            first.get('motor',{}).get('output')!=[0,0]):
        raise RuntimeError('probe requires healthy armed service and stopped motors')
    start=float(first['imu']['euler'][0])
    observed=pulse(args.duty,-args.duty,args.seconds,token)
    middle=call('status');outbound=wrap(float(middle['imu']['euler'][0])-start)
    if not any(o and (abs(o[0])>.01 or abs(o[1])>.01) for o in observed):
        raise RuntimeError('wheel-response failed: motor watchdog never reported the accepted nonzero command')
    if abs(outbound)<args.minimum_degrees:
        raise RuntimeError('wheel-response failed: only %.2f deg at %.2f duty'%(outbound,args.duty))
    pulse(-args.duty,args.duty,args.seconds,token)
    final=call('status');returned=wrap(float(final['imu']['euler'][0])-start)
    if final.get('motor',{}).get('output')!=[0,0]:
        raise RuntimeError('motor output nonzero after wheel-response check')
    print(json.dumps(dict(passed=True,duty=args.duty,pulse_seconds=args.seconds,
                          outbound_degrees=outbound,return_error_degrees=returned,
                          final_motor_output=final['motor']['output'],
                          pack_voltage_v=final.get('power',{}).get('pack_voltage_v'))))

if __name__=='__main__':main()
