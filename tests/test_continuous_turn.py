import math
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(__file__)),'local_nav'))
from continuous_turn import TurnPolicy,execute,validate_status


class PolicyTests(unittest.TestCase):
    def test_signed_arbitrary_targets_and_limits(self):
        self.assertEqual(TurnPolicy(-137).direction,-1)
        self.assertEqual(TurnPolicy(180).target,180)
        for degrees in (0,181,-181,float('nan')):
            with self.assertRaises(ValueError): TurnPolicy(degrees)

    def test_predictive_braking_uses_rate_and_settles(self):
        policy=TurnPolicy(90)
        phase,_=policy.update(.1,1.,84.,60.,0.)
        self.assertEqual(phase,'brake')
        self.assertGreater(policy.history[-1]['predicted_braking_degrees'],3.)
        answer=None
        for i in range(6):
            phase,answer=policy.update(.2+i*.05,1.05+i*.05,89.2,0.,0.)
        self.assertEqual(phase,'settled')
        self.assertAlmostEqual(answer['final_error_degrees'],-.8)

    def test_acceptance_tolerance_does_not_cause_early_braking(self):
        policy=TurnPolicy(30,tolerance=3,brake_margin=1)
        phase,_=policy.update(.45,1.,21.7,57.,0.)
        self.assertEqual(phase,'drive')
        phase,_=policy.update(.48,1.03,24.2,60.,0.)
        self.assertEqual(phase,'brake')

    def test_coast_model_matches_measured_carpet_braking(self):
        # Two powered turns at 0.14: 57.8 deg/s coasted 4.63 deg, 95.5 coasted
        # 7.59. The model must not over-predict enough to brake far too early.
        policy=TurnPolicy(30)
        for rate,measured in ((57.8,4.63),(95.45,7.59)):
            predicted=policy.braking_degrees(rate)
            self.assertGreaterEqual(predicted,measured,'must not brake late')
            self.assertLess(predicted-measured,1.5,
                            'over-prediction at %.1f deg/s forces an early stop'%rate)

    def test_undershoot_is_usable_progress_but_overshoot_is_rejected(self):
        def settle(policy,progress):
            # Commit to braking first, then hold at the final settled angle.
            policy.update(.30,1.,24.,60.,0.)
            answer=None
            for i in range(12):
                phase,answer=policy.update(.35+i*.05,1.05+i*.05,progress,0.,0.)
            return phase,answer
        # Short of the request: inside the already-checked sweep, so usable.
        phase,answer=settle(TurnPolicy(30,tolerance=3),26.4)
        self.assertEqual(phase,'settled')
        self.assertAlmostEqual(answer['progress_error_degrees'],-3.6,places=6)
        self.assertEqual(answer['requested_degrees'],30.)
        # Past the request: can leave the swept corridor the caller verified.
        phase,answer=settle(TurnPolicy(30,tolerance=3),34.2)
        self.assertGreater(answer['progress_error_degrees'],3)

    def test_wrong_way_tilt_rate_stall_overshoot_and_timeout_guards(self):
        cases=[lambda p:p.update(.1,1,-3,0,0),lambda p:p.update(.1,1,1,0,5.1),
               lambda p:p.update(.1,1,1,121,0),lambda p:p.update(.1,1,40,0,0)]
        for invoke in cases:
            with self.assertRaises(RuntimeError): invoke(TurnPolicy(30))
        # A measured scan turn peaks near 80 deg/s and must stay accepted.
        self.assertEqual(TurnPolicy(30).update(.1,1,1,80,0)[0],'drive')
        p=TurnPolicy(30);p.update(.1,1,.1,1,0)
        with self.assertRaisesRegex(RuntimeError,'progress'):p.update(.41,1.1,.2,1,0)
        with self.assertRaisesRegex(RuntimeError,'deadline'):TurnPolicy(30,timeout=1).update(1.1,1,2,2,0)

    def test_generation_health_and_power_guards(self):
        base=dict(session_id='s',control_epoch=2,healthy=True,motion_enabled=True,
                  power=dict(motion_allowed=True,stale=False,service_stop_latched=False))
        validate_status(base,dict(session_id='s',control_epoch=2))
        for change in (dict(control_epoch=3),dict(healthy=False),
                       dict(power=dict(motion_allowed=False))):
            status=dict(base);status.update(change)
            with self.assertRaises(RuntimeError):validate_status(status,dict(session_id='s',control_epoch=2))


class FakeTimeline:
    def __init__(self):
        self.last=None;self.yaw=0.;self.up=np.array([0.,1.,0.]);self.gyro=np.zeros(3)
    def feed(self,samples):
        sample=samples[-1];self.last=sample['time'];self.yaw=math.radians(sample['yaw'])
        # Timeline's yaw-rate extraction is -gyro dot up.
        self.gyro=np.array([0.,-math.radians(sample['rate']),0.])


class ExecuteTests(unittest.TestCase):
    def test_one_process_turn_renews_lease_then_proves_settling(self):
        clock=[0.]; yaw=[0.]; stopped=[False]; commands=[]
        def now(): return clock[0]
        def sleep(dt): clock[0]+=.05
        def call(action,**fields):
            commands.append((action,fields))
            if action=='motors_hold' and fields['left']==0: stopped[0]=True
            if action=='status':
                if not stopped[0] and commands: yaw[0]=min(96.,yaw[0]+8.)
                return dict(session_id='s',control_epoch=1,healthy=True,motion_enabled=True,
                    power=dict(motion_allowed=True,stale=False,service_stop_latched=False),
                    motor=dict(output=[0,0] if stopped[0] or not any(c[0]=='motors_hold' for c in commands) else [.16,-.16]),
                    imu=dict(time=clock[0]+1,yaw=yaw[0],rate=0. if stopped[0] else 60.))
            return dict(session_id='s',control_epoch=1,accepted=True)
        with tempfile.NamedTemporaryFile() as log:
            result=execute(dict(degrees=90,session_id='s',control_epoch=1),log.name,
                           call_fn=call,monotonic=now,sleep=sleep,timeline=FakeTimeline())
        self.assertEqual(result['outcome'],'turn_reached_imu_estimate',result)
        powered=[f for a,f in commands if a=='motors_hold' and f['left']!=0]
        self.assertGreater(len(powered),2)
        self.assertTrue(all(f['control_epoch']==1 for f in powered))
        self.assertGreaterEqual(result['settling_sample_count'],6)
        self.assertEqual(commands[-2][0],'stop')

    def _run_to_settle(self, degrees, settle_at, step=8.):
        """Fake plant: ramps while powered, then coasts to settle_at and holds.

        settle_at must be at or above the progress reached when the controller
        cuts power, so the recorded motion stays physical.
        """
        clock=[0.]; yaw=[0.]; stopped=[False]; commands=[]
        def now(): return clock[0]
        def sleep(dt): clock[0]+=.05
        def call(action,**fields):
            commands.append((action,fields))
            if action=='motors_hold' and fields['left']==0: stopped[0]=True
            if action=='status':
                # Hold still for the pre-turn status so the origin is zero.
                if stopped[0]: yaw[0]=settle_at
                elif any(c[0]=='motors_hold' for c in commands): yaw[0]=yaw[0]+step
                return dict(session_id='s',control_epoch=1,healthy=True,motion_enabled=True,
                    power=dict(motion_allowed=True,stale=False,service_stop_latched=False),
                    motor=dict(output=[0,0] if stopped[0] or not any(c[0]=='motors_hold' for c in commands) else [.14,-.14]),
                    imu=dict(time=clock[0]+1,yaw=yaw[0],rate=0. if stopped[0] else 60.))
            return dict(session_id='s',control_epoch=1,accepted=True)
        with tempfile.NamedTemporaryFile() as log:
            return execute(dict(degrees=degrees,session_id='s',control_epoch=1),log.name,
                           call_fn=call,monotonic=now,sleep=sleep,timeline=FakeTimeline())

    def test_short_turn_is_accepted_so_the_search_can_keep_scanning(self):
        # The real 22:38 failure: 30 deg requested, settled at 26.0.
        result=self._run_to_settle(30,26.)
        self.assertEqual(result['outcome'],'turn_reached_imu_estimate',result.get('reason'))
        self.assertAlmostEqual(result['shortfall_degrees'],4.,places=6)
        self.assertAlmostEqual(result['final_angle_degrees'],26.,places=6)

    def test_gross_undershoot_is_still_a_failure(self):
        # Brakes at 4 of a requested 10 and stops dead: too little rotation to
        # count as scan progress, so it must not be reported as a reached turn.
        result=self._run_to_settle(10,4.,step=2.)
        self.assertEqual(result['outcome'],'stopped')
        self.assertIn('short of',result['reason'])

    def test_cancellation_stops_without_further_power(self):
        clock=[0.]; calls=[]
        def call(action,**fields):
            calls.append((action,fields)); epoch=2 if len(calls)>2 else 1
            return dict(session_id='s',control_epoch=epoch,healthy=True,motion_enabled=True,
                power=dict(motion_allowed=True),motor=dict(output=[0,0]),
                imu=dict(time=len(calls),yaw=0.,rate=0.))
        with tempfile.NamedTemporaryFile() as log:
            result=execute(dict(degrees=30,session_id='s',control_epoch=1),log.name,
                call_fn=call,monotonic=lambda:clock[0],sleep=lambda x:None,timeline=FakeTimeline())
        self.assertIn('cancelled',result['reason'])
        self.assertEqual(calls[-2][0],'stop')

    def test_lease_delay_is_fail_closed(self):
        clock=[0.];calls=[]
        def call(action,**fields):
            calls.append((action,fields))
            return dict(session_id='s',control_epoch=1,healthy=True,motion_enabled=True,
                power=dict(motion_allowed=True),motor=dict(output=[0,0]),
                imu=dict(time=len(calls),yaw=0.,rate=0.))
        def delayed_sleep(_): clock[0]+=.16
        with tempfile.NamedTemporaryFile() as log:
            result=execute(dict(degrees=30),log.name,call_fn=call,
                monotonic=lambda:clock[0],sleep=delayed_sleep,timeline=FakeTimeline())
        self.assertIn('lease renewal delayed',result['reason'])
        self.assertEqual(calls[-2][0],'stop')

    def test_blocked_status_call_cannot_hide_expired_lease(self):
        clock=[0.];calls=[]
        def call(action,**fields):
            calls.append((action,fields))
            if action=='status' and any(a=='motors_hold' for a,_ in calls): clock[0]+=.16
            return dict(session_id='s',control_epoch=1,healthy=True,motion_enabled=True,
                power=dict(motion_allowed=True),motor=dict(output=[0,0]),
                imu=dict(time=len(calls),yaw=0.,rate=0.))
        with tempfile.NamedTemporaryFile() as log:
            result=execute(dict(degrees=30),log.name,call_fn=call,
                monotonic=lambda:clock[0],sleep=lambda _:None,timeline=FakeTimeline())
        self.assertIn('lease renewal delayed',result['reason'])
        self.assertFalse(any(a=='motors_hold' for a,_ in calls[2:]))


if __name__=='__main__':unittest.main()
