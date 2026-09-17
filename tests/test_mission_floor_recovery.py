"""Fault-injected motor-off recovery, with the real accumulating PlanarState."""
import math
import os
import sys
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','local_nav'))
from object_mission import recover_floor
from state_estimator import PlanarState


class RecoveryHarness:
    def __init__(self,bridge_cm=1.,persistent=False,cancel=False,tilt=False,
                 clear=True,quiet=True,duplicate=False):
        self.now=1.
        self.bridge_cm=bridge_cm
        self.persistent=persistent
        self.cancel=cancel
        self.tilt=tilt
        self.clear=clear
        self.quiet=quiet
        self.duplicate=duplicate
        self.commands=[]
        self.checked=[]
        self.pairs=[]
        self.state=PlanarState()
        self.state.position=np.array([1.,2.])
        self.state.position_variance=.25
        self.state.yaw_variance=math.radians(1)**2
        self.up=np.array([0.,1.,0.])
        self.states=[]
        self.refresh()

    def refresh(self):
        samples={sample['time']:sample for sample in self.states}
        for i in range(10):
            timestamp=round(self.now-.16+i*.02,9)
            samples[timestamp]=dict(time=timestamp,yaw=0.,acceleration=np.array([0.,9.80665,0.]),
                                    gyro=np.array([0.,0. if self.quiet else .1,0.]))
        self.states=[samples[key] for key in sorted(samples)]

    def motion(self,before,after,*args):
        self.pairs.append((before,after))
        if before=='anchor':
            if self.persistent:raise RuntimeError('Camera tilt/height or tracking changed (scale1.037)')
            return np.eye(2),np.array([0.,-self.bridge_cm]),dict(residual_cm=.05,yaw_variance=.001)
        return np.eye(2),np.zeros(2),dict(residual_cm=.02)

    def frame(self,timeline,settled=False):
        self.now+=.1
        self.refresh()
        if self.tilt:self.up=np.array([0.,math.cos(.2),math.sin(.2)])
        return str(self.now),1. if self.duplicate else self.now,{'yaw':0.}

    def call(self,action,**fields):
        self.commands.append((action,fields))
        if self.cancel:raise RuntimeError('Control cancelled')

    def check(self,body):
        self.checked.append(body)
        return self.clear

    def run(self):
        class Planner:
            clear=staticmethod(self.check)
        with patch('object_mission.time.monotonic',side_effect=lambda:self.now):
            return recover_floor(self,self,'anchor',.9,{'yaw':0.},'failed',1.,{'yaw':0.},np.array([0.,1.,0.]),
                                 self.state,Planner(),dict(session_id='test',control_epoch=0),
                                 self.frame,self.call)


class MissionFloorRecoveryTests(unittest.TestCase):
    def test_bridge_preserves_anchor_and_accumulated_uncertainty(self):
        harness=RecoveryHarness()
        image,timestamp,attitude,r,t,q=harness.run()
        self.assertGreaterEqual(timestamp,1.19)
        self.assertEqual(harness.pairs[-1][0],'anchor')
        np.testing.assert_equal(harness.state.position,[1.,2.])
        self.assertEqual(harness.state.position_variance,.25)
        self.assertGreater(q['residual_cm'],.05)
        harness.state.update(r,t,timestamp-.9,q)
        np.testing.assert_allclose(harness.state.position,[1.,3.])
        self.assertGreater(harness.state.position_variance,.25)
        self.assertEqual(len(harness.checked),22)
        self.assertTrue(all(fields['left']==fields['right']==0 for _,fields in harness.commands))

    def test_persistent_scale_fault_is_not_accepted(self):
        harness=RecoveryHarness(persistent=True)
        with self.assertRaisesRegex(RuntimeError,'deadline'):harness.run()
        np.testing.assert_equal(harness.state.position,[1.,2.])
        self.assertLess(harness.now,2.11)

    def test_nonquiet_sensor_never_bridges(self):
        harness=RecoveryHarness(quiet=False)
        with self.assertRaisesRegex(RuntimeError,'deadline'):harness.run()
        self.assertEqual(harness.pairs,[])

    def test_cancellation_prevents_any_recovery_work(self):
        harness=RecoveryHarness(cancel=True)
        with self.assertRaisesRegex(RuntimeError,'cancelled'):harness.run()
        self.assertEqual(harness.pairs,[])

    def test_tilt_guard_remains_active_between_recovery_frames(self):
        harness=RecoveryHarness(tilt=True)
        with self.assertRaisesRegex(RuntimeError,'tilt guard'):harness.run()
        self.assertNotIn('anchor',[before for before,_ in harness.pairs])

    def test_duplicate_frames_do_not_count_as_quiet_observations(self):
        harness=RecoveryHarness(duplicate=True)
        with self.assertRaisesRegex(RuntimeError,'deadline'):harness.run()
        self.assertEqual(harness.pairs,[])

    def test_coast_rotation_cannot_leave_reserved_envelope(self):
        harness=RecoveryHarness()
        original=harness.frame
        def turning_frame(*args,**kwargs):
            image,timestamp,attitude=original(*args,**kwargs)
            attitude['yaw']=math.radians(21.)
            return image,timestamp,attitude
        harness.frame=turning_frame
        with self.assertRaisesRegex(RuntimeError,'twenty-degree'):harness.run()
        self.assertNotIn('anchor',[before for before,_ in harness.pairs])

    def test_actual_yaw_allows_observation_before_map_verification(self):
        from spatial_planner import sweep
        from route_geometry import contains
        harness=RecoveryHarness()
        start=(1.,2.,0.)
        def envelope(limit):
            boxes=[sweep(start,('turn',angle),precise_turn=True) for angle in (-limit,limit)]
            return [min(b[0] for b in boxes)-4.,min(b[1] for b in boxes)-4.,
                    max(b[2] for b in boxes)+4.,max(b[3] for b in boxes)+4.]
        free=envelope(2.)
        self.assertFalse(contains(free,envelope(20.)))
        def check(body):
            self.assertGreaterEqual(harness.now,1.19)
            return contains(free,body)
        harness.check=check
        harness.run()
        self.assertTrue(all(fields['left']==fields['right']==0 for _,fields in harness.commands))

    def test_missing_endpoint_or_internal_imu_history_rejects(self):
        for mode in ('start','end','gap'):
            harness=RecoveryHarness()
            original=harness.refresh
            def refresh():
                original()
                if mode=='start':harness.states=[s for s in harness.states if s['time']>.9]
                elif mode=='end':harness.states=[s for s in harness.states if s['time']<harness.now-.001]
                else:harness.states=[s for s in harness.states if not .90<s['time']<1.02]
            harness.refresh=refresh
            harness.refresh()
            with self.assertRaisesRegex(RuntimeError,'Incomplete IMU history'):harness.run()
            np.testing.assert_equal(harness.state.position,[1.,2.])

    def test_yaw_excursion_between_frames_is_included_in_envelope(self):
        from spatial_planner import sweep
        harness=RecoveryHarness()
        original=harness.refresh
        def refresh():
            original()
            for sample in harness.states:
                if .94<=sample['time']<=.98:sample['yaw']=math.radians(10.)
        harness.refresh=refresh
        harness.refresh()
        boxes=[sweep((1.,2.,0.),('turn',angle),precise_turn=True) for angle in (-2.,2.)]
        narrow_width=max(b[2] for b in boxes)-min(b[0] for b in boxes)+8.
        harness.check=lambda body:body[2]-body[0]<=narrow_width+1e-9
        with self.assertRaisesRegex(RuntimeError,'coast corridor'):harness.run()

    def test_arrival_floor_fault_recovers_with_plant_imu_history(self):
        import cv2
        from simulate_object_mission import ObjectPlant,run_case
        from point_controller import FloorTracker
        cv2.setNumThreads(1)
        original_frame,original_motion=ObjectPlant.frame,FloorTracker.motion
        context={'plant':None,'injected':False}
        def frame(plant,timeline,settled=False):
            observation=original_frame(plant,timeline,settled=settled)
            context['plant']=plant
            return observation
        def motion(floor,*args,**kwargs):
            plant=context['plant']
            if (plant is not None and not context['injected'] and plant.left==plant.right==0
                    and math.hypot(plant.target[0]-plant.x,plant.target[1]-plant.z)<23
                    and abs(plant.velocity)>1):
                context['injected']=True
                raise RuntimeError('Camera tilt/height or tracking changed (scale1.030 injected)')
            return original_motion(floor,*args,**kwargs)
        with patch.object(ObjectPlant,'frame',frame),patch.object(FloorTracker,'motion',motion):
            run=run_case(dict(speed=11,coast=.05,seed=1,pivot_cm=6),target=(0,60),perception_compute=.045)
        self.assertTrue(context['injected'])
        result=run['result']
        self.assertEqual(result['outcome'],'object_reached_estimate',result.get('reason'))
        self.assertEqual(result['floor_recoveries'],1)
        events=result['events']
        begin=next(event['elapsed_seconds'] for event in events if event['event']=='floor_recovery_started')
        end=next(event['elapsed_seconds'] for event in events if event['event']=='floor_recovered')
        origin=run['commands'][0]['time']
        commands=[command for command in run['commands'] if origin+begin<=command['time']<=origin+end]
        self.assertTrue(commands)
        self.assertFalse(any(command['action']=='motors_hold' and
                             (command['fields']['left']!=0 or command['fields']['right']!=0)
                             for command in commands))
        self.assertEqual(run['clearance_violations'],[])
        self.assertEqual(run['watchdog_stops'],0)
        self.assertEqual(run['final_motor_output'],[0.,0.])
        self.assertLess(abs(run['true_target_range_cm']-20.),3.)

    def test_excessive_coast_and_blocked_corridor_reject(self):
        for harness,reason in [(RecoveryHarness(bridge_cm=4.1),'braking allowance'),
                               (RecoveryHarness(clear=False),'coast corridor')]:
            with self.assertRaisesRegex(RuntimeError,reason):harness.run()
            np.testing.assert_equal(harness.state.position,[1.,2.])


if __name__=='__main__':unittest.main()
