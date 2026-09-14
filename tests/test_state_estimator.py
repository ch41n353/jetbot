import os,sys,unittest,math
import numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(__file__)),'local_nav'))
from calibrate_imu_mount import solve
from state_estimator import AttitudeTimeline,PlanarState,fuse_yaw,rotation2


def mount():
    m=solve([0,0,9.8],[3.35,0,9.2],14.25)
    m.update(verified=True,gyro_bias_deg_s=[0,0,0],level_acceleration=[0,0,9.8])
    return m

def sample(t,gyro=(0,0,0),acc=(0,0,9.8)):
    return dict(time=t,gyro=gyro,gyro_units='deg/s',acceleration=acc,error=0,system_status=5)

class EstimatorTests(unittest.TestCase):
    def test_quiet_start_averages_and_bounds_tilt(self):
        def samples(degrees):
            a=math.radians(degrees)
            return [sample(1+i*.02,acc=(9.8*math.sin(a),0,9.8*math.cos(a))) for i in range(11)]
        f=AttitudeTimeline(mount())
        with self.assertRaises(RuntimeError):f.feed(samples(4.2)[:1])
        f=AttitudeTimeline(mount());f.initialize_stationary(samples(4.2),1.2)
        self.assertAlmostEqual(math.degrees(math.acos(f.up[2])),4.2,places=2)
        self.assertEqual(f.startup_limit_degrees,4)
        with self.assertRaises(RuntimeError):
            AttitudeTimeline(mount()).initialize_stationary(samples(6),1.2)

    def test_stationary_start_rejects_motion_and_short_history(self):
        for readings in ([sample(1+i*.02,gyro=(0,0,3)) for i in range(11)],
                         [sample(1+i*.02,acc=(i*.1,0,9.8)) for i in range(11)],
                         [sample(1),sample(1.02)]):
            with self.assertRaises(RuntimeError):
                AttitudeTimeline(mount()).initialize_stationary(readings,readings[-1]['time'])
    def test_calibration_geometry(self):
        m=mount();r=np.array(m['imu_to_camera'])
        np.testing.assert_allclose(r@r.T,np.eye(3),atol=1e-6)
        np.testing.assert_allclose(-r@np.array([0,0,1]),[0,math.cos(math.radians(14.25)),math.sin(math.radians(14.25))],atol=1e-6)
    def test_reject_missing_excitation(self):
        with self.assertRaises(ValueError):solve([0,0,9.8],[0,0,9.8],14.25)
    def test_gyro_yaw_and_interpolation(self):
        f=AttitudeTimeline(mount())
        f.feed([sample(1+i*.02,(0,0,10)) for i in range(11)])
        a=f.at(1.15)
        self.assertAlmostEqual(math.degrees(a['yaw']),-1.5,places=5)
        self.assertEqual(a['extrapolation_ms'],0)
    def test_pitch_propagation(self):
        f=AttitudeTimeline(mount())
        f.feed([sample(1+i*.02,(0,-10,0)) for i in range(11)])
        down=f.at(1.2)['down_camera']
        self.assertAlmostEqual(math.degrees(math.atan2(down[2],down[1])),12.25,places=4)
    def test_gap_and_old_frame_fail(self):
        f=AttitudeTimeline(mount());f.feed([sample(1)])
        with self.assertRaises(RuntimeError):f.feed([sample(1.2)])
        with self.assertRaises(RuntimeError):f.at(.9)
        with self.assertRaises(RuntimeError):f.at(1.1)
    def test_translation_and_uncertainty(self):
        state=PlanarState()
        result=state.update(np.eye(2),np.array([0,-2]),.2,dict(residual_cm=.1,yaw_variance=.001))
        self.assertAlmostEqual(result['position_cm'][1],2)
        self.assertGreater(result['position_sigma_cm'],0)
    def test_yaw_disagreement_fails(self):
        with self.assertRaises(RuntimeError):fuse_yaw(math.radians(10),0,.1)
    def test_yaw_fusion_uses_both_sources(self):
        angle,var=fuse_yaw(math.radians(2),math.radians(1),.1)
        self.assertGreater(angle,math.radians(1));self.assertLess(angle,math.radians(2))

class AccelerationGatingTests(unittest.TestCase):
    def test_settled_gravity_corrects_tilt_without_changing_yaw(self):
        f=AttitudeTimeline(mount())
        f.feed([sample(1+i*.02,(0,-10,0)) for i in range(6)])
        f.feed([sample(1+i*.02) for i in range(6,21)])
        raw=f.at(1.4);settled=f.settled_at(1.4)
        self.assertTrue(settled['settled_gravity_used'])
        self.assertGreater(settled['settled_correction_degrees'],.5)
        np.testing.assert_allclose(settled['down_camera'],-f.matrix@np.array([0,0,1]),atol=1e-6)
        self.assertEqual(settled['yaw'],raw['yaw'])
    def test_moving_window_does_not_use_gravity(self):
        f=AttitudeTimeline(mount())
        f.feed([sample(1+i*.02,(0,-10,0)) for i in range(11)])
        self.assertFalse(f.settled_at(1.2)['settled_gravity_used'])
    def test_large_settled_disagreement_stops(self):
        f=AttitudeTimeline(mount())
        f.feed([sample(1+i*.02,(0,-20,0)) for i in range(11)])
        f.feed([sample(1+i*.02) for i in range(11,26)])
        with self.assertRaises(RuntimeError):f.settled_at(1.5)
    def test_brief_impulse_rejected_as_gravity(self):
        f=AttitudeTimeline(mount())
        f.feed([sample(1),sample(1.02,acc=(0,0,13.4)),sample(1.04)])
        self.assertEqual(f.rejected_acceleration_samples,1)
        np.testing.assert_allclose(f.up,[0,0,1])
    def test_sustained_acceleration_stops(self):
        f=AttitudeTimeline(mount());f.feed([sample(1)])
        with self.assertRaises(RuntimeError):f.feed([sample(1+i*.02,acc=(0,0,14)) for i in range(1,8)])
    def test_severe_impulse_stops(self):
        f=AttitudeTimeline(mount());f.feed([sample(1)])
        with self.assertRaises(RuntimeError):f.feed([sample(1.02,acc=(0,0,21))])
