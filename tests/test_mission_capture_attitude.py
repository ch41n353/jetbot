"""Capture geometry uses the same stationary IMU initialization as execution."""
import os
import sys
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','local_nav'))
from object_mission import capture_attitude,prepare,floor_alignment
from route_executor import load_json
from point_controller import ROOT


class CaptureAttitudeTests(unittest.TestCase):
    def setUp(self):
        self.mount=load_json(os.path.join(ROOT,'calibration/imu_mount.json'))
        acceleration=np.array(self.mount['level_acceleration'])
        acceleration*=9.80665/np.linalg.norm(acceleration)
        self.capture=dict(metadata_path='capture.json',captured_monotonic=1.,session_id='session',control_epoch=4)
        self.metadata=dict(time=1.,session_id='session',control_epoch=4,imu_samples=[
            dict(time=.8+i*.02,acceleration=acceleration.tolist(),gyro=self.mount['gyro_bias_deg_s'],
                 gyro_units='deg/s',error=0,system_status=5) for i in range(11)])

    def load(self,path):
        return self.metadata if path=='capture.json' else self.mount

    def test_exact_capture_attitude_is_serializable_and_calibrated(self):
        with patch('route_executor.load_json',side_effect=self.load):
            attitude=capture_attitude(self.capture)
        expected=-np.array(self.mount['imu_to_camera'])@np.array(self.mount['level_acceleration'])
        expected/=np.linalg.norm(expected)
        np.testing.assert_allclose(attitude['down_camera'],expected,atol=1e-7)
        self.assertEqual(attitude['time'],1.)
        self.assertIsInstance(attitude['down_camera'],list)

    def test_timestamp_and_generation_mismatch_reject(self):
        for field,value in [('time',1.01),('session_id','other'),('control_epoch',5)]:
            original=self.metadata[field]
            self.metadata[field]=value
            with patch('route_executor.load_json',side_effect=self.load):
                with self.assertRaisesRegex(ValueError,'mismatch'):capture_attitude(self.capture)
            self.metadata[field]=original

    def test_prepare_projects_with_frozen_capture_attitude(self):
        capture=dict(self.capture,image_path='capture.jpg')
        request=dict(image_path='capture.jpg',target_box=[10,10,30,40],target_label='bottle',
                     inspected_free_rectangle_cm=[-20,-20,20,100],obstacle_rectangles_cm=[])
        attitude=dict(time=1.,down_camera=[0.,.95,.3122499],yaw=0.,variance=0.)
        with patch('object_mission.time.monotonic',return_value=1.1), \
             patch('object_mission.validate',return_value=(None,20)), \
             patch('object_mission.capture_attitude',return_value=attitude), \
             patch('route_executor.load_json',return_value={'intrinsics_path':'intrinsics'}), \
             patch('point_controller.FloorTracker') as floor:
            floor.return_value.ground.return_value=np.array([[0.,50.]])
            plan,preview=prepare(capture,request)
        floor.return_value.ground.assert_called_once_with([[20.,40]],attitude)
        self.assertEqual(plan['capture_attitude'],attitude)
        self.assertEqual(preview['capture_attitude'],attitude)
        self.assertEqual(plan['obstacle_rectangles_cm'],[])
        self.assertEqual(plan['target_ground_cm'],[0.,50.])

    def test_local_alignment_rejects_invalid_or_large_rotations(self):
        import cv2
        for matrix in (np.zeros((3,3)),np.diag([1.,1.,-1.]),
                       np.eye(3)*1.01,np.full((3,3),np.nan),
                       cv2.Rodrigues(np.array([np.radians(8.1),0.,0.]))[0]):
            with self.assertRaises(ValueError):
                floor_alignment(dict(floor_alignment_rotation=matrix.tolist()))
        self.assertIsNone(floor_alignment({}))

    def test_local_alignment_changes_preview_geometry_without_changing_capture(self):
        import cv2
        capture=dict(self.capture,image_path='capture.jpg')
        rotation=cv2.Rodrigues(np.array([np.radians(-4.),0.,0.]))[0]
        request=dict(image_path='capture.jpg',target_box=[10,10,30,40],target_label='bottle',
                     inspected_free_rectangle_cm=[-20,-20,20,100],obstacle_rectangles_cm=[],
                     floor_alignment_rotation=rotation.tolist())
        attitude=dict(time=1.,down_camera=[0.,.95,.3122499],yaw=0.,variance=0.)
        original=list(attitude['down_camera'])
        with patch('object_mission.time.monotonic',return_value=1.1), \
             patch('object_mission.validate',return_value=(None,20)), \
             patch('object_mission.capture_attitude',return_value=attitude), \
             patch('route_executor.load_json',return_value={'intrinsics_path':'intrinsics'}), \
             patch('point_controller.FloorTracker') as floor:
            floor.return_value.ground.return_value=np.array([[0.,50.]])
            plan,preview=prepare(capture,request)
        np.testing.assert_allclose(plan['capture_attitude']['down_camera'],rotation@original)
        self.assertEqual(attitude['down_camera'],original)
        self.assertEqual(preview['capture_attitude'],plan['capture_attitude'])
        np.testing.assert_allclose(floor_alignment(plan),rotation)


if __name__=='__main__':unittest.main()
