import json,os,sys,unittest,tempfile
import cv2
import numpy as np
ROOT=os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0,os.path.join(ROOT,'local_nav'))
from point_controller import FloorTracker
from unittest.mock import patch
from point_controller import settled_frame, record_observation

class ObservationTests(unittest.TestCase):
    def test_duplicate_frame_preserves_startup_and_imu_history(self):
        image=np.zeros((480,640,3),dtype=np.uint8)
        first=dict(time=1.,imu_samples=[{'time':.8},{'time':.9}],attitude_initialization='stationary_5deg')
        second=dict(time=1.,imu_samples=[{'time':.9},{'time':1.}])
        with tempfile.TemporaryDirectory() as directory:
            record_observation(directory,first,image)
            record_observation(directory,second,image)
            with open(os.path.join(directory,'1.000000.json')) as source:record=json.load(source)
        self.assertEqual(record['attitude_initialization'],'stationary_5deg')
        self.assertEqual([s['time'] for s in record['imu_samples']],[.8,.9,1.])

class SettledFrameTests(unittest.TestCase):
    def test_waits_for_quiet_observation(self):
        moving=(None,1,{'settled_gravity_used':False})
        quiet=(None,2,{'settled_gravity_used':True})
        with patch('point_controller.frame',side_effect=[moving,quiet]), patch('point_controller.time.sleep'):
            self.assertIs(settled_frame(None),quiet)
    def test_stops_if_quiet_observation_unavailable(self):
        with patch('point_controller.frame',return_value=(None,1,{'settled_gravity_used':False})), patch('point_controller.time.monotonic',side_effect=[0,1]):
            with self.assertRaisesRegex(RuntimeError,'did not settle'):
                settled_frame(None)

class FloorMotionTests(unittest.TestCase):
    def test_known_forward_translation(self):
        p=dict(camera_height_cm=9.5,pitch_degrees=14.25)
        i=dict(K=[[306.8,0,323.4],[0,308,252.1],[0,0,1]],D=[-.045,.045,-.057,.021])
        tracker=FloorTracker(p,i)
        rng=np.random.RandomState(3)
        image=rng.randint(0,256,(480,640),dtype=np.uint8)
        image=cv2.GaussianBlur(image,(3,3),.7)
        before=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        yy,xx=np.mgrid[0:480,0:640]
        ground=tracker.ground(np.column_stack((xx.ravel(),yy.ravel())))
        ground[:,1]+=1.0
        angle=np.radians(p['pitch_degrees'])
        points=np.column_stack((ground[:,0],p['camera_height_cm']*np.cos(angle)-ground[:,1]*np.sin(angle),
            p['camera_height_cm']*np.sin(angle)+ground[:,1]*np.cos(angle)))
        pixels,_=cv2.fisheye.projectPoints(points.reshape(-1,1,3),np.zeros(3),np.zeros(3),np.array(i['K']),np.array(i['D']))
        mapping=pixels.reshape(480,640,2).astype(np.float32)
        after=cv2.remap(before,mapping,None,cv2.INTER_LINEAR)
        r,t,q=tracker.motion(before,after)
        self.assertAlmostEqual(t[0],0,delta=.15)
        self.assertAlmostEqual(t[1],-1,delta=.15)
        self.assertGreater(q['matches'],20)
        # Auto-exposure changes must not hide the known physical translation.
        brighter=cv2.add(after,np.full_like(after,20))
        r,t,q=tracker.motion(before,brighter)
        self.assertAlmostEqual(t[0],0,delta=.15)
        self.assertAlmostEqual(t[1],-1,delta=.15)
        self.assertGreater(q['matches'],20)
    def test_blank_floor_stops(self):
        tracker=FloorTracker({}, {})
        with self.assertRaises(RuntimeError):tracker.motion(np.zeros((480,640,3),np.uint8),np.zeros((480,640,3),np.uint8))

class TiltCompensationTests(unittest.TestCase):
    def test_pitch_change_with_forward_motion(self):
        import math
        p=dict(camera_height_cm=9.5,pitch_degrees=14.25)
        i=dict(K=[[306.8,0,323.4],[0,308,252.1],[0,0,1]],D=[-.045,.045,-.057,.021])
        tracker=FloorTracker(p,i)
        rng=np.random.RandomState(12)
        image=cv2.GaussianBlur(rng.randint(0,256,(480,640),dtype=np.uint8),(3,3),.7)
        before=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        def attitude(deg,t):return dict(down_camera=np.array([0,math.cos(math.radians(deg)),math.sin(math.radians(deg))]),yaw=0,time=t)
        old,new=attitude(14.25,1),attitude(12.25,1.2)
        yy,xx=np.mgrid[0:480,0:640]
        points=np.column_stack((xx.ravel(),yy.ravel()))
        xy=cv2.fisheye.undistortPoints(points.astype(float).reshape(-1,1,2),np.array(i['K']),np.array(i['D'])).reshape(-1,2)
        rays=np.column_stack((xy,np.ones(len(xy))))
        down=new['down_camera'];forward=np.array([0.,0.,1.])-down*down[2];forward/=np.linalg.norm(forward)
        right=np.cross(down,forward)
        ground=np.column_stack((rays@right,rays@forward))*9.5/(rays@down)[:,None]
        ground[:,1]+=1
        angle=math.radians(14.25)
        xyz=np.column_stack((ground[:,0],9.5*math.cos(angle)-ground[:,1]*math.sin(angle),9.5*math.sin(angle)+ground[:,1]*math.cos(angle)))
        pixels,_=cv2.fisheye.projectPoints(xyz.reshape(-1,1,3),np.zeros(3),np.zeros(3),np.array(i['K']),np.array(i['D']))
        mapping=pixels.reshape(480,640,2).astype(np.float32)
        after=cv2.remap(before,mapping,None,cv2.INTER_LINEAR)
        rotation,translation,q=tracker.motion(before,after,old,new)
        self.assertAlmostEqual(translation[1],-1,delta=.2)
        self.assertAlmostEqual(translation[0],0,delta=.2)
        self.assertTrue(.97<q['scale']<1.03)
