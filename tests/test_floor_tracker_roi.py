"""Cropped LK keeps full-image geometry and floor exclusion semantics."""
import os,sys,json,math,unittest
import numpy as np
import cv2
sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(__file__)),'local_nav'))
from point_controller import FloorTracker,ROOT
from simulate_route import Plant

class FloorRoiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads=cv2.getNumThreads();cv2.setNumThreads(1)
        with open(os.path.join(ROOT,'calibration/floor_geometry.json')) as f:cls.profile=json.load(f)
        with open(cls.profile['intrinsics_path']) as f:cls.intrinsics=json.load(f)
        cls.plant=Plant(seed=11)
        cls.before=cls.plant.render()
    @classmethod
    def tearDownClass(cls):cv2.setNumThreads(cls.old_threads)
    def compare(self,z,yaw,box=None,attitudes=True,exposure=0):
        plant=self.plant;plant.z=z;plant.yaw=math.radians(yaw)
        after=plant.render();plant.z=plant.yaw=0
        if exposure:after=np.clip(after.astype(float)+exposure,0,255).astype(np.uint8)
        a=dict(time=1.,yaw=0.,down_camera=plant.down)
        b=dict(time=1.1,yaw=math.radians(yaw),down_camera=plant.down)
        outputs=[]
        for crop in (False,True):
            tracker=FloorTracker(self.profile,self.intrinsics,125,crop)
            tracker.excluded_box=box
            outputs.append(tracker.motion(self.before,after,a if attitudes else None,b if attitudes else None))
        np.testing.assert_allclose(outputs[0][0],outputs[1][0],atol=.001)
        np.testing.assert_allclose(outputs[0][1],outputs[1][1],atol=.015)
        self.assertLess(abs(outputs[0][2]['scale']-outputs[1][2]['scale']),.002)
    def test_translation_and_imu_yaw_coordinates(self):
        for z,yaw in ((1.,0.),(-1.,0.),(.5,3.),(.5,-3.)):
            with self.subTest(z=z,yaw=yaw):self.compare(z,yaw)
    def test_no_initial_flow_and_exposure(self):self.compare(.5,0.,attitudes=False,exposure=15)
    def test_exclusions_inside_and_outside_crop(self):
        for box in ([250,310,330,420],[0,0,640,100],[0,200,30,400],[610,200,640,400]):
            with self.subTest(box=box):self.compare(.5,1.,box=box)
    def test_texture_failure_remains_closed(self):
        image=np.full((480,640,3),128,dtype=np.uint8)
        for crop in (False,True):
            with self.assertRaisesRegex(RuntimeError,'Insufficient carpet texture'):
                FloorTracker(self.profile,self.intrinsics,125,crop).motion(image,image)
    def test_explicit_boolean(self):
        with self.assertRaises(ValueError):FloorTracker(self.profile,self.intrinsics,crop_flow=1)

if __name__=='__main__':unittest.main()
