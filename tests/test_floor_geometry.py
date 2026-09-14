import os,sys,unittest
import numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(__file__)),'local_nav'))
from floor_geometry import estimate_pitch,project_floor

class FloorGeometryTests(unittest.TestCase):
    def setUp(self):
        self.i=dict(K=[[300,0,320],[0,300,240],[0,0,1]],D=[0,0,0,0],image_width=640,image_height=480)
        self.p=dict(camera_height_cm=9.5,pitch_degrees=0,roll_degrees=0,mounting_verified=True)
    def test_known_reference(self):
        self.p['pitch_degrees']=estimate_pitch([320,280],9.5,30,self.i)
        point=project_floor([320,280],self.p,self.i)
        self.assertAlmostEqual(point['forward_cm'],30,places=6)
        self.assertAlmostEqual(point['lateral_cm'],0,places=6)
    def test_requires_verification(self):
        self.p['mounting_verified']=False
        with self.assertRaises(ValueError):project_floor([320,300],self.p,self.i)
    def test_horizon_and_bounds(self):
        for pixel in ([320,240],[-1,300],[320,float('nan')]):
            with self.assertRaises(ValueError):project_floor(pixel,self.p,self.i)
    def test_height_scales_distance(self):
        first=project_floor([320,300],self.p,self.i)
        self.p['camera_height_cm']=19
        second=project_floor([320,300],self.p,self.i)
        self.assertAlmostEqual(second['forward_cm'],2*first['forward_cm'])
