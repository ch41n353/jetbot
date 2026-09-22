import unittest,sys,pathlib,base64
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'local_nav'))
import planner_demo as p
class Snapshots(unittest.TestCase):
 def test_boundary_pair_and_retained_stop(self):
  b=p.Bench.__new__(p.Bench)
  class Warp:
   def apply(self,image):return image.copy()
  b.warp=Warp();b.frame=np.zeros((480,640,3),np.uint8);b.last_note='clear route';b.to_pixel=lambda x,z,mode:[320+x,480-z];b.log=lambda:'mission ended'
  b.publish_snapshot('replanning',[]);a=b.planner_snapshot
  b.publish_snapshot('executing',[(0,40),(10,80)]);c=b.planner_snapshot
  self.assertEqual(c['phase'],'executing');self.assertEqual(len(c['route_cm']),2)
  self.assertNotEqual(c['images']['camera'],a['images']['camera'])
  self.assertEqual(c['images']['camera'],c['images']['floor'])
  b.frame[:]=255
  self.assertEqual(c['images']['camera'],b.planner_snapshot['images']['camera'])
  b.publish_snapshot('stopped');d=b.planner_snapshot
  self.assertEqual(d['images'],c['images']);self.assertTrue(d['retained_plan']);self.assertGreater(d['id'],c['id'])
  b.publish_snapshot('stopped',[]);self.assertNotEqual(b.planner_snapshot['images'],d['images'])
if __name__=='__main__':unittest.main()
