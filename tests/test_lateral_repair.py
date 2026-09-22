import os,sys,unittest
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','local_nav'))
import fetch
class LateralRepair(unittest.TestCase):
 def test_clear_routes_preserve_waypoints_on_both_sides(self):
  route=[(0,10),(0,30)]
  limit=fetch.OBSTACLE_RADIUS_CM+fetch.CORRIDOR_HALF_CM
  for sign in [-1,1]:
   fixed,n=fetch.avoid(route,[('side block',(sign*(limit+1),15))])
   self.assertEqual(fixed,route);self.assertEqual(n,0)
 def test_segment_collision_still_triggers_repair(self):
  route=[(0,30)]
  for sign in [-1,1]:
   fixed,n=fetch.avoid(route,[('block',(sign*5,15))])
   self.assertGreater(n,0)
   self.assertNotEqual(fixed,route)
 def test_short_clear_approach_is_not_rounded_or_dropped(self):
  route=[(1,5),(2,10)]
  fixed,n=fetch.avoid(route,[('side',(25,8))],(2,30))
  self.assertEqual(fixed,route)
if __name__=='__main__':unittest.main()
