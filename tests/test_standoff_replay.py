import os,sys,math,unittest
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','local_nav'))
import fetch
from evaluate_gpt_routes import Lens
class StandoffReplay(unittest.TestCase):
 def test_saved_final_response_preserves_approach_side(self):
  r=Lens();contact=(286,286)
  g=r.ground(*contact);near=fetch.nearest_range(r,*contact);g=tuple(v*near/math.hypot(*g) for v in g)
  route=[r.ground(*p) for p in [(382,446),(361,382),(333,326),(286,286)]]
  first=fetch.stop_short(route,g,20)
  old=fetch.stop_short(first,g,20)
  self.assertGreater(first[-1][0],0);self.assertLess(old[-1][0],-19)
  fixed=fetch.clip_standoff(first,g,20)
  self.assertAlmostEqual(fixed[-1][0],first[-1][0],places=5)
  self.assertAlmostEqual(fixed[-1][1],first[-1][1],places=5)
 def test_segment_crossing_clipped_even_with_endpoints_outside(self):
  self.assertEqual(fetch.clip_standoff([(0,50)],(0,25),5),[(0.,20.)])
 def test_empty_safe_route_not_revived(self):
  self.assertEqual(fetch.clip_standoff([(0,50)],(0,5),10),[])
 def test_outside_route_unchanged(self):
  route=[(20,10),(20,30)]
  self.assertEqual(fetch.clip_standoff(route,(0,25),5),route)
 def test_actual_mission_does_not_escape_turn_when_close_and_stalled(self):
  from unittest.mock import patch
  from test_planner_motion_contract import FakeRobot, planner_demo
  robot=FakeRobot()
  answer=dict(visible=True,all_done=True,contact_pixel={'x':286,'y':286},route_pixels=[{'x':x,'y':y} for x,y in [(382,446),(361,382),(333,326),(286,286)]],obstacles=[],note='saved route',motion='follow')
  with patch.object(fetch,'Robot',return_value=robot),patch.object(fetch,'Odometer',return_value=object()),patch.object(fetch,'call',return_value={'power':{},'imu':{}}),patch.dict(os.environ,OPENAI_API_KEY='test'):
   bench=planner_demo.Bench(False)
   with patch.object(fetch,'recognize',return_value=answer),patch.object(fetch,'follow',return_value=[0.,0.,0.]) as follow:
    bench.mission('Advil',3)
    self.assertEqual(follow.call_count,1)
    self.assertGreater(follow.call_args[0][2][-1][0],0)
    self.assertIn('stopping without recovery turn',bench.log())
    self.assertNotIn('mission complete',bench.log())
if __name__=='__main__':unittest.main()
