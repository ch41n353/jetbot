"""No-motor arrival replay through GPT Drive and the real turn executor."""
import sys,unittest,threading,tempfile,os
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'local_nav'),str(Path(__file__).parent)]
import planner_demo,trajectory_executor as ex,fetch
from test_trajectory_executor import Robot

class FacingTests(unittest.TestCase):
    def bench(self):
        b=planner_demo.Bench.__new__(planner_demo.Bench);b.robot=Robot()
        b.robot.halt=lambda:b.robot.hold(0,0)
        b.frame=np.zeros((480,640,3),np.uint8);b.frame_token=1;b.abort=threading.Event();b.lock=threading.Lock();b.lines=[]
        return b
    def test_real_executor_turn_only_and_blocked_sweep(self):
        b=self.bench()
        with tempfile.TemporaryDirectory() as tmp,patch.object(ex,'artifact_root',return_value=Path(tmp)):
            pose=b.align_mission_target([8,16],[])
            self.assertGreater(pose[2],10);self.assertEqual(b.local_execution.travelled,0)
            self.assertEqual(b.robot.outputs[-1],[0,0])
            with self.assertRaises(fetch.Stop):b.align_mission_target([8,16],[('close obstacle',[5,0])])
    def test_mission_reobserves_before_claiming_arrival(self):
        b=self.bench();b.robot.ground=lambda x,y:(x,y)
        b.restage=lambda image:setattr(b,'frame',image)
        b.show_request=lambda *args:None
        b.publish_snapshot=lambda *args:None
        turns=[];observations=[]
        def align(goal,obstacles):turns.append(goal);b.local_execution=type('Job',(),{'id':'replay'})();return [1,0,20]
        b.align_mission_target=align
        def recognize(*args,**kwargs):
            observations.append(len(turns));x=8 if len(observations)==1 else 0
            return dict(motion='follow',goal='red block',visible=True,contact_pixel=dict(x=x,y=16),route_pixels=[dict(x=0,y=10)],obstacles=[],all_done=True,note='replay')
        import paired_planning
        with patch.dict(os.environ,OPENAI_API_KEY='test'),patch.object(fetch,'recognize',recognize),patch.object(fetch,'recall',return_value=None),patch.object(paired_planning,'observation',return_value=(None,None,{})):
            b.mission('red block',seconds=3)
        self.assertEqual(observations,[0,1]);self.assertEqual(len(turns),1)
        self.assertTrue(any('REACHED:' in line for line in b.lines))
