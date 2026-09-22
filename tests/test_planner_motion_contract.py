"""No-motor API replay of explicit planner intent, including the real HTTP path."""
import json
import os
import sys
import threading
import time
import unittest
import urllib.request
from unittest.mock import patch
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'local_nav'))
import fetch
import planner_demo
from evaluate_gpt_routes import Lens

class Reply:
    def __init__(self, answer): self.answer=answer
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def read(self): return json.dumps(dict(status='completed',output=[dict(type='message',content=[dict(type='output_text',text=json.dumps(self.answer))])])).encode()

class FakeRobot(Lens):
    def __init__(self):
        super().__init__(); self.dry_run=True; self.speed=7.5; self.outputs=[]
    def frame(self): return np.zeros((480,640,3),np.uint8), {}
    def hold(self,l,r): self.outputs.append((l,r))
    def halt(self): self.outputs.append((0,0))
    def turn(self,degrees): raise AssertionError('Unexpected turn')
    def pixel(self, right, forward): return (320+int(right), 400-int(forward))
    @staticmethod
    def heading(imu): return 0.

class MotionContractTests(unittest.TestCase):
    def recognize(self,answer):
        with patch.dict(os.environ,OPENAI_API_KEY='test'), patch.object(fetch.urllib.request,'urlopen',return_value=Reply(answer)):
            return fetch.recognize(np.zeros((480,640,3),np.uint8),'Approach bottle',
                                   {'route_pixels':[{'x':320,'y':300}]})

    def test_hold_cannot_reuse_even_a_nonempty_route(self):
        with self.assertRaises(fetch.PlannerHold):
            self.recognize(dict(motion='hold',route_pixels=[{'x':320,'y':300}],
                                turn_degrees=90,note='Blocked',visible=False))

    def test_follow_cannot_also_turn(self):
        result=self.recognize(dict(motion='follow',turn_degrees=90,visible=False))
        self.assertIsNone(result['turn_degrees'])

    def test_turn_cannot_also_drive(self):
        result=self.recognize(dict(motion='turn',turn_degrees=-20,route_pixels=[{'x':320,'y':300}],visible=False))
        self.assertEqual(result['route_pixels'],[])

    def test_dashboard_mission_hold_preserves_zero_motion_and_reports_failure(self):
        robot=FakeRobot()
        status={'power':{'pack_voltage_v':12.,'motion_allowed':True},'imu':{}}
        with patch.object(fetch,'Robot',return_value=robot),patch.object(fetch,'Odometer',return_value=object()),patch.object(fetch,'call',return_value=status),patch.dict(os.environ,OPENAI_API_KEY='test'):
            bench=planner_demo.Bench(False)
            bench.live_route=[(0,30)]
            with patch.object(fetch,'recognize',side_effect=fetch.PlannerHold({'note':'Bin fills view'})),patch.object(fetch,'follow') as follow:
                server=planner_demo.Server(('127.0.0.1',0),planner_demo.Handler)
                server.bench=bench;server.open_lan=False
                thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
                try:
                    request=urllib.request.Request('http://127.0.0.1:%d/api/mission'%server.server_address[1],
                        data=b'{"target":"Approach bin","seconds":8}',headers={'Content-Type':'application/json'})
                    with urllib.request.urlopen(request,timeout=3) as response:self.assertTrue(json.load(response)['mission'])
                    if bench.flight:bench.flight.join(2)
                    self.assertFalse(bench.running())
                    follow.assert_not_called()
                    self.assertIn('PLANNER HOLD: Bin fills view',bench.log())
                    self.assertIn('mission ended',bench.log())
                    self.assertNotIn('mission complete',bench.log())
                    self.assertFalse(bench.live_route)
                    self.assertTrue(all(pair==(0,0) for pair in robot.outputs))
                finally:server.shutdown();server.server_close();thread.join(2)

if __name__=='__main__':unittest.main()
