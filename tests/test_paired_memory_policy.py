import sys,json,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'local_nav'))
import fetch
from paired_planning import INSTRUCTIONS,observation,reference_reliability
from evaluate_gpt_routes import Lens

class MemoryPolicy(unittest.TestCase):
    def setUp(self):
        self.lens=Lens();self.lens.pixel=lambda x,z:fetch.Robot.pixel(self.lens,x,z)
        self.image=np.zeros((480,640,3),np.uint8)
    def test_visible_memory_suppressed_offscreen_retained(self):
        m=dict(goal=[0,50],obstacles=[('visible',[0,50]),('behind',[0,-50])])
        _,_,c=observation(self.lens,self.image,m)
        self.assertIsNone(c['target_cm']);self.assertTrue(c['target_requires_fresh_rgb'])
        self.assertEqual([o['label'] for o in c['obstacles']],['behind'])
        m['goal']=[0,-50]
        self.assertEqual(list(observation(self.lens,self.image,m)[2]['target_cm']),[0,-50])
    def test_reference_confidence_does_not_reset_between_looks(self):
        m=dict(reference_distance_cm=100,reference_turn_deg=90)
        c=observation(self.lens,self.image,m,pose=[0,20,30])[2]
        self.assertEqual(c['reference_distance_cm'],120)
        self.assertEqual(c['reference_turn_deg'],120)
        self.assertLess(c['reference_reliability'],reference_reliability(100,90))
    def test_offscreen_target_contract_continues_top_down_belief(self):
        compact=' '.join(INSTRUCTIONS.split())
        self.assertIn('Target loss by itself is NEVER a reason to hold', compact)
        self.assertIn('Continue a plausible remembered route', compact)
        self.assertIn('top-down belief has no target', compact)
    def test_request_does_not_leak_visible_target_memory(self):
        pair=observation(self.lens,self.image,dict(goal=[0,50],route=[[0,10],[30,80]]))
        sent=[]
        class Reply:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self):return json.dumps(dict(status='completed',output=[dict(type='message',content=[dict(type='output_text',text=json.dumps(dict(motion='follow',route_pixels=[])))])])).encode()
        def send(req,**kwargs):sent.append(json.loads(req.data));return Reply()
        with patch.dict('os.environ',{'OPENAI_API_KEY':'test','JETBOT_GPT_AUDIT':''}),patch.object(fetch.urllib.request,'urlopen',send):
            fetch.recognize(self.image,'target',dict(target_was={'contact_pixel':{'x':320,'y':300}}),paired=pair)
        parts=sent[0]['input'][0]['content'];context=json.loads(parts[0]['text'])
        self.assertNotIn('target_was',context['last_time'])
        self.assertNotIn('_previous_route_visualization',context['local_map'])
        points=context['local_map']['previous_route_cm'];distance=0.;last=[0,0]
        for point in points:distance+=float(np.linalg.norm(np.array(point)-last));last=point
        self.assertGreater(distance,15.)
        self.assertIn('only when',context['local_map']['previous_route_role'])
        self.assertEqual(sum(p['type']=='input_image' for p in parts),2)
