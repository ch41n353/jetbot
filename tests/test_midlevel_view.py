"""Replay reference/proposal rendering without hardware or GPT access."""
import sys,json,tempfile,unittest
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from midlevel_view import local_view,call_view
from trajectory_executor import encode

class ReferenceViews(unittest.TestCase):
    def test_local_reference_uses_current_pose_and_clean_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'command.json').write_text(json.dumps({'trajectory':{'waypoints_cm':[[0,50],[20,70]]}}))
            snapshot={'source_rgb':encode(np.zeros((480,640,3),np.uint8)), 'execution':{'pose_cm_deg':[0,10,0]}}
            d=local_view(tmp,0,json.dumps(snapshot))
            self.assertEqual([list(x) for x in d['reference_cm']],[[0,-10],[0,40],[20,60]])
            self.assertEqual(d['source'],'top_level_local_instruction')
            self.assertEqual(set(d['images']),{'camera','floor'})
    def test_call_does_not_invent_top_level_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);cv2.imwrite(str(p/'clean.jpg'),np.zeros((480,640,3),np.uint8))
            prior={'route_pixels':[{'x':320,'y':340},{'x':340,'y':300}]}
            (p/'request.json').write_text(json.dumps({'input':[{'content':[{'type':'input_text','text':json.dumps({'last_time':prior})}]}]}))
            (p/'response.json').write_text(json.dumps({'output':[{'content':[{'type':'output_text','text':json.dumps({'route_pixels':[{'x':310,'y':330},{'x':300,'y':310}]})}]}]}))
            d=call_view(tmp,0)
            self.assertTrue(d['available']);self.assertEqual(len(d['proposal_cm']),2)
            self.assertIn('no separate top-level',d['note'])
            self.assertEqual(len(d['previous_cm']),2)
            self.assertEqual(d['reference_cm'],[])
    def test_paired_context_keeps_three_distinct_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);cv2.imwrite(str(p/'clean.jpg'),np.zeros((480,640,3),np.uint8))
            metric=dict(reference_cm=[[0,20],[20,60]],previous_route_cm=[[1,20],[30,70]],target_cm=[0,-60],
                        obstacles=[dict(label='old box',position_cm=[-20,-40],uncertainty_cm=5)])
            (p/'request.json').write_text(json.dumps({'input':[{'content':[{'type':'input_text','text':json.dumps({'local_map':metric})}]}]}))
            (p/'response.json').write_text(json.dumps({'output':[{'content':[{'type':'output_text','text':json.dumps({'route_pixels':[{'x':320,'y':360},{'x':330,'y':310}], 'obstacles':[{'label':'new box','contact_pixel':{'x':200,'y':340}}]})}]}]}))
            d=call_view(tmp,0)
            self.assertEqual(d['reference_cm'],metric['reference_cm'])
            self.assertEqual(d['previous_cm'],metric['previous_route_cm'])
            self.assertEqual(d['remembered_obstacles'],metric['obstacles'])
            self.assertEqual(len(d['proposal_cm']),2)
            self.assertEqual(set(d['images']),{'camera','floor'})
    def test_missing_clean_image_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(call_view(tmp,0)['available'])
