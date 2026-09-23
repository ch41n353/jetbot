"""Offline contract checks for the replay experiment; no API or robot access."""
import base64
import copy
import importlib.util
import json
import pathlib
import unittest

spec=importlib.util.spec_from_file_location('ablation',str(pathlib.Path(__file__).resolve().parents[1]/'scripts/ablate_clearance.py'))
ablation=importlib.util.module_from_spec(spec);spec.loader.exec_module(ablation)

class AblationTests(unittest.TestCase):
    def test_inputs_change_only_intended_factors(self):
        original={'model':'fixed-model','reasoning':{'effort':'none'},'max_output_tokens':600,
                  'instructions':'old','text':{'schema':'fixed'},'input':[{'role':'user','content':[
                      {'type':'input_text','text':json.dumps({'instruction':'Go right of carton','last_time':{'route_pixels':[{'x':1,'y':2}]}})},
                      {'type':'input_image','detail':'high','image_url':'original overlay'}]}]}
        saved=copy.deepcopy(original)
        for v in ablation.VARIANTS:
            body=ablation.prepare(original,b'clean frame','base','extreme',v)
            for k in ('model','reasoning','max_output_tokens','text'):self.assertEqual(body[k],original[k])
            ctx=json.loads(body['input'][0]['content'][0]['text'])
            self.assertEqual(ctx['instruction'],'Go right of carton')
            self.assertEqual('last_time' in ctx,not v.endswith('no_prior'))
            self.assertEqual(body['input'][0]['content'][1]['image_url'],
                             'original overlay' if v.endswith('_reference') else
                             'data:image/jpeg;base64,'+base64.b64encode(b'clean frame').decode())
        self.assertEqual(original,saved)

    def test_crossing_segment_detected_when_both_waypoints_outside(self):
        poly=[[300,240],[340,240],[340,280],[300,280]]
        a={'motion':'follow','route_pixels':[{'x':320,'y':320},{'x':320,'y':200}]}
        self.assertTrue(ablation.score(a,poly)['crosses'])
        a['route_pixels']=[{'x':450,'y':340},{'x':450,'y':200}]
        self.assertFalse(ablation.score(a,poly)['near'])
        a={'motion':'hold','route_pixels':[]}
        self.assertFalse(ablation.score(a,poly)['route'])

if __name__=='__main__':unittest.main()
