import json
import pathlib
import sys
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
import time
from unittest.mock import patch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
import control_visualizer as visual
import mission_log_server as server

class VisualizerTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.temp.name)
  self.ident='a'*32;d=self.root/'gpt'/self.ident;d.mkdir(parents=True)
  (d/'request.json').write_text(json.dumps({'model':'test','instructions':'exact prompt','input':[{'content':[{'type':'input_text','text':'Approach Advil'},{'type':'input_image','image_url':'data:image/jpeg;base64,eA=='}]}]}))
  (d/'image-0.jpg').write_bytes(b'x')
  (d/'response.json').write_text(json.dumps({'output':[{'content':[{'type':'output_text','text':json.dumps({'motion':'follow','route_pixels':[{'x':320,'y':400}],'obstacles':[]})}]}],'usage':{'total_tokens':5}}))
  (d/'events.jsonl').write_text('\n'.join(json.dumps(x) for x in [{'event':'dispatch','time':'2026-09-20T01:00:00.000000+00:00'},{'event':'response','time':'2026-09-20T01:00:04.000000+00:00'}]))
  (self.root/'commands.jsonl').write_text(json.dumps({'issued_at':'2026-09-20T01:00:00+00:00','url':'/api/mission','body':{'target':'Advil'},'rationale':'visible bottle'})+'\n'+json.dumps({'issued_at':'2026-09-20T01:00:00+00:00','response':{'mission':True}})+'\n{"partial":')
 def tearDown(self):self.temp.cleanup()
 def test_exact_request_image_response_and_timing(self):
  call=visual.calls(self.root)[0];self.assertEqual(call['latency'],4);self.assertEqual(call['answer']['route_pixels'],[{'x':320,'y':400}])
  detail=visual.detail(self.root,self.ident);self.assertEqual(detail['request']['instructions'],'exact prompt');self.assertEqual(detail['request']['input'][0]['content'][1]['image_url'],'/api/gpt/'+self.ident+'/image-0.jpg')
  self.assertIn('base64',json.loads((self.root/'gpt'/self.ident/'request.json').read_text())['input'][0]['content'][1]['image_url'])
  self.assertEqual(len(visual.interventions(self.root)),1)
 def test_partial_response_remains_pending(self):
  (self.root/'gpt'/self.ident/'response.json').write_text('{')
  self.assertEqual(visual.calls(self.root)[0]['status'],'awaiting response')
 def test_saved_local_snapshot_restores_executor_grid(self):
  execution=self.root/'local-executions'/('b'*32);shots=execution/'snapshots';shots.mkdir(parents=True)
  (execution/'result.json').write_text(json.dumps({'phase':'completed','execution_id':'b'*32,'pose_cm_deg':[1,2,3]}))
  (shots/'0012-camera.jpg').write_bytes(b'camera');(shots/'0012-floor.jpg').write_bytes(b'floor')
  snapshot=server.saved_local_snapshot(self.root)
  self.assertTrue(snapshot['id'].endswith('-0012'))
  self.assertEqual(snapshot['execution']['pose_cm_deg'],[1,2,3])
  self.assertIn('4 m x 4 m',snapshot['note'])
  self.assertTrue(snapshot['images']['floor'].startswith('data:image/jpeg;base64,'))
 def test_midlevel_floor_has_independent_pan_and_zoom_controls(self):
  html=pathlib.Path(server.__file__).with_name('static').joinpath('control_visualizer.html').read_text()
  for ident in ('midFloorViewport','midFloorZoom','midZoomIn','midZoomOut','midFloorCenter','midFloorFit'):
   self.assertIn('id="'+ident+'"',html)
  self.assertIn("function zoomMidFloor",html)
 def test_operator_http_path(self):
  server.latest.update(root=str(self.root),text='live mission log',error=None)
  http=server.Server(('127.0.0.1',0),server.Handler);thread=threading.Thread(target=http.serve_forever,daemon=True);thread.start();base='http://127.0.0.1:'+str(http.server_address[1])
  try:
   with patch.object(server,'planner_state',return_value={'running':False,'power':'11 V','log':'stopped'}):
    html=urllib.request.urlopen(base+'/').read();self.assertIn(b'Assistant interventions',html)
    data=json.load(urllib.request.urlopen(base+'/api/control'));self.assertEqual(len(data['calls']),1);self.assertEqual(data['interventions'][0]['rationale'],'visible bottle')
    self.assertEqual(urllib.request.urlopen(base+data['calls'][0]['image']).read(),b'x')
    detail=json.load(urllib.request.urlopen(base+'/api/gpt/'+self.ident+'/detail'));self.assertEqual(detail['response']['usage']['total_tokens'],5)
    with self.assertRaises(urllib.error.HTTPError):urllib.request.urlopen(base+'/api/gpt/../../etc/passwd')
    server.latest['error']='storage unavailable'
    self.assertEqual(json.load(urllib.request.urlopen(base+'/api/control'))['error'],'storage unavailable')
  finally:http.shutdown();http.server_close();thread.join(2);server.latest['error']=None
if __name__=='__main__':unittest.main()
