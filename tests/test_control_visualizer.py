import json
import pathlib
import sys
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
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
