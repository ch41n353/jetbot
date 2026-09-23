import threading,subprocess,urllib.request,pathlib,json,re,tempfile
from http.server import BaseHTTPRequestHandler,HTTPServer
from socketserver import ThreadingMixIn
sample=json.load(urllib.request.urlopen('http://127.0.0.1:8772/api/planner-snapshot'));assert sample.get('images')
seen=[];live=[]
class H(BaseHTTPRequestHandler):
 def do_GET(self):
  if self.path.startswith('/api/planner-snapshot'):
   phase=['replanning','executing','stopped'][min(len(seen),2)];seen.append(phase);s=dict(sample,id=min(len(seen),3),phase=phase,retained_plan=phase=='stopped');data=json.dumps(s).encode();kind='application/json'
  else:
   if '/api/live/' in self.path:live.append(self.path)
   with urllib.request.urlopen('http://127.0.0.1:8772'+self.path,timeout=5) as r:data=r.read();kind=r.headers.get('Content-Type')
  self.send_response(200);self.send_header('Content-Type',kind);self.end_headers();self.wfile.write(data)
 def log_message(self,*args):pass
class S(ThreadingMixIn,HTTPServer):daemon_threads=True
s=S(('127.0.0.1',18775),H);threading.Thread(target=s.serve_forever,daemon=True).start()
p=subprocess.run(['chromium-browser','--user-data-dir='+tempfile.mkdtemp(prefix='snap-browser-'),'--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--virtual-time-budget=8000','--dump-dom','http://127.0.0.1:18775/'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=35)
dom=p.stdout.decode();pathlib.Path('/tmp/snapshot-browser-dom.html').write_text(dom)
print(p.stderr.decode()[-1500:])
assert seen[:3]==['replanning','executing','stopped'],seen
assert not live,live
assert dom.count('data-snapshot-id="3"')==2
assert 'STOPPED · snapshot 3' in dom
print('PASS: replanning → executing → stopped; both panels have snapshot 3; no live camera requests.')
s.shutdown()
