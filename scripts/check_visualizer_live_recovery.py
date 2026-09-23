import threading,subprocess,urllib.request,pathlib,time,json,re,tempfile
from http.server import BaseHTTPRequestHandler,HTTPServer
from socketserver import ThreadingMixIn
counts={'camera':0,'floor':0};failures={'camera':0,'floor':0}
class H(BaseHTTPRequestHandler):
 def do_GET(self):
  path=self.path.split('?')[0]
  if path.startswith('/api/live/'):
   view=path.split('/')[-1].split('.')[0];counts[view]+=1
   if counts[view] in (1,2,6,7):
    failures[view]+=1;self.send_error(503,'Simulated service restart');return
  try:
   with urllib.request.urlopen('http://127.0.0.1:8772'+self.path,timeout=5) as r:
    data=r.read();self.send_response(200)
    for key in ['Content-Type','X-Camera-Time']:
     if r.headers.get(key):self.send_header(key,r.headers[key])
    self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
  except Exception:self.send_error(503)
 def log_message(self,*args):pass
class S(ThreadingMixIn,HTTPServer):daemon_threads=True
s=S(('127.0.0.1',18774),H);threading.Thread(target=s.serve_forever,daemon=True).start()
profile=tempfile.mkdtemp(prefix='jetbot-live-browser-')
p=subprocess.run(['chromium-browser','--user-data-dir='+profile,'--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--virtual-time-budget=12000','--dump-dom','http://127.0.0.1:18774/'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45)
text=p.stdout.decode();pathlib.Path('/tmp/live-view-recovery-dom.html').write_text(text)
print('Requests:',counts,'Injected failures:',failures)
for view in counts:
 tag=re.search(r'<img id="'+view+r'"[^>]+>',text)
 print(view,tag.group(0) if tag else 'MISSING')
 assert tag and 'data-frame-count=' in tag.group(0)
 assert counts[view]>8 and failures[view]==4
print('PASS: both panels resumed after two forced outages.')
s.shutdown()
