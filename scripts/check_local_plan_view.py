"""Browser interaction acceptance for local plan zoom/pan; never controls motors."""
import json,subprocess,tempfile,threading,urllib.request
from http.server import BaseHTTPRequestHandler,HTTPServer
from socketserver import ThreadingMixIn

TEST=r'''<script>
(function check(attempt){
const image=document.getElementById('floor'),v=document.getElementById('floorViewport');
if(!image.dataset.snapshotId||!document.getElementById('midCamera').naturalWidth||!document.getElementById('midFloor').naturalWidth){if(attempt<100)return setTimeout(()=>check(attempt+1),200);document.body.dataset.viewTest='FAIL:no snapshot';return}
try{
const assert=(x,m)=>{if(!x)throw Error(m)};
assert(document.getElementById('midCamera').dataset.source==='top_level_local_instruction','missing reference provenance');
assert(document.getElementById('midFloor').naturalWidth===640,'missing top-down reference');
const button=id=>document.getElementById(id).click();
button('zoomIn');button('zoomIn');assert(image.clientWidth>v.clientWidth,'zoom does not overflow');
button('floorCenter');assert(v.scrollLeft>0&&v.scrollTop>0,'center not scrollable');
const before=v.scrollLeft;v.dispatchEvent(new PointerEvent('pointerdown',{pointerId:1,button:0,clientX:150,clientY:150}));
v.dispatchEvent(new PointerEvent('pointermove',{pointerId:1,clientX:100,clientY:110}));
v.dispatchEvent(new PointerEvent('pointerup',{pointerId:1}));assert(v.scrollLeft>before,'drag does not pan');
const old=image.clientWidth;v.dispatchEvent(new WheelEvent('wheel',{deltaY:-100,cancelable:true}));assert(image.clientWidth>old,'wheel does not zoom');
v.scrollLeft=0;assert(v.scrollLeft===0,'left scroll failed');v.scrollLeft=10000;assert(v.scrollLeft>0,'right scroll failed');
button('floorFit');assert(image.clientWidth<=v.clientWidth&&v.scrollLeft===0,'fit failed');
button('zoomIn');const width=image.clientWidth;
setTimeout(()=>{document.body.dataset.viewTest=image.clientWidth===width?'PASS:paired reference RGB/top-down,provenance,zoom,pan,wheel,scroll,fit,poll persistence':'FAIL:poll reset zoom'},2200);
}catch(e){document.body.dataset.viewTest='FAIL:'+e.message}
})(0);
</script>'''
class H(BaseHTTPRequestHandler):
 def do_GET(self):
  try:
   with urllib.request.urlopen('http://127.0.0.1:8772'+self.path,timeout=6) as r:data=r.read();kind=r.headers.get('Content-Type')
   if self.path=='/':data=data.replace(b'</html>',TEST.encode()+b'</html>')
   self.send_response(200);self.send_header('Content-Type',kind);self.end_headers();self.wfile.write(data)
  except Exception:self.send_error(502)
 def log_message(self,*args):pass
class S(ThreadingMixIn,HTTPServer):daemon_threads=True
server=S(('127.0.0.1',0),H);threading.Thread(target=server.serve_forever,daemon=True).start()
try:
 with tempfile.TemporaryDirectory(prefix='local-plan-browser-') as profile:
  result=subprocess.run(['chromium-browser','--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--user-data-dir='+profile,'--virtual-time-budget=30000','--dump-dom','http://127.0.0.1:%d/'%server.server_port],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=55)
  dom=result.stdout.decode();open('/tmp/local-plan-browser-test.html','w').write(dom)
  import re
  answer=re.search(r'data-view-test="([^"]+)"',dom)
  assert answer, 'Browser test did not finish'
  print(answer.group(1));assert answer.group(1).startswith('PASS:')
finally:server.shutdown();server.server_close()
