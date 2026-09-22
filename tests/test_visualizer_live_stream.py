"""Motion-free replay: camera frames must advance while the motion lock is held."""
import base64
import sys
import pathlib
import threading
import unittest
import urllib.request
from unittest.mock import patch
import cv2
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'local_nav'))
import planner_demo as planner

class LiveStreamTests(unittest.TestCase):
 def test_both_http_streams_advance_while_motion_lock_held(self):
  bench=planner.Bench.__new__(planner.Bench)
  bench._live_route=[]
  bench.lock=threading.Lock();bench.lock.acquire()
  class Warp:
   def apply(self,image):return cv2.resize(image,(480,480))
  bench.warp=Warp()
  count=[0]
  def observation(*args,**kwargs):
   count[0]+=1
   frame=np.full((480,640,3),30+(count[0]%4)*40,np.uint8)
   return {'time':float(count[0]),'jpeg_base64':base64.b64encode(cv2.imencode('.jpg',frame)[1]).decode()}
  server=planner.Server(('127.0.0.1',0),planner.Handler);server.bench=bench;server.open_lan=False
  thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  try:
   with patch.object(planner.fetch,'call',side_effect=observation), patch.object(planner,'STREAM_WIDTH',0):
    for view,width in [('camera',640),('floor',480)]:
     with urllib.request.urlopen('http://127.0.0.1:%d/api/stream.mjpg?view=%s'%(server.server_address[1],view),timeout=5) as response:
      frames=[]
      for i in range(2):
       length=None
       while True:
        line=response.readline()
        if line.lower().startswith(b'content-length:'):length=int(line.split(b':')[1])
        if line==b'\r\n' and length is not None:break
       frames.append(response.read(length))
      self.assertNotEqual(frames[0],frames[1])
      self.assertEqual(cv2.imdecode(np.frombuffer(frames[0],np.uint8),1).shape[1],width)
    for view in ['camera','floor']:
     timestamps=[]
     for i in range(2):
      with urllib.request.urlopen('http://127.0.0.1:%d/api/live.jpg?view=%s'%(server.server_address[1],view),timeout=5) as response:
       timestamps.append(float(response.headers['X-Camera-Time']))
       self.assertGreater(len(response.read()),100)
     self.assertGreater(timestamps[1],timestamps[0])
   self.assertTrue(bench.lock.locked())
  finally:server.shutdown();server.server_close();bench.lock.release()
if __name__=='__main__':unittest.main()
