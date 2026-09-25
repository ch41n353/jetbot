#!/usr/bin/env python3
"""Update the current run's newest-first file and serve an auto-refreshing view."""
import datetime
import json
import os
import threading
import time
import urllib.request
import pathlib
import re
import base64
import hashlib
from control_visualizer import calls, interventions, detail, planner_state
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from render_mission_log import render

CURRENT = '/mnt/robotlogs/current-search.json'
lock = threading.Lock()
latest = {'text': 'Waiting for a run.', 'error': None}
PAGE = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>JetBot live mission log</title><style>body{max-width:1100px;margin:auto;padding:24px;background:#111820;color:#e9f1f6;font:15px system-ui}pre{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.5}a{color:#85d6ff}</style>
<h1>Mission log · newest first</h1><p id="status">Connecting…</p><a href="/mission.md">Download Markdown</a><pre id="log"></pre>
<script>async function refresh(){try{let r=await fetch('/api/log',{cache:'no-store'});let d=await r.json();document.getElementById('status').textContent=d.error?'Update failed: '+d.error:'Live · refreshed '+new Date().toLocaleTimeString();if(!d.error)document.getElementById('log').textContent=d.text;}catch(e){document.getElementById('status').textContent='Connection lost: '+e.message}}refresh();setInterval(refresh,2000);</script>'''

def refresh():
    if not os.path.ismount('/mnt/robotlogs'):
        raise RuntimeError('USB volume unavailable')
    with open(CURRENT) as f: root=json.load(f)['root']
    if not os.path.realpath(root).startswith('/mnt/robotlogs/goals/'):
        raise ValueError('Invalid run directory')
    with open(root+'/mission-history.md') as f: history=f.read()
    text=render(history).replace('# Advil mission — newest first','# JetBot mission — newest first')
    path=root+'/mission.md'
    old=''
    if os.path.exists(path):
        with open(path) as f: old=f.read()
    if old!=text:
        with open(path+'.live.tmp','w') as f:f.write(text)
        os.replace(path+'.live.tmp',path)
    with lock: latest.update(text=text,error=None,root=root)

def watch():
    while True:
        try:refresh()
        except Exception as exc:
            with lock:latest['error']=str(exc)
        time.sleep(2)

def saved_local_snapshot(root):
    """Restore the last executor grid after planner restarts or GPT previews."""
    results=list((root/'local-executions').glob('*/result.json'))
    if not results:return None
    result_path=max(results,key=lambda p:p.stat().st_mtime_ns)
    directory=result_path.parent;snapshots=directory/'snapshots'
    floors=list(snapshots.glob('*-floor.jpg'))
    if not floors:return None
    floor=max(floors,key=lambda p:int(p.name.split('-',1)[0]))
    stem=floor.name.split('-',1)[0];camera=snapshots/(stem+'-camera.jpg')
    if not camera.exists():return None
    result=json.loads(result_path.read_text())
    uri=lambda p:'data:image/jpeg;base64,'+base64.b64encode(p.read_bytes()).decode('ascii')
    return dict(id='saved-'+directory.name+'-'+stem,phase=result.get('phase','stopped'),
                image_time=max(camera.stat().st_mtime,floor.stat().st_mtime),
                note='Restored last local executor snapshot from USB; 4 m x 4 m camera-centered grid.',
                retained_plan=True,images=dict(camera=uri(camera),floor=uri(floor)),
                execution=result)

class Handler(BaseHTTPRequestHandler):
    # HTTP/1.0 (the BaseHTTPRequestHandler default) closes after every
    # response, so a dashboard polling a few times a second costs a new TCP
    # connection and a new ThreadingMixIn thread per poll. Measured
    # 2026-09-24: one browser tab held this process at a full core, 480
    # minutes of CPU. Every response below sets Content-Length, which is
    # what keepalive needs to frame the body, so this is safe here.
    protocol_version = 'HTTP/1.1'

    def do_GET(self):
        path=self.path.split('?')[0]
        with lock:state=dict(latest)
        if path=='/clearance-priority':
            try:
                data=pathlib.Path('/mnt/robotlogs/goals/advil-flow-20260922-060659/clearance-ablation-priority-03/comparison.html').read_bytes()
                kind='text/html; charset=utf-8'
            except OSError:
                self.send_error(404, 'Priority comparison unavailable');return
        elif path=='/clearance-ablation':
            try:
                data=pathlib.Path('/mnt/robotlogs/goals/advil-flow-20260922-060659/clearance-ablation-02/comparison.html').read_bytes()
                kind='text/html; charset=utf-8'
            except OSError:
                self.send_error(404, 'Ablation comparison unavailable');return
        elif path=='/clearance-comparison':
            try:
                data=pathlib.Path('/mnt/robotlogs/goals/advil-flow-20260922-060659/clearance-replay-01/comparison.html').read_bytes()
                kind='text/html; charset=utf-8'
            except OSError:
                self.send_error(404, 'Replay comparison unavailable');return
        elif path=='/gallery':
            data=pathlib.Path(__file__).with_name('static').joinpath('gpt_gallery.html').read_bytes()
            kind='text/html; charset=utf-8'
        elif path=='/':
            data=pathlib.Path(__file__).with_name('static').joinpath('control_visualizer.html').read_bytes()
            kind='text/html; charset=utf-8'
        elif path=='/api/highlevel-view':
            try:
                from highlevel_view import view
                payload=view(pathlib.Path(state['root']))
            except Exception as exc:payload=dict(available=False,note='High-level view unavailable: '+str(exc))
            data,kind=json.dumps(payload).encode(),'application/json'
        elif path=='/api/midlevel-view':
            try:
                from midlevel_view import local_view,call_view,seeded_view
                from urllib.parse import parse_qs,urlsplit
                query=parse_qs(urlsplit(self.path).query)
                root=pathlib.Path(state['root'])
                ident=query.get('call',[''])[0]
                if ident:
                    if not re.fullmatch('[a-f0-9]{32}',ident):raise ValueError('Invalid call')
                    directory=root/'gpt'/ident
                    version=tuple(p.stat().st_mtime_ns for p in directory.glob('*.json'))
                    payload=call_view(str(directory),version)
                else:
                    directories=list((root/'local-executions').glob('*/command.json'))
                    if not directories:payload=seeded_view(root)
                    else:
                        command=max(directories,key=lambda p:p.stat().st_mtime);snapshot={}
                        try:
                            with urllib.request.urlopen('http://127.0.0.1:8770/api/planner-snapshot',timeout=2) as response:snapshot=json.load(response)
                        except Exception:pass
                        if snapshot.get('execution',{}).get('execution_id')!=command.parent.name:snapshot={}
                        # local_view is lru_cached on its arguments and reads only
                        # these two fields. Passing the whole snapshot made the key
                        # include every rendered pane image, so it changed on every
                        # poll and the cache never hit -- while also thrashing a
                        # four-entry cache of large payloads.
                        lean={} if not snapshot else dict(
                            source_rgb=snapshot.get('source_rgb'),
                            execution=dict(pose_cm_deg=(snapshot.get('execution') or {}).get('pose_cm_deg')))
                        payload=local_view(str(command.parent),command.stat().st_mtime_ns,
                                           json.dumps(lean,sort_keys=True))
                data,kind=json.dumps(payload).encode(),'application/json'
            except Exception as exc:
                data,kind=json.dumps(dict(available=False,note='Reference view unavailable: '+str(exc))).encode(),'application/json'
        elif path=='/api/planner-snapshot':
            try:
                with urllib.request.urlopen('http://127.0.0.1:8770/api/planner-snapshot',timeout=3) as response:
                    live=json.load(response)
                # ASK/GPT previews and planner restarts must not replace the
                # local controller's durable 4 m executor representation.
                if not live.get('execution'):
                    live=saved_local_snapshot(pathlib.Path(state['root'])) or live
                data=json.dumps(live).encode()
                kind='application/json'
            except Exception:
                self.send_error(503, 'Planner snapshot unavailable');return
        elif path in ('/api/live/camera.jpg', '/api/live/floor.jpg'):
            view = path.split('/')[-1].split('.')[0]
            try:
                with urllib.request.urlopen('http://127.0.0.1:8770/api/live.jpg?view='+view, timeout=3) as response:
                    data = response.read()
                    camera_time = response.headers.get('X-Camera-Time', '')
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Camera-Time', camera_time)
                self.end_headers(); self.wfile.write(data)
            except Exception:
                self.send_error(503, 'Live camera unavailable')
            return
        elif path=='/api/control':
            try:
                if state.get('error'): raise RuntimeError(state['error'])
                root=pathlib.Path(state['root'])
                # `power` is a live pack-voltage string that changes on every
                # poll. Left in this 300 KB body it invalidated the ETag every
                # time, so the slow-changing 99% of the payload could never be
                # revalidated away. It is served on its own below instead.
                planner=planner_state();planner.pop('power',None)
                payload=dict(state, calls=calls(root), interventions=interventions(root), planner=planner)
                data,kind=json.dumps(payload).encode(),'application/json'
            except Exception as exc:
                data,kind=json.dumps({'error':str(exc)}).encode(),'application/json'
        elif path=='/api/power':
            state_=planner_state()
            data=json.dumps(dict(power=state_.get('power'),running=state_.get('running'),
                                 error=state_.get('error'))).encode()
            kind='application/json'
        elif path.startswith('/api/gpt/'):
            match=re.fullmatch(r'/api/gpt/([a-f0-9]{32})/(detail|request.json|response.json|clean.jpg|image-\d+\.jpg)',path)
            if not match or state.get('error') or not state.get('root'):
                self.send_error(404);return
            ident,name=match.groups()
            root=pathlib.Path(state['root'])
            try:
                if name=='detail':
                    value=detail(root,ident)
                    if value is None:raise FileNotFoundError()
                    data,kind=json.dumps(value).encode(),'application/json'
                else:
                    file=root/'gpt'/ident/name
                    if os.path.commonpath([str(file.resolve()),str(root.resolve())])!=str(root.resolve()):
                        raise FileNotFoundError()
                    data=file.read_bytes()
                    kind='image/jpeg' if name.endswith('.jpg') else 'application/json'
            except (OSError,ValueError):
                self.send_error(404);return
        elif path=='/api/log':data,kind=json.dumps(state).encode(),'application/json'
        elif path=='/mission.md':data,kind=state['text'].encode(),'text/markdown; charset=utf-8'
        else:self.send_error(404);return
        # Most panes change far more slowly than the page polls them. Tag every
        # body so an unchanged pane costs a 304 with no body instead of a few
        # hundred KB of JSON to serialise, send and re-parse. 'no-cache' rather
        # than 'no-store': the client must revalidate every time, but it has to
        # be allowed to keep the copy in order to revalidate at all.
        tag='"%s"'%hashlib.md5(data).hexdigest()
        if self.headers.get('If-None-Match')==tag:
            self.send_response(304);self.send_header('ETag',tag)
            self.send_header('Cache-Control','no-cache');self.end_headers();return
        self.send_response(200);self.send_header('Content-Type',kind)
        self.send_header('Content-Length',str(len(data)));self.send_header('ETag',tag)
        self.send_header('Cache-Control','no-cache');self.end_headers();self.wfile.write(data)
    def log_message(self,*args):pass

class Server(ThreadingMixIn,HTTPServer):
    daemon_threads=True
    allow_reuse_address=True

if __name__=='__main__':
    refresh()
    threading.Thread(target=watch,daemon=True).start()
    Server(('0.0.0.0',8772),Handler).serve_forever()
