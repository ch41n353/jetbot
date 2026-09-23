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

class Handler(BaseHTTPRequestHandler):
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
        elif path=='/api/planner-snapshot':
            try:
                with urllib.request.urlopen('http://127.0.0.1:8770/api/planner-snapshot',timeout=3) as response:
                    data=response.read()
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
                payload=dict(state, calls=calls(root), interventions=interventions(root), planner=planner_state())
                data,kind=json.dumps(payload).encode(),'application/json'
            except Exception as exc:
                data,kind=json.dumps({'error':str(exc)}).encode(),'application/json'
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
        self.send_response(200);self.send_header('Content-Type',kind)
        self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
    def log_message(self,*args):pass

class Server(ThreadingMixIn,HTTPServer):
    daemon_threads=True
    allow_reuse_address=True

if __name__=='__main__':
    refresh()
    threading.Thread(target=watch,daemon=True).start()
    Server(('0.0.0.0',8772),Handler).serve_forever()
