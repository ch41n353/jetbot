#!/usr/bin/env python3
"""Read-only LAN viewer for the JetBot live camera and USB recordings."""
import json
import os
import subprocess
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import unquote, urlparse

ROOT = '/mnt/robotlogs/recordings'
PAGE = '''<!doctype html><html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>JetBot camera recordings</title>
<style>body{background:#111820;color:#e9f1f6;font:16px system-ui;max-width:1000px;margin:auto;padding:24px}h1{margin-bottom:8px}img,video{width:100%;max-width:640px;background:#000;border-radius:10px}a{color:#85d6ff}button{padding:9px 16px;margin-right:12px;cursor:pointer}li{margin:18px 0}small,p{color:#abbcc8}#player[hidden]{display:none}</style>
<h1>JetBot camera</h1><p>Live view and recordings saved on the USB drive.</p>
<img id="live" alt="Live JetBot camera"><p id="status">Loading recordings…</p>
<section id="player" hidden><h2 id="playing"></h2><video id="video" controls playsinline></video>
<p>Replay starts from the beginning. For an ongoing recording, reopen Replay to include newer footage.</p>
<button id="close">Close replay</button></section>
<h2>Recordings</h2><ul id="files"></ul>
<script>
const live=document.getElementById('live'), video=document.getElementById('video');
live.src=location.protocol+'//'+location.hostname+':8770/api/stream.mjpg';
document.getElementById('close').onclick=()=>{video.pause();video.removeAttribute('src');video.load();document.getElementById('player').hidden=true;live.src=location.protocol+'//'+location.hostname+':8770/api/stream.mjpg'};
async function refresh(){try{const r=await fetch('/api/recordings');if(!r.ok)throw Error(await r.text());const d=await r.json();document.getElementById('status').textContent=d.some(x=>x.growing)?'Recording is receiving video.':'Saved recordings';const ul=document.getElementById('files');ul.replaceChildren();for(const f of d){const li=document.createElement('li'),name=document.createElement('p'),b=document.createElement('button'),a=document.createElement('a');name.textContent=f.name+' — '+(f.bytes/1048576).toFixed(1)+' MB'+(f.growing?' · recording':'');b.textContent='Replay';b.onclick=()=>{live.removeAttribute('src');document.getElementById('playing').textContent=f.name;document.getElementById('player').hidden=false;video.src='/play/'+encodeURIComponent(f.name);video.play().catch(()=>{});};a.textContent='Download original';a.href='/download/'+encodeURIComponent(f.name);li.append(name,b,a);ul.append(li);}}catch(e){document.getElementById('status').textContent='Storage unavailable: '+e.message;}}
refresh();setInterval(refresh,10000);
</script></html>'''

class Handler(BaseHTTPRequestHandler):
    def send(self, code, data, kind):
        self.send_response(code)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path == '/':
            return self.send(200, PAGE.encode(), 'text/html; charset=utf-8')
        if not os.path.ismount('/mnt/robotlogs'):
            return self.send(503, b'USB volume is not mounted', 'text/plain')
        if path == '/api/recordings':
            try:
                rows = []
                for name in sorted(os.listdir(ROOT), reverse=True):
                    if not name.endswith('.mkv'):
                        continue
                    stat = os.stat(os.path.join(ROOT, name))
                    rows.append(dict(name=name, bytes=stat.st_size,
                                     growing=time.time()-stat.st_mtime < 15))
                return self.send(200, json.dumps(rows).encode(), 'application/json')
            except OSError as exc:
                return self.send(503, str(exc).encode(), 'text/plain')
        mode, _, name = path.lstrip('/').partition('/')
        if mode not in ('play', 'download') or name != os.path.basename(name) or not name.endswith('.mkv'):
            return self.send(404, b'Not found', 'text/plain')
        file = os.path.join(ROOT, name)
        if not os.path.isfile(file) or os.path.islink(file):
            return self.send(404, b'Not found', 'text/plain')
        if mode == 'download':
            with open(file, 'rb') as source:
                remaining = os.fstat(source.fileno()).st_size
                self.send_response(200)
                self.send_header('Content-Type', 'video/x-matroska')
                self.send_header('Content-Disposition', 'attachment; filename="'+name+'"')
                self.send_header('Content-Length', str(remaining))
                self.end_headers()
                try:
                    while remaining:
                        chunk = source.read(min(65536, remaining))
                        if not chunk: break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            return
        # Convert to browser-compatible fragmented MP4 in memory; original stays untouched.
        proc = subprocess.Popen(['ffmpeg','-nostdin','-hide_banner','-loglevel','error',
            '-i',file,'-an','-c:v','libx264','-preset','ultrafast','-threads','1',
            '-pix_fmt','yuv420p','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2',
            '-movflags','frag_keyframe+empty_moov+default_base_moof','-f','mp4','pipe:1'],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.send_response(200)
        self.send_header('Content-Type','video/mp4')
        self.send_header('Cache-Control','no-store')
        self.send_header('Connection','close')
        self.end_headers()
        self.close_connection = True
        try:
            while True:
                chunk = proc.stdout.read(32768)
                if not chunk: break
                self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            proc.stdout.close()
            if proc.poll() is None: proc.terminate()
            try: proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()

    def log_message(self, *args):
        pass

class Server(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

if __name__ == '__main__':
    Server(('0.0.0.0', 8771), Handler).serve_forever()
