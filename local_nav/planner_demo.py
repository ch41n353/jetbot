#!/usr/bin/env python3
"""Draw a trajectory on the camera image and watch the local planner drive it.

A bench for the execution half of the stack, with the recognizer taken out of
the loop: you supply the path, so whatever the robot gets wrong is the planner,
the odometry or the plant, not a model's opinion about where an object is.

    turn left/right by N degrees
    take a picture, click a path on the carpet, run it

Both feedback loops are on while it drives: the IMU holds heading by trimming
the wheels every cycle, and carpet optical flow measures distance every 0.12 s.
Each leg reports what was asked for against what was measured, so the gap
between the two is the thing you are actually looking at.

Loopback only unless --host is given. Motors need --enable-motion.
"""
import argparse
import base64
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import urlparse

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import explore
import fetch

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Local planner bench</title><style>
*{box-sizing:border-box}
body{margin:0;background:#0a0d10;color:#c8d2d8;font:13px ui-monospace,SFMono-Regular,Menlo,monospace}
header{padding:10px 16px;border-bottom:1px solid #1e2830;display:flex;gap:18px;align-items:center}
h1{font-size:13px;margin:0;color:#d6ff3f;letter-spacing:.08em}
main{display:grid;grid-template-columns:500px 1fr;gap:16px;padding:16px}
.stage{position:relative;width:480px;height:480px;background:#05070a}
.stage.camera{width:640px;height:480px}
/* Scoped to the stage on purpose: a bare img,canvas rule also catches the
   filmstrip in the side panel and stacks it on top of the live view. */
.stage img,.stage canvas{position:absolute;left:0;top:0;width:480px;height:480px}
/* No display here: an id selector outranks [hidden], so setting .hidden would
   stop working and the element would show as a broken image with no frames. */
#strip,#livecam{position:static;width:100%;height:auto;border-radius:8px}
[hidden]{display:none!important}
.stage.camera img,.stage.camera canvas{width:640px;height:480px}
canvas{cursor:crosshair}
.controls{display:flex;flex-wrap:wrap;gap:8px 14px;margin-top:10px;align-items:center}
/* Related controls stay on one line together; only whole groups wrap, so
   ASK GPT never ends up on a different row from the box it reads. */
.group{display:flex;gap:6px;align-items:center}
#target{flex:1;min-width:150px}
main{display:flex;gap:18px;align-items:flex-start}
/* The left column is exactly as wide as the picture it holds, so the controls
   wrap under it in whole groups instead of stretching the column and pushing
   the log off the screen. */
.pane{flex:0 0 auto;width:480px}
.pane.camera{width:640px}
.side{flex:1;min-width:300px;display:flex;flex-direction:column;gap:10px}
#log{flex:1;min-height:420px;overflow:auto;white-space:pre-wrap}
button{background:#141b22;color:#c8d2d8;border:1px solid #2b3742;padding:7px 12px;
 font:12px ui-monospace,monospace;cursor:pointer}
button:hover{border-color:#d6ff3f;color:#d6ff3f}
button.go{border-color:#d6ff3f;color:#d6ff3f}
button:disabled{opacity:.4;cursor:default}
input{width:64px;background:#05070a;color:#c8d2d8;border:1px solid #2b3742;padding:6px}
.side{min-width:0}
pre{background:#05070a;border:1px solid #1e2830;padding:10px;max-height:560px;
 overflow:auto;white-space:pre-wrap;font-size:11px;line-height:1.5;margin:0}
.row{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:8px;font-size:11px;color:#7d8a93}
.row b{color:#c8d2d8;font-weight:500}
.hint{color:#7d8a93;font-size:11px;margin-top:6px;line-height:1.6}
.k{display:inline-block;width:9px;height:9px;vertical-align:middle;margin-right:4px}
</style></head><body>
<header><h1>LOCAL PLANNER BENCH</h1>
<span id="mode" class="hint"></span><span id="power" class="hint"></span></header>
<main>
<div id="pane" class="pane">
  <div class="stage"><img id="shot"><canvas id="pad" width="480" height="480"></canvas></div>
  <div class="controls">
    <div class="group">
      <button id="grab">TAKE PICTURE</button>
      <button id="view">VIEW: FLOOR</button>
    </div>
    <div class="group">
      <span>turn</span><input id="deg" type="number" value="30" min="1" max="180">
      <button id="left">&#8630; LEFT</button><button id="right">RIGHT &#8631;</button>
    </div>
    <div class="group">
      <input id="target" type="text" style="min-width:300px"
        placeholder="tell it what to do"
        value="reach the can of nuts"
        ><button id="ask">ASK GPT</button>
      <button id="go" class="go">GPT DRIVE</button>
      <span>drive</span><input id="every" type="number" min="1" max="60"
        placeholder="once" style="width:62px" title="seconds of driving between
        looks; leave empty for a single look and drive"><span>s</span>
    </div>
    <div class="group">
      <button id="flowgo" class="go">FLOW</button>
      <button id="find" class="go">FIND IT</button>
      <span>look every</span><input id="flowevery" type="number" min="1" max="30"
        value="1" style="width:56px" title="how often a look is issued; several
        run at once and the wheels never stop"><span>s</span>
    </div>
    <div class="group">
      <button id="run" class="go">RUN TRAJECTORY</button>
      <button id="clear">CLEAR</button><button id="stop">STOP</button>
    </div>
  </div>
  <div class="hint" id="floorhint">
    Looking down on the 2 m of floor in front of the robot; grid lines every
    25 cm, heavier every metre. A centimetre is the same size anywhere on this
    view, so spacing your clicks evenly really does space the waypoints evenly.
    <span class="k" style="background:#43d4ff"></span>path to drive
    <span class="k" style="background:#ffa53d"></span>obstacle GPT reported
    <span class="k" style="background:#9d7bff"></span>same points, reprojected<br>
    The picture you draw on stays frozen, so waypoints keep meaning the floor
    they were placed on. START the live camera in the panel beside it to watch
    what the robot sees now, during a run included.<br>
    VIEW switches between this plan view and the raw camera image. Draw on the
    camera image, run it, and the purple markers are those same floor points
    redrawn in the new photograph &mdash; they should land on the same carpet.<br>
    The box takes an instruction in plain words, not just a name &mdash; "reach
    the can of nuts", "go to the pink bin by the desk". The model reads the goal
    out of it and the log says what it understood, so a misread shows up before
    the wheels turn. Obstacles are always avoided: clearance is enforced here
    rather than by the model, and an instruction to ignore something on the
    floor does not change that.<br>
    ASK GPT fills in the waypoints instead of you drawing them. Its route is widened to the clearance the wheels need before it is
    shown &mdash; the model has no scale, so the margins are ours.<br>
    GPT DRIVE with a number in the box drives for that many seconds, stops, asks again
    from where it now stands &mdash; handing the model its own unreached
    waypoints redrawn in the new picture &mdash; and repeats until it arrives.
    If the object leaves the frame it keeps following the remembered route
    rather than giving up, and it keeps avoiding obstacles that have gone out of
    shot. Leave the box empty for a single look and drive. STOP ends it.<br>
    GPT DRIVE's <em>drive N s</em> is how long to drive between looks; leave it
    empty for a single look and drive. FLOW's <em>look every N s</em> is how
    often a look is issued &mdash; smaller is more responsive and costs more.<br>
    FLOW never stops the wheels: it keeps several looks in flight, so a fresh
    trajectory lands about once a second and takes over the moment it does. Each one is reprojected by the distance
    driven while it was being thought about, and an answer that arrives out of
    order, describing an older picture than the plan already in use, is thrown
    away.<br>
    Dark areas are floor the camera cannot see. Near the top the picture is
    built from very few camera pixels, so it looks smeared &mdash; that blur is
    a fair picture of how little the robot really knows out there.
  </div>
  <div class="hint" id="camhint" hidden>
    The raw camera image. Clicks are projected onto the carpet through the lens,
    so a click near the bottom is worth a couple of centimetres and the same
    click near the horizon is worth tens &mdash; at this camera height the lower
    half of the picture is only the first 34 cm of floor. Draw here, run it,
    then step the frames below to see those same floor points redrawn from
    wherever the robot then stood.
  </div>
</div>
<div class="side">
  <div class="row"><span>waypoints <b id="n">0</b></span><span>path <b id="len">0</b> cm</span>
    <span>heading <b id="hdg">0.0</b>&deg;</span><span>speed <b id="spd">-</b> cm/s</span></div>
  <div class="row"><span>live camera <b id="livestate">off</b></span>
    <button id="live">START</button>
    <span class="hint">what the robot sees now, with your route drawn into it
    &mdash; including while it drives</span></div>
  <img id="livecam" hidden>
  <div class="row" id="stripbar" style="display:none">
    <button id="prev">&#8592;</button><span>frame <b id="si">1</b>/<b id="sn">1</b></span>
    <button id="next">&#8594;</button><span class="hint">route redrawn from where
    the robot stood at each waypoint</span></div>
  <img id="strip" hidden>
  <div class="row"><span>model says</span><b id="note">&mdash;</b></div>
  <pre id="log">Take a picture, then click a path on the carpet.</pre>
</div>
</main>
<script>
const pad=document.getElementById('pad'), ctx=pad.getContext('2d');
let points=[], busy=false, hazards=[], reprojected=false, mode='floor';
function draw(){
  ctx.clearRect(0,0,pad.width,pad.height);
  if(points.length){
    const tint = reprojected ? '#9d7bff' : '#43d4ff';
    ctx.strokeStyle=tint; ctx.lineWidth=2.5; ctx.setLineDash([9,5]);
    ctx.beginPath(); ctx.moveTo(pad.width/2,pad.height-1);
    points.forEach(p=>ctx.lineTo(p[0],p[1])); ctx.stroke(); ctx.setLineDash([]);
    points.forEach((p,i)=>{ctx.beginPath();ctx.arc(p[0],p[1],5,0,7);
      if(p[2]){ctx.strokeStyle=tint;ctx.lineWidth=2;ctx.stroke();}
      else {ctx.fillStyle=tint;ctx.fill();}
      ctx.fillStyle=p[2]?tint:'#0a0d10';ctx.font='10px monospace';
      ctx.fillText(i+1,p[0]-2,p[1]+3);});
  }
  hazards.forEach(h=>{
    ctx.strokeStyle='#ffa53d'; ctx.lineWidth=1.5;
    ctx.beginPath(); ctx.arc(h[0],h[1],h[2],0,7); ctx.stroke();
    ctx.fillStyle='#ffa53d'; ctx.beginPath(); ctx.arc(h[0],h[1],3,0,7); ctx.fill();
    ctx.font='10px monospace'; ctx.fillText(h[3],h[0]+7,h[1]-5);
  });
  document.getElementById('n').textContent=points.length;
}
pad.addEventListener('click',e=>{
  if(busy) return;
  const r=pad.getBoundingClientRect();
  if(reprojected){ points=[]; reprojected=false; }
  points.push([Math.round(e.clientX-r.left),Math.round(e.clientY-r.top)]);
  draw(); project();
});
async function post(path,body){
  busy=true; buttons();
  try{
    const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body||{})});
    const d=await r.json(); show(d); return d;
  }catch(err){ document.getElementById('log').textContent='request failed: '+err; }
  finally{ busy=false; buttons(); }
}
// 'live' is deliberately absent: the stream is the one thing that should keep
// running while the robot is busy.
function buttons(){['grab','left','right','run','clear','ask','view','go','flowgo','find'].forEach(
  id=>document.getElementById(id).disabled=busy);}
function show(d){
  if(d.log) document.getElementById('log').textContent=d.log;
  if(d.frame) document.getElementById('shot').src='/api/frame?t='+Date.now();
  if(d.heading!==undefined) document.getElementById('hdg').textContent=d.heading.toFixed(1);
  if(d.speed!==undefined) document.getElementById('spd').textContent=d.speed.toFixed(1);
  if(d.length!==undefined) document.getElementById('len').textContent=d.length.toFixed(0);
  if(d.hazards){ hazards=d.hazards; draw(); }
  if(d.points){ points=d.points; reprojected=!!d.reprojected; draw(); }
  if(d.strip!==undefined){ strip=d.strip; si=0; showStrip(); }
  if(d.view_mode && d.view_mode!==mode){
    // The server remembers the view across reloads; the page must not assume
    // FLOOR, or a 640-wide camera image lands in a 480-wide stage and every
    // click is projected from the wrong pixel.
    mode=d.view_mode; applyView();
  }
  if(d.note!==undefined) document.getElementById('note').textContent=d.note;
  if(d.power) document.getElementById('power').textContent=d.power;
  if(d.mode) document.getElementById('mode').textContent=d.mode;
}
async function project(){ show(await post('/api/project',{points})); }
document.getElementById('grab').onclick=async()=>{points=[];hazards=[];draw();
  show(await post('/api/capture'));};
// The drawing surface stays frozen on purpose: waypoints are placed against
// the picture they were drawn on, and refreshing it under the cursor would
// move the floor out from under them. The live feed goes beside it instead.
let wantLive=false, retry=null;
function liveState(text){document.getElementById('livestate').textContent=text;}
function startLive(){
  const img=document.getElementById('livecam');
  img.hidden=false; img.src='/api/stream.mjpg?t='+Date.now();
  liveState('on');
}
// An MJPEG connection dies whenever the bench restarts, and a dead <img> just
// keeps showing its last frame -- looking exactly like a camera that has frozen
// or an overlay that never appears. So reconnect rather than sit there.
document.getElementById('livecam').onerror=()=>{
  if(!wantLive) return;
  liveState('reconnecting...');
  clearTimeout(retry); retry=setTimeout(startLive, 1500);
};
document.getElementById('live').onclick=()=>{
  const img=document.getElementById('livecam'), b=document.getElementById('live');
  wantLive=!wantLive;
  if(!wantLive){
    clearTimeout(retry); img.src=''; img.hidden=true;
    b.textContent='START'; liveState('off'); return;
  }
  b.textContent='STOP'; startLive();
};
let strip=0, si=0;
function showStrip(){
  const bar=document.getElementById('stripbar'), img=document.getElementById('strip');
  if(!strip){ bar.style.display='none'; img.hidden=true; return; }
  bar.style.display=''; img.hidden=false;
  si=Math.max(0,Math.min(si,strip-1));
  // Only ever point at a frame that exists; an empty src renders as broken.
  document.getElementById('si').textContent=si+1;
  document.getElementById('sn').textContent=strip;
  img.src='/api/step?i='+si+'&t='+Date.now();
}
document.getElementById('prev').onclick=()=>{si--;showStrip();};
document.getElementById('next').onclick=()=>{si++;showStrip();};
function applyView(){
  document.getElementById('view').textContent =
    'VIEW: '+(mode==='floor'?'FLOOR':'CAMERA');
  // The canvas backing store has to match the picture underneath it: the plan
  // view is 480 square, the camera image is 640x480. Getting this wrong puts
  // every click and every drawn marker in the wrong place.
  document.querySelector('.stage').classList.toggle('camera', mode==='camera');
  document.getElementById('pane').classList.toggle('camera', mode==='camera');
  document.getElementById('floorhint').hidden = mode==='camera';
  document.getElementById('camhint').hidden = mode!=='camera';
  pad.width = mode==='camera' ? 640 : 480;
  pad.height = 480;
  draw();
}
document.getElementById('view').onclick=async()=>{
  mode = mode==='floor' ? 'camera' : 'floor';
  points=[]; hazards=[]; reprojected=false;
  applyView();
  await post('/api/look',{mode});
};
let watching=null, seenFrame=-1;
function watch(){
  clearInterval(watching);
  watching=setInterval(async()=>{
    try{
      const d=await (await fetch('/api/progress')).json();
      if(d.log) document.getElementById('log').textContent=d.log;
      // Draw what the model just said onto the still picture too, not only
      // into the stream: this panel is where you look to see its answer.
      if(d.points){ points=d.points; reprojected=false; }
      if(d.hazards) hazards=d.hazards;
      if(d.points||d.hazards) draw();
      if(d.note!==undefined) document.getElementById('note').textContent=d.note;
      if(d.frame_token!==undefined && d.frame_token!==seenFrame){
        seenFrame=d.frame_token;
        document.getElementById('shot').src='/api/frame?t='+Date.now();
      }
      if(d.power) document.getElementById('power').textContent=d.power;
      if(d.heading!==undefined)
        document.getElementById('hdg').textContent=d.heading.toFixed(1);
      if(d.strip!==undefined && d.strip!==strip){ strip=d.strip; si=0; showStrip(); }
      if(!d.running){
        clearInterval(watching); watching=null;
        document.getElementById('go').textContent='GPT DRIVE';
        document.getElementById('flowgo').textContent='FLOW';
        document.getElementById('find').textContent='FIND IT';
        busy=false; buttons();
      }
    }catch(err){ /* the bench restarted; the next tick will pick it up */ }
  }, 700);
}
// A field each. One box serving both buttons meant the same number was the
// drive slice for one and the issue interval for the other, which nobody
// should have to remember.
document.getElementById('find').onclick=async()=>{
  // For something that is not in the picture at all. It turns on the spot
  // looking for it, and hands over to the reach controller once it has a
  // direction. The drive box is its handover slice, not a sweep interval.
  document.getElementById('find').textContent='LOOKING...';
  busy=true; buttons();
  await post('/api/search',{target:document.getElementById('target').value,
                            seconds:document.getElementById('every').value.trim()});
  watch();
};
document.getElementById('flowgo').onclick=async()=>{
  // Flowing keeps several looks in the air so the wheels never stop. The
  // number box is reused as the issue interval rather than the drive slice.
  const every=document.getElementById('flowevery').value.trim();
  document.getElementById('flowgo').textContent='FLOWING...';
  busy=true; buttons();
  await post('/api/flow',{target:document.getElementById('target').value,
                          every:every===''?null:+every});
  watch();
};
document.getElementById('go').onclick=async()=>{
  // Blank means one look, drive what it planned, stop. A number means keep
  // looking again after that many seconds of motion, until it arrives.
  const every=document.getElementById('every').value.trim();
  document.getElementById('go').textContent = every ? 'DRIVING...' : 'ONE LOOK...';
  busy=true; buttons();
  await post('/api/mission',{target:document.getElementById('target').value,
                             seconds:every===''?null:+every});
  watch();
};
document.getElementById('ask').onclick=async()=>{
  document.getElementById('log').textContent='asking the model...';
  await post('/api/ask',{target:document.getElementById('target').value});
};
document.getElementById('left').onclick=()=>post('/api/turn',
  {degrees:-Math.abs(+document.getElementById('deg').value)});
document.getElementById('right').onclick=()=>post('/api/turn',
  {degrees:Math.abs(+document.getElementById('deg').value)});
document.getElementById('run').onclick=async()=>{
  // The run ends with a fresh picture from wherever the robot now is, and the
  // server sends the same floor points reprojected into it. Those replace the
  // drawn ones: same carpet, new viewpoint. Clicking again starts over.
  await post('/api/run',{points});
  hazards=[]; draw();
};
document.getElementById('clear').onclick=()=>{points=[];hazards=[];draw();project();};
document.getElementById('stop').onclick=()=>fetch('/api/halt',{method:'POST'});
applyView();
post('/api/capture');
// Started after load, never inline: an MJPEG connection stays open for as long
// as it is watched, so pointing an <img> at it before the load event leaves the
// page reporting itself as still loading forever.
window.addEventListener('load', ()=>document.getElementById('live').click());
</script></body></html>"""


STREAM_FPS = 3.            # Frames a second offered to a viewer. The service's
                           # camera worker grabs
                           # continuously whatever anyone asks, so this does not
                           # take frames from the odometry; it costs one small
                           # socket round trip each, against the roughly eight a
                           # second a driving leg already makes.
STREAM_WIDTH = 320         # Measured on this machine, 3 fps to one viewer:
                           #   pass-through (0): bench 1.2% CPU, 241 KB/s
                           #   320 wide:         bench 3.8% CPU,  20 KB/s
                           # Shrinking costs a decode, a resize and an encode
                           # that passing through does not, so it trades CPU for
                           # link. The Jetson has CPU to spare and the robot is
                           # on wifi, so the trade is worth taking; set 0 to hand
                           # the camera's own JPEG straight on instead.
STREAM_QUALITY = 70
MISSION_SECONDS = 4.       # wheels turn for about this long between looks. Long
                           # enough that a look is worth its latency, short
                           # enough that the picture still resembles the plan.
MISSION_CYCLES = 25        # bound on a run, so a mission that is getting nowhere
                           # ends by itself rather than driving until the battery
MISSION_STANDOFF_CM = 20.
MISSION_COMMIT_CM = 30.
FLOW_INTERVAL_S = 1.        # how often a look is issued when flowing. At ~3 s a
                            # call that keeps about three in the air at once.
FLOW_MAX_IN_FLIGHT = 4      # a slow call must not let requests pile up
FLOW_SECONDS = 120.         # a flowing run costs a call a second; bound it     # furthest the robot drives on one measurement. Range
                            # from a single pixel is poor at distance (34 px of
                            # contact error measured as 80 cm here), so a wrong
                            # reading must cost one short leg, not a crash.
MISSION_TURN_MIN_DEG = 5.   # smaller than this is not worth a look
MISSION_TURN_MAX_DEG = 120. # the model is told to stay inside this; enforce it
MISSION_TURN_STEP_DEG = 30. # how much of a requested turn to do before looking
                            # again. A 90 degree rotation done blind spends four
                            # seconds and every view along the way; in steps the
                            # robot stops as soon as open floor appears, and a
                            # heading chosen from one frame is not committed to.
MISSION_TURN_BUDGET_DEG = 270.  # total rotation allowed without driving. Enough
                                # to look all the way round and a bit more; past
                                # that the robot is searching, not escaping.


def fatal(exc):
    """Whether a Stop means the run is over, or just this leg.

    The guards that protect the hardware end a mission. The ones that describe a
    maneuver going badly -- a leg that ran long, a frame that would not decode --
    describe exactly the situation a recurrent planner exists to recover from.
    """
    text = str(exc).lower()
    return any(mark.lower() in text for mark in Bench.FATAL)
STREAM_LIMIT = 3           # concurrent viewers, so a forgotten tab cannot pile
                           # threads onto the service

BEV_PIXELS = 480           # square canvas
BEV_EXTENT_CM = 200.       # 2 m across and 2 m ahead
BEV_SCALE = BEV_PIXELS / BEV_EXTENT_CM        # pixels per centimetre


def bev_to_floor(x, y):
    """Canvas pixel to floor position: right of centre, forward of the robot."""
    return ((x - BEV_PIXELS / 2.) / BEV_SCALE,
            (BEV_PIXELS - y) / BEV_SCALE)


def floor_to_bev(right, forward):
    return (right * BEV_SCALE + BEV_PIXELS / 2.,
            BEV_PIXELS - forward * BEV_SCALE)


class Warp(object):
    """Bird's-eye view of the floor, built once from the calibration.

    Straightens out the thing that made drawing on the camera image awkward: in
    perspective, 30 px near the bottom of the frame is under 2 cm of carpet
    while 30 px near the top is tens of centimetres, so evenly spaced clicks
    landed bunched. Here a centimetre is a centimetre everywhere.

    Floor that the camera cannot see stays black, which is the honest part: the
    far corners of a 2 m square are outside the lens, and the top of the view is
    built from very few source pixels, so it is stretched and soft. That blur is
    a fair picture of how much the robot really knows out there.
    """

    def __init__(self, robot):
        grid = np.indices((BEV_PIXELS, BEV_PIXELS), dtype=np.float32)
        ys, xs = grid[0], grid[1]
        right = (xs - BEV_PIXELS / 2.) / BEV_SCALE
        forward = (BEV_PIXELS - ys) / BEV_SCALE
        down = np.array([0., math.cos(robot.pitch), math.sin(robot.pitch)])
        ahead = np.array([0., 0., 1.]) - down * down[2]
        ahead /= np.linalg.norm(ahead)
        side = np.cross(down, ahead)
        rays = (side[None, None, :] * right[..., None]
                + ahead[None, None, :] * forward[..., None]
                + down[None, None, :] * robot.height)
        norm = np.linalg.norm(rays, axis=2, keepdims=True)
        rays = rays / np.maximum(norm, 1e-9)
        flat, _ = cv2.fisheye.projectPoints(
            rays.reshape(1, -1, 3).astype(np.float64), np.zeros(3), np.zeros(3),
            robot.K, robot.D)
        flat = flat.reshape(BEV_PIXELS, BEV_PIXELS, 2)
        self.mapx = flat[..., 0].astype(np.float32)
        self.mapy = flat[..., 1].astype(np.float32)
        # Anything that projects off the sensor, or behind it, is unknown floor.
        behind = rays[..., 2] <= .05
        self.unknown = (behind | (self.mapx < 0) | (self.mapx > 639)
                        | (self.mapy < 0) | (self.mapy > 479))

    def apply(self, image):
        out = cv2.remap(image, self.mapx, self.mapy, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(12, 14, 17))
        out[self.unknown] = (12, 14, 17)
        return self.decorate(out)

    @staticmethod
    def decorate(view):
        """Metre grid, so distance is readable rather than guessed."""
        for centimetres in range(25, int(BEV_EXTENT_CM) + 1, 25):
            major = centimetres % 100 == 0
            shade = (70, 86, 96) if major else (34, 44, 52)
            y = int(BEV_PIXELS - centimetres * BEV_SCALE)
            cv2.line(view, (0, y), (BEV_PIXELS, y), shade, 2 if major else 1)
            for sign in (-1, 1):
                x = int(BEV_PIXELS / 2. + sign * centimetres * BEV_SCALE)
                if 0 <= x < BEV_PIXELS:
                    cv2.line(view, (x, 0), (x, BEV_PIXELS), shade, 2 if major else 1)
            if major:
                cv2.putText(view, '%dm' % (centimetres // 100), (6, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (120, 140, 150), 1)
        # Where the floor projection stops being worth trusting. Past this a
        # pixel of error is worth several centimetres, so the bench refuses
        # waypoints out here rather than pretending to place them.
        limit = int(BEV_PIXELS - fetch.ROUTE_RANGE_CM * BEV_SCALE)
        if 0 <= limit < BEV_PIXELS:
            for x in range(0, BEV_PIXELS, 18):
                cv2.line(view, (x, limit), (min(BEV_PIXELS, x + 9), limit),
                         (70, 90, 200), 2)
            cv2.putText(view, 'beyond here not trusted', (8, limit - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, .42, (120, 140, 220), 1)
        # The robot: 12 cm wide, lens at the front centre, body behind the view.
        half = int(6. * BEV_SCALE)
        cv2.rectangle(view, (BEV_PIXELS // 2 - half, BEV_PIXELS - 10),
                      (BEV_PIXELS // 2 + half, BEV_PIXELS - 1), (90, 190, 240), -1)
        cv2.putText(view, 'robot', (BEV_PIXELS // 2 - 20, BEV_PIXELS - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (90, 190, 240), 1)
        return view


class Bench(object):
    """Owns the robot and the last captured frame. One run at a time."""

    def __init__(self, enable_motion):
        self.lock = threading.Lock()
        self.enable_motion = enable_motion
        self.robot = fetch.Robot(dry_run=not enable_motion)
        self.odometer = fetch.Odometer(self.robot)
        self.warp = Warp(self.robot)
        self.frame = None
        self.view = None
        self.heading = 0.          # accumulated since the last picture
        self.lines = []
        # What the model last said, in floor coordinates of the frame it was
        # said in, plus where the robot has moved since. Carried into the next
        # ask so each look is not blind to what has left the picture.
        self.memory = None
        self.since = [0., 0., 0.]
        # 'floor' is the metric bird's-eye; 'camera' is the raw photograph.
        # Drawing on the photograph is the only way to see whether a route
        # reprojected into a later frame lands on the same carpet.
        self.view_mode = 'floor'
        # A frame per waypoint, with the remaining route redrawn in it. This is
        # how you see whether the reprojection is right: the markers should
        # stay on the same carpet while the picture moves underneath them.
        self.strip = []
        # What the live stream draws: the route being executed, in the frame it
        # was drawn in, plus where the robot believes it stands in that frame.
        # `pose_heading` is the IMU reading when the pose was last recorded, so
        # the stream can carry the heading forward from its own observation and
        # turn smoothly instead of in steps.
        self.live_route = []
        self.live_pose = [0., 0., 0.]
        self.pose_heading = None
        # A mission drives for minutes, so it runs on its own thread and the
        # page polls it. Holding the request open instead would freeze the log
        # for the whole run -- exactly when there is most to watch.
        self.abort = threading.Event()
        self.flight = None
        self.live_obstacles = []     # what the last look said is on the floor
        self.last_note = ''          # and what it said about it, in its words
        self.frame_token = 0         # bumped whenever the still picture changes

    def say(self, text):
        self.lines.append(text)
        self.lines = self.lines[-200:]

    def log(self):
        return '\n'.join(self.lines) or 'ready'

    def power(self):
        try:
            status = fetch.call('status')
            supply = status.get('power') or {}
            return 'pack %.2f V  motors %s' % (
                supply.get('pack_voltage_v', 0.),
                'armed' if supply.get('motion_allowed') else 'BLOCKED')
        except Exception as exc:
            return 'service: %s' % exc

    def state(self, **extra):
        out = dict(log=self.log(), heading=self.heading, speed=self.robot.speed,
                   power=self.power(), view_mode=self.view_mode,
                   strip=len(self.strip), note=self.last_note,
                   mode='MOTION ENABLED' if self.enable_motion else 'PREVIEW ONLY')
        # Hand back whatever route is armed, so a reloaded page draws the same
        # thing the live stream is drawing. Without this the canvas comes back
        # empty while the stream still shows the marks, which reads as stray
        # graphics nobody put there.
        if self.live_route and 'points' not in extra:
            shown = []
            for spot in self.live_route:
                place = self.to_pixel(spot[0], spot[1])
                if place and 0 <= place[0] < (640 if self.view_mode == 'camera'
                                              else BEV_PIXELS) \
                        and 0 <= place[1] < (480 if self.view_mode == 'camera'
                                             else BEV_PIXELS):
                    shown.append([int(place[0]), int(place[1])])
            if len(shown) == len(self.live_route):
                out['points'] = shown
        out.update(extra)
        return out

    def capture(self):
        image, _ = self.robot.frame()
        self.frame = image
        self.view = self.warp.apply(image)
        self.heading = 0.
        self.lines = []
        self.live_route = []
        self.live_pose = [0., 0., 0.]
        self.pose_heading = None
        self.strip = []          # frames from an older run are not about this
                                 # picture, and outlived their run confusingly
        if self.memory:
            self.say('picture taken; carrying %d obstacle(s) from the last answer'
                     % len(self.memory.get('obstacles') or []))
        else:
            self.say('picture taken; heading reset to 0')
        return self.state(frame=True, length=0.)

    FATAL = ('power guard', 'stale', 'unhealthy', 'control generation', 'tilt',
             'service disconnected', 'OPENAI_API_KEY')

    def running(self):
        return self.flight is not None and self.flight.is_alive()

    def hunt(self, target, seconds=MISSION_SECONDS):
        """Start a search on its own thread; the page polls /api/progress."""
        if self.running():
            self.say('a mission is already running; press STOP first')
            return self.state(mission=True)
        self.abort.clear()
        self.flight = threading.Thread(target=self.search, args=(target, seconds),
                                       daemon=True)
        self.flight.start()
        time.sleep(.4)                # let the first line reach the log
        return self.state(mission=True)

    def search(self, target, seconds=MISSION_SECONDS):
        """Find something that is not in the picture, then drive to it.

        mission() is the controller that arrives, and it gives up immediately
        when the target has never been in frame -- there is nothing to carry
        forward and guessing a direction is worse than saying so. So the two are
        stacked rather than merged: explore turns on the spot until a look comes
        back sure, points the robot at what it found, and hands the wheels over
        mid-run. Everything below the handover is the code that already works.

        The search log is lost at the handover, because mission() clears it to
        write its own. The journal is in the run's events either way.
        """
        target = (target or '').strip()
        if not target:
            self.say('name something to look for first')
            return
        if not os.environ.get('OPENAI_API_KEY'):
            self.say('OPENAI_API_KEY is not set, so the model cannot be asked')
            return

        self.lines = []
        self.say('searching for %s: turning on the spot and looking' % target)

        def watch(image):
            # Show each photograph as it is taken. A sweep is a minute of
            # turning, and the panel is the only way to see what it is seeing.
            self.frame = image
            if self.view_mode != 'camera':
                self.view = self.warp.apply(image)
            self.frame_token += 1

        def record(event, **fields):
            if self.abort.is_set():
                raise fetch.Stop('stopped by the operator')
            if event == 'look':
                self.say('  %+04.0f  %s%s%s'
                         % (fields.get('heading', 0.), fields.get('scene', ''),
                            '' if fields.get('confidence') != 'sure' else '  <- SEEN',
                            '' if fields.get('confidence') != 'unsure'
                            else '  <- might be it'))
                self.last_note = str(fields.get('scene', ''))[:120]
            elif event == 'sweep':
                self.say('looking round %.0f degrees in %.0f degree steps'
                         % (fields.get('arc', 0.), fields.get('step', 0.)))
            elif event == 'plan':
                if fields.get('rejected'):
                    self.say('  refused that plan: %s'
                             % '; '.join(fields['rejected'])[:120])
            elif event == 'planned':
                self.say('plan (%d step(s)): %s' % (fields.get('steps', 0),
                                                    fields.get('note', '')))
            elif event == 'hop':
                self.say('driving %.0f cm along %+.0f to look from somewhere else'
                         % (fields.get('distance_cm', 0.), fields.get('heading', 0.)))
            elif event in ('hop_blocked', 'replanning'):
                self.say('  %s' % fields.get('why', event))
            elif event == 'found':
                self.say('')
                self.say('FOUND: %s is %+.0f degrees away; handing over to the '
                         'reach controller' % (target, fields.get('heading', 0.)))
            elif event == 'gave_up':
                self.say('')
                self.say('gave up after %d look(s) from %d place(s): %s'
                         % (fields.get('looks_used', 0), fields.get('stations', 0),
                            fields.get('why', '')))
                self.say('that is a budget running out, not a proof it is absent')

        hunt = explore.Search(self.robot, target, self.odometer, record,
                              watch=watch)
        try:
            result = hunt.run(handoff=lambda: self.mission(target, seconds))
        except fetch.Stop as exc:
            self.robot.halt()
            self.say('STOPPED: %s' % exc)
            return
        if result.get('outcome') == 'not_found':
            self.say('')
            self.say(hunt.journal.render())

    def launch_flow(self, target, every=FLOW_INTERVAL_S):
        """Start a flowing run on its own thread; the page polls /api/progress."""
        if self.running():
            self.say('something is already running; press STOP first')
            return self.state(mission=True)
        self.abort.clear()
        self.flight = threading.Thread(target=self.flow, args=(target, every))
        self.flight.daemon = True
        self.flight.start()
        time.sleep(.4)
        return self.state(mission=True)

    def launch(self, target, seconds=MISSION_SECONDS):
        """Start a mission on its own thread; the page polls /api/progress.

        `seconds` of None means do not plan recurrently: take one look, drive
        what it planned, and stop.
        """
        if self.running():
            self.say('a mission is already running; press STOP first')
            return self.state(mission=True)
        self.abort.clear()
        self.flight = threading.Thread(target=self.mission, args=(target, seconds),
                                       daemon=True)
        self.flight.start()
        time.sleep(.4)                # let the first line reach the log
        return self.state(mission=True)

    def compose(self, first, second):
        """`second`, given in the frame `first` ends in, expressed from the start."""
        heading = math.radians(first[2])
        return [first[0] + math.cos(heading) * second[0] + math.sin(heading) * second[1],
                first[1] + math.cos(heading) * second[1] - math.sin(heading) * second[0],
                first[2] + second[2]]

    def since_pose(self, earlier, now):
        """The motion from `earlier` to `now`, in the frame `earlier` ends in.

        This is what a trajectory has to be rebased by. A route comes back in
        the frame of the picture it was planned from, and by the time it arrives
        the robot has driven for the length of the call -- roughly 3 s, about
        25 cm. Rebasing needs that displacement expressed in the older frame,
        not the difference of two world positions.
        """
        heading = math.radians(earlier[2])
        dx, dz = now[0] - earlier[0], now[1] - earlier[1]
        return [math.cos(heading) * dx - math.sin(heading) * dz,
                math.cos(heading) * dz + math.sin(heading) * dx,
                now[2] - earlier[2]]

    def ask_later(self, image, target, prior):
        """Start a recognizer call on its own thread and hand back a handle.

        The point of the whole exercise: the wheels keep turning while this is
        in flight. Measured, a call takes 3.15 s against a 4 s drive, so serial
        looking leaves the robot standing still 44 per cent of the time.
        """
        holder = {}

        def work():
            try:
                holder['answer'] = fetch.recognize(image, target, prior)
            except Exception as exc:
                holder['error'] = exc

        worker = threading.Thread(target=work)
        worker.daemon = True
        worker.start()
        return dict(thread=worker, holder=holder, issued=time.monotonic())

    @staticmethod
    def freshest(pending, applied):
        """Pick the newest finished look; report what to keep and what to drop.

        Latency is not constant -- 2.18 to 4.56 s measured on this robot -- so
        with several calls in the air the answers do not come back in the order
        they were asked. Applying whichever finished last would sometimes steer
        the robot with an older picture than the one it already used, which is
        worse than not asking at all.

        Returns (chosen, still_pending, dropped). `chosen` is None when nothing
        has finished, or when everything that has is older than what is already
        applied.
        """
        finished = [job for job in pending if not job['thread'].is_alive()]
        if not finished:
            return None, list(pending), 0
        newest = max(finished, key=lambda job: job['seq'])
        keep = [job for job in pending if job['seq'] > newest['seq']]
        dropped = len(finished) - 1
        if newest['seq'] <= applied:
            return None, keep, dropped + 1
        return newest, keep, dropped

    def flow(self, target, every=FLOW_INTERVAL_S, span=FLOW_SECONDS):
        """Drive without ever standing still, keeping several looks in flight.

        Serial looking stops the wheels for the length of a call: measured,
        3.15 s against a 4 s drive, so the robot idles 44 per cent of the time.
        Issuing one call and waiting for it fixes the idling but only refreshes
        the plan every 4 s. Issuing one every `every` seconds instead keeps
        about latency/every of them in the air at once, so a fresh answer lands
        every second and the robot corrects course that often.

        Two things this forces. Latency is not constant -- 2.18 to 4.56 s
        measured -- so answers come back out of order, and an older one must be
        dropped rather than applied. And every answer describes the frame of the
        picture it was planned from, so it is rebased by the motion since that
        shot before any of it is driven; likewise the surviving route is rebased
        after each slice of driving. Serial needs neither, because a stopped
        robot is still where its picture was taken.
        """
        target = (target or '').strip()
        if not target:
            self.say('name something to drive to first')
            return
        if not os.environ.get('OPENAI_API_KEY'):
            self.say('OPENAI_API_KEY is not set, so the model cannot be asked')
            return

        world = [0., 0., 0.]     # where the robot is, from where it began
        route, obstacles, goal = [], [], None
        memory = None
        pending, issued, applied = [], 0, -1
        stale, trouble = 0, 0
        last_issue = 0.
        started_at = time.monotonic()
        self.lines = []
        self.say('flowing to %s: a look every %.1f s, wheels never stop'
                 % (target, every))

        def build(answer, shot_pose):
            """An answer, moved from the frame it was planned in into this one."""
            drift = self.since_pose(shot_pose, world)
            spot = None
            contact = answer.get('contact_pixel')
            if answer.get('visible') and isinstance(contact, dict):
                try:
                    spot = self.robot.ground(float(contact['x']), float(contact['y']))
                    near = fetch.nearest_range(self.robot, contact['x'], contact['y'])
                    if near is not None and near < math.hypot(*spot):
                        scale = near / max(1e-6, math.hypot(*spot))
                        spot = (spot[0] * scale, spot[1] * scale)
                    spot = fetch.rebase([spot], drift)[0]
                except (fetch.Stop, KeyError, TypeError, ValueError):
                    spot = None
            points = []
            for point in answer.get('route_pixels') or []:
                if not isinstance(point, dict) or 'x' not in point:
                    continue
                try:
                    points.append(self.robot.ground(float(point['x']),
                                                    float(point['y'])))
                except (fetch.Stop, TypeError, ValueError):
                    continue
            found = fetch.project_obstacles(self.robot, answer)
            points = [p for p in fetch.rebase(points, drift) if p[1] > 0.]
            found = list(zip([o[0] for o in found],
                             fetch.rebase([o[1] for o in found], drift)))
            return points, found, spot, math.hypot(drift[0], drift[1])

        while time.monotonic() - started_at < span:
            if self.abort.is_set():
                self.say('stopped by the operator')
                break
            now = time.monotonic()

            if now - last_issue >= every and len(pending) < FLOW_MAX_IN_FLIGHT:
                try:
                    image, _ = self.robot.frame()
                    self.frame = image
                    if self.view_mode != 'camera':
                        # /api/frame serves the warp in floor view, so setting
                        # only self.frame leaves that panel on a stale picture
                        # for the whole run.
                        self.view = self.warp.apply(image)
                    # The page only reloads the still picture when this changes.
                    # Without it the left panel freezes on whatever was there
                    # when the run began, which looks exactly like a hung feed.
                    self.frame_token += 1
                except Exception as exc:
                    self.say('lost the camera: %s' % exc)
                    break
                prior = (fetch.recall(self.robot, memory, [0., 0., 0.])
                         if memory else None)
                job = self.ask_later(image, target, prior)
                job.update(seq=issued, shot=list(world))
                pending.append(job)
                issued += 1
                last_issue = now

            newest, pending, dropped = self.freshest(pending, applied)
            stale += dropped
            if newest is not None:
                if 'answer' in newest['holder']:
                    applied = newest['seq']
                    answer = newest['holder']['answer']
                    points, found, spot, moved = build(answer, newest['shot'])
                    if memory:
                        found = fetch.merge_obstacles(
                            memory['obstacles'],
                            self.since_pose(newest['shot'], world), found)
                    goal, obstacles = spot, found
                    self.last_note = str(answer.get('note', ''))[:120]
                    self.live_obstacles = list(obstacles)
                    if goal is not None and math.hypot(*goal) <= MISSION_STANDOFF_CM:
                        self.say('')
                        self.say('REACHED: %s is %.0f cm away'
                                 % (target, math.hypot(*goal)))
                        break
                    if goal is not None:
                        points = fetch.stop_short(points, goal, MISSION_STANDOFF_CM)
                    if points:
                        points, _ = fetch.avoid(points, obstacles, goal)
                        if goal is not None:
                            points = fetch.stop_short(points, goal,
                                                      MISSION_STANDOFF_CM) or points
                        points = fetch.cap_path(points, MISSION_COMMIT_CM) or points
                    route = points
                    if answer.get('goal'):
                        self.last_note = ('%s | %s' % (str(answer['goal'])[:40],
                                                       self.last_note))[:120]
                    self.say('look %d %s | rebased %.0f cm | %d in flight%s'
                             % (newest['seq'] + 1,
                                ('target %.0f cm' % math.hypot(*goal)) if goal
                                else ('seen, unplaceable' if answer.get('visible')
                                      else 'not seen'),
                                moved, len(pending),
                                ', %d stale dropped' % stale if stale else ''))
                elif 'error' in newest['holder']:
                    self.say('  a look failed: %s'
                             % str(newest['holder']['error'])[:60])

            if not route:
                time.sleep(.1)
                continue

            self.live_route = list(route)
            self.live_pose = [0., 0., 0.]
            self.pose_heading = self.imu_heading()

            def record(event, **fields):
                if event == 'pose':
                    self.live_pose = [fields.get('x', 0.), fields.get('z', 0.),
                                      fields.get('heading', 0.)]
                    self.pose_heading = self.imu_heading()
                elif event == 'leg_blocked':
                    self.say('  blocked by "%s"' % fields.get('blocked_by'))

            def fresher():
                """True once a look lands that this drive should give way to.

                Only for one that will actually be used. An answer older than
                the plan being driven is discarded rather than applied, so
                stopping the wheels for it would cost motion and buy nothing.
                """
                for job in pending:
                    if (job['seq'] > applied and not job['thread'].is_alive()
                            and 'answer' in job['holder']):
                        return True
                return False

            slice_end = time.monotonic() + every
            try:
                pose = fetch.follow(self.robot, self.odometer, route, record,
                                    obstacles, deadline=slice_end,
                                    interrupt=fresher)
                trouble = 0
            except fetch.Stop as exc:
                self.robot.halt()
                if fatal(exc):
                    self.say('STOPPED: %s' % exc)
                    break
                trouble += 1
                self.say('  cut short: %s' % exc)
                if trouble >= 3:
                    self.say('three slices in a row went wrong; stopping')
                    break
                pose = list(self.live_pose)
            world = self.compose(world, pose)
            # The part of the route still ahead belongs to the old frame too.
            route = [p for p in fetch.rebase(route, pose) if p[1] > 0.]
            obstacles = list(zip([o[0] for o in obstacles],
                                 fetch.rebase([o[1] for o in obstacles], pose)))

        try:
            self.robot.hold(0., 0.)
        except Exception:
            pass
        self.live_route = []
        elapsed = time.monotonic() - started_at
        self.say('')
        self.say('ended after %.0f s: %d looks issued, %d applied, %d stale dropped'
                 % (elapsed, issued, applied + 1, stale))

    def mission(self, target, seconds=MISSION_SECONDS):
        """Drive to a named object, asking the model again every few seconds.

        One look is a plan made from one photograph, and it stops being true as
        soon as the wheels turn. So the wheels are given a fixed slice of time,
        then stopped, and the model is asked again from where the robot now
        stands -- with its own previous route redrawn in the new picture, so it
        can continue a plan rather than invent one each time.

        The object going out of frame is not a failure. It is what happens when
        a small robot drives at something: it leaves the top of the picture long
        before it is reached. The memory is what carries the mission across that
        gap, and the prompt says so.
        """
        target = (target or '').strip()
        if not target:
            self.say('name something to drive to first')
            return
        if not os.environ.get('OPENAI_API_KEY'):
            self.say('OPENAI_API_KEY is not set, so the model cannot be asked')
            return

        memory, pose, reached, trouble = None, [0., 0., 0.], False, 0
        closest, idle = None, 0       # nearest the target has come, and how many
                                      # looks since that last improved
        spins = 0.                    # degrees turned since anything was driven
        once = seconds is None
        self.lines = []
        if once:
            self.say('one look: plan a route to %s and drive it' % target)
        else:
            self.say('mission: reach %s, re-planning every %.0f s of motion'
                     % (target, seconds))
        for cycle in range(1 if once else MISSION_CYCLES):
            if self.abort.is_set():
                self.say('stopped by the operator')
                break
            try:
                image, _ = self.robot.frame()
                self.frame = image
                if self.view_mode != 'camera':
                    self.view = self.warp.apply(image)
            except Exception as exc:
                self.say('cycle %d: no picture (%s)' % (cycle + 1, exc))
                break

            prior = fetch.recall(self.robot, memory, pose) if memory else None
            try:
                answer = fetch.recognize(self.frame, target, prior)
            except fetch.Stop as exc:
                self.say('cycle %d: the model could not be asked: %s'
                         % (cycle + 1, exc))
                break

            goal = None
            contact = answer.get('contact_pixel')
            if answer.get('visible') and isinstance(contact, dict):
                try:
                    goal = self.robot.ground(float(contact['x']), float(contact['y']))
                    # Plan against the near edge of the range bracket, not its
                    # middle. A contact pixel carries a spread, not a number,
                    # and the spread is 21 per cent of the range at a metre.
                    near = fetch.nearest_range(self.robot, contact['x'],
                                               contact['y'])
                    if near is not None and near < math.hypot(*goal):
                        scale = near / max(1e-6, math.hypot(*goal))
                        goal = (goal[0] * scale, goal[1] * scale)
                except (fetch.Stop, KeyError, TypeError, ValueError):
                    goal = None

            route = []
            for point in answer.get('route_pixels') or []:
                if not isinstance(point, dict) or 'x' not in point:
                    continue
                try:
                    route.append(self.robot.ground(float(point['x']),
                                                   float(point['y'])))
                except (fetch.Stop, TypeError, ValueError):
                    continue
            obstacles = fetch.project_obstacles(self.robot, answer)
            if memory:
                # Carry what was on the floor last time into this frame. The
                # things the robot is about to touch are the first to leave the
                # picture, so a look alone cannot be trusted to list them.
                was = len(obstacles)
                obstacles = fetch.merge_obstacles(memory['obstacles'], pose,
                                                  obstacles)
                if len(obstacles) > was:
                    self.say('  still avoiding %d obstacle(s) now out of shot'
                             % (len(obstacles) - was))

            if not route and memory:
                # It answered with nothing usable. The remembered route is still
                # a description of this room, so carry it rather than stop.
                route = [spot for spot in fetch.rebase(memory['route'], pose)
                         if spot[1] > 0.]
                if route:
                    self.say('  no usable route returned; carrying %d remembered '
                             'waypoint(s)' % len(route))

            self.say('')
            if goal is not None:
                sighting = 'sees it at %.0f cm' % math.hypot(*goal)
            elif answer.get('visible'):
                # It can see the thing but its contact with the floor is at or
                # above the horizon, where one pixel is worth metres. That is
                # not the same as not seeing it, and saying so hid a working
                # look behind a failure message.
                sighting = 'sees it, but too far off to place on the floor'
            else:
                sighting = 'cannot see it'
            if answer.get('goal'):
                self.say('  it read the goal as: %s' % str(answer['goal'])[:60])
            self.say('look %d: %s%s' % (
                cycle + 1, sighting,
                ' (%d waypoint(s) remembered)' % len(prior['route_pixels'])
                if prior else ''))
            self.last_note = str(answer.get('note', ''))[:120]
            self.live_obstacles = list(obstacles)
            self.frame_token += 1
            if answer.get('note'):
                self.say('  "%s"' % str(answer['note'])[:80])
            for label, spot in obstacles:
                self.say('  avoiding "%s" at %.0f cm' % (label, math.hypot(*spot)))

            if goal is not None:
                range_now = math.hypot(*goal)
                if closest is None or range_now < closest - 3.:
                    closest, idle = range_now, 0
                else:
                    idle += 1
                    if idle >= 4:
                        self.say('')
                        self.say('four looks without getting nearer than %.0f cm; '
                                 'stopping rather than working at it forever'
                                 % closest)
                        break
            # A rotation the model asked for. It is the only motion it may
            # request, and the only one that helps when something is too close
            # to drive around: at 10 cm an obstacle blocks every heading until
            # it is nearly behind, so no route exists to draw.
            asked_turn = answer.get('turn_degrees')
            if (isinstance(asked_turn, (int, float)) and math.isfinite(asked_turn)
                    and abs(asked_turn) >= MISSION_TURN_MIN_DEG
                    and (goal is None or math.hypot(*goal) > MISSION_STANDOFF_CM)):
                asked_turn = max(-MISSION_TURN_MAX_DEG,
                                 min(MISSION_TURN_MAX_DEG, float(asked_turn)))
                step = math.copysign(min(abs(asked_turn), MISSION_TURN_STEP_DEG),
                                     asked_turn)
                spins += abs(step)
                if spins > MISSION_TURN_BUDGET_DEG:
                    self.say('  %.0f deg of turning without driving; stopping'
                             % spins)
                    break
                self.say('  it asks to turn %+.0f deg; turning %+.0f of it and '
                         'looking again' % (asked_turn, step)
                         if abs(step) < abs(asked_turn)
                         else '  it asks to turn %+.0f deg and look again' % step)
                try:
                    turned = self.robot.turn(step)
                except fetch.Stop as exc:
                    self.robot.halt()
                    if fatal(exc):
                        self.say('STOPPED: %s' % exc)
                        break
                    self.say('  could not turn: %s' % exc)
                    break
                shift = fetch.pivot_shift(turned)
                pose = [shift[0], shift[1], turned]
                memory = dict(route=list(route), obstacles=list(obstacles))
                self.live_route = []
                continue
            spins = 0.

            if goal is not None and math.hypot(*goal) <= MISSION_STANDOFF_CM:
                self.say('')
                self.say('REACHED: %s is %.0f cm away' % (target, math.hypot(*goal)))
                reached = True
                break
            if not route:
                self.say('nothing to drive and nothing remembered; stopping')
                break

            # End the route at the standoff, not on the object. Checking the
            # range only at look time and then driving a fixed slice blind is
            # how the robot ended up touching what it was sent to reach.
            if goal is not None:
                before = len(route)
                route = fetch.stop_short(route, goal, MISSION_STANDOFF_CM)
                if not route:
                    self.say('')
                    self.say('REACHED: already within %.0f cm of %s'
                             % (MISSION_STANDOFF_CM, target))
                    reached = True
                    break
                if len(route) < before:
                    self.say('  trimmed %d waypoint(s) to stop %.0f cm short of it'
                             % (before - len(route), MISSION_STANDOFF_CM))

            widened, nudged = fetch.avoid(route, obstacles, goal)
            widened = fetch.stop_short(widened, goal, MISSION_STANDOFF_CM) or widened
            # Commit only a bounded distance before looking again, however far
            # the target is believed to be. A range read from a contact pixel
            # near the horizon can be wrong by a factor of three, and a standoff
            # subtracted from a wrong range is wrong by the same amount -- that
            # is how "stop 20 cm short of a bin 73 cm away" became a collision
            # with a bin about 25 cm away. This cap is the thing that survives a
            # bad measurement, because it does not depend on it.
            capped = fetch.cap_path(widened, MISSION_COMMIT_CM)
            if capped and len(capped) < len(widened):
                self.say('  committing %.0f cm of it before looking again'
                         % MISSION_COMMIT_CM)
            widened = capped or widened
            if nudged:
                self.say('  widened %d waypoint(s) to the clearance the wheels need'
                         % nudged)
            self.live_route = list(widened)
            self.live_pose = [0., 0., 0.]
            self.pose_heading = self.imu_heading()
            self.frame_token += 1

            legs = [0]

            def record(event, **fields):
                if event == 'pose':
                    self.live_pose = [fields.get('x', 0.), fields.get('z', 0.),
                                      fields.get('heading', 0.)]
                    self.pose_heading = self.imu_heading()
                    return
                if event == 'leg':
                    legs[0] += 1
                    self.say('  drove %.1f cm of %.1f asked (%s)'
                             % (fields.get('moved_cm', 0.), fields.get('wanted_cm', 0.),
                                fields.get('by', '')))
                elif event == 'leg_blocked':
                    self.say('  blocked by "%s" with %.0f cm of room'
                             % (fields.get('blocked_by'), fields.get('clear_cm', 0.)))
                elif event in ('stalled', 'no_progress', 'cannot_face'):
                    self.say('  %s' % event.replace('_', ' '))

            started = time.monotonic()
            try:
                pose = fetch.follow(self.robot, self.odometer, widened, record,
                                    obstacles,
                                    deadline=None if once else started + seconds)
                trouble = 0
            except fetch.Stop as exc:
                self.robot.halt()
                if fatal(exc):
                    self.say('STOPPED: %s' % exc)
                    break
                # A leg that ran long, or a frame that would not decode, is a
                # reason to look again -- which is the whole point of planning
                # recurrently. Only give up if it keeps happening.
                trouble += 1
                self.say('  cut short: %s' % exc)
                if trouble >= 3:
                    self.say('three cycles in a row went wrong; stopping')
                    break
                pose = list(self.live_pose)
                memory = dict(route=list(widened), obstacles=list(obstacles))
                continue
            self.say('  %.1f s of motion, now %.0f cm from where the look started'
                     % (time.monotonic() - started, math.hypot(pose[0], pose[1])))
            memory = dict(route=list(widened), obstacles=list(obstacles))
            # Turning does not count as progress. While blocked, follow still
            # rotates to face each waypoint, so requiring zero rotation here
            # meant the escape never fired and the give-up counter reset every
            # cycle -- nine looks in a row against one cable, going nowhere.
            if legs[0] == 0:
                # Nothing moved. Usually that means the robot has closed inside
                # the keep-back distance of something, where every heading whose
                # corridor still holds it is refused -- at 10 cm that is a cone
                # of about 42 degrees, so most of them. Turning is the only move
                # that is always safe here: it shifts the corridor without
                # carrying the robot into anything.
                here = fetch.merge_obstacles(memory['obstacles'], pose, [])
                turn = fetch.escape_heading(here)
                if turn is None:
                    self.say('nothing moved and nothing is in the way; stopping')
                    break
                step = math.copysign(min(abs(turn), MISSION_TURN_STEP_DEG), turn)
                self.say('  boxed in; %+.0f deg would clear it, turning %+.0f now'
                         % (turn, step))
                try:
                    turned = self.robot.turn(step)
                except fetch.Stop as exc:
                    self.say('could not turn clear: %s' % exc)
                    break
                shift = fetch.pivot_shift(turned)
                pose = [shift[0], shift[1], turned]
                # Counted in degrees, not attempts. Turning in 30 degree steps
                # means an escape that needs 120 takes four cycles, and a limit
                # of "three tries" would have given up part way round.
                spins += abs(turned)
                if spins > MISSION_TURN_BUDGET_DEG:
                    self.say('%.0f deg of turning without driving; stopping'
                             % spins)
                    break
        else:
            if not once:
                self.say('gave up after %d looks' % MISSION_CYCLES)

        try:
            self.robot.hold(0., 0.)
        except Exception:
            pass
        self.live_route = []
        self.say('')
        self.say('mission %s' % ('complete' if reached else 'ended'))

    def ask(self, target):
        """Let the model place the waypoints instead of the operator.

        It answers in pixels of the camera image, never in centimetres: one
        photograph carries no scale, and every distance here comes from the
        calibration. Its route is then widened to the clearance the wheels
        need, because asking it to "leave room for a 12 cm robot" is asking it
        to judge a distance it cannot see -- measured over stored frames it
        leaves 3 to 14 cm where 15 is needed, consistently.
        """
        target = (target or '').strip()
        if not target:
            self.say('name something to drive to first')
            return self.state()
        if self.frame is None:
            self.say('take a picture first')
            return self.state()
        if not os.environ.get('OPENAI_API_KEY'):
            self.say('OPENAI_API_KEY is not set, so the model cannot be asked')
            return self.state()

        self.lines = []
        prior = None
        if self.memory and self.memory.get('target') == target:
            prior = fetch.recall(self.robot, self.memory, self.since)
            if prior:
                self.say('reminding it of %d waypoint(s) and %d obstacle(s) '
                         'from last time; it %s'
                         % (len(prior['route_pixels']), len(prior['obstacles']),
                            prior['since_then']))
                if prior['out_of_frame']:
                    self.say('  now out of shot: %s'
                             % ', '.join(prior['out_of_frame'][:4]))
        try:
            answer = fetch.recognize(self.frame, target, prior)
        except fetch.Stop as exc:
            self.say('the model could not be asked: %s' % exc)
            return self.state()

        self.last_note = str(answer.get('note', ''))[:120]
        self.say('asked for: %s' % target)
        self.say('model says: %s' % (answer.get('note') or '(nothing)'))
        if not answer.get('visible'):
            self.say('it cannot see it from here -- turn and take another picture')
            return self.state(points=[], hazards=[], length=0.)

        goal = None
        contact = answer.get('contact_pixel')
        if isinstance(contact, dict):
            try:
                goal = self.robot.ground(float(contact['x']), float(contact['y']))
            except (fetch.Stop, KeyError, TypeError, ValueError):
                self.say('it sees the object but not where it meets the floor, '
                         'so there is no range to it')

        route = []
        for point in answer.get('route_pixels') or []:
            if not isinstance(point, dict) or 'x' not in point or 'y' not in point:
                continue
            try:
                route.append(self.robot.ground(float(point['x']), float(point['y'])))
            except (fetch.Stop, TypeError, ValueError):
                continue      # above the horizon: not a place on the floor
        obstacles = fetch.project_obstacles(self.robot, answer)

        widened, nudged = fetch.avoid(route, obstacles, goal)
        if goal is not None:
            self.say('target is %.0f cm away' % math.hypot(*goal))
        for label, spot in obstacles:
            self.say('  obstacle "%s" at %.0f cm, %+.0f cm across'
                     % (label, math.hypot(*spot), spot[0]))
        if not obstacles:
            self.say('  it reported no obstacles on the floor')
        self.say('%d waypoints from the model%s'
                 % (len(route),
                    '; %d widened for clearance' % nudged if nudged else
                    ', already clear of what it reported'))

        def on_view(spot):
            """Canvas pixel, pulled to the border when it is off the view.

            to_pixel is pure arithmetic and will happily return a coordinate
            past the edge; drawing there loses the point silently. Most of a
            route can fall outside the camera image at close range, so the
            off-frame ones are marked at the border rather than dropped.
            """
            pixel = self.to_pixel(spot[0], spot[1])
            if not pixel:
                return None
            wide = 640 if self.view_mode == 'camera' else BEV_PIXELS
            tall = 480 if self.view_mode == 'camera' else BEV_PIXELS
            inside = 0 <= pixel[0] < wide and 0 <= pixel[1] < tall
            x = max(3, min(wide - 4, int(pixel[0])))
            y = max(3, min(tall - 4, int(pixel[1])))
            return [x, y] if inside else [x, y, 1]

        hazards = []
        for label, spot in obstacles:
            pixel = on_view(spot)
            if pixel:
                hazards.append([pixel[0], pixel[1],
                                (fetch.OBSTACLE_RADIUS_CM + fetch.CORRIDOR_HALF_CM)
                                * BEV_SCALE, label[:16]])
        drawn = []
        for spot in widened:
            pixel = on_view(spot)
            if pixel:
                drawn.append(pixel)
        edge = sum(1 for p in drawn if len(p) > 2)
        if edge:
            self.say('%d of %d waypoint(s) are outside the %s; those are marked '
                     'hollow at the edge, in the direction they lie'
                     % (edge, len(drawn),
                        'camera image' if self.view_mode == 'camera'
                        else "2 m bird's-eye view"))
        if not drawn:
            self.say('nothing left to drive to on this view')
            return self.state(points=[], hazards=hazards, length=0.)
        self.live_route = list(widened)
        self.live_pose = [0., 0., 0.]
        self.pose_heading = self.imu_heading()
        self.memory = dict(target=target, route=list(widened),
                           obstacles=list(obstacles))
        self.since = [0., 0., 0.]     # the memory is now in this frame
        self.live_obstacles = list(obstacles)
        self.say('press RUN TRAJECTORY to drive it, or click to edit first')
        return self.state(points=drawn, hazards=hazards,
                          length=self.measure(widened))

    def measure(self, route):
        """Length of a floor route in centimetres, from the robot outward."""
        total, previous = 0., (0., 0.)
        for spot in route:
            total += math.hypot(spot[0] - previous[0], spot[1] - previous[1])
            previous = spot
        return total

    def to_floor(self, x, y):
        """Canvas click to floor position, in whichever view is on show."""
        if self.view_mode == 'camera':
            try:
                return self.robot.ground(float(x), float(y))
            except fetch.Stop:
                return None            # at or above the horizon: not floor
        return bev_to_floor(float(x), float(y))

    def project(self, points):
        """Canvas clicks to floor positions."""
        spots, total, previous = [], 0., (0., 0.)
        for index, point in enumerate(points or []):
            spot = self.to_floor(point[0], point[1])
            if spot is None:
                self.say('point %d is at or above the horizon, so it is not a '
                         'place on the floor' % (index + 1))
                break
            if spot[1] <= 0.:
                self.say('point %d is level with or behind the robot' % (index + 1))
                break
            if math.hypot(*spot) > fetch.ROUTE_RANGE_CM:
                self.say('point %d is %.0f cm away, past the %.0f cm the floor '
                         'projection is trusted to' %
                         (index + 1, math.hypot(*spot), fetch.ROUTE_RANGE_CM))
                break
            leg = math.hypot(spot[0] - previous[0], spot[1] - previous[1])
            self.say('point %d -> (%6.1f, %6.1f) cm   %5.1f cm from the last%s'
                     % (index + 1, spot[0], spot[1], leg,
                        '   <-- too close, will be skipped'
                        if leg < fetch.ROUTE_MIN_LEG_CM else ''))
            total += leg
            previous = spot
            spots.append(spot)
        # Arm the live overlay with what has just been drawn, so the stream
        # shows the route on the carpet before anything is driven.
        self.live_route = list(spots)
        self.live_pose = [0., 0., 0.]
        self.pose_heading = self.imu_heading() if spots else None
        return spots, total

    def to_pixel(self, right, forward):
        """Floor position to canvas pixel, in whichever view is on show."""
        if self.view_mode == 'camera':
            place = self.robot.pixel(right, forward)
            return list(place) if place else None
        return list(floor_to_bev(right, forward))

    def look(self, mode):
        """Switch between the metric plan view and the raw photograph."""
        if mode not in ('floor', 'camera'):
            return self.state()
        self.view_mode = mode
        self.lines = []
        self.say('showing the %s' % ('camera image' if mode == 'camera'
                                     else "bird's-eye floor view"))
        if mode == 'camera':
            self.say('clicks are projected onto the carpet through the lens, so '
                     'the same click is worth more centimetres near the horizon')
        return self.state(frame=True, points=[], hazards=[], length=0.)

    def _camera_pixel(self, right, forward):
        down = np.array([0., math.cos(self.robot.pitch), math.sin(self.robot.pitch)])
        ahead = np.array([0., 0., 1.]) - down * down[2]
        ahead /= np.linalg.norm(ahead)
        side = np.cross(down, ahead)
        ray = side * right + ahead * forward + down * self.robot.height
        norm = np.linalg.norm(ray)
        if norm < 1e-6:
            return None
        out, _ = cv2.fisheye.projectPoints((ray / norm).reshape(1, 1, 3),
                                           np.zeros(3), np.zeros(3),
                                           self.robot.K, self.robot.D)
        x, y = out.reshape(2)
        return [float(x), float(y)]

    def turn(self, degrees):
        if not -180. <= degrees <= 180. or degrees == 0.:
            return self.state()
        try:
            turned = self.robot.turn(degrees)
            self.heading += turned
            shift = fetch.pivot_shift(turned)
            self.advance([shift[0], shift[1], turned])
            self.say('turn: asked %+.0f deg, measured %+.1f deg (error %+.1f)'
                     % (degrees, turned, turned - degrees))
        except fetch.Stop as exc:
            self.robot.halt()
            self.say('turn STOPPED: %s' % exc)
        return self.capture_after()

    def snapshot(self, route, pose, reached):
        """Photograph from here, with the whole route redrawn in this frame.

        Drawn from the floor positions the operator chose, moved into the pose
        the robot believes it now holds. Waypoints already driven past fall
        behind the camera and simply have no pixel, so they drop out; the ones
        still ahead should land on exactly the carpet they were put on. If they
        drift as the strip goes on, the odometry is lying about the motion.
        """
        try:
            image, _ = self.robot.frame()
        except Exception:
            return
        shown = 0
        for index, spot in enumerate(fetch.rebase(route, pose)):
            if spot[1] <= 0.:
                continue
            place = self.robot.pixel(spot[0], spot[1])
            if not place or not (0 <= place[0] < image.shape[1]
                                 and 0 <= place[1] < image.shape[0]):
                continue
            shown += 1
            cv2.drawMarker(image, (int(place[0]), int(place[1])), (255, 170, 60),
                           cv2.MARKER_CROSS, 24, 2)
            cv2.putText(image, str(index + 1),
                        (int(place[0]) + 9, int(place[1]) - 9),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 170, 60), 2)
        cv2.putText(image, 'after waypoint %d  (%.0f cm, %+.0f deg travelled)  '
                           '%d of %d still ahead'
                    % (reached, math.hypot(pose[0], pose[1]), pose[2], shown,
                       len(route)),
                    (8, 22), cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 170, 60), 2)
        ok, encoded = cv2.imencode('.jpg', image)
        if ok:
            self.strip.append(encoded.tobytes())

    def advance(self, pose):
        """Fold a motion into where the robot stands relative to its memory.

        The memory is held in the frame it was answered in, so every motion
        since has to be composed onto the same running pose -- a turn from the
        buttons counts just as much as a driven route.
        """
        heading = math.radians(self.since[2])
        self.since = [
            self.since[0] + math.cos(heading) * pose[0] + math.sin(heading) * pose[1],
            self.since[1] + math.cos(heading) * pose[1] - math.sin(heading) * pose[0],
            self.since[2] + pose[2]]

    def imu_heading(self):
        """The IMU's fused heading right now, or None if it cannot be read."""
        try:
            samples = fetch.call('observation', timeout=2.).get('imu_samples') or []
            return self.robot.heading(samples[-1]) if samples else None
        except Exception:
            return None

    def overlay_route(self, image, heading_now=None):
        """Draw the running route into a live frame, from where the robot is.

        The waypoints were placed on a photograph taken from somewhere the robot
        has since left, so they are moved into the pose it believes it now holds
        and projected through the lens again. Points it has driven past fall
        behind the camera and have no pixel, so they drop out by themselves.

        Heading is taken from the observation that carried this very frame when
        one is available: pose updates arrive per odometry sample, but a turn
        moves the view between them, and using the stale heading makes the marks
        visibly lag the picture.
        """
        if not self.live_route:
            return image
        pose = list(self.live_pose)
        if heading_now is not None and self.pose_heading is not None:
            pose[2] += fetch.Robot.unwrap(heading_now - self.pose_heading)
        drawn = 0
        previous = None
        for index, spot in enumerate(fetch.rebase(self.live_route, pose)):
            if spot[1] <= 0.:
                previous = None
                continue
            place = self.robot.pixel(spot[0], spot[1])
            if not place or not (0 <= place[0] < image.shape[1]
                                 and 0 <= place[1] < image.shape[0]):
                previous = None
                continue
            here = (int(place[0]), int(place[1]))
            if previous:
                cv2.line(image, previous, here, (255, 210, 90), 2)
            cv2.drawMarker(image, here, (255, 210, 90), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(image, str(index + 1), (here[0] + 8, here[1] - 8),
                        cv2.FONT_HERSHEY_PLAIN, 1.1, (255, 210, 90), 2)
            previous = here
            drawn += 1
        cv2.putText(image,
                    'your route: %d of %d waypoints still ahead'
                    % (drawn, len(self.live_route)),
                    (8, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 210, 90), 2)
        return image

    def live_jpeg(self, width=None):
        """The newest camera frame, with the running route drawn into it.

        The service runs its own camera worker that grabs continuously, so an
        observation returns the most recent frame rather than taking a new one:
        streaming does not compete with the odometry for the sensor. The same
        reply carries the IMU, so the heading used to place the route is the one
        that belongs to this picture rather than the last one recorded.

        With nothing to draw and no width set, the bytes are handed on exactly
        as the camera made them -- no decode, no encode. Otherwise the frame is
        decoded once, drawn on at full resolution (the projection speaks in the
        camera's own pixels), and shrunk last.
        """
        try:
            observation = fetch.call('observation', timeout=2.)
        except Exception:
            return None
        blob = observation.get('jpeg_base64')
        if not blob:
            return None
        try:
            jpeg = base64.b64decode(blob)
        except (ValueError, TypeError):
            return None

        width = STREAM_WIDTH if width is None else width
        drawing = bool(self.live_route)
        if not width and not drawing:
            return jpeg          # the cheap path: hand the bytes straight on

        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jpeg
        if drawing:
            samples = observation.get('imu_samples') or []
            heading = None
            if samples:
                try:
                    heading = self.robot.heading(samples[-1])
                except Exception:
                    heading = None
            self.overlay_route(image, heading)
        scale = float(width) / image.shape[1] if width else 1.
        if scale < 1.:
            image = cv2.resize(image, (int(width), int(round(image.shape[0] * scale))),
                               interpolation=cv2.INTER_AREA)
        ok, small = cv2.imencode('.jpg', image,
                                 [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_QUALITY])
        return small.tobytes() if ok else jpeg

    def refresh(self):
        """A new picture for the live view, without disturbing the run state.

        Deliberately not capture(): that resets the heading and the pose the
        remembered answer is measured against. This only replaces pixels.
        """
        try:
            image, _ = self.robot.frame()
        except Exception:
            return
        self.frame = image
        if self.view_mode != 'camera':
            self.view = self.warp.apply(image)   # the warp is not free; skip it
                                                 # when it is not on screen

    def capture_after(self):
        try:
            image, _ = self.robot.frame()
            self.frame = image
            self.view = self.warp.apply(image)
        except Exception as exc:
            self.say('could not refresh the picture: %s' % exc)
        return self.state(frame=True)

    def run(self, points):
        """Drive the drawn path, reporting asked-for against measured per leg."""
        route, planned = self.project(points)
        if len(route) < 1:
            self.say('nothing to drive: put at least one usable point on the carpet')
            return self.state(length=0.)
        self.say('running %d waypoints, %.0f cm of path' % (len(route), planned))
        self.say('asked      measured   drift   vision')
        travelled = [0.]
        moved = []          # the same floor points, seen from where it ends up
        self.strip = []
        self.live_route = list(route)
        self.live_pose = [0., 0., 0.]
        self.pose_heading = self.imu_heading()

        def record(event, **fields):
            if event == 'pose':
                # Silent on purpose: this arrives about eight times a second
                # while a leg runs, and it is for the picture, not the log.
                self.live_pose = [fields.get('x', 0.), fields.get('z', 0.),
                                  fields.get('heading', 0.)]
                self.pose_heading = self.imu_heading()
                return
            if event == 'leg':
                travelled[0] += fields.get('moved_cm', 0.)
                self.say('%6.1f cm %7.1f cm %6.1f deg %5s'
                         % (fields.get('wanted_cm', 0.), fields.get('moved_cm', 0.),
                            fields.get('drift_deg', 0.), fields.get('by', '')))
            elif event == 'faced':
                # Show all three numbers: a bearing larger than what was asked
                # means the turn was chunked (another follows), while asked but
                # not turned is the wheels falling short.
                self.say('turn to waypoint %d: need %+.0f, asked %+.0f, got %+.0f'
                         '%s'
                         % (fields.get('waypoint', 0) + 1,
                            fields.get('bearing_deg', 0.),
                            fields.get('asked_deg', 0.),
                            fields.get('turned_deg', 0.),
                            '  (%+.0f still to go)' % fields['remaining_deg']
                            if abs(fields.get('remaining_deg', 0.)) > 1. else ''))
            elif event == 'waypoint':
                pose = fields.get('pose') or [0., 0., 0.]
                self.snapshot(route, pose, fields.get('n', 0) + 1)
                self.say('waypoint %d %s, %.1f cm from where it was drawn'
                         % (fields.get('n', 0) + 1,
                            'reached' if fields.get('reached') else 'NOT reached',
                            fields.get('short_by_cm', 0.)))
            elif event == 'stalled':
                self.say('waypoint %d STALLED: asked %.1f cm, moved %.1f cm, '
                         'still %.1f cm short -- wheels are not turning the floor'
                         % (fields.get('waypoint', 0) + 1, fields.get('asked_cm', 0.),
                            fields.get('moved_cm', 0.), fields.get('short_by_cm', 0.)))
            elif event == 'unreached':
                self.say('waypoint %d GAVE UP after %d legs, still %.1f cm short'
                         % (fields.get('waypoint', 0) + 1, fields.get('legs', 0),
                            fields.get('short_by_cm', 0.)))
            elif event == 'skipped':
                self.say('waypoint %d SKIPPED: only %.1f cm from the one before '
                         '(minimum %.0f cm) -- perspective bunches clicks near '
                         'the bottom of the frame'
                         % (fields.get('waypoint', 0) + 1, fields.get('leg_cm', 0.),
                            fields.get('minimum_cm', 0.)))
            elif event in ('shortened', 'leg_blocked'):
                self.say('%s: %s' % (event, json.dumps(fields)))
        try:
            pose = fetch.follow(self.robot, self.odometer, route, record)
            self.heading += pose[2]
            self.advance(pose)
            moved = fetch.rebase(route, pose)
            asked_end = route[-1]
            error = math.hypot(pose[0] - asked_end[0], pose[1] - asked_end[1])
            self.say('')
            self.say('asked to finish at (%.1f, %.1f) cm, believes it is at (%.1f, %.1f)'
                     % (asked_end[0], asked_end[1], pose[0], pose[1]))
            self.say('miss %.1f cm over %.0f cm of path (%.0f%%), final heading %+.1f deg'
                     % (error, planned, 100. * error / max(1., planned), pose[2]))
            self.say('NOTE: that is where the robot THINKS it is, from its own '
                     'odometry. Compare it against the picture.')
        except fetch.Stop as exc:
            self.robot.halt()
            self.say('')
            self.say('STOPPED: %s' % exc)
        # The whole point of re-photographing here: the route was drawn on a
        # picture taken from somewhere the robot no longer is. Carrying it into
        # the new frame is what lets the next look build on this one, and
        # drawing it on the new photograph is how you can see whether the
        # arithmetic is right -- the markers should land on the same carpet.
        state = self.capture_after()
        shown, lost, behind = [], 0, 0
        wide = self.frame.shape[1] if self.view_mode == 'camera' else BEV_PIXELS
        tall = self.frame.shape[0] if self.view_mode == 'camera' else BEV_PIXELS
        for spot in moved:
            if spot[1] <= 0.:
                behind += 1
                continue
            pixel = self.to_pixel(spot[0], spot[1])
            if pixel and 0 <= pixel[0] < wide and 0 <= pixel[1] < tall:
                shown.append([int(pixel[0]), int(pixel[1])])
            else:
                lost += 1
        if moved:
            self.say('')
            self.say('reprojected the same %d floor points into the new picture: '
                     '%d still in view, %d driven past, %d outside the frame'
                     % (len(moved), len(shown), behind, lost))
            if shown:
                self.say('they should sit on the same carpet as before -- if '
                         'they have slid, the odometry or the projection is off')
            else:
                self.say('none are left to draw: the robot drove the whole route, '
                         'so every waypoint is now behind the camera. Draw a '
                         'longer path, or STOP part way, to see this work.')
        # capture_after() snapshotted the log before those lines were written.
        if self.strip:
            self.say('%d frames taken along the way, each with the route '
                     'redrawn from where the robot then stood -- step through '
                     'them to see whether the marks stay on the same carpet'
                     % len(self.strip))
        state.update(length=planned, points=shown, reprojected=True,
                     strip=len(self.strip), log=self.log())
        return state

    def halt(self):
        self.abort.set()
        self.robot.halt()
        self.say('operator stop')
        return self.state()


class Handler(BaseHTTPRequestHandler):
    server_version = 'PlannerBench/1'

    def _local(self):
        return self.client_address[0] in ('127.0.0.1', '::1') or self.server.open_lan

    def _stream(self):
        """Push frames until the viewer goes away.

        multipart/x-mixed-replace, so the browser renders it in a plain <img>
        and holds one connection instead of asking for a new picture several
        times a second. Nothing here touches the bench lock: the stream has to
        keep running while a trajectory does, which is the only time watching it
        is interesting.
        """
        bench = self.server.bench
        with self.server.streams_lock:
            if self.server.streams >= STREAM_LIMIT:
                return self._send(b'{"error":"too many viewers"}')
            self.server.streams += 1
        try:
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type',
                             'multipart/x-mixed-replace; boundary=jetbotframe')
            self.end_headers()
            while True:
                jpeg = bench.live_jpeg()
                if jpeg:
                    self.wfile.write(b'--jetbotframe\r\nContent-Type: image/jpeg\r\n')
                    self.wfile.write(('Content-Length: %d\r\n\r\n'
                                      % len(jpeg)).encode('ascii'))
                    self.wfile.write(jpeg)
                    self.wfile.write(b'\r\n')
                time.sleep(1. / max(.5, STREAM_FPS))
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass                 # the tab was closed; that is how a stream ends
        finally:
            with self.server.streams_lock:
                self.server.streams -= 1

    def _send(self, body, kind='application/json'):
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self._local():
            return self._send(b'{"error":"local only"}')
        path = urlparse(self.path).path
        if path == '/':
            return self._send(PAGE.encode('utf-8'), 'text/html; charset=utf-8')
        if path == '/api/progress':
            # Deliberately outside the lock: a mission holds the robot for
            # minutes, and this is the only way to watch it do so.
            bench = self.server.bench
            def on_view(spot):
                """Waypoint as a pixel, pulled to the border if it is off-frame.

                At this camera height the near field is narrow, so a route that
                swings around anything leaves the picture almost at once -- five
                of six waypoints in one measured answer. Dropping them showed
                the model's route as a single dot, which read as the overlay not
                working at all. An edge marker at least says which way it went.
                """
                place = bench.to_pixel(spot[0], spot[1])
                if not place:
                    return None
                wide = 640 if bench.view_mode == 'camera' else BEV_PIXELS
                tall = 480 if bench.view_mode == 'camera' else BEV_PIXELS
                inside = 0 <= place[0] < wide and 0 <= place[1] < tall
                x = max(3, min(wide - 4, int(place[0])))
                y = max(3, min(tall - 4, int(place[1])))
                return [x, y] if inside else [x, y, 1]

            route = [p for p in (on_view(s) for s in bench.live_route) if p]
            hazards = []
            for label, spot in bench.live_obstacles:
                place = on_view(spot)
                if place:
                    hazards.append([place[0], place[1],
                                    (fetch.OBSTACLE_RADIUS_CM
                                     + fetch.CORRIDOR_HALF_CM) * BEV_SCALE,
                                    label[:16]])
            return self._send(json.dumps(dict(
                log=bench.log(), running=bench.running(),
                power=bench.power(), heading=bench.heading,
                strip=len(bench.strip), points=route, hazards=hazards,
                note=bench.last_note, frame_token=bench.frame_token)).encode())
        if path == '/api/stream.mjpg':
            return self._stream()
        if path.startswith('/api/step'):
            bench = self.server.bench
            try:
                index = int((urlparse(self.path).query.split('i=')[1]).split('&')[0])
            except (IndexError, ValueError):
                index = 0
            strip = list(bench.strip)   # a list of finished JPEG bytes; never
                                        # block a picture request on the run
            if not strip:
                return self._send(b'', 'image/jpeg')
            return self._send(strip[max(0, min(index, len(strip) - 1))], 'image/jpeg')
        if path == '/api/frame':
            bench = self.server.bench
            # Never block here. The run holds the lock for the whole trajectory,
            # so waiting on it froze the picture until the robot stopped -- and
            # grabbing a frame while the wheels are turning steals camera time
            # from the odometry, which is what starves the IMU loop. So: take a
            # new picture only when nothing else is using the robot, and serve
            # the last one otherwise.
            fresh = 'live=1' in (urlparse(self.path).query or '')
            held = bench.lock.acquire(False)
            try:
                if held and fresh:
                    bench.refresh()
                image = bench.frame if bench.view_mode == 'camera' else bench.view
            finally:
                if held:
                    bench.lock.release()
            if image is None:
                return self._send(b'', 'image/jpeg')
            ok, encoded = cv2.imencode('.jpg', image)
            return self._send(encoded.tobytes() if ok else b'', 'image/jpeg')
        return self._send(b'{"error":"not found"}')

    def do_POST(self):
        if not self._local():
            return self._send(b'{"error":"local only"}')
        length = int(self.headers.get('Content-Length', '0') or 0)
        try:
            body = json.loads(self.rfile.read(length) or b'{}')
        except ValueError:
            body = {}
        bench = self.server.bench
        path = urlparse(self.path).path
        if path == '/api/halt':          # never queues behind a running leg
            return self._send(json.dumps(bench.halt()).encode())
        if bench.running() and path not in ('/api/mission', '/api/search',
                                            '/api/flow'):
            # A mission owns the robot for minutes and runs on its own thread,
            # so the request lock is free the whole time. Without this, a page
            # reload takes a picture mid-mission: the camera moves under the
            # planner and the log it is writing is cleared. STOP still works.
            return self._send(json.dumps(bench.state(
                log=bench.log() + '\n(a mission is running; press STOP first)'
            )).encode())
        if not bench.lock.acquire(False):
            return self._send(json.dumps(dict(log=bench.log() + '\nbusy')).encode())
        try:
            if path == '/api/capture':
                out = bench.capture()
            elif path == '/api/turn':
                out = bench.turn(float(body.get('degrees', 0.)))
            elif path == '/api/run':
                out = bench.run(body.get('points') or [])
            elif path == '/api/search':
                # Same shape as a mission: it owns the robot for minutes and
                # reports through the same log, so the page treats it the same.
                every = body.get('seconds')
                out = bench.hunt(body.get('target') or '',
                                 MISSION_SECONDS if every in (None, '')
                                 else float(every))
            elif path == '/api/flow':
                every = body.get('every')
                out = bench.launch_flow(body.get('target') or '',
                                        FLOW_INTERVAL_S if every in (None, '')
                                        else float(every))
            elif path == '/api/mission':
                every = body.get('seconds')
                out = bench.launch(body.get('target') or '',
                                   None if every in (None, '') else float(every))
            elif path == '/api/look':
                out = bench.look(body.get('mode') or 'floor')
            elif path == '/api/ask':
                out = bench.ask(body.get('target') or '')
            elif path == '/api/project':
                spots, total = bench.project(body.get('points') or [])
                out = bench.state(length=total)
            else:
                out = dict(error='not found')
        except Exception as exc:
            bench.say('error: %s: %s' % (type(exc).__name__, exc))
            out = bench.state()
        finally:
            bench.lock.release()
        return self._send(json.dumps(out).encode())

    def log_message(self, *args):
        pass


class Server(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    streams = 0                 # live viewers; guarded by streams_lock
    streams_lock = threading.Lock()


def apply_stream_settings(fps, width):
    """Module level so the stream loop and live_jpeg both see the choice."""
    global STREAM_FPS, STREAM_WIDTH
    STREAM_FPS, STREAM_WIDTH = float(fps), int(width)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1',
                        choices=('127.0.0.1', '0.0.0.0'))
    parser.add_argument('--port', type=int, default=8770)
    parser.add_argument('--enable-motion', action='store_true')
    parser.add_argument('--stream-fps', type=float, default=STREAM_FPS,
                        help='frames a second offered to a live viewer')
    parser.add_argument('--stream-width', type=int, default=STREAM_WIDTH,
                        help='shrink streamed frames to this width; 0 passes the '
                             'camera JPEG through without decoding it')
    args = parser.parse_args()
    apply_stream_settings(args.stream_fps, args.stream_width)
    server = Server((args.host, args.port), Handler)
    server.bench = Bench(args.enable_motion)
    server.open_lan = args.host == '0.0.0.0'
    print('planner bench: http://%s:%d/  (%s)' % (
        args.host, args.port,
        'motors armed' if args.enable_motion else 'preview only'), flush=True)
    try:
        server.serve_forever()
    finally:
        try:
            server.bench.robot.halt()
        except Exception:
            pass


if __name__ == '__main__':
    main()
