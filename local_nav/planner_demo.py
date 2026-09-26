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
import inspection_sequence

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
.pane{flex:0 0 auto;width:1130px;max-width:100%}
.views{display:flex;flex-wrap:wrap;gap:10px;align-items:flex-start}
.viewbox{flex:0 0 auto}
.vlabel{font:10px monospace;letter-spacing:.08em;color:#7d8a93;padding:0 0 4px 2px}
.stage.alt{width:640px;height:480px}
.stage.alt img,.stage.alt canvas{width:640px;height:480px}
.stage.alt.floorview{width:480px;height:480px}
.stage.alt.floorview img,.stage.alt.floorview canvas{width:480px;height:480px}
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
  <div class="views">
    <div class="viewbox"><div class="vlabel" id="mainlabel">FLOOR &mdash; click to draw</div>
      <div class="stage"><img id="shot"><canvas id="pad" width="480" height="480"></canvas></div></div>
    <div class="viewbox"><div class="vlabel" id="altlabel">CAMERA</div>
      <div class="stage alt"><img id="altshot"><canvas id="altpad" width="640" height="480"></canvas></div></div>
  </div>
  <div class="controls">
    <div class="group">
      <button id="grab">TAKE PICTURE</button>
      <button id="view">VIEW: FLOOR</button>
    </div>
    <div class="group">
      <span>turn</span><input id="deg" type="number" value="30" min="1" max="180">
      <button id="left">&#8630; LEFT</button><button id="right">RIGHT &#8631;</button>
      <button id="check" title="mark the carpet, turn, and redraw the same floor
        points -- the crosses should stay put">CHECK REPROJ</button>
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
      <span>for</span><input id="runfor" type="number" min="1" max="60"
        placeholder="all" style="width:54px" title="drive only this many seconds
        of the trajectory, then redraw what is left"><span>s</span>
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
    RUN TRAJECTORY with a number in <em>for N s</em> drives only that much of
    the path and then redraws what is left from where it stopped &mdash; the
    leftover waypoints should sit on the same carpet they were drawn on.<br>
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
const apad=document.getElementById('altpad'), actx=apad.getContext('2d');
let altPoints=[], altHazards=[];
let points=[], busy=false, hazards=[], reprojected=false, mode='floor';
// Both views are painted by this one function. Two copies of the drawing code
// would drift apart, and the whole point of showing the plan view beside the
// photograph is that they are the same floor points -- so any difference on
// screen has to come from the projection, never from the painting.
function paint(c,pts,haz){
  c.clearRect(0,0,c.canvas.width,c.canvas.height);
  if(pts && pts.length){
    const tint = reprojected ? '#9d7bff' : '#43d4ff';
    c.strokeStyle=tint; c.lineWidth=2.5; c.setLineDash([9,5]);
    c.beginPath(); c.moveTo(c.canvas.width/2,c.canvas.height-1);
    pts.forEach(p=>c.lineTo(p[0],p[1])); c.stroke(); c.setLineDash([]);
    pts.forEach((p,i)=>{c.beginPath();c.arc(p[0],p[1],5,0,7);
      if(p[2]){c.strokeStyle=tint;c.lineWidth=2;c.stroke();}
      else {c.fillStyle=tint;c.fill();}
      c.fillStyle=p[2]?tint:'#0a0d10';c.font='10px monospace';
      c.fillText(i+1,p[0]-2,p[1]+3);});
  }
  (haz||[]).forEach(h=>{
    c.strokeStyle='#ffa53d'; c.lineWidth=1.5;
    c.beginPath(); c.arc(h[0],h[1],h[2],0,7); c.stroke();
    c.fillStyle='#ffa53d'; c.beginPath(); c.arc(h[0],h[1],3,0,7); c.fill();
    c.font='10px monospace'; c.fillText(h[3],h[0]+7,h[1]-5);
  });
}
function draw(){
  paint(ctx,points,hazards);
  paint(actx,altPoints,altHazards);
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
function buttons(){['grab','left','right','run','clear','ask','view','go','flowgo','find','check'].forEach(
  id=>document.getElementById(id).disabled=busy);}
function show(d){
  if(d.log) document.getElementById('log').textContent=d.log;
  if(d.frame) refreshShots();
  if(d.heading!==undefined) document.getElementById('hdg').textContent=d.heading.toFixed(1);
  if(d.speed!==undefined) document.getElementById('spd').textContent=d.speed.toFixed(1);
  if(d.length!==undefined) document.getElementById('len').textContent=d.length.toFixed(0);
  if(d.alt_hazards) altHazards=d.alt_hazards;
  if(d.alt_points) altPoints=d.alt_points;
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
// Both pictures come from the same capture, so they are refreshed together.
// Letting one lag leaves markers drawn over floor the robot has left.
function refreshShots(){
  const t=Date.now();
  document.getElementById('shot').src='/api/frame?t='+t;
  document.getElementById('altshot').src=
    '/api/frame?view='+(mode==='camera'?'floor':'camera')+'&t='+t;
}
function applyView(){
  document.getElementById('view').textContent =
    'VIEW: '+(mode==='floor'?'FLOOR':'CAMERA');
  // The canvas backing store has to match the picture underneath it: the plan
  // view is 480 square, the camera image is 640x480. Getting this wrong puts
  // every click and every drawn marker in the wrong place.
  document.querySelector('.stage').classList.toggle('camera', mode==='camera');
  document.getElementById('floorhint').hidden = mode==='camera';
  document.getElementById('camhint').hidden = mode!=='camera';
  pad.width = mode==='camera' ? 640 : 480;
  pad.height = 480;
  // The second view is always the other one, and its canvas has to match its
  // own picture for exactly the reason the first one does: a backing store
  // that disagrees with the image under it puts every marker in the wrong
  // place. VIEW no longer chooses what you can see -- both are on screen -- it
  // chooses which picture your clicks land on.
  const altIsFloor = mode==='camera';
  document.querySelector('.stage.alt').classList.toggle('floorview', altIsFloor);
  apad.width = altIsFloor ? 480 : 640;
  apad.height = 480;
  document.getElementById('mainlabel').innerHTML =
    (mode==='camera'?'CAMERA':'FLOOR')+' &mdash; click to draw';
  document.getElementById('altlabel').textContent = altIsFloor ? 'FLOOR' : 'CAMERA';
  document.getElementById('altshot').src =
    '/api/frame?view='+(altIsFloor?'floor':'camera')+'&t='+Date.now();
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
      if(d.alt_points) altPoints=d.alt_points;
      if(d.alt_hazards) altHazards=d.alt_hazards;
      if(d.points||d.hazards) draw();
      if(d.note!==undefined) document.getElementById('note').textContent=d.note;
      if(d.frame_token!==undefined && d.frame_token!==seenFrame){
        seenFrame=d.frame_token;
        refreshShots();
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
document.getElementById('check').onclick=async()=>{
  // Marks floor points, turns by the degrees box, and redraws the same points.
  // Two frames in the stepper: the crosses should stay on the same carpet.
  document.getElementById('check').textContent='CHECKING...';
  busy=true; buttons();
  await post('/api/check',{degrees:+document.getElementById('deg').value});
  document.getElementById('check').textContent='CHECK REPROJ';
  busy=false; buttons();
};
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
  await post('/api/run',{points,
    seconds:document.getElementById('runfor').value.trim()||null});
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
MISSION_STANDOFF_CM = 20.  # range to the target's contact pixel at which the
# A leg shorter than this is not progress, whatever the executor reports it
# as. Below fetch.ROUTE_MIN_LEG_CM, so a leg the router would not have
# bothered to plan does not count as having driven one.
MISSION_MIN_PROGRESS_CM = 3.
                           # mission declares arrival. For a wide flat object the
                           # contact pixel migrates to its near edge as the robot
                           # closes, so the robot ends up nearer than this number.
                           # Measured 2026-09-24 on the same box: aiming for 20 cm
                           # from a point ranged at 46 cm left it 5 cm away. 35 cm
                           # over-corrected and stopped it well short. Until the
                           # standoff is measured to the nearest part of the target
                           # rather than one pixel, 20 is the operator's choice.
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
# How far the final leg may be off the bearing to the target before it is
# re-aimed, and how long the leg added to re-aim it is. The tolerance is
# wide enough that a route already pointing roughly at the goal is left
# alone; the leg is short enough to stay inside one look.
FACING_TOLERANCE_DEG = 15.
FACING_LEG_CM = 12.
MISSION_HOLD_FALLBACKS = 3  # consecutive holds answered with the remembered
                            # route before the mission really does give up.
                            # Bounded so a stale route cannot drive forever.
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
        self.live_drawn = False
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
        self._live_route = []
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
                ('armed' if status.get('motion_enabled')
                 else 'preview only') if supply.get('motion_allowed')
                else 'BLOCKED')
        except Exception as exc:
            return 'service: %s' % exc

    # live_route is armed from a dozen places -- every loop, every turn, every
    # clear -- and only one of them is the operator drawing. Tracking that with
    # a flag each site had to remember to set is how this file produced the same
    # class of bug three times, so the flag rides on the assignment: arming a
    # route clears it and project() alone sets it again afterwards. A rebase is
    # the one case that must carry it across, and does so explicitly.
    @property
    def live_route(self):
        return self._live_route

    @live_route.setter
    def live_route(self, spots):
        self._live_route = spots
        self.live_drawn = False

    def on_view(self, spot, mode=None):
        """Floor position as a canvas pixel, or None when it is off the view."""
        mode = mode or self.view_mode
        place = self.to_pixel(spot[0], spot[1], mode)
        if not place:
            return None
        wide = 640 if mode == 'camera' else BEV_PIXELS
        tall = 480 if mode == 'camera' else BEV_PIXELS
        if 0 <= place[0] < wide and 0 <= place[1] < tall:
            return [int(place[0]), int(place[1])]
        return None

    def drawable(self, mode=None):
        """The armed route and obstacles as pixels of the view on show.

        One place, because there are two callers -- the reply to a command and
        the progress poll -- and when only one of them sent the obstacles the
        page redrew a turned route over obstacle circles still at their old
        screen positions. The robot had them right; the picture did not.
        """
        mode = mode or self.view_mode
        points = [p for p in (self.on_view(s, mode) for s in self.live_route) if p]
        if len(points) != len(self.live_route):
            points = None            # part of it is off view; do not half-draw
        hazards = []
        for item in self.live_obstacles:
            # Obstacles carry a measured half-extent since the metric contract;
            # unpacking two names off a three-tuple threw on every view request
            # and took the whole dashboard down mid-drive.
            label, spot = item[0], item[1]
            radius = float(item[2]) if len(item) > 2 and item[2] else fetch.OBSTACLE_RADIUS_CM
            radius = min(radius, fetch.OBSTACLE_MAX_RADIUS_CM)
            place = self.on_view(spot, mode)
            if place:
                hazards.append([place[0], place[1],
                                (radius + fetch.CORRIDOR_HALF_CM) * BEV_SCALE,
                                label[:16]])
        return points, hazards

    def both_views(self):
        """The armed route and obstacles drawn for each of the two views.

        A helper rather than two copies because there are two callers -- the
        reply to a command and the progress poll -- and the poll is the one
        that runs while the robot drives. Last time a field was added to the
        reply and missed on the poll, the page redrew a turned route over
        obstacle circles still at their old screen positions.
        """
        points, hazards = self.drawable()
        other = 'floor' if self.view_mode == 'camera' else 'camera'
        alt_points, alt_hazards = self.drawable(other)
        return (points, hazards, other, alt_points, alt_hazards)

    def state(self, **extra):
        out = dict(log=self.log(), heading=self.heading, speed=self.robot.speed,
                   power=self.power(), view_mode=self.view_mode,
                   strip=len(self.strip), note=self.last_note,
                   frame_token=self.frame_token,
                   mode='MOTION ENABLED' if self.enable_motion else 'PREVIEW ONLY')
        # Hand back whatever is armed, so a reloaded page draws the same thing
        # the live stream is drawing, and so a turn moves the obstacle circles
        # as well as the route.
        points, hazards, other, alt_points, alt_hazards = self.both_views()
        if points and 'points' not in extra:
            out['points'] = points
        if 'hazards' not in extra:
            out['hazards'] = hazards
        out.update(extra)
        # The other view, drawn from the same floor points. Sent alongside
        # rather than instead, because both are on screen: the plan view says
        # where things are in centimetres, the photograph says what they are,
        # and a route that looks right in one and wrong in the other is the
        # projection telling you something.
        out['alt_mode'] = other
        out['alt_points'] = alt_points
        out['alt_hazards'] = alt_hazards
        # A caller that clears the drawing clears both of them, or the second
        # view keeps showing a route the first has already dropped.
        if 'points' in extra and not extra['points']:
            out['alt_points'] = []
        if 'hazards' in extra and not extra['hazards']:
            out['alt_hazards'] = []
        return out

    def publish_snapshot(self, phase, route=None):
        """Publish one immutable image/geometry pair at a planner boundary."""
        previous = getattr(self, 'planner_snapshot', None)
        if phase == 'stopped' and previous and route is None:
            result = dict(previous)
            result.update(phase=phase, status_time=time.time(), note=self.log().split('\n')[-1],
                          retained_plan=True)
        else:
            if self.frame is None:
                return
            route = list(route or [])
            camera = self.frame.copy()
            floor = self.warp.apply(camera)
            images = {}
            for mode, image in [('camera', camera), ('floor', floor)]:
                last = None
                for index, spot in enumerate(route):
                    pixel = self.to_pixel(spot[0], spot[1], mode)
                    if pixel is None:
                        last = None; continue
                    here = tuple(int(v) for v in pixel)
                    if last is not None:
                        cv2.line(image, last, here, (255, 210, 90), 3)
                    cv2.circle(image, here, 5, (255, 210, 90), -1)
                    cv2.putText(image, str(index+1), (here[0]+7,here[1]-7),
                                cv2.FONT_HERSHEY_PLAIN, 1, (255,210,90), 2)
                    last = here
                ok, encoded = cv2.imencode('.jpg', image)
                if not ok: return
                images[mode] = 'data:image/jpeg;base64,'+base64.b64encode(encoded).decode('ascii')
            result = dict(phase=phase, image_time=getattr(self,"staged_at",time.time()), status_time=time.time(),
                          note=getattr(self,'last_note',''), route_cm=route,
                          images=images, retained_plan=False)
        result['id'] = (previous or {}).get('id', 0)+1
        self.planner_snapshot = result

    def restage(self, image):
        """Adopt `image` as the still picture behind both views.

        One method because seven places take a fresh photograph, and each has
        to do the same three things: keep the frame, rebuild the bird's-eye
        warp, and bump the token the page watches. Sites doing two of the three
        are why the top-down sat frozen for a whole run while the camera pane
        moved beside it -- two of them rebuilt the warp and never bumped the
        token, so the page was never told to reload the picture.

        The warp used to be skipped whenever the camera was the selected view,
        on the grounds that it was not on screen. Both views are on screen now,
        so it always is.
        """
        self.frame = image
        self.staged_at = time.time()
        self.view = self.warp.apply(image)
        self.frame_token += 1
        self.publish_snapshot("replanning" if self.running() else "stopped", [])

    def capture(self):
        image, _ = self.robot.frame()
        self.restage(image)
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
            self.restage(image)

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

    def remembered_plan(self, memory, pose, answer=None):
        """The previous route, rebased into this frame, as a plan to re-offer.

        Returns None when nothing usable survives -- no remembered route, or
        every point of it now falls outside the image. The caller still gets
        the ordinary clearance and commit treatment, so an unsafe remembered
        route is refused downstream exactly like a fresh one.
        """
        try:
            route = fetch.rebase(list(memory.get('route') or []), pose)
        except Exception:
            return None
        pixels = []
        for spot in route:
            if spot[1] <= 0.:
                continue                       # behind the robot: no pixel
            try:
                point = self.robot.pixel(spot[0], spot[1])
            except Exception:
                point = None
            if point and all(math.isfinite(v) for v in point) \
                    and 0 <= point[0] < 640 and 0 <= point[1] < 480:
                pixels.append(dict(x=round(float(point[0]), 1),
                                   y=round(float(point[1]), 1)))
        if not pixels:
            return None
        carried = (answer or {}).get('obstacles') or []
        return dict(motion='follow', visible=False, all_done=True,
                    contact_pixel=None, route_pixels=pixels, turn_degrees=None,
                    goal=(answer or {}).get('goal', 'remembered target'),
                    obstacles=carried,
                    note='controller fallback: remembered route re-offered after a planner hold')

    def launch(self, target, seconds=MISSION_SECONDS, initial_plan=None,
               seed_frame_token=None, carry_memory=True, metric=False):
        """Start a mission on its own thread; the page polls /api/progress.

        `seconds` of None means do not plan recurrently: take one look, drive
        what it planned, and stop.

        `initial_plan` is a high-level trajectory drawn on the currently staged
        camera frame.  It replaces only the first GPT request; after that first
        bounded movement the normal recurrent planner takes over.  Requiring
        the frame token prevents coordinates from an older picture being
        applied to a newer scene.
        """
        if self.running():
            self.say('a mission is already running; press STOP first')
            return self.state(mission=True)
        if initial_plan is not None:
            if not isinstance(initial_plan, dict):
                self.say('initial plan must be an object')
                return self.state(mission=False)
            points = initial_plan.get('route_pixels')
            if not isinstance(points, list) or not points:
                self.say('initial plan needs at least one route pixel')
                return self.state(mission=False)
            if self.frame is None or seed_frame_token != self.frame_token:
                self.say('initial plan rejected: its frame token is stale')
                return self.state(mission=False)
            for point in points:
                if (not isinstance(point, dict)
                        or not isinstance(point.get('x'), (int, float))
                        or not isinstance(point.get('y'), (int, float))
                        or not 0 <= point['x'] < 640 or not 0 <= point['y'] < 480):
                    self.say('initial plan has an invalid route pixel')
                    return self.state(mission=False)
            # The command body belongs to the HTTP request thread.  Copy it so
            # later caller mutation cannot change a trajectory already armed.
            initial_plan = json.loads(json.dumps(initial_plan))
            initial_plan.setdefault('motion', 'follow')
            initial_plan.setdefault('visible', bool(initial_plan.get('contact_pixel')))
            initial_plan.setdefault('obstacles', [])
            initial_plan.setdefault('note', 'high-level seeded trajectory')
        self.abort.clear()
        if not carry_memory:
            # Goal-only: no seed is used and none is accepted silently, so the
            # first look is the model's own and every look after it is too.
            if initial_plan is not None:
                self.say('goal-only run: the seeded trajectory is ignored')
            initial_plan = None
            self.say('goal-only: no remembered trajectory or obstacles are sent; '
                     'the model plans every look from the picture and the goal')
        if metric:
            self.say('metric contract: nothing is drawn on either picture; '
                     'the model answers in centimetres in the robot frame')
        self.flight = threading.Thread(target=self.mission,
                                       args=(target, seconds, initial_plan),
                                       kwargs=dict(carry_memory=carry_memory,
                                                   metric=metric),
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
    def discard_after_turn(job):
        """Keep the full response audit, but never execute this pre-turn view."""
        def finish():
            job['thread'].join()
            from gpt_audit import plan_event
            answer = job['holder'].get('answer')
            if answer is None:
                answer = getattr(job['holder'].get('error'), 'answer', {})
            plan_event(answer, 'discarded', reason='pre-turn plan invalidated',
                       mode='flow', sequence=job['seq'])
        worker = threading.Thread(target=finish)
        worker.daemon = True
        worker.start()

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

    def turn_then_follow_plan(self, answer, route, obstacles, goal, drift_heading=0.):
        """Execute one bounded planned turn, then rebase its paired floor route."""
        asked = answer.get('turn_degrees')
        if (not isinstance(asked, (int, float)) or isinstance(asked, bool)
                or not math.isfinite(asked) or not 0 < abs(asked) <= MISSION_TURN_STEP_DEG
                or not route):
            raise fetch.Stop('combined plan needs a route and a turn within 30 degrees')
        # The requested angle belongs to the capture heading, not reply arrival.
        remaining = (asked - drift_heading + 180.) % 360. - 180.
        if abs(remaining) > MISSION_TURN_STEP_DEG:
            raise fetch.Stop('combined turn became stale; needs a fresh plan')
        if self.abort.is_set():
            raise fetch.Stop('combined plan cancelled')
        shift = fetch.pivot_shift(remaining)
        predicted = fetch.rebase(route, [shift[0], shift[1], remaining])
        useful = next((p for p in predicted if math.hypot(*p) >= fetch.ROUTE_MIN_LEG_CM), None)
        if (useful is None or useful[1] <= 0 or
                abs(math.degrees(math.atan2(*useful))) > fetch.INITIAL_TURN_LIMIT_DEG or
                abs(fetch.aim_turn([0., 0., 0.], useful)) > fetch.INITIAL_TURN_LIMIT_DEG):
            raise fetch.Stop('combined route does not start near the post-turn heading')
        self.live_route = list(route)
        self.live_obstacles = list(obstacles)
        self.publish_snapshot('turn_then_follow', route)
        self.say('combined plan: turn %+.1f deg, then follow reprojected route' % remaining)
        turned = self.robot.turn(remaining) if abs(remaining) > fetch.HEADING_TOLERANCE_DEG else 0.
        shift = fetch.pivot_shift(turned)
        motion = [shift[0], shift[1], turned]
        self.advance(motion)
        route = fetch.rebase(route, motion)
        obstacles = list(zip([o[0] for o in obstacles],
                             fetch.rebase([o[1] for o in obstacles], motion)))
        goal = fetch.rebase([goal], motion)[0] if goal is not None else None
        from gpt_audit import plan_event
        plan_event(answer, 'turn_then_follow', requested_degrees=asked,
                   remaining_degrees=remaining, measured_pose=motion,
                   points_cm=route, obstacles_cm=obstacles, goal_cm=goal)
        return route, obstacles, goal, motion

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
        stale, trouble, spun = 0, 0, 0.
        rejected_starts = 0
        done = []                    # steps of the instruction already reached
        self.strip = []
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
                    self.restage(image)
                except Exception as exc:
                    self.say('lost the camera: %s' % exc)
                    break
                prior = (fetch.recall(self.robot, memory, [0., 0., 0.])
                         if memory else None)
                self.show_request(image, prior, 'look %d: what was sent' % (issued + 1))
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
                    from gpt_audit import plan_event
                    plan_event(answer, 'applied', mode='flow', sequence=newest['seq'])
                    points, found, spot, moved = build(answer, newest['shot'])
                    if answer.get('motion') == 'turn' and points:
                        drift = self.since_pose(newest['shot'], world)
                        try:
                            points, found, spot, turn_pose = self.turn_then_follow_plan(
                                answer, points, found, spot, drift[2])
                        except fetch.Stop as exc:
                            self.robot.halt()
                            self.say('PLANNER HOLD: %s' % exc)
                            break
                        world = self.compose(world, turn_pose)
                        for job in pending:
                            self.discard_after_turn(job)
                        stale += len(pending)
                        pending = []
                        applied = issued - 1
                        # Execute the paired route before scheduling another look.
                        last_issue = time.monotonic()
                    # Each accepted observation replaces the obstacle set.
                    # build() already compensates for motion since its image.
                    goal, obstacles = spot, found
                    self.last_note = str(answer.get('note', ''))[:120]
                    self.live_obstacles = list(obstacles)
                    if goal is not None and math.hypot(*goal) <= MISSION_STANDOFF_CM:
                        here = str(answer.get('goal') or target)[:40]
                        if answer.get('all_done', True):
                            self.say('')
                            self.say('REACHED: %s is %.0f cm away -- that was '
                                     'the last step' % (here, math.hypot(*goal)))
                            break
                        # One step of the instruction, not the end of the run.
                        if here not in done:
                            done.append(here)
                            self.say('reached %s (%d step(s) done); looking for '
                                     'what comes next' % (here, len(done)))
                        goal, route = None, []
                        memory = dict(route=[], obstacles=list(obstacles),
                                      goal=None, done=list(done))
                        continue
                    # The plan as the model drew it, kept apart from the
                    # trimmed thing the wheels get. Handing back the stub loses
                    # the shape it chose: ask for an approach from the left and
                    # the stub is a couple of points aimed straight ahead, so
                    # the next look has none of the curve to continue.
                    planned = list(points)
                    if goal is not None:
                        points = fetch.stop_short(points, goal, MISSION_STANDOFF_CM)
                    if points:
                        points, _ = fetch.avoid(points, obstacles, goal)
                        if goal is not None:
                            points = fetch.clip_standoff(points, goal,
                                                        MISSION_STANDOFF_CM)
                        points = fetch.cap_path(points, MISSION_COMMIT_CM) or points
                    route = points
                    memory = dict(route=list(planned), obstacles=list(obstacles),
                                  goal=goal, done=list(done))
                    # A turn the model asked for. Flowing has no way to drive
                    # out of being boxed in -- at 10 cm an obstacle blocks every
                    # heading until it is nearly behind -- so without this the
                    # wheels keep grinding at a route the executor refuses.
                    asked = answer.get('turn_degrees')
                    if (answer.get('motion') != 'turn_then_follow' and not route
                            and isinstance(asked, (int, float))
                            and math.isfinite(asked)
                            and abs(asked) >= MISSION_TURN_MIN_DEG):
                        asked = max(-MISSION_TURN_MAX_DEG,
                                    min(MISSION_TURN_MAX_DEG, float(asked)))
                        step = math.copysign(
                            min(abs(asked), MISSION_TURN_STEP_DEG), asked)
                        spun += abs(step)
                        if spun > MISSION_TURN_BUDGET_DEG:
                            self.say('  %.0f deg of turning without driving; '
                                     'stopping' % spun)
                            break
                        self.say('  no route; turning %+.0f of the %+.0f it asked'
                                 % (step, asked))
                        try:
                            turned = self.robot.turn(step)
                        except fetch.Stop as exc:
                            self.robot.halt()
                            if fatal(exc):
                                self.say('STOPPED: %s' % exc)
                                break
                            self.say('  could not turn: %s' % exc)
                            turned = 0.
                        if turned:
                            spot = fetch.pivot_shift(turned)
                            moved_by = [spot[0], spot[1], turned]
                            world = self.compose(world, moved_by)
                            # A pre-turn answer can immediately undo this view
                            # change. Keep audit records, but require a new image.
                            for job in pending:
                                self.discard_after_turn(job)
                            stale += len(pending)
                            pending = []
                            applied = issued - 1
                            last_issue = 0.
                            # Everything held in the old frame follows the turn.
                            obstacles = list(zip(
                                [o[0] for o in obstacles],
                                fetch.rebase([o[1] for o in obstacles], moved_by)))
                            if goal is not None:
                                goal = fetch.rebase([goal], moved_by)[0]
                            memory = dict(route=[], obstacles=list(obstacles),
                                          goal=goal, done=list(done))
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
                    if isinstance(newest['holder']['error'], fetch.PlannerHold):
                        self.robot.halt()
                        self.say('PLANNER HOLD: %s' % newest['holder']['error'].answer.get(
                            'note', 'no supported route'))
                        break
                    self.say('  a look failed: %s'
                             % str(newest['holder']['error'])[:60])

            if not route:
                time.sleep(.1)
                continue

            self.live_route = list(route)
            self.live_pose = [0., 0., 0.]
            self.pose_heading = self.imu_heading()
            self.publish_snapshot("executing", route)

            start_rejected = [False]

            def record(event, **fields):
                if event == 'route_start_rejected':
                    start_rejected[0] = True
                    self.say('  route start rejected: %.1f deg exceeds %.0f deg; replanning'
                             % (fields['bearing_deg'], fields['limit_deg']))
                elif event == 'pose':
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
                            and ('answer' in job['holder'] or isinstance(
                                job['holder'].get('error'), fetch.PlannerHold))):
                        return True
                return False

            slice_end = time.monotonic() + every
            try:
                pose = fetch.follow(self.robot, self.odometer, route, record,
                                    obstacles, deadline=slice_end,
                                    interrupt=fresher,
                                    initial_turn_limit_deg=fetch.INITIAL_TURN_LIMIT_DEG)
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
            if start_rejected[0]:
                rejected_starts += 1
                route = []
                last_issue = 0.
                if memory:
                    memory = dict(memory, route=[])
                if rejected_starts >= 3:
                    self.say('PLANNER HOLD: three routes require a sharp initial turn')
                    break
            else:
                rejected_starts = 0
            world = self.compose(world, pose)
            if math.hypot(pose[0], pose[1]) > 1.:
                spun = 0.
            # The part of the route still ahead belongs to the old frame too.
            route = [p for p in fetch.rebase(route, pose) if p[1] > 0.]
            obstacles = list(zip([o[0] for o in obstacles],
                                 fetch.rebase([o[1] for o in obstacles], pose)))
            # The goal moves with the robot too. Rebasing the route and the
            # obstacles but not the goal left the memory pointing at where the
            # target used to be, which is worse than having none.
            if goal is not None:
                goal = fetch.rebase([goal], pose)[0]
            if memory:
                # The remembered plan sits in the old frame like everything
                # else, so it is rebased rather than replaced by what is left
                # of the trimmed route.
                memory = dict(
                    route=[p for p in fetch.rebase(memory.get('route') or [], pose)
                           if p[1] > 0.],
                    obstacles=list(obstacles), goal=goal, done=list(done))

        try:
            self.robot.hold(0., 0.)
        except Exception:
            pass
        self.live_route = []
        elapsed = time.monotonic() - started_at
        self.say('')
        self.say('ended after %.0f s: %d looks issued, %d applied, %d stale dropped'
                 % (elapsed, issued, applied + 1, stale))

    def execute_mission_trajectory(self, route, seconds, record, obstacles):
        """Last-line clearance check, then hand a trajectory to Execution.

        This is a backstop, not a planner. Obstacle avoidance is the job of the
        layers that can see -- the mid-level planner and the operator above it
        -- and this must never rewrite a route, only refuse one. It tests the
        planner's own detections against the chassis, which is arithmetic this
        layer legitimately owns; it does not decide where to go.

        It was briefly deleted on 2026-09-24 on the argument that a layer with
        no vision should not overrule one with vision. The very next measurement
        refuted that: on call 099667ca the planner drew a route passing 5.9 cm
        from a green block with a 6 cm chassis half-width -- the robot's edge
        through it by a millimetre -- while its own note read "green block
        remains well right of approach". In pixels, the units the model is told
        to judge in, it left 44 px where scale_px_by_row required 125. The
        refusal was correct and it was the only thing that caught it.

        `along` is bounded by the leg length below. Without that an obstacle
        beyond the end of a leg still truncated it, because clear_distance
        projects onto an infinite heading ray; that is what produced deadlocks
        earlier the same day.
        """
        start=[0.,0.]
        for point in route:
            dx,dz=point[0]-start[0],point[1]-start[1]
            distance=math.hypot(dx,dz)
            room,blame=fetch.clear_distance(obstacles,start,math.degrees(math.atan2(dx,dz)),distance)
            if room+0.1<distance:raise fetch.Stop('mid-level clearance rejected trajectory: '+str(blame))
            start=point
        from trajectory_executor import Execution
        body=dict(frame_token=self.frame_token,trajectory=dict(waypoints_cm=[list(p) for p in route]))
        job=Execution(self,body,seconds=seconds)
        self.local_execution=job
        job.run()
        result=job.result()
        self.live_pose=list(result['pose_cm_deg'])
        self.frame=job.camera.copy()
        record('pose',x=self.live_pose[0],z=self.live_pose[1],heading=self.live_pose[2])
        if job.travelled>0:record('leg',moved_cm=job.travelled,wanted_cm=job.travelled,by='new executor vision + IMU')
        if job.phase not in ('completed','paused'):raise fetch.Stop(job.error or job.phase)
        return list(job.pose)

    def align_mission_target(self, goal, obstacles):
        """Bounded final facing turn; caller must acquire a fresh target observation."""
        from trajectory_executor import Execution,pivot_aim
        turn=max(-30.,min(30.,pivot_aim(goal)))
        # Check contact points against the rotating body plus pivot uncertainty.
        # Do not apply forward-corridor clearance plus a second assumed object
        # radius to in-place turns: that rejected separated side objects.
        for angle in np.linspace(0.,turn,max(2,int(abs(turn))+1)):
            shift=fetch.pivot_shift(float(angle))
            # Obstacles carry a measured half-extent, so they are three-tuples.
            # Unpacking two names here raised ValueError, and the caller only
            # catches fetch.Stop -- so the mission thread died silently right
            # after announcing the final facing turn, and the robot never made
            # it. Measured 2026-09-26 arriving at a can of nuts 17 cm away,
            # 49 degrees off-axis.
            for item in [(o[0],o[1]) for o in obstacles]+[('target',goal)]:
                label,point=item
                x,z=fetch.rebase([point],[shift[0],shift[1],float(angle)])[0]
                margin=2.  # measured pivot uncertainty; points represent nearest contacts
                if abs(x)<6.+margin and -15.-margin<z<margin:
                    raise fetch.Stop('final facing turn blocked by '+label)
        job=Execution(self,dict(frame_token=self.frame_token,
                     trajectory=dict(initial_turn_deg=turn,waypoints_cm=[])))
        self.local_execution=job;job.run()
        self.live_pose=list(job.pose);self.frame=job.camera.copy()
        if job.phase!='completed':raise fetch.Stop(job.error or job.phase)
        return list(job.pose)

    def mission(self, target, seconds=MISSION_SECONDS, initial_plan=None,
                carry_memory=True, metric=False):
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

        carry_memory=False runs the goal-only variant: nothing remembered is
        handed to the model except the goal itself. No high-level seed, no
        previous route of its own, no obstacle it reported on an earlier look.
        It receives the goal and the current picture, and draws the whole
        trajectory from that, every look, starting with the first -- which is
        not skipped. An obstacle it cannot see now does not exist. Measured 2026-09-24 on one replayed call: with the reference
        present the route passed 6.1 cm from an obstacle across four samples and
        none cleared the chassis; with the reference removed and nothing else
        changed the median was 14.2 cm and three of four cleared it. Continuity
        is what is traded away: each look is planned afresh, so the approach
        side can change between looks and a target that leaves the frame is not
        carried by a remembered path.
        """
        target = (target or '').strip()
        if not target:
            self.say('name something to drive to first')
            return
        if not os.environ.get('OPENAI_API_KEY'):
            self.say('OPENAI_API_KEY is not set, so the model cannot be asked')
            return

        memory, pose, reached, trouble = None, [0., 0., 0.], False, 0
        facing_attempts=0
        reference=[]
        reference_distance=0.;reference_turn=0.
        rejected_starts = 0
        self.strip = []
        closest, idle = None, 0       # nearest the target has come, and how many
                                      # looks since that last improved
        done, repeats = [], 0         # steps of the instruction already reached
        held = 0                      # consecutive holds answered with the
                                      # remembered route; reset by any look that
                                      # comes back with a plan of its own
        spins = 0.                    # degrees turned since anything was driven
        facing_turns = 0              # rotations spent aligning on arrival
        turn_way = 0.                 # committed rotation direction, if any
        turned_since_drive_reset = True   # cleared on a turn, set by real motion
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
            reference_distance=(memory or {}).get('reference_distance_cm',0.)+math.hypot(pose[0],pose[1])
            reference_turn=(memory or {}).get('reference_turn_deg',0.)+abs(pose[2])
            remembered_goal=fetch.rebase([memory['goal']],pose)[0] if memory and memory.get('goal') is not None else None
            reference=fetch.rebase(memory.get('top_reference',[]),pose) if memory else reference
            if not carry_memory:
                # Goal-only: no trajectory the model drew before, in text or as
                # ink. Remembered obstacles are kept, but only as rings on the
                # top-down map -- the map presents them as past sightings with
                # an uncertainty radius, where the text list reads as a fact
                # about the present. The goal survives; that is the point.
                reference=[]
                if memory is not None:
                    memory=dict(memory,route=[],top_reference=[])
            seeded = cycle == 0 and initial_plan is not None and carry_memory
            if seeded:
                # launch() already tied these pixels to this staged frame.  Do
                # not capture again here: doing so would silently change the
                # image underneath the high-level trajectory.
                answer = initial_plan
                prior = None
                self.say('seed 1: using high-level trajectory; first GPT call skipped')
            else:
                try:
                    image, _ = self.robot.frame()
                    self.restage(image)
                except Exception as exc:
                    self.say('cycle %d: no picture (%s)' % (cycle + 1, exc))
                    break

                prior = fetch.recall(self.robot, memory, pose) if memory else None
                self.show_request(self.frame, prior, 'look %d: what was sent'
                                  % (cycle + 1))
                try:
                    from paired_planning import observation
                    pair=observation(self.robot,self.frame,memory,pose,reference,
                                     obstacles_in_text=carry_memory)
                    answer = fetch.recognize(self.frame, target, prior, paired=pair,
                                             metric=metric)
                except fetch.PlannerHold as exc:
                    # A hold used to end the mission outright. But the payload's
                    # own rule is "preserve the previous route only when the
                    # current route to goal is uncertain", and a hold is exactly
                    # that state -- the model could not identify the target, not
                    # that the way is blocked. Measured 2026-09-24: on one such
                    # call the remembered route ended 22 cm from the target and
                    # was simply discarded, four cycles running. Re-offer it as
                    # the plan and let the ordinary clearance, widening and
                    # commit path decide whether any of it is still safe; that
                    # is a decision the controller can make and the model
                    # repeatedly would not, however the prompt was worded.
                    fallback = self.remembered_plan(memory, pose, answer=exc.answer)
                    if fallback is None or held >= MISSION_HOLD_FALLBACKS:
                        self.say('PLANNER HOLD: %s'
                                 % exc.answer.get('note', 'no supported route'))
                        break
                    held += 1
                    self.say('  hold: "%s"' % str(exc.answer.get('note', ''))[:70])
                    self.say('  following the remembered route instead (%d of %d)'
                             % (held, MISSION_HOLD_FALLBACKS))
                    answer = fallback
                except fetch.Stop as exc:
                    self.say('cycle %d: the model could not be asked: %s'
                             % (cycle + 1, exc))
                    break

            if not str(answer.get('note','')).startswith('controller fallback'):
                held = 0

            if metric:
                # The model answered in the frame the robot drives in, so there
                # is nothing to project. Every range error that came from
                # reading a contact pixel near the horizon -- where one row is
                # worth tens of centimetres -- is simply absent here.
                answer = dict(answer)
                answer['route_pixels'] = []
                answer['contact_pixel'] = None
            goal = None
            contact = answer.get('contact_pixel')
            if metric:
                # A fresh sighting arrives as a pixel and is projected here; the
                # remembered position stands only when nothing was seen.
                seen = answer.get('target_px')
                if answer.get('visible') and isinstance(seen, dict):
                    try:
                        goal = self.robot.ground(float(seen['x']), float(seen['y']))
                    except (fetch.Stop, KeyError, TypeError, ValueError):
                        goal = None
                if goal is None and remembered_goal is not None:
                    goal = remembered_goal
            elif answer.get('visible') and isinstance(contact, dict):
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
            if metric:
                # The route is drawn in image 1, like everything else the model
                # returns, and projected here. Pointing is what it does
                # reliably; the metric waypoints it used to return were
                # arithmetic, and on 2026-09-26 one ended 6 cm past a target it
                # had itself placed at 42 cm.
                for point in answer.get('route_px') or []:
                    if not isinstance(point, dict) or 'x' not in point:
                        continue
                    try:
                        route.append(self.robot.ground(float(point['x']),
                                                       float(point['y'])))
                    except (fetch.Stop, TypeError, ValueError):
                        continue
            else:
                for point in answer.get('route_pixels') or []:
                    if not isinstance(point, dict) or 'x' not in point:
                        continue
                    try:
                        route.append(self.robot.ground(float(point['x']),
                                                       float(point['y'])))
                    except (fetch.Stop, TypeError, ValueError):
                        continue
            if seeded:reference=list(route)
            if metric:
                # Each obstacle carries the half-extent the model measured, so
                # the controller stops substituting one constant for the size of
                # everything it never saw. A carton and a Lego brick were both
                # 4 cm discs until now.
                # Two base pixels per obstacle, projected here. The separation
                # is the measured width, so the half-extent is a measurement
                # rather than the one constant that made a 12.3 cm carton an
                # 8 cm disc. Remembered obstacles are merged in below: the
                # model is told not to repeat them, so they arrive only from
                # memory and only while out of view.
                obstacles = []
                far = 0
                for item in answer.get('obstacles') or []:
                    left, right = item.get('base_left_px'), item.get('base_right_px')
                    if not (isinstance(left, dict) and isinstance(right, dict)):
                        continue
                    try:
                        a = self.robot.ground(float(left['x']), float(left['y']))
                        b = self.robot.ground(float(right['x']), float(right['y']))
                    except (fetch.Stop, KeyError, TypeError, ValueError):
                        continue
                    middle = ((a[0] + b[0]) / 2., (a[1] + b[1]) / 2.)
                    half = math.hypot(a[0] - b[0], a[1] - b[1]) / 2.
                    # Everything the model can see is kept, however far: the
                    # robot is told to find as many obstacles as it can, and a
                    # distant one still becomes near once it drives at it. Only
                    # the extent is bounded, by the same ceiling every obstacle
                    # gets -- see fetch.OBSTACLE_MAX_RADIUS_CM.
                    if math.hypot(*middle) > fetch.ROUTE_RANGE_CM:
                        far += 1
                    obstacles.append((str(item.get('label') or 'obstacle'), middle,
                                      max(3., min(fetch.OBSTACLE_MAX_RADIUS_CM, half))))
                if far:
                    self.say('  %d obstacle(s) reported beyond %.0f cm; kept, '
                             'position uncertain' % (far, fetch.ROUTE_RANGE_CM))
                for item in (memory or {}).get('obstacles', []) or []:
                    spot = fetch.rebase([item[1]], pose)[0]
                    radius = float(item[2]) if len(item) > 2 and item[2] else fetch.OBSTACLE_RADIUS_CM
                    radius = min(radius, fetch.OBSTACLE_MAX_RADIUS_CM)
                    if any(math.hypot(spot[0] - o[1][0], spot[1] - o[1][1]) < 10.
                           for o in obstacles):
                        continue      # seen again this look; the fresh one wins
                    obstacles.append((str(item[0]), (spot[0], spot[1]), radius))
            else:
                obstacles = fetch.project_obstacles(self.robot, answer)
            # The plan as the model drew it, before any trimming. This is what
            # goes back to it next look: handing back the truncated stub the
            # wheels were given loses the shape it chose -- ask for an approach
            # from the left and the stub is a couple of points aimed straight
            # ahead, so the next look has nothing of the curve to continue.
            if metric:
                route, dropped_unsafe = self.trim_to_own_obstacles(route, obstacles)
                route = self.face_the_target(route, goal, obstacles)
            planned = list(route)
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

            # A turn carrying a route: turn first, then drive what was drawn.
            # There is no separate motion name -- the route's presence is what
            # says the robot has somewhere to go once it has turned.
            if answer.get('motion') == 'turn' and route:
                try:
                    route, obstacles, goal, turn_pose = self.turn_then_follow_plan(
                        answer, route, obstacles, goal)
                except fetch.Stop as exc:
                    # Not drivable after the turn; take the turn alone rather
                    # than abandoning the cycle.
                    self.say('  %s; turning without a route' % exc)
                    route = []
                    turn_pose = None
                planned = list(route)
                if turn_pose is not None:
                    reference=fetch.rebase(reference,turn_pose)
                if turn_pose is not None:
                    if remembered_goal is not None:
                        remembered_goal=fetch.rebase([remembered_goal],turn_pose)[0]
                    pose = [0., 0., 0.]  # memory below is now in the post-turn frame

            if not route and memory and answer.get('motion') != 'turn':
                # A turn is an answer, not an absence of one. This fallback was
                # written for a call that came back with nothing usable, but an
                # empty route is exactly what motion="turn" looks like, so it
                # refilled the route from memory and the mission drove on --
                # the turn branch below never saw an empty route to act on.
                # Measured 2026-09-25: a call asked to turn 10 degrees off a
                # block 14.5 cm dead ahead and the robot drove the remembered
                # route straight past it instead.
                #
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
            for item in obstacles:
                self.say('  avoiding "%s" at %.0f cm'
                         % (item[0], math.hypot(*item[1])))

            # Inside standoff, progress is angular rather than translational.
            # Let the bounded facing branch verify bearing before declaring
            # distance stagnation; it already limits alignment to three tries.
            if goal is not None and math.hypot(*goal) > MISSION_STANDOFF_CM:
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
            if (not (answer.get('motion') == 'turn' and route)
                    and isinstance(asked_turn, (int, float)) and math.isfinite(asked_turn)
                    and abs(asked_turn) >= MISSION_TURN_MIN_DEG
                    and (goal is None or math.hypot(*goal) > MISSION_STANDOFF_CM)):
                asked_turn = max(-MISSION_TURN_MAX_DEG,
                                 min(MISSION_TURN_MAX_DEG, float(asked_turn)))
                # Each look is judged on its own, so the model can ask to turn
                # back the way it just came without knowing it. Measured
                # 2026-09-26 reaching for a tissue box: turn left to face the
                # target, which puts a block in the path; turn right to clear
                # the block, which loses the target; four looks, 0 cm covered,
                # the two reasons stated plainly in its own notes. Hold the
                # committed direction until something is actually driven --
                # going the long way round a block reaches the target, and
                # alternating never does.
                if (turn_way and asked_turn * turn_way < 0
                        and not turned_since_drive_reset):
                    self.say('  it now asks %+.0f deg, reversing the last turn; '
                             'holding %s until something moves'
                             % (asked_turn, 'right' if turn_way > 0 else 'left'))
                    asked_turn = math.copysign(abs(asked_turn), turn_way)
                turn_way = math.copysign(1., asked_turn)
                turned_since_drive_reset = False
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
                memory = dict(route=list(route), obstacles=list(obstacles),
                              goal=goal if goal is not None else remembered_goal, done=list(done), top_reference=list(reference),
                              reference_distance_cm=reference_distance,reference_turn_deg=reference_turn)
                self.live_route = []
                continue
            spins = 0.

            # Arrival uses this fresh observation, never the requested turn alone.
            bearing=math.degrees(math.atan2(goal[0],goal[1])) if goal is not None else None
            if goal is not None and math.hypot(*goal)<=MISSION_STANDOFF_CM and abs(bearing)>8.:
                if facing_attempts>=3:
                    self.say('HOLD: final facing did not converge; not reached')
                    break
                facing_attempts+=1
                self.say('within range but target is %+.1f degrees off-axis; aligning then rechecking'%bearing)
                try:
                    pose=self.align_mission_target(goal,obstacles)
                except fetch.Stop as exc:
                    self.robot.halt();self.say('HOLD: '+str(exc));break
                except Exception as exc:
                    # Anything else here used to kill the mission thread with
                    # no line in the log at all.
                    self.robot.halt()
                    self.say('HOLD: final facing failed (%s: %s)'
                             % (type(exc).__name__, exc))
                    break
                from gpt_audit import plan_event
                plan_event(answer,'final_facing',execution_id=self.local_execution.id,
                           measured_pose=pose,verified_arrival=False)
                memory=dict(route=list(planned),obstacles=list(obstacles),goal=goal,
                            done=list(done),top_reference=list(reference),
                            reference_distance_cm=reference_distance,reference_turn_deg=reference_turn)
                if once:self.say('Facing turn complete; arrival unverified until a fresh observation')
                continue

            if goal is not None and math.hypot(*goal) <= MISSION_STANDOFF_CM:
                here = str(answer.get('goal') or target)[:40]
                if answer.get('all_done', True):
                    self.say('')
                    self.say('REACHED: %s is %.0f cm away -- that was the last '
                             'step' % (here, math.hypot(*goal)))
                    reached = True
                    break
                # More of the instruction to go. Arriving is not the end of the
                # run, it is the end of one step: note it, so the next look is
                # told not to come back here, and carry on.
                if here not in done:
                    done.append(here)
                    repeats = 0
                    self.say('')
                    self.say('reached %s (%d step(s) done); looking for what '
                             'comes next' % (here, len(done)))
                    closest, idle = None, 0     # the next step earns its own
                    memory = dict(route=[], obstacles=list(obstacles),
                                  goal=None, done=list(done))
                    pose = [0., 0., 0.]   # memory is in the frame we stand in
                    continue
                repeats += 1
                if repeats >= 3:
                    self.say('')
                    self.say('it keeps arriving at %s without saying the '
                             'instruction is finished; stopping' % here)
                    break
                continue
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
                    self.say('HOLD: no approach remains after standoff trimming; arrival not verified')
                    break
                if len(route) < before:
                    self.say('  trimmed %d waypoint(s) to stop %.0f cm short of it'
                             % (before - len(route), MISSION_STANDOFF_CM))

            widened, nudged = fetch.avoid(route, obstacles, goal)
            widened = fetch.clip_standoff(widened, goal, MISSION_STANDOFF_CM)
            if not widened:
                self.say('no safe route remains after standoff check; stopping')
                break
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
            from gpt_audit import plan_event
            plan_event(answer, 'controller_route', mode='mission', cycle=cycle+1,
                       points_cm=widened, obstacles_cm=obstacles, goal_cm=goal)
            self.live_pose = [0., 0., 0.]
            self.pose_heading = self.imu_heading()
            self.frame_token += 1

            self.publish_snapshot("executing", widened)
            legs = [0]
            moved = [0.]          # centimetres, not leg count -- see below
            start_rejected = [False]

            def record(event, **fields):
                if event == 'route_start_rejected':
                    start_rejected[0] = True
                    self.say('  route start rejected: %.1f deg exceeds %.0f deg; replanning'
                             % (fields['bearing_deg'], fields['limit_deg']))
                elif event == 'pose':
                    self.live_pose = [fields.get('x', 0.), fields.get('z', 0.),
                                      fields.get('heading', 0.)]
                    self.pose_heading = self.imu_heading()
                    return
                if event == 'leg':
                    legs[0] += 1
                    moved[0] += float(fields.get('moved_cm', 0.) or 0.)
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
                pose = self.execute_mission_trajectory(widened,None if once else seconds,record,obstacles)
                plan_event(answer,'local_execution',execution_id=self.local_execution.id,
                           result=self.local_execution.result())
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
                memory = dict(route=list(planned), obstacles=list(obstacles),
                              goal=goal if goal is not None else remembered_goal, done=list(done), top_reference=list(reference),
                              reference_distance_cm=reference_distance,reference_turn_deg=reference_turn)
                continue
            self.say('  %.1f s of motion, now %.0f cm from where the look started'
                     % (time.monotonic() - started, math.hypot(pose[0], pose[1])))
            memory = dict(route=list(planned), obstacles=list(obstacles),
                          goal=goal if goal is not None else remembered_goal, done=list(done), top_reference=list(reference),
                              reference_distance_cm=reference_distance,reference_turn_deg=reference_turn)
            if start_rejected[0]:
                rejected_starts += 1
                memory = dict(memory, route=[])
                if once or rejected_starts >= 3:
                    self.say('PLANNER HOLD: route requires a sharp initial turn')
                    break
                continue  # Do not interpret rejection as an escape-turn request.
            rejected_starts = 0
            # Turning does not count as progress. While blocked, follow still
            # rotates to face each waypoint, so requiring zero rotation here
            # meant the escape never fired and the give-up counter reset every
            # cycle -- nine looks in a row against one cable, going nowhere.
            # Counting legs missed the way this actually fails. With something
            # 15 cm ahead the clearance check leaves about 4 cm of room, so the
            # executor drives a real but tiny leg -- 3.4, then 2.9, 1.2, 0.6 --
            # and legs[0] is 1 every time, so the escape turn never fires and
            # the robot crawls at the obstacle instead of going around it.
            # Measured 2026-09-25 against a red block on the approach to the
            # tissue box: six looks, 56 cm covered, then stuck at 74 cm out.
            if moved[0] >= MISSION_MIN_PROGRESS_CM:
                # Real ground covered: the robot is somewhere new, so a request
                # to turn the other way is no longer a contradiction.
                turn_way = 0.
                turned_since_drive_reset = True
            if legs[0] == 0 or moved[0] < MISSION_MIN_PROGRESS_CM:
                if goal is not None and math.hypot(*goal) <= MISSION_STANDOFF_CM + 15.:
                    # Arrived on distance. Arriving also means facing it: the
                    # camera only looks forward, so stopping beside the target
                    # while aimed elsewhere leaves it out of the next picture.
                    # There is no route that fixes this -- at this range a route
                    # is a few centimetres long and driving it leaves the
                    # standoff -- so rotate on the spot, which costs no ground.
                    bearing = math.degrees(math.atan2(goal[0], goal[1]))
                    if abs(bearing) > FACING_TOLERANCE_DEG and facing_turns < 3:
                        step = math.copysign(
                            min(abs(bearing), MISSION_TURN_STEP_DEG), bearing)
                        self.say('  at the target but %+.0f deg off it; turning '
                                 '%+.0f to face it' % (bearing, step))
                        try:
                            turned = self.robot.turn(step)
                        except fetch.Stop as exc:
                            self.say('could not turn to face it: %s' % exc)
                            break
                        facing_turns += 1
                        shift = fetch.pivot_shift(turned)
                        from trajectory_executor import compose
                        pose = compose(pose, [shift[0], shift[1], turned])
                        continue
                    self.say('arrived: %.0f cm from the target, %+.0f deg off'
                             % (math.hypot(*goal), bearing))
                    break
                # Nothing moved. Usually that means the robot has closed inside
                # the keep-back distance of something, where every heading whose
                # corridor still holds it is refused -- at 10 cm that is a cone
                # of about 42 degrees, so most of them. Turning is the only move
                # that is always safe here: it shifts the corridor without
                # carrying the robot into anything.
                # The thing blocking the robot is usually the one it just saw,
                # not one it remembers: on the first look memory is empty, so
                # escaping from memory alone reported "nothing is in the way"
                # while a block sat 15 cm ahead. Give it this cycle's obstacles
                # as the fresh list and let memory fill in what has since left
                # the frame.
                here = fetch.merge_obstacles((memory or {}).get('obstacles', []),
                                             pose, obstacles)
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
                from trajectory_executor import compose
                pose = compose(pose,[shift[0],shift[1],turned])
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
        self.publish_snapshot('stopped')

    def floor_map(self):
        """The image-2 projection, built once -- it costs a fisheye solve."""
        made = getattr(self, '_floor_map', None)
        if made is None:
            from trajectory_executor import ZonedFloorProjection
            made = self._floor_map = ZonedFloorProjection(self.robot)
        return made

    def trim_to_own_obstacles(self, route, obstacles):
        """Cut a route where it runs into something the model itself reported.

        The model is told to keep 12 cm from every obstacle it lists, and it
        does not always do it: measured 2026-09-25, a route came back with its
        second waypoint 0.5 cm from a yellow block named in the same answer.
        Everything downstream then behaved correctly and uselessly -- widening
        pushed the route around the block until the first leg was 1.1 cm, the
        executor drove it, and the escape turn found nothing blocking straight
        ahead, because the obstruction was on the route rather than in front of
        the robot. The mission stopped with every component having done its job.

        Checking the answer against its own obstacle list costs nothing and
        needs no cooperation: keep the part of the route that can actually be
        driven and drop the rest, so what reaches the wheels is a short honest
        route rather than a long one that collapses.
        """
        if not route or not obstacles:
            return route, 0
        kept, start = [], (0., 0.)
        for point in route:
            dx, dz = point[0] - start[0], point[1] - start[1]
            span = math.hypot(dx, dz)
            if span < 1e-6:
                continue
            room, blame = fetch.clear_distance(
                obstacles, start, math.degrees(math.atan2(dx, dz)), span)
            if room < span:
                # Keep the drivable part of the blocked leg rather than throwing
                # the leg away. Discarding it whole left nothing to drive at all
                # when the very first leg was long and the obstacle was most of
                # the way along it -- 56 clear centimetres binned because the
                # 80 cm leg ended in a block. clear_distance has already taken
                # off the obstacle's extent and the keep-back margin, so `room`
                # is a distance the wheels can have.
                if room >= fetch.ROUTE_MIN_LEG_CM:
                    kept.append((start[0] + dx / span * room,
                                 start[1] + dz / span * room))
                self.say('  route runs into "%s" after %.0f of %.0f cm; '
                         'keeping %d of %d waypoint(s)'
                         % (blame or 'something', room, span,
                            len(kept), len(route)))
                return kept, len(route) - len(kept)
            kept.append(point)
            start = point
        return kept, 0

    def face_the_target(self, route, goal, obstacles):
        """Make the last leg point at the goal, so the robot stops looking at it.

        The robot faces along whatever its final segment was, and the camera
        only looks forward. A route that swings wide of an obstacle and never
        swings back finishes aimed past the target, which drops it out of the
        next picture and sends the planner searching for something the robot is
        standing beside.

        Asking for this in the prompt was measured on 2026-09-25 over three
        calls on one scene: the final-leg error went from -51 degrees to a
        median of -32, with one call in three actually complying. That is a
        nudge, not a constraint, so it is imposed here instead -- a short final
        leg straight at the goal, kept only if it is as clear as the route it
        extends.
        """
        if goal is None or not route:
            return route
        last = route[-1]
        gap = math.hypot(goal[0] - last[0], goal[1] - last[1])
        if gap <= MISSION_STANDOFF_CM:
            return route              # already inside the standoff; nothing to aim
        before = route[-2] if len(route) > 1 else (0., 0.)
        leg = (last[0] - before[0], last[1] - before[1])
        toward = (goal[0] - last[0], goal[1] - last[1])
        if math.hypot(*leg) < 1e-6:
            return route
        error = abs(math.degrees(math.atan2(leg[0] * toward[1] - leg[1] * toward[0],
                                            leg[0] * toward[0] + leg[1] * toward[1])))
        if error <= FACING_TOLERANCE_DEG:
            return route
        reach = min(FACING_LEG_CM, gap - MISSION_STANDOFF_CM)
        if reach < fetch.ROUTE_MIN_LEG_CM:
            return route
        aimed = (last[0] + toward[0] / gap * reach, last[1] + toward[1] / gap * reach)
        # The added leg is a claim about floor the model did not route over, so
        # it gets the same clearance test as every other leg rather than a free
        # pass for being ours.
        room, blame = fetch.clear_distance(obstacles, last,
                                           math.degrees(math.atan2(toward[0], toward[1])),
                                           reach)
        if room < reach:
            self.say('  final leg would face the target but %s blocks it at '
                     '%.1f of %.1f cm; left as answered (%.0f deg off)'
                     % (blame or 'something', room, reach, error))
            return route
        extended = list(route) + [aimed]
        self.say('  final leg aimed at the target: %.0f deg off, +%.0f cm'
                 % (error, reach))
        return extended

    def plan_once(self, target, metric=True):
        """One planning call and nothing else -- no motor service is opened.

        The mission loop is the only other way to reach the model, and it
        drives. That makes a prompt change expensive to test: every check of
        whether the model obeys a new rule costs a drive across the carpet and
        a scene to reset. This asks once from where the robot stands and hands
        back the raw answer, so a rule can be measured before anything moves.
        """
        target = (target or '').strip()
        if not target:
            return dict(error='name something to drive to')
        if not os.environ.get('OPENAI_API_KEY'):
            return dict(error='OPENAI_API_KEY is not set')
        if self.running():
            return dict(error='a run is in progress; stop it first')
        try:
            image, _ = self.robot.frame()
            self.restage(image)
        except Exception as exc:
            return dict(error='no picture (%s)' % exc)
        try:
            from paired_planning import observation
            pair = observation(self.robot, self.frame)
            answer = fetch.recognize(self.frame, target, paired=pair, metric=metric)
        except (fetch.Stop, fetch.PlannerHold) as exc:
            return dict(error=str(exc))
        return dict(answer=answer)

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
                # Obstacles are no longer sent: the model is looking at the
                # floor and lists them afresh. Only the unexecuted plan and
                # where the goal was go back to it.
                target_was = prior.get('target_was')
                self.say('reminding it of %d waypoint(s) it had not driven yet;'
                         ' it %s'
                         % (len(prior.get('route_pixels') or []),
                            prior.get('since_then', '')))
                if target_was:
                    self.say('  the goal was %s' % target_was['where'])
        if prior is None and self.live_route and self.live_drawn:
            # A route drawn by hand is a plan too. It is on the carpet in front
            # of the robot in this very frame, it is the operator saying which
            # way they want this done, and until now the model never saw it --
            # it was drawn, displayed, and driveable, but absent from the
            # request. Fed in the same way as the model's own leftover route,
            # it is something to refine rather than something to replace.
            #
            # live_route is already in the current frame: it is rebased by
            # every turn and every run, so nothing has happened since that the
            # model needs telling about.
            prior = fetch.recall(self.robot,
                                 dict(route=list(self.live_route), obstacles=[],
                                      goal=None, done=[]),
                                 [0., 0., 0.])
            if prior:
                self.say('handing it the %d waypoint(s) already drawn, to '
                         'refine rather than replace'
                         % len(prior.get('route_pixels') or []))
        try:
            answer = fetch.recognize(self.frame, target, prior)
        except fetch.PlannerHold as exc:
            self.live_route = []
            self.memory = None
            self.say('PLANNER HOLD: %s' % exc.answer.get('note', 'no supported route'))
            return self.state(points=[])
        except fetch.Stop as exc:
            self.say('the model could not be asked: %s' % exc)
            return self.state()

        self.last_note = str(answer.get('note', ''))[:120]
        self.say('asked for: %s' % target)
        self.say('model says: %s' % (answer.get('note') or '(nothing)'))
        if not answer.get('visible'):
            # Losing sight of the goal is not a dead end. The prompt tells the
            # model that with a last_time it must NOT return an empty route --
            # carry on along the one it already gave, because following it is
            # what brings the object back into view. It does exactly that, and
            # this used to throw the answer away, so the log read "continuing
            # prior route" with nothing drawn and nothing to drive.
            #
            # The driving loops never did this: for them `visible` gates the
            # goal, not the plan. Now this agrees with them.
            if answer.get('route_pixels'):
                self.say('it cannot see the goal from here, so it is carrying '
                         'on along the route it already had')
            else:
                self.say('it cannot see the goal and has no route left -- '
                         'turn and take another picture')

        goal = None
        contact = answer.get('contact_pixel')
        if answer.get('visible') and isinstance(contact, dict):
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
        for item in obstacles:
            label, spot = item[0], item[1]
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
        for item in obstacles:
            label, spot = item[0], item[1]
            radius = float(item[2]) if len(item) > 2 and item[2] else fetch.OBSTACLE_RADIUS_CM
            pixel = on_view(spot)
            if pixel:
                hazards.append([pixel[0], pixel[1],
                                (min(radius, fetch.OBSTACLE_MAX_RADIUS_CM)
                                 + fetch.CORRIDOR_HALF_CM)
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
        # The model's own route, not `widened`. The widened one has been
        # pushed out for clearance and had its corners cut, so handing it back
        # returns twice the points it gave and a shape it did not choose. The
        # instruction is stored with it so that changing the instruction
        # retires the plan instead of silently inheriting it.
        self.memory = dict(target=target, route=list(route),
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
        self.live_drawn = bool(spots)
        if not spots:
            # CLEAR comes through here with an empty list. It used to empty the
            # canvas and leave the remembered plan untouched, so the view went
            # blank and the model still received the whole previous route --
            # an operator staring at an empty screen with every reason to think
            # they were asking cold.
            self.memory = None
            self.live_obstacles = []
            self.say('cleared: the model will be asked with no prior route')
        self.live_pose = [0., 0., 0.]
        self.pose_heading = self.imu_heading() if spots else None
        return spots, total

    def to_pixel(self, right, forward, mode=None):
        """Floor position to canvas pixel, in `mode`'s view.

        The view is an argument rather than only self.view_mode because both
        views are on screen at once now: the same floor point has to be placed
        in the plan view and in the photograph, and the two projections are
        nothing like each other.
        """
        if (mode or self.view_mode) == 'camera':
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
            moved = [shift[0], shift[1], turned]
            self.advance(moved)
            # Everything held in floor coordinates was measured in the frame the
            # robot occupied before this turn, so it all moves. Without this the
            # drawn trajectory and the obstacle circles stay where they were on
            # screen while the picture rotates underneath them, which is the
            # opposite of reprojection.
            was_drawn = self.live_drawn
            self.live_route = [spot for spot in fetch.rebase(self.live_route, moved)]
            self.live_drawn = was_drawn
            self.live_obstacles = [
                (item[0], spot) + tuple(item[2:])
                for item, spot in zip(
                    self.live_obstacles,
                    fetch.rebase([o[1] for o in self.live_obstacles], moved))]
            self.live_pose = [0., 0., 0.]     # the route is in this frame now
            self.pose_heading = self.imu_heading()
            if self.memory:
                self.memory = dict(
                    self.memory,
                    route=fetch.rebase(self.memory.get('route') or [], moved),
                    goal=(fetch.rebase([self.memory['goal']], moved)[0]
                          if self.memory.get('goal') is not None else None))
                self.since = [0., 0., 0.]     # memory is in this frame now too
            self.say('turn: asked %+.0f deg, measured %+.1f deg (error %+.1f); '
                     'moved %d waypoint(s) and %d obstacle(s) into the new frame'
                     % (degrees, turned, turned - degrees,
                        len(self.live_route), len(self.live_obstacles)))
        except fetch.Stop as exc:
            self.robot.halt()
            self.say('turn STOPPED: %s' % exc)
        return self.capture_after()

    STRIP_MAX = 40             # a flowing run makes a look a second; keep the
                               # recent ones rather than the whole session

    def show_request(self, image, prior, label):
        """Keep the picture being sent, with the prior drawn on it.

        The drawing is fetch.annotate -- the very function that marks up the
        frame going to the model -- so this filmstrip is what was sent, not a
        second rendering of it that could drift. Only the caption and the
        already-reached line are added here; the model does not get those.

        This is the one place reprojection is visible in context: every pixel
        drawn here was computed by moving something the model said earlier into
        the frame of the photograph it is about to be asked about. If a waypoint
        sits on the carpet it was put on, the reprojection is right; if it has
        slid, the amount it slid is the error, and it is visible rather than
        inferred.
        """
        shot = fetch.annotate(image, prior)
        if shot is image:
            shot = image.copy()
        cv2.putText(shot, label, (8, 20), cv2.FONT_HERSHEY_PLAIN, 1.0,
                    (240, 240, 240), 1)
        if prior:
            if prior.get('already_reached'):
                cv2.putText(shot, 'done: ' + ', '.join(
                    prior['already_reached'])[:46], (8, shot.shape[0] - 10),
                    cv2.FONT_HERSHEY_PLAIN, .9, (170, 230, 170), 1)
        else:
            cv2.putText(shot, 'nothing remembered -- asked cold', (8, 42),
                        cv2.FONT_HERSHEY_PLAIN, 1., (170, 170, 170), 1)
        ok, encoded = cv2.imencode('.jpg', shot)
        if ok:
            self.strip.append(encoded.tobytes())
            del self.strip[:-self.STRIP_MAX]

    CHECK_POINTS = [(-25., 45.), (0., 45.), (25., 45.),
                    (-15., 75.), (15., 75.), (0., 110.)]

    def mark_floor(self, spots, colour, caption):
        """Photograph with a cross at each floor position, and keep it."""
        try:
            image, _ = self.robot.frame()
        except Exception:
            return 0
        drawn = 0
        for index, spot in enumerate(spots):
            if spot is None or spot[1] <= 0.:
                continue
            place = self.to_pixel(spot[0], spot[1], view)
            if not place or not (0 <= place[0] < image.shape[1]
                                 and 0 <= place[1] < image.shape[0]):
                continue
            drawn += 1
            at = (int(place[0]), int(place[1]))
            cv2.drawMarker(image, at, colour, cv2.MARKER_CROSS, 24, 2)
            cv2.putText(image, str(index + 1), (at[0] + 9, at[1] - 9),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, colour, 2)
        cv2.putText(image, caption, (8, 22), cv2.FONT_HERSHEY_PLAIN, 1.1, colour, 2)
        ok, encoded = cv2.imencode('.jpg', image)
        if ok:
            self.strip.append(encoded.tobytes())
        self.restage(image)
        return drawn

    def check_reprojection(self, degrees):
        """Turn a known amount and redraw the same floor points afterwards.

        This is the only way to see whether reprojection works, as opposed to
        believing the arithmetic. Points are marked on the carpet, the robot
        turns, and the marks are drawn again from where it now believes it
        stands. If the geometry is right they land on the same carpet -- the
        picture moves underneath and the crosses stay put.

        A turn rather than a drive, because a turn is measured by the IMU to
        about 0.7 degrees while distance comes from carpet optical flow that
        lands anywhere from one sample in eight to five in five. Testing both at
        once would not say which was wrong.
        """
        self.lines = []
        self.strip = []
        spots = list(self.CHECK_POINTS)
        before = self.mark_floor(spots, (60, 240, 60),
                                 'BEFORE: marks on fixed floor points')
        self.say('marked %d point(s) on the carpet' % before)
        self.say('turning %+.0f deg, then drawing the same floor points again'
                 % degrees)
        try:
            turned = self.robot.turn(degrees)
        except fetch.Stop as exc:
            self.robot.halt()
            self.say('STOPPED: %s' % exc)
            return self.state(strip=len(self.strip))
        shift = fetch.pivot_shift(turned)
        pose = [shift[0], shift[1], turned]
        self.say('asked %+.0f, measured %+.1f deg; the lens also slid '
                 '(%+.1f, %+.1f) cm about the pivot'
                 % (degrees, turned, shift[0], shift[1]))
        moved = fetch.rebase(spots, pose)
        after = self.mark_floor(moved, (60, 200, 255),
                                'AFTER: the same floor points, reprojected')
        self.say('%d of %d still in frame and redrawn' % (after, len(spots)))
        self.say('')
        self.say('step the two frames below. A cross that sits on the same bit '
                 'of carpet in both is reprojection working; one that has slid '
                 'is the error, and you can see how big it is.')
        return self.state(strip=len(self.strip))

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
            place = self.to_pixel(spot[0], spot[1], view)
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

    def overlay_route(self, image, heading_now=None, view="camera"):
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
            place = self.to_pixel(spot[0], spot[1], view)
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

    def live_jpeg(self, width=None, view="camera", metadata=None):
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
        if metadata is not None:
            metadata['camera_time'] = observation.get('time')
        blob = observation.get('jpeg_base64')
        if not blob:
            return None
        try:
            jpeg = base64.b64decode(blob)
        except (ValueError, TypeError):
            return None

        width = STREAM_WIDTH if width is None else width
        drawing = bool(self.live_route)
        if not width and not drawing and view == "camera":
            return jpeg          # the cheap path: hand the bytes straight on

        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jpeg
        if view == "floor":
            image = self.warp.apply(image)
        if drawing:
            samples = observation.get('imu_samples') or []
            heading = None
            if samples:
                try:
                    heading = self.robot.heading(samples[-1])
                except Exception:
                    heading = None
            self.overlay_route(image, heading, view)
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
        self.restage(image)

    def capture_after(self):
        try:
            image, _ = self.robot.frame()
            self.restage(image)
        except Exception as exc:
            self.say('could not refresh the picture: %s' % exc)
        return self.state(frame=True)

    def run(self, points, seconds=None):
        """Drive the drawn path, reporting asked-for against measured per leg."""
        route, planned = self.project(points)
        if len(route) < 1:
            self.say('nothing to drive: put at least one usable point on the carpet')
            return self.state(length=0.)
        self.say('running %d waypoints, %.0f cm of path%s'
                 % (len(route), planned,
                    ', stopping after %.0f s' % seconds if seconds else ''))
        self.say('asked      measured   drift   vision')
        travelled = [0.]
        moved = []          # the same floor points, seen from where it ends up
        pose = [0., 0., 0.]  # how far it got; a Stop leaves follow without one
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
            # A bounded run stops part way on purpose. That is the only way to
            # see the reprojection do its job: what is left of the trajectory
            # has to be redrawn from where the robot stopped, and if the
            # arithmetic is wrong the leftover waypoints land on the wrong
            # carpet where anyone can see it.
            pose = fetch.follow(self.robot, self.odometer, route, record,
                                deadline=(time.monotonic() + seconds)
                                if seconds else None,
                                initial_turn_limit_deg=fetch.INITIAL_TURN_LIMIT_DEG)
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
            # A cut-short run still moved. live_pose is what the drive reported
            # as it went, so the picture can still be put right.
            pose = list(self.live_pose)
            moved = fetch.rebase(route, pose)
            self.say('')
            self.say('STOPPED: %s' % exc)
        # The whole point of re-photographing here: the route was drawn on a
        # picture taken from somewhere the robot no longer is. Carrying it into
        # the new frame is what lets the next look build on this one, and
        # drawing it on the new photograph is how you can see whether the
        # arithmetic is right -- the markers should land on the same carpet.
        # Everything the bench holds in floor coordinates was measured before
        # this drive, so it all moves -- not just the waypoints. Leaving the
        # obstacles behind drew their circles at pre-drive screen positions,
        # which after 30 cm of travel is very visibly wrong.
        was_drawn = self.live_drawn
        self.live_route = list(moved)
        self.live_drawn = was_drawn
        self.live_obstacles = [
            (item[0], spot) + tuple(item[2:])
            for item, spot in zip(
                self.live_obstacles,
                fetch.rebase([o[1] for o in self.live_obstacles], pose))]
        self.live_pose = [0., 0., 0.]     # what is held is in this frame now
        self.pose_heading = self.imu_heading()

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
                jpeg = bench.live_jpeg(view="floor" if "view=floor" in self.path else "camera")
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
        if urlparse(self.path).path == '/api/inspection':
            if not self._local():
                return self._send(b'{"error":"local only"}')
            return self._send(json.dumps(getattr(self.server.bench,
                'inspection_result', {'status': 'idle'})).encode())
        if not self._local():
            return self._send(b'{"error":"local only"}')
        path = urlparse(self.path).path
        if path == '/':
            return self._send(PAGE.encode('utf-8'), 'text/html; charset=utf-8')
        if path == '/api/progress':
            # Deliberately outside the lock: a mission holds the robot for
            # minutes, and this is the only way to watch it do so.
            bench = self.server.bench
            route, hazards, other, alt, alt_hazards = bench.both_views()
            route = route or []
            return self._send(json.dumps(dict(
                log=bench.log(), running=bench.running(),
                power=bench.power(), heading=bench.heading,
                strip=len(bench.strip), points=route, hazards=hazards,
                view_mode=bench.view_mode, alt_mode=other,
                alt_points=alt or [], alt_hazards=alt_hazards,
                note=bench.last_note, frame_token=bench.frame_token)).encode())
        if path == '/api/planner-snapshot':
            bench = self.server.bench
            if not bench.running() and getattr(bench, 'planner_snapshot', {}).get('phase') in ('executing','replanning'):
                bench.publish_snapshot('stopped')
            return self._send(json.dumps(getattr(self.server.bench, 'planner_snapshot', {'phase':'waiting','id':0})).encode())
        if path == '/api/local-execution':
            job = getattr(self.server.bench, 'local_execution', None)
            return self._send(json.dumps(job.result() if job else {'phase':'idle'}).encode())
        if path == '/api/live.jpg':
            view = 'floor' if 'view=floor' in self.path else 'camera'
            metadata = {}
            jpeg = self.server.bench.live_jpeg(width=0, view=view, metadata=metadata)
            if not jpeg:
                self.send_error(503, 'Live camera unavailable'); return
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(jpeg)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Camera-Time', str(metadata.get('camera_time', '')))
            self.end_headers(); self.wfile.write(jpeg); return
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
                # Either view can be asked for by name, since both are on
                # screen; no name means whichever is the interactive one.
                query = urlparse(self.path).query or ''
                wanted = 'camera' if 'view=camera' in query else (
                    'floor' if 'view=floor' in query else bench.view_mode)
                image = bench.frame if wanted == 'camera' else bench.view
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
            stopped = bench.halt()
            try:
                import gpt_audit
                gpt_audit.command(path, body)
            except Exception as exc:
                bench.say('stop command audit failed: %s' % exc)
            return self._send(json.dumps(stopped).encode())
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
            import gpt_audit
            gpt_audit.command(path, body)
            if path == '/api/capture':
                out = bench.capture()
            elif path == '/api/inspection':
                out = inspection_sequence.launch(bench, body)
            elif path == '/api/turn':
                out = bench.turn(float(body.get('degrees', 0.)))
            elif path == '/api/run':
                every = body.get('seconds')
                out = bench.run(body.get('points') or [],
                                None if every in (None, '') else float(every))
            elif path == '/api/local-execution':
                import trajectory_executor
                out = trajectory_executor.launch(bench, body)
            elif path == '/api/check':
                out = bench.check_reprojection(
                    float(body.get('degrees') or 30.))
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
                                   None if every in (None, '') else float(every),
                                   body.get('initial_plan'),
                                   body.get('seed_frame_token'),
                                   carry_memory=bool(body.get('carry_memory', True)),
                                   metric=bool(body.get('metric', False)))
            elif path == '/api/look':
                out = bench.look(body.get('mode') or 'floor')
            elif path == '/api/ask':
                out = bench.ask(body.get('target') or '')
            elif path == '/api/plan-once':
                out = bench.plan_once(body.get('target') or '',
                                      bool(body.get('metric', True)))
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
    parser.add_argument('--restore-snapshot', help='Restore a saved visualization snapshot while idle')
    args = parser.parse_args()
    apply_stream_settings(args.stream_fps, args.stream_width)
    server = Server((args.host, args.port), Handler)
    server.bench = Bench(args.enable_motion)
    if args.restore_snapshot:
        with open(args.restore_snapshot) as handle:
            server.bench.planner_snapshot = json.load(handle)
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
