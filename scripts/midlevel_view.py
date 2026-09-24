"""Read-only calibrated reference/proposal views; never opens the motor service."""
import base64
import json
import math
import sys
from pathlib import Path
from functools import lru_cache
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'local_nav'))
import fetch
from evaluate_gpt_routes import Lens
from trajectory_executor import CameraFloorProjection, encode

@lru_cache(maxsize=1)
def geometry():
    lens=Lens()
    lens.pixel=lambda x,z:fetch.Robot.pixel(lens,x,z)
    return lens,CameraFloorProjection(lens)

def render_pair(camera,reference,proposal=(),target=None,previous=(),obstacles=(),remembered=(),target_cm=None,reference_weight=1.):
    lens,projection=geometry()
    rgb=camera.copy();floor=projection.apply(camera)
    for v in range(0,640,40):
        cv2.line(floor,(v,0),(v,639),(65,65,65),1)
        cv2.line(floor,(0,v),(639,v),(65,65,65),1)
    for points,color,width in [(reference,tuple(int(v*(.25+.75*reference_weight)) for v in (220,100,255)),9),(previous,(0,160,255),6),(proposal,(255,210,90),3)]:
        for a,b in zip(points,points[1:]):
            cv2.line(floor,(round(320+a[0]*1.6),round(320-a[1]*1.6)),
                     (round(320+b[0]*1.6),round(320-b[1]*1.6)),color,width)
            last=None
            for t in np.linspace(0,1,min(800,max(2,int(math.hypot(b[0]-a[0],b[1]-a[1])*2)))):
                p=[a[i]+t*(b[i]-a[i]) for i in range(2)]
                q=lens.pixel(*p) if p[1]>0 else None
                q=tuple(int(round(x)) for x in q) if q and all(math.isfinite(x) and abs(x)<1e6 for x in q) else None
                if q and last:
                    ok,start,end=cv2.clipLine((0,0,640,480),last,q)
                    if ok:cv2.line(rgb,start,end,color,width)
                last=q
    if target:
        cv2.drawMarker(rgb,(int(target['x']),int(target['y'])),(60,220,220),cv2.MARKER_STAR,22,2)
        x,z=lens.ground(float(target['x']),float(target['y']))
        cv2.drawMarker(floor,(int(round(320+x*1.6)),int(round(320-z*1.6))),(60,220,220),cv2.MARKER_STAR,18,2)
    def gp(p):return (int(round(320+p[0]*1.6)),int(round(320-p[1]*1.6)))
    for o in obstacles:
        p=o.get('contact_pixel')
        if not p:continue
        q=(int(p['x']),int(p['y']))
        cv2.drawMarker(rgb,q,(80,80,255),cv2.MARKER_SQUARE,14,2)
        cv2.putText(rgb,o.get('label','obstacle')[:28],(q[0]+8,q[1]-8),0,.4,(80,80,255),1)
        try:cv2.circle(floor,gp(lens.ground(p['x'],p['y'])),6,(80,80,255),2)
        except (fetch.Stop,ValueError):pass
    for o in remembered:
        p=o['position_cm'];q=lens.pixel(*p) if p[1]>0 else None
        if q and 0<=q[0]<640 and 0<=q[1]<480:continue
        at=gp(p);radius=int(max(6,min(50,o.get('uncertainty_cm',5)*1.6)))
        cv2.circle(floor,at,radius,(180,180,180),1)
        cv2.drawMarker(floor,at,(180,180,180),cv2.MARKER_TILTED_CROSS,12,2)
        cv2.putText(floor,o.get('label','remembered')[:24],at,0,.35,(200,200,200),1)
    if target_cm is not None:cv2.drawMarker(floor,gp(target_cm),(60,220,220),cv2.MARKER_STAR,22,2)
    cv2.arrowedLine(floor,(320,320),(320,296),(0,220,255),3)
    cv2.putText(floor,'4 m x 4 m | 25 cm grid | camera at center',(8,20),0,.45,(255,255,255),1)
    return dict(camera=encode(rgb),floor=encode(floor))

def pixels_to_floor(points):
    lens,_=geometry();result=[]
    for p in points:
        result.append(list(lens.ground(float(p['x']),float(p['y']))))
    return result

@lru_cache(maxsize=12)
def call_view(directory,version):
    directory=Path(directory)
    camera=cv2.imread(str(directory/'clean.jpg'))
    if camera is None:return dict(available=False,note='Clean RGB unavailable for this historical call; no synthetic overlay shown.')
    request=json.loads((directory/'request.json').read_text());context={}
    for message in request.get('input',[]):
        for part in message.get('content',[]):
            if part.get('type')=='input_text':
                try:context.update(json.loads(part['text']))
                except (ValueError,TypeError):pass
    prior=context.get('last_time') or {}
    metric=context.get('local_map') or {}
    if (directory/'local-map.json').exists():metric=json.loads((directory/'local-map.json').read_text())
    top=prior.get('source')=='top_level_synthetic_demo'
    previous=metric.get('_previous_route_visualization',metric.get('previous_route_cm',pixels_to_floor(prior.get('route_pixels') or [])))
    reference=metric.get('reference_cm',previous if top else [])
    if top:previous=[]
    proposal=[];answer={}
    if (directory/'response.json').exists():
        response=json.loads((directory/'response.json').read_text())
        answer=json.loads(''.join(c.get('text','') for x in response.get('output',[]) for c in x.get('content',[]) if c.get('type')=='output_text'))
        proposal=pixels_to_floor(answer.get('route_pixels') or [])
    target=answer.get('contact_pixel') or (prior.get('target_was') or {}).get('contact_pixel')
    note='Same captured observation for all layers; previous plan and memory are in this camera frame using the saved motion estimate.'
    if not metric and not top:note+=' Historical call: no separate top-level reference or metric obstacle memory was recorded.'
    if '_previous_route_visualization' in metric:note+=' Orange shows the full previous plan for inspection; GPT received only its first 15 cm as a faint hint (see exact inputs).'
    if 'reference_reliability' in metric:note+=' Reference heuristic weight %.2f after %.0f cm estimated displacement / %.0f degrees accumulated yaw; fresh RGB takes priority.'%(metric['reference_reliability'],metric['reference_distance_cm'],metric['reference_turn_deg'])
    if top:note='Synthetic stationary demo; no motion executed.'
    return dict(available=True,images=render_pair(camera,reference,proposal,target,previous,
                answer.get('obstacles',[]),metric.get('obstacles',[]),metric.get('target_cm'),metric.get('reference_reliability',1.)),
                reference_cm=reference,previous_cm=previous,proposal_cm=proposal,
                remembered_obstacles=metric.get('obstacles',[]),source='saved_gpt_request',note=note)


@lru_cache(maxsize=4)
def local_view(directory,version,snapshot_json):
    directory=Path(directory);snapshot=json.loads(snapshot_json)
    command=json.loads((directory/'command.json').read_text())
    reference=command['trajectory']['waypoints_cm']
    if snapshot.get('source_rgb'):
        camera=cv2.imdecode(np.frombuffer(base64.b64decode(snapshot['source_rgb'].split(',',1)[1]),np.uint8),1)
        reference=fetch.rebase([[0.,0.]]+reference,snapshot['execution']['pose_cm_deg'])
        frame='current executor observation; reference transformed using measured pose'
    else:
        camera=cv2.imread(str(directory/'initial.jpg'));reference=[[0.,0.]]+reference
        frame='original instruction camera frame'
    if camera is None:return dict(available=False,note='Reference source image unavailable')
    return dict(available=True,images=render_pair(camera,reference),reference_cm=reference,proposal_cm=[],
                source='top_level_local_instruction',note='Top-level reference in '+frame+'. Magenta: full supplied trajectory. No GPT proposal: this was a local-only test.')


def seeded_view(root):
    """A seed can be rejected before either a GPT call or local job exists."""
    from control_visualizer import interventions
    for command in interventions(root):
        seed=(command.get('body') or {}).get('initial_plan')
        if not seed:continue
        camera=cv2.imread(str(root/command.get('handoff_image','start.jpg')))
        if camera is None:return dict(available=False,note='Seed handoff image unavailable.')
        route=pixels_to_floor(seed.get('route_pixels') or [])
        return dict(available=True,images=render_pair(camera,reference=route,
                    target=seed.get('contact_pixel'),obstacles=seed.get('obstacles',[])),
                    reference_cm=route,proposal_cm=[],source='high_level_seed',
                    note='High-level seed supplied to GPT Drive; no GPT call. Magenta: supplied trajectory; yellow: target. See controller output for acceptance or rejection; this image does not imply execution.')
    return dict(available=False,note='No seeded handoff or local execution recorded.')
