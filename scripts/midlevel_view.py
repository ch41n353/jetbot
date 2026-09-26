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
from trajectory_executor import ZonedFloorProjection, encode

@lru_cache(maxsize=1)
def geometry():
    lens=Lens()
    lens.pixel=lambda x,z:fetch.Robot.pixel(lens,x,z)
    return lens,ZonedFloorProjection(lens)

def render_pair(camera,reference,proposal=(),target=None,previous=(),obstacles=(),remembered=(),target_cm=None,reference_weight=1.):
    lens,projection=geometry()
    rgb=camera.copy();floor=projection.apply(camera)
    Z=ZonedFloorProjection
    hub=projection.place(0.,0.)
    def ring_px(cm):return int(round(Z.radius_px(cm)*projection.zoom))
    for degrees in range(-90,91,15):
        a=math.radians(degrees);reach=ring_px(Z.OUTER_CM)
        cv2.line(floor,hub,(int(hub[0]+reach*math.sin(a)),
                            int(hub[1]-reach*math.cos(a))),(60,60,60),1)
    for cm in (20,40,60,80,125,150,175):
        cv2.circle(floor,hub,ring_px(cm),(86,86,86),1)
    cv2.circle(floor,hub,ring_px(Z.INNER_CM),(90,200,255),2)
    # A route is a list of places to go, so the leg the robot actually drives
    # first -- from where it stands to waypoint one -- is not in it. Drawing
    # only the supplied points renders a two-point route as one short segment
    # floating in the scene, which reads as a plan that goes nowhere near the
    # robot. Put the origin back for the drawing only.
    def from_robot(points):
        points=[list(p) for p in points or []]
        if not points:return points
        return points if math.hypot(points[0][0],points[0][1])<1. else [[0.,0.]]+points
    reference,previous,proposal=from_robot(reference),from_robot(previous),from_robot(proposal)
    for points,color,width in [(reference,tuple(int(v*(.25+.75*reference_weight)) for v in (220,100,255)),9),(previous,(0,160,255),6),(proposal,(255,210,90),3)]:
        for a,b in zip(points,points[1:]):
            cv2.line(floor,projection.place(*a),projection.place(*b),color,width)
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
        q=(int(target['x']),int(target['y']))
        cv2.drawMarker(rgb,q,(60,220,220),cv2.MARKER_STAR,22,2)
        cv2.putText(rgb,'TARGET',(q[0]+12,q[1]+4),0,.45,(60,220,220),1)
        # The pixel is always drawable; its floor position is not. A target
        # reported at or above the horizon has no ground intersection, and
        # projecting it unguarded took the whole view down -- 9 of 252 saved
        # calls rendered nothing at all because of this one line.
        try:
            x,z=lens.ground(float(target['x']),float(target['y']))
        except (fetch.Stop,ValueError,TypeError):
            cv2.putText(rgb,'(above the horizon: no floor position)',
                        (q[0]+12,q[1]+20),0,.38,(60,220,220),1)
        else:
            at=projection.place(x,z)
            cv2.drawMarker(floor,at,(60,220,220),cv2.MARKER_STAR,18,2)
            cv2.putText(floor,'TARGET %.0f,%.0f'%(x,z),(at[0]+10,at[1]+4),0,.4,(60,220,220),1)
            target_cm=None        # already drawn from the pixel; no second star
    def gp(p):return projection.place(p[0],p[1])
    drawn=[]                      # answer obstacles, to dedupe the remembered ones
    dropped=[]                    # reported past the range the planner trusts
    unplaced=[]                   # base pixel above the horizon: no floor point
    for o in obstacles:
        p=o.get('contact_pixel')
        if p:
            q=(int(p['x']),int(p['y']))
            cv2.drawMarker(rgb,q,(80,80,255),cv2.MARKER_SQUARE,14,2)
            cv2.putText(rgb,o.get('label','obstacle')[:28],(q[0]+8,q[1]-8),0,.4,(80,80,255),1)
            try:cv2.circle(floor,gp(lens.ground(p['x'],p['y'])),6,(80,80,255),2)
            except (fetch.Stop,ValueError):pass
            continue
        # Metric contract: the model reports where an object meets the floor as
        # two pixels in image 1, left and right edge. Project both, take the
        # midpoint as its position and half the separation as its extent --
        # the same arithmetic the controller does, so the picture shows what
        # the controller acted on rather than a redrawing of it.
        left,right=o.get('base_left_px'),o.get('base_right_px')
        if isinstance(left,dict) and isinstance(right,dict):
            try:
                a2=lens.ground(float(left['x']),float(left['y']))
                b2=lens.ground(float(right['x']),float(right['y']))
            except (fetch.Stop,ValueError,KeyError,TypeError):
                # A base pixel at or above the horizon has no floor
                # intersection, so there is no position to place it at. The
                # model did see the thing -- show that it did, on the only
                # picture where it can be shown, rather than dropping it
                # silently and leaving the view looking like it was never
                # reported.
                for q in (left,right):
                    try:pt=(int(q['x']),int(q['y']))
                    except (KeyError,TypeError,ValueError):continue
                    if 0<=pt[0]<640 and 0<=pt[1]<480:
                        cv2.drawMarker(rgb,pt,(120,120,120),cv2.MARKER_DIAMOND,7,1)
                unplaced.append(o.get('label','obstacle'))
                continue
            spot=((a2[0]+b2[0])/2.,(a2[1]+b2[1])/2.)
            radius=max(1.,math.hypot(a2[0]-b2[0],a2[1]-b2[1])/2.)
            edges=(left,right)
        else:
            m=o.get('position_cm')
            if not isinstance(m,dict) or 'right' not in m:continue
            spot=(float(m['right']),float(m['forward']))
            edges=()
            try:radius=max(1.,float(o.get('radius_cm') or 6.))
            except (TypeError,ValueError):radius=6.
        # Every obstacle is drawn at the extent the planner will actually use
        # it at, capped the same way, so the picture is not more confident than
        # the clearance check that acts on it.
        radius=min(radius,fetch.OBSTACLE_MAX_RADIUS_CM)
        if math.hypot(*spot) > fetch.ROUTE_RANGE_CM:dropped.append(o.get('label','obstacle'))
        # Mark the two edges in the camera view, so a badly picked edge shows
        # up as a wrong-sized box rather than hiding inside a number.
        for q in edges:
            if 0<=q['x']<640 and 0<=q['y']<480:
                cv2.drawMarker(rgb,(int(q['x']),int(q['y'])),(80,80,255),
                               cv2.MARKER_TILTED_CROSS,10,2)
        c=gp(spot)
        edge=projection.place(spot[0]+radius,spot[1])
        cv2.circle(floor,c,max(3,int(math.hypot(edge[0]-c[0],edge[1]-c[1]))),(80,80,255),2)
        cv2.putText(floor,o.get('label','obstacle')[:24],(c[0]+6,c[1]-6),0,.35,(100,100,255),1)
        drawn.append(spot)
        if spot[1] > 0:
            try:q=lens.pixel(*spot)
            except Exception:q=None
            if q and all(math.isfinite(v) for v in q) and 0<=q[0]<640 and 0<=q[1]<480:
                q=(int(q[0]),int(q[1]))
                # The circle's radius in pixels at that range, so the drawing
                # says how big the thing was reported to be, not just where.
                edge=lens.pixel(spot[0]+radius,spot[1])
                px=int(abs(edge[0]-q[0])) if edge else 8
                cv2.circle(rgb,q,max(4,min(120,px)),(80,80,255),2)
                cv2.putText(rgb,o.get('label','obstacle')[:28],(q[0]+8,q[1]-8),0,.4,(80,80,255),1)
    for o in remembered:
        p=o['position_cm']
        if isinstance(p,dict) and 'right' in p:p=[float(p['right']),float(p['forward'])]
        # The model echoes remembered obstacles back in its own list, so without
        # this every carried obstacle was drawn twice, once per source.
        if any(math.hypot(p[0]-d[0],p[1]-d[1]) < 8. for d in drawn):continue
        q=lens.pixel(*p) if p[1]>0 else None
        if q and 0<=q[0]<640 and 0<=q[1]<480:continue
        at=gp(p)
        widest=float(o.get('radius_cm') or o.get('uncertainty_cm') or 5)
        widest=min(widest,fetch.OBSTACLE_MAX_RADIUS_CM)
        rim=projection.place(p[0]+widest,p[1])
        radius=int(max(6,math.hypot(rim[0]-at[0],rim[1]-at[1])))
        cv2.circle(floor,at,radius,(180,180,180),1)
        cv2.drawMarker(floor,at,(180,180,180),cv2.MARKER_TILTED_CROSS,12,2)
        cv2.putText(floor,o.get('label','remembered')[:24],at,0,.35,(200,200,200),1)
    if target_cm is not None:cv2.drawMarker(floor,gp(target_cm),(60,220,220),cv2.MARKER_STAR,22,2)
    cv2.arrowedLine(floor,hub,(hub[0],hub[1]-24),(0,220,255),3)
    cv2.putText(floor,'radial | blue ring 100 cm, scale halves outside it '
                '(as sent to GPT)',(8,20),0,.42,(255,255,255),1)
    if dropped:
        cv2.putText(floor,'%d obstacle(s) past %.0f cm: position uncertain'
                    %(len(dropped),fetch.ROUTE_RANGE_CM),(8,636),0,.4,(140,140,255),1)
    if unplaced:
        cv2.putText(rgb,'%d obstacle(s) above the horizon, ignored'%len(unplaced),
                    (8,474),0,.4,(150,150,150),1)
    return dict(camera=encode(rgb),floor=encode(floor))

def pixels_to_floor(points):
    # Historical calls answered in pixels, and any one of them can sit above
    # the horizon. Drop those rather than raise: a route missing a point still
    # renders, where an exception loses the whole view.
    lens,_=geometry();result=[]
    for p in points or []:
        try:result.append(list(lens.ground(float(p['x']),float(p['y']))))
        except (fetch.Stop,ValueError,TypeError,KeyError):continue
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
    # The metric contract (2026-09-24) renamed these and answers in centimetres.
    # Accept both shapes so historical calls still render.
    def as_cm(points):
        out=[]
        for q in points or []:
            if isinstance(q,dict) and 'right' in q:out.append([float(q['right']),float(q['forward'])])
            elif isinstance(q,(list,tuple)) and len(q)>=2:out.append([float(q[0]),float(q[1])])
        return out
    previous=metric.get('_previous_route_visualization')
    if previous is None:
        previous=as_cm(metric.get('previous_cm')) or metric.get('previous_route_cm') \
                 or pixels_to_floor(prior.get('route_pixels') or [])
    reference=as_cm(metric.get('reference_cm')) or metric.get('reference_cm') or (previous if top else [])
    if top:previous=[]
    proposal=[];answer={}
    if (directory/'response.json').exists():
        response=json.loads((directory/'response.json').read_text())
        answer=json.loads(''.join(c.get('text','') for x in response.get('output',[]) for c in x.get('content',[]) if c.get('type')=='output_text'))
        proposal=as_cm(answer.get('route_cm')) or pixels_to_floor(answer.get('route_pixels') or [])
    # The metric contract answers with target_px, not contact_pixel. Reading
    # only the old name left every metric call with no target drawn at all --
    # the one mark that says what the route is for.
    target=answer.get('target_px') or answer.get('contact_pixel') \
           or (prior.get('target_was') or {}).get('contact_pixel')
    goal_cm=metric.get('target_cm')
    if goal_cm is None and answer.get('visible') and isinstance(target,dict):
        # Project the pixel the model claimed, so the top-down view marks where
        # the robot now thinks the goal is rather than where it was told.
        try:goal_cm=list(geometry()[0].ground(float(target['x']),float(target['y'])))
        except (fetch.Stop,ValueError,KeyError,TypeError):goal_cm=None
    if isinstance(goal_cm,dict) and 'right' in goal_cm:
        goal_cm=[float(goal_cm['right']),float(goal_cm['forward'])]
    if isinstance(answer.get('target_cm'),dict) and 'right' in (answer.get('target_cm') or {}):
        goal_cm=[float(answer['target_cm']['right']),float(answer['target_cm']['forward'])]
    note='Same captured observation for all layers; previous plan and memory are in this camera frame using the saved motion estimate.'
    if not metric and not top:note+=' Historical call: no separate top-level reference or metric obstacle memory was recorded.'
    if '_previous_route_visualization' in metric:note+=' Orange shows the full previous plan for inspection; GPT received only its first 15 cm as a faint hint (see exact inputs).'
    if 'reference_reliability' in metric:note+=' Reference heuristic weight %.2f after %.0f cm estimated displacement / %.0f degrees accumulated yaw; fresh RGB takes priority.'%(metric['reference_reliability'],metric['reference_distance_cm'],metric['reference_turn_deg'])
    if top:note='Synthetic stationary demo; no motion executed.'
    return dict(available=True,images=render_pair(camera,reference,proposal,target,previous,
                answer.get('obstacles',[]),metric.get('obstacles',[]),goal_cm,metric.get('reference_reliability',1.)),
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
