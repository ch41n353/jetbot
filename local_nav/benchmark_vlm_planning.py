"""Offline, motor-free Responses API comparison on frozen JetBot scenes.

Does not import the robot service or execute model plans. Credentials stay in
OPENAI_API_KEY. References are never included in model requests.
"""
import argparse
import base64
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def obj(properties):
    return dict(type='object', properties=properties,
                required=list(properties), additionalProperties=False)


BOX = obj({k: {'type': 'number'} for k in ('x0', 'y0', 'x1', 'y1')})
POINT = obj({k: {'type': 'number'} for k in ('x', 'y')})
SCHEMA = obj({
    'target_visible': {'type': 'boolean'},
    'target_box': {'anyOf': [BOX, {'type': 'null'}]},
    'contact_pixel': {'anyOf': [POINT, {'type': 'null'}]},
    'obstacles': {'type': 'array', 'items': obj({
        'label': {'type': 'string'}, 'box': BOX})},
    'spatial_answers': {'type': 'array', 'items': obj({
        'id': {'type': 'string'},
        'decision': {'type': 'string', 'enum': ['execute', 'hold']}})},
    'approach_side': {'type': 'string', 'enum': ['left', 'center', 'right', 'not_visible']},
    # Proposed floor-contact pixels only; strict mode keeps every property
    # required, so "no proposal" is the empty array, as target_box uses null.
    'route_pixels': {'type': 'array', 'items': POINT},
})

PROMPT = '''You are the high-level planner of a JetBot. Produce one compact JSON
answer; no tools or follow-up questions. The photograph is 640x480, with pixel
origin top-left. Locate the requested target, give a tight full-object box and
the center of its actual floor contact. Null both if not visible. Identify up to
8 relevant floor obstacles (especially LEGO pieces, bins and boxes near the
approach), excluding the target. Use tight object boxes, not free-space boxes.
approach_side is the target's horizontal image third (left x<213.33, center
x<=426.67, right otherwise); not_visible when absent.

route_pixels proposes a walking path as successive points where the path touches
the FLOOR in this image: first point just ahead of the robot near the bottom
centre, last point at the target's floor contact, intermediate points stepping
around the obstacle footprints you reported. Every point must lie on visible
floor. Do not estimate distances in centimetres; the local controller projects,
validates and may reject these pixels. Use an empty array when no floor path is
visible. It is a proposal, not a motion command.

Also answer each INDEPENDENT SYNTHETIC spatial check supplied below. These maps
are NOT estimated from the photograph. Robot width=12cm, length=15cm; camera is
front-center. Coordinates x right and z forward, rectangles [xmin,zmin,xmax,zmax].
The proposed swept rectangles ALREADY include the whole chassis, 5cm clearance,
position uncertainty and braking. Execute only if EVERY swept rectangle is
contained in inspected free space and intersects NO retained obstacle. Touching
counts as collision. Obstacles persist even when offscreen. The provided swept
rectangles include all turn/rear extents; do not add margins again.
This is an offline planning benchmark, not a command to motors.'''

SPATIAL = [
    dict(id='clear_drive', free=[-35,-35,55,120], obstacles=[[30,70,40,80]],
         swept=[[-16,-24,16,73]]),
    dict(id='offscreen_rear_turn', free=[-50,-50,70,120],
         obstacles=[[-29,-22,-23,-16]], obstacle_visibility='offscreen',
         swept=[[-31,-31,31,31],[-16,-24,16,73]]),
    dict(id='narrow_gap', free=[-40,-35,40,100],
         obstacles=[[-40,30,-10,45],[10,30,40,45]],
         swept=[[-16,-24,16,73]]),
    dict(id='unknown_space', free=[-20,-20,20,80], obstacles=[],
         swept=[[-16,-24,16,73]]),
    dict(id='detour_clear', free=[-60,-40,60,110], obstacles=[[15,40,30,55]],
         swept=[[-31,-31,10,31],[-45,-25,-15,90]]),
]


def freeze(destination):
    """Hand-inspected root-assistant references, fixed before candidate calls."""
    entries = [
        ('far_left','session-20260914-224255-1ac1c2-29024.854240',
         [262,171,281,206], [271.5,206],
         [[250,194,263,208],[198,169,263,207],[299,188,350,206]]),
        ('far_right','session-20260914-224848-999cf7-29162.462213',
         [399,171,417,203], [408,203],
         [[390,190,402,204],[337,168,399,200],[430,191,480,208]]),
        ('near_left','session-20260914-224255-1ac1c2-29062.113672',
         [195,145,279,307], [237,307],
         [[150,248,200,304],[116,163,283,240],[344,220,551,320]]),
        ('near_right','session-20260914-224848-999cf7-29278.973969',
         [530,151,612,329], [571,329],
         [[490,264,533,331],[410,259,482,294],[276,155,463,241]]),
        ('far_center_new','session-20260914-225729-037635-29683.720780',
         [302,168,322,202], [312,202],
         [[291,189,304,203],[234,166,302,201],[336,186,391,204]]),
        ('near_center_new','session-20260914-225729-037635-29752.710527',
         [294,141,380,293], [337,293],
         [[254,241,297,293],[217,241,272,271],[152,157,304,237],[405,213,585,294]]),
    ]
    cases=[]
    for name, stem, box, contact, obstacles in entries:
        path=ROOT/'local_nav/goals'/(stem+'.jpg')
        cases.append(dict(id=name,image=str(path),image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          target='Advil bottle',reference=dict(target_visible=True,target_box=box,
                          contact_pixel=contact,obstacle_boxes=obstacles)))
    absent=dict(cases[-1],id='coke_absent',target='Coca-Cola can',
                reference=dict(target_visible=False,target_box=None,contact_pixel=None,
                               obstacle_boxes=cases[-1]['reference']['obstacle_boxes']))
    cases.append(absent)
    data=dict(version=1,reference_author='root assistant, hand-inspected before candidate responses',
              limitations='Six correlated room photos plus one absent-target task; annotations are not independent physical ground truth. Spatial checks are synthetic, not image-derived.',
              cases=cases,spatial_checks=SPATIAL,prompt=PROMPT,schema=SCHEMA)
    destination.write_text(json.dumps(data,indent=2)+'\n')


def overlap(a,b):
    return not (a[2]<b[0] or b[2]<a[0] or a[3]<b[1] or b[3]<a[1])


def spatial_reference(task):
    f=task['free']
    for b in task['swept']:
        if not (f[0]<=b[0] and f[1]<=b[1] and f[2]>=b[2] and f[3]>=b[3]):return 'hold'
        if any(overlap(b,o) for o in task['obstacles']):return 'hold'
    return 'execute'


def box_values(box):
    if not isinstance(box,dict):raise ValueError('Missing box')
    b=[float(box[k]) for k in ('x0','y0','x1','y1')]
    if not all(math.isfinite(v) for v in b) or not (0<=b[0]<b[2]<=640 and 0<=b[1]<b[3]<=480):
        raise ValueError('Invalid pixel box')
    return b


def iou(a,b):
    area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return area/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area)


def score(answer,case,tasks):
    ref=case['reference']; visible=ref['target_visible']
    visibility=answer['target_visible']==visible
    quality=dict(visibility_correct=visibility)
    if visible and answer['target_visible']:
        box=box_values(answer['target_box']);contact=answer['contact_pixel']
        error=math.hypot(contact['x']-ref['contact_pixel'][0],contact['y']-ref['contact_pixel'][1])
        quality.update(target_iou=iou(box,ref['target_box']),contact_error_px=error)
        quality['target_pass']=quality['target_iou']>=.5 and error<=5
    else:
        quality['target_pass']=visibility and answer['target_box'] is None and answer['contact_pixel'] is None
    boxes=[box_values(o['box']) for o in answer['obstacles']]
    # One-to-one matching prevents one huge detection from satisfying many obstacles.
    matches=[];available=set(range(len(boxes)))
    for refbox in ref['obstacle_boxes']:
        best=max(available,key=lambda j:iou(refbox,boxes[j])) if available else None
        ok=best is not None and iou(refbox,boxes[best])>=.3
        matches.append(ok)
        if ok:available.remove(best)
    quality['obstacles_found']=sum(matches);quality['obstacles_expected']=len(matches)
    spatial={a['id']:a['decision'] for a in answer['spatial_answers']}
    quality['spatial_correct']=sum(spatial.get(t['id'])==spatial_reference(t) for t in tasks)
    quality['spatial_total']=len(tasks)
    quality['unsafe_approvals']=sum(spatial.get(t['id'])=='execute' and spatial_reference(t)=='hold' for t in tasks)
    quality['all_pass']=quality['target_pass'] and all(matches) and quality['spatial_correct']==len(tasks)
    return quality


def canonical_answer(answer, coordinate_space='pixels'):
    """Convert declared normalized coordinates, retaining raw model output."""
    if coordinate_space=='pixels':return answer
    if coordinate_space!='normalized_1000':raise ValueError('Unknown coordinate convention')
    result=json.loads(json.dumps(answer))
    boxes=[o['box'] for o in result['obstacles']]
    if result['target_box'] is not None:boxes.append(result['target_box'])
    for b in boxes:
        for k in ('x0','y0','x1','y1'):
            v=b[k]
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1000:
                raise ValueError('Invalid normalized coordinate')
            b[k]=v*(.64 if k.startswith('x') else .48)
    points=list(result.get('route_pixels') or [])
    if result['contact_pixel'] is not None:points.append(result['contact_pixel'])
    for point in points:
        for k,factor in [('x',.64),('y',.48)]:
            v=point[k]
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1000:
                raise ValueError('Invalid normalized contact')
            point[k]=v*factor
    return result


def request_one(config,case,data,repeat,timeout):
    model,effort=config;started=time.monotonic()
    record=dict(model=model,effort=effort,case=case['id'],repeat=repeat)
    encoded=base64.b64encode(Path(case['image']).read_bytes()).decode('ascii')
    body=dict(model=model,reasoning=dict(effort=effort),store=False,max_output_tokens=4500,
              instructions=data['prompt'],input=[dict(role='user',content=[
                  dict(type='input_text',text=json.dumps(dict(target=case['target'],spatial_checks=data['spatial_checks']))),
                  dict(type='input_image',image_url='data:image/jpeg;base64,'+encoded,detail='high')])],
              text=dict(format=dict(type='json_schema',name='robot_scene_plan',strict=True,schema=data['schema'])))
    examples=[]
    for example in data.get('examples',[]):
        if Path(example['image']).resolve()==Path(case['image']).resolve():
            raise ValueError('Example cannot be the query image')
        example_image=base64.b64encode(Path(example['image']).read_bytes()).decode('ascii')
        examples.extend([dict(role='user',content=[
            dict(type='input_text',text='Labeled visual example only. Target: '+example['target']+'. No spatial checks for this example.'),
            dict(type='input_image',image_url='data:image/jpeg;base64,'+example_image,detail='high')]),
            dict(role='assistant',content=json.dumps(example['answer']))])
    body['input']=examples+body['input']
    request=urllib.request.Request('https://api.openai.com/v1/responses',
        data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+os.environ['OPENAI_API_KEY'],
                                               'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(request,timeout=timeout) as response:raw=json.load(response)
        record['response_seconds']=time.monotonic()-started
        record.update(resolved_model=raw.get('model'),status=raw.get('status'),usage=raw.get('usage'),response=raw)
        if raw.get('status')!='completed':raise ValueError('Response not completed')
        answer=json.loads(''.join(c.get('text','') for item in raw.get('output',[]) if item.get('type')=='message'
                                 for c in item.get('content',[]) if c.get('type')=='output_text'))
        record['answer_native']=answer
        record['coordinate_space']=data.get('coordinate_space','pixels')
        answer=canonical_answer(answer,record['coordinate_space'])
        record['answer']=answer;record['score']=score(answer,case,data['spatial_checks'])
    except urllib.error.HTTPError as error:
        record.update(error='HTTP '+str(error.code),error_body=error.read().decode()[:1000])
    except Exception as error:record['error']=type(error).__name__+': '+str(error)[:250]
    record['total_seconds']=time.monotonic()-started
    return record


def summarize(records):
    result=[]
    for model,effort in sorted(set((r['model'],r['effort']) for r in records)):
        rows=[r for r in records if (r['model'],r['effort'])==(model,effort)]
        valid=[r for r in rows if 'score' in r];scores=[r['score'] for r in valid]
        times=sorted(r['total_seconds'] for r in rows)
        contacts=[s['contact_error_px'] for s in scores if 'contact_error_px' in s]
        result.append(dict(model=model,effort=effort,calls=len(rows),valid=len(valid),
            median_seconds=statistics.median(times),max_seconds=max(times),
            target_pass=sum(s['target_pass'] for s in scores),
            all_pass=sum(s['all_pass'] for s in scores),
            obstacles_found=sum(s['obstacles_found'] for s in scores),
            obstacles_expected=sum(s['obstacles_expected'] for s in scores),
            spatial_correct=sum(s['spatial_correct'] for s in scores),
            spatial_total=sum(s['spatial_total'] for s in scores),
            unsafe_approvals=sum(s['unsafe_approvals'] for s in scores),
            median_contact_error_px=statistics.median(contacts) if contacts else None))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--freeze',action='store_true');p.add_argument('--run',action='store_true')
    p.add_argument('--repeats',type=int,default=2);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--models',nargs='+',default=['gpt-5.4','gpt-5.4-mini'])
    p.add_argument('--efforts',nargs='+',default=['none','low','medium'])
    p.add_argument('--timeout',type=float,default=90)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    manifest=a.output/'manifest.json'
    if a.freeze:
        if manifest.exists():raise ValueError('Refusing to replace frozen manifest')
        freeze(manifest)
    if not a.run:return
    if not os.environ.get('OPENAI_API_KEY'):raise SystemExit('OPENAI_API_KEY is not configured')
    data=json.loads(manifest.read_text())
    for case in data['cases']:
        if hashlib.sha256(Path(case['image']).read_bytes()).hexdigest()!=case['image_sha256']:
            raise ValueError('Image changed since reference freeze')
    path=a.output/'responses.jsonl'
    records=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    done={(r['model'],r['effort'],r['case'],r['repeat']) for r in records}
    jobs=[((m,e),c,rep) for rep in range(a.repeats) for c in data['cases']
          for m in a.models for e in a.efforts
          if (m,e,c['id'],rep) not in done]
    random.Random(20260914).shuffle(jobs)
    with path.open('a') as out, concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(request_one,config,case,data,rep,a.timeout) for config,case,rep in jobs]
        for future in concurrent.futures.as_completed(futures):
            r=future.result();records.append(r);out.write(json.dumps(r)+'\n');out.flush()
            (a.output/'summary.json').write_text(json.dumps(summarize(records),indent=2)+'\n')
            print(json.dumps({k:r[k] for k in ('model','effort','case','repeat','total_seconds','score','error') if k in r}),flush=True)


if __name__=='__main__':main()
