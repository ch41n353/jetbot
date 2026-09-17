"""Motor-free prompt comparison with frozen development/held-out splits."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import random
import statistics
from benchmark_vlm_planning import ROOT, PROMPT, SCHEMA, SPATIAL, request_one

APPEARANCE='''The requested Advil bottle, if present, is a white plastic medicine
bottle with a dark blue cap and a green label with yellow Advil lettering. The
label may be tiny, partly hidden or facing away. Match the whole object using
visible evidence; do not substitute a carton. Coca-Cola means a Coke beverage
can, not a medicine bottle or a red LEGO. Locate only what the image supports.'''

SPATIAL_PROMPT=PROMPT[PROMPT.index('Also answer'):]
PIXELS='''Return JSON for the attached 640-column by 480-row photograph.
All visual coordinates are ORIGINAL image pixels, measured directly on the image
plane. Top-left=(0,0), bottom-right=(640,480), center=(320,240).
Quarter-height rows are 120,240,360; quarter-width columns are 160,320,480.
Estimate horizontal and vertical fractions independently; multiply horizontal
fractions by 640 and vertical fractions by 480. Do not use a square canvas,
1000-unit coordinates, perspective-corrected coordinates, or floor distances.
Bound each object tightly by its visible top, bottom, left and right edges.
For a bottle include the cap and body. The floor-contact point lies on the
bottle's bottom rim, not at the image bottom or the label's bottom.
Check that boxes match the object's apparent size and that the contact sits on
the observed bottom edge. Null target_box and contact_pixel if target is absent.
List up to 8 nearby floor obstacles, prioritizing LEGO, bins and boxes. Exclude
the target. Each obstacle gets its own tight box. approach_side is the image
third containing the target center: left, center, right; otherwise not_visible.
'''+APPEARANCE+'\n'+SPATIAL_PROMPT

NORMALIZED='''Return only the requested JSON for the attached image. All visual
coordinates use INDEPENDENT normalized axes from 0 to 1000, regardless of image
aspect ratio: left=0, right=1000, top=0, bottom=1000. Center=(500,500).
For example, an object spanning 20% to 30% of image width and 40% to 60% of image
height has x0=200,x1=300,y0=400,y1=600. This is a coordinate example, not an object
location in the photograph. Do not convert to pixels yourself; the caller will.
The legacy field contact_pixel ALSO uses 0..1000 normalized coordinates.
Find each object's visual edges directly in the image, then express them as
fractions of the FULL image width/height. Do not rectify perspective. A bottle
box includes its cap and body; its contact is centered on the visible bottom
rim. Do not extend the box to the image bottom unless it is actually clipped.
Return null box and contact when absent. List up to 8 nearby floor obstacles,
prioritizing individual LEGO pieces, bins and boxes; exclude the target.
approach_side is the horizontal image third: left, center, right, or not_visible.
'''+APPEARANCE+'\n'+SPATIAL_PROMPT

COMPACT='''Label the target and nearby floor obstacles in the photograph.
Return JSON only. Every visual coordinate, including contact_pixel, is normalized
0..1000 separately for width and height. Top-left=(0,0), bottom-right=(1000,1000).
Boxes tightly cover visible objects; include a bottle's cap. Contact is the
center of the bottom rim touching carpet. Missing target: false and nulls.
Identify up to 8 obstacles, prioritizing LEGO beside the target, then bins and
boxes. Exclude target. approach_side: horizontal image third or not_visible.
'''+APPEARANCE+'''
The spatial_checks are independent synthetic maps, NOT image coordinates.
All their rectangles already include body size, clearance, uncertainty and
braking. For every proposed swept rectangle b and inspected rectangle f:
require f.xmin<=b.xmin, f.zmin<=b.zmin, b.xmax<=f.xmax, b.zmax<=f.zmax.
For each retained obstacle o require at least one STRICT separation:
b.xmax<o.xmin OR o.xmax<b.xmin OR b.zmax<o.zmin OR o.zmax<b.zmin.
If any requirement fails, hold. Otherwise execute. Check every swept rectangle,
including rear and turning extents. Keep obstacles that are offscreen.
'''


def freeze(out):
    source=ROOT/'reports/vlm-comparison-2026-09-14/expanded/manifest.json'
    old=json.loads(source.read_text())
    dev=[c for c in old['cases'] if c['id'] in ('far_left','near_center_new','near_right','coke_absent')]
    refs=[
        ('held_mid_right','session-20260914-191854-d9cf7e-16569.301958',
         [495,162,538,239],[516.5,239],[[356,154,473,214],[537,205,636,250],[294,190,309,202]]),
        ('held_far_left','session-20260914-202836-4a4deb-20750.308779',
         [122,178,140,211],[131,211],[[78,178,123,214],[157,190,201,206],[50,211,62,221]]),
        ('held_near_center','session-20260914-203315-d1f494-22629.274970',
         [281,143,365,307],[323,307],[[214,249,276,287],[187,159,290,236],[429,230,640,340]])]
    held=[]
    for name,stem,box,contact,obstacles in refs:
        path=ROOT/'local_nav/goals'/(stem+'.jpg')
        held.append(dict(id=name,image=str(path),image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),target='Advil bottle',
                         reference=dict(target_visible=True,target_box=box,contact_pixel=contact,obstacle_boxes=obstacles)))
    held.append(dict(held[0],id='held_coke_absent',target='Coca-Cola can',reference=dict(target_visible=False,
                     target_box=None,contact_pixel=None,obstacle_boxes=held[0]['reference']['obstacle_boxes'])))
    data=dict(dev=dev,held=held,schema=SCHEMA,spatial_checks=SPATIAL,
              variants={k:dict(prompt=p,coordinate_space=c) for k,p,c in [
                  ('baseline',PROMPT,'pixels'),('pixel_contract',PIXELS,'pixels'),
                  ('normalized',NORMALIZED,'normalized_1000'),('compact_normalized',COMPACT,'normalized_1000')]},
              selection='Minimize unsafe approvals, maximize target/contact passes, mean target IoU (misses zero), obstacle matches, then minimize median latency. Choose from tuned variants using dev only.',
              limitations='Held-out photos are previously unused in model benchmarks, but from the same room and historical robot runs. Not an independent environment. References are assistant annotations.')
    p=out/'manifest.json'
    if p.exists():raise ValueError('Manifest already frozen')
    p.write_text(json.dumps(data,indent=2)+'\n')


def groups(rows):
    result=[]
    for model,variant in sorted(set((r['model'],r['variant']) for r in rows)):
        rr=[r for r in rows if (r['model'],r['variant'])==(model,variant)]
        present=[r for r in rr if 'absent' not in r['case']]
        ss=[r.get('score',{}) for r in rr]
        result.append(dict(model=model,variant=variant,calls=len(rr),errors=sum('error' in r for r in rr),
            unsafe=sum(s.get('unsafe_approvals',0) for s in ss),
            target_pass=sum(r.get('score',{}).get('target_pass',False) for r in present),
            target_box_pass=sum(r.get('score',{}).get('target_iou',0)>=.5 for r in present),
            present=len(present),mean_iou=statistics.mean(r.get('score',{}).get('target_iou',0) for r in present) if present else 0.,
            obstacles=sum(s.get('obstacles_found',0) for s in ss),obstacle_total=sum(s.get('obstacles_expected',0) for s in ss),
            absent_correct=sum(r.get('score',{}).get('target_pass',False) for r in rr if 'absent' in r['case']),
            median_seconds=statistics.median(r['total_seconds'] for r in rr)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['freeze','dev','fewshot','select','held'],required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.stage=='freeze':freeze(a.output);return
    data=json.loads((a.output/'manifest.json').read_text())
    extra=a.output/'fewshot-variant.json'
    if a.stage=='fewshot' and not extra.exists():
        extra.write_text(json.dumps(dict(prompt=PIXELS+'\nA labeled visual example precedes the query. Use it to understand coordinates and object identity; estimate new coordinates from the query image, never copy the example locations.',
                                        coordinate_space='pixels',fewshot=True),indent=2)+'\n')
    if extra.exists():data['variants']['fewshot_pixels']=json.loads(extra.read_text())
    if a.stage=='select':
        if (a.output/'selection.json').exists():raise ValueError('Selection is already frozen')
        rows=list(map(json.loads,(a.output/'dev.jsonl').read_text().splitlines()))
        assert len(rows)==(80 if extra.exists() else 64)
        selected={}
        for model in ('gpt-5.4','gpt-5.4-mini'):
            candidates=[g for g in groups(rows) if g['model']==model and g['variant']!='baseline']
            best=max(candidates,key=lambda g:(-g['errors'],-g['unsafe'],g['target_pass'],g['mean_iou'],g['obstacles'],-g['median_seconds']))
            selected[model]=best['variant']
        (a.output/'selection.json').write_text(json.dumps(selected,indent=2)+'\n');print(selected);return
    split='dev' if a.stage=='fewshot' else a.stage
    rows_path=a.output/(split+'.jsonl')
    rows=list(map(json.loads,rows_path.read_text().splitlines())) if rows_path.exists() else []
    done={(r['model'],r['variant'],r['case'],r['repeat']) for r in rows}
    selection=json.loads((a.output/'selection.json').read_text()) if a.stage=='held' else None
    jobs=[]
    for model in ('gpt-5.4','gpt-5.4-mini'):
        variants=(['fewshot_pixels'] if a.stage=='fewshot' else
                  ['baseline',selection[model]] if selection else [v for v in data['variants'] if v!='fewshot_pixels'])
        for v in variants:
            for case in data[split]:
                assert hashlib.sha256(Path(case['image']).read_bytes()).hexdigest()==case['image_sha256']
                for rep in range(3 if selection else 2):
                    if (model,v,case['id'],rep) not in done:jobs.append((model,v,case,rep))
    random.Random(20260915).shuffle(jobs)
    def run(job):
        m,v,c,rep=job
        d=dict(data,**data['variants'][v])
        if d.get('fewshot'):
            example=next(e for e in data['dev'] if e['id'] in ('far_left','near_right') and e['image']!=c['image'])
            ref=example['reference'];b=ref['target_box'];pt=ref['contact_pixel']
            answer=dict(target_visible=True,target_box=dict(zip(('x0','y0','x1','y1'),b)),
                contact_pixel=dict(zip(('x','y'),pt)),
                obstacles=[dict(label='floor obstacle',box=dict(zip(('x0','y0','x1','y1'),o))) for o in ref['obstacle_boxes']],
                spatial_answers=[],route_pixels=[],approach_side='left' if pt[0]<213.33 else 'center' if pt[0]<=426.67 else 'right')
            d['examples']=[dict(image=example['image'],target=example['target'],answer=answer)]
        r=request_one((m,'none'),c,d,rep,90);r['variant']=v
        return r
    with rows_path.open('a') as out,concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for f in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):
            r=f.result();rows.append(r);out.write(json.dumps(r)+'\n');out.flush()
            (a.output/(split+'-summary.json')).write_text(json.dumps(groups(rows),indent=2)+'\n')
            print(json.dumps({k:r[k] for k in ('model','variant','case','repeat','total_seconds','score','error') if k in r}),flush=True)


if __name__=='__main__':main()
