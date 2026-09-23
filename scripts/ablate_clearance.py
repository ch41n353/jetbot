"""Stationary saved-image prompt ablation. No robot service imports or commands."""
import argparse,base64,concurrent.futures,copy,hashlib,json,os,pathlib,time,urllib.request
import cv2
import numpy as np

ROOT=pathlib.Path('/mnt/robotlogs/goals/advil-flow-20260922-060659')
# Human-labelled visible carton silhouettes in the original 640x480 clean frames.
# Conservative 12px image dilation is a diagnostic, NOT a metric body margin.
POLYGONS={4:[[245,196],[292,190],[300,193],[301,211],[247,217]],
          6:[[438,238],[480,225],[552,238],[550,277],[500,290],[435,283]]}
VARIANTS=['buffer_reference','buffer_clean_text_prior','buffer_no_prior',
          'extreme_reference','extreme_no_prior']

def parse(raw):
    return json.loads(''.join(c.get('text','') for o in raw.get('output',[]) for c in o.get('content',[]) if c.get('type')=='output_text'))

def score(a,polygon):
    if polygon is None:return None
    mask=np.zeros((480,640),np.uint8);cv2.fillPoly(mask,[np.array(polygon,np.int32)],1)
    path=np.zeros_like(mask);points=[(320,479)]+[(round(p['x']),round(p['y'])) for p in a.get('route_pixels',[])]
    if len(points)<2:return dict(motion=a.get('motion'),crosses=False,near=False,route=False)
    cv2.polylines(path,[np.array(points,np.int32)],False,1,1)
    expanded=cv2.dilate(mask,np.ones((25,25),np.uint8))
    return dict(motion=a.get('motion'),crosses=bool(np.any(path&mask)),near=bool(np.any(path&expanded)),route=True)

def prepare(original,clean,prompt,extreme,variant):
    b=copy.deepcopy(original);b['instructions']=prompt
    if variant.startswith('extreme'):b['instructions']+='\n\n'+extreme
    for message in b['input']:
        for part in message.get('content',[]):
            if part['type']=='input_image' and variant!='buffer_reference' and variant!='extreme_reference':
                part['image_url']='data:image/jpeg;base64,'+base64.b64encode(clean).decode()
            if part['type']=='input_text' and variant.endswith('no_prior'):
                ctx=json.loads(part['text']);ctx.pop('last_time',None);part['text']=json.dumps(ctx)
    return b

def gallery(rows,out):
    payload=[]
    for row in rows:
        r=dict(row);folder=out/r['artifact'];b=json.loads((folder/'request.json').read_text())
        r['input']=next(p['image_url'] for m in b['input'] for p in m['content'] if p['type']=='input_image')
        r['clean']='data:image/jpeg;base64,'+base64.b64encode((ROOT/'gpt'/r['source_id']/'clean.jpg').read_bytes()).decode()
        r['polygon']=POLYGONS.get(r['call']);r['request']=copy.deepcopy(b)
        for m in r['request']['input']:
            for p in m['content']:
                if p['type']=='input_image':p['image_url']='[exact image shown on left; full payload archived in request.json]'
        payload.append(r)
    html='''<!doctype html><meta charset="utf-8"><title>Trajectory reference ablation</title><style>body{background:#111820;color:#eee;font:16px system-ui;margin:24px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}img,canvas{width:100%}article{border-top:1px solid #789;padding:12px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}select{padding:8px}summary{cursor:pointer}</style><h1>Reference trajectory and clearance ablation</h1><p>Exact input at left; returned path on the clean SAME FRAME at right. Cyan: new route. Red: manually labelled carton silhouette (calls 4 and 6 only, never supplied to GPT). The dotted line from camera bottom centre is an approximate start connector. Image-space overlap is a diagnostic, not a metric collision certificate. Full requests and responses are archived.</p><select id="filter"><option value="all">All variants</option></select><main id="rows"></main><script>const rows=DATA;const make=(t,s)=>{let n=document.createElement(t);if(s!==undefined)n.textContent=s;return n};const filter=document.getElementById('filter');for(const v of [...new Set(rows.map(r=>r.variant))]){let o=make('option',v);o.value=v;filter.append(o)}function render(){document.getElementById('rows').replaceChildren();for(const r of rows){if(filter.value!=='all'&&filter.value!==r.variant)continue;const art=make('article');art.append(make('h2','Call '+r.call+' · '+r.variant+' · repeat '+r.repeat));art.append(make('p',JSON.stringify(r.score)+' · '+r.seconds.toFixed(2)+' s'));const pair=make('div');pair.className='pair';const left=make('div'),right=make('div');left.append(make('h3','Exact input'));const inp=make('img');inp.src=r.input;left.append(inp);right.append(make('h3','New output'));const c=make('canvas');right.append(c);pair.append(left,right);art.append(pair);const im=new Image();im.onload=()=>{c.width=640;c.height=480;const x=c.getContext('2d');x.drawImage(im,0,0);if(r.polygon){x.strokeStyle='#ff5555';x.lineWidth=2;x.beginPath();r.polygon.forEach((p,i)=>i?x.lineTo(...p):x.moveTo(...p));x.closePath();x.stroke()}const pts=(r.answer||{}).route_pixels||[];x.strokeStyle='#00eaff';x.fillStyle='#00eaff';x.lineWidth=3;if(pts.length){x.setLineDash([5,5]);x.beginPath();x.moveTo(320,479);x.lineTo(pts[0].x,pts[0].y);x.stroke();x.setLineDash([])}x.beginPath();pts.forEach((p,i)=>i?x.lineTo(p.x,p.y):x.moveTo(p.x,p.y));x.stroke();pts.forEach((p,i)=>{x.beginPath();x.arc(p.x,p.y,4,0,7);x.fill();x.fillText(i+1,p.x+6,p.y-6)})};im.src=r.clean;art.append(make('p',(r.answer||{}).note||r.error));const d=make('details');d.append(make('summary','Full prompt, context and returned JSON'),make('pre',JSON.stringify({request:r.request,answer:r.answer,error:r.error},null,2)));art.append(d);document.getElementById('rows').append(art)}}filter.onchange=render;render();</script>'''
    html=html.replace('filter.onchange=render;render();',
                      "filter.value=new URLSearchParams(location.search).get('variant')||'all';filter.onchange=render;render();")
    html=html.replace('<h1>Reference trajectory and clearance ablation</h1>',
                      '<h1>Reference trajectory and clearance ablation</h1><p>Experiment: '+out.name+'</p>')
    (out/'comparison.html').write_text(html.replace('DATA',json.dumps(payload).replace('</','<\\/')))

def main():
    p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('--all',action='store_true');p.add_argument('--variant');p.add_argument('--repeats',type=int,default=3);args=p.parse_args()
    out=pathlib.Path(args.out);out.mkdir(parents=True,exist_ok=False)
    prompt=pathlib.Path('local_nav/prompts/trajectory-20260920.txt').read_text();extreme=pathlib.Path('local_nav/prompts/clearance-candidate-20260922.txt').read_text()
    (out/'base-prompt.txt').write_text(prompt);(out/'candidate.txt').write_text(extreme)
    paths=sorted((ROOT/'gpt').glob('*/request.json'),key=lambda x:x.stat().st_mtime)
    jobs=[(i+1,path,v,rep+1) for rep in range(args.repeats) for i,path in enumerate(paths) if args.all or i+1 in (4,6) for v in ([args.variant] if args.variant else VARIANTS)]
    def run(job):
        i,path,v,rep=job;folder=out/('%02d-%s-%d'%(i,v,rep));folder.mkdir();original=json.loads(path.read_text());body=prepare(original,(path.parent/'clean.jpg').read_bytes(),prompt,extreme,v)
        for k in ('model','reasoning','text','max_output_tokens'):assert body[k]==original[k]
        (folder/'request.json').write_text(json.dumps(body));row=dict(call=i,source_id=path.parent.name,variant=v,repeat=rep,artifact=folder.name);start=time.time()
        try:
            request=urllib.request.Request('https://api.openai.com/v1/responses',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+os.environ['OPENAI_API_KEY']})
            with urllib.request.urlopen(request,timeout=90) as response:raw=response.read()
            (folder/'response.json').write_bytes(raw);row['answer']=parse(json.loads(raw));row['score']=score(row['answer'],POLYGONS.get(i))
        except Exception as e:row.update(error=str(e),score=None)
        row['seconds']=time.time()-start;(folder/'result.json').write_text(json.dumps(row));print(json.dumps({k:v for k,v in row.items() if k!='answer'}),flush=True);return row
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(run,jobs))
    (out/'results.json').write_text(json.dumps(rows,indent=2));gallery(rows,out)
if __name__=='__main__':main()
