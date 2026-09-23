"""Replay saved requests changing only instructions; never connects to motors."""
import concurrent.futures
import copy
import json
import os
import pathlib
import sys
import time
import urllib.request
import base64
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]/'local_nav'))
from evaluate_gpt_routes import Lens
import fetch

root=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=False)
prompt=pathlib.Path('local_nav/prompts/trajectory-20260920.txt').read_text()
def answer(raw):
    return json.loads(''.join(c.get('text','') for o in raw.get('output',[]) for c in o.get('content',[]) if c.get('type')=='output_text'))
cases=sorted((root/'gpt').glob('*/request.json'),key=lambda p:p.stat().st_mtime)
def run(item):
    i,path=item;dest=out/path.parent.name;dest.mkdir()
    body=json.loads(path.read_text());body['instructions']=prompt
    original=json.loads(path.read_text());assert {k:v for k,v in body.items() if k!='instructions'}=={k:v for k,v in original.items() if k!='instructions'}
    (dest/'request.json').write_text(json.dumps(body))
    old=answer(json.loads((path.parent/'response.json').read_text()))
    start=time.time();row=dict(index=i+1,id=path.parent.name,old=old)
    try:
        req=urllib.request.Request('https://api.openai.com/v1/responses',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+os.environ['OPENAI_API_KEY']})
        raw=urllib.request.urlopen(req,timeout=90).read();(dest/'response.json').write_bytes(raw)
        row.update(new=answer(json.loads(raw)),seconds=time.time()-start)
    except Exception as e:row.update(error=str(e),seconds=time.time()-start)
    lens=Lens()
    # Fixed original carton contacts for both routes: changes in detection must
    # not masquerade as a clearance improvement. This is a point proxy only.
    contacts=[]
    for o in old.get('obstacles',[]):
        if 'carton' in o.get('label','').lower():
            try:contacts.append(lens.ground(**dict(x=o['contact_pixel']['x'],y=o['contact_pixel']['y'])))
            except Exception:
                try:contacts.append(lens.ground(o['contact_pixel']['x'],o['contact_pixel']['y']))
                except Exception:pass
    def metric(a):
        points=[]
        for p in a.get('route_pixels',[]):
            try:points.append(lens.ground(p['x'],p['y']))
            except Exception:pass
        if not points or not contacts:return None
        return min(fetch._near_segment(s,e,c)[1] for s,e in zip([(0,0)]+points,points) for c in contacts)
    row['old_contact_clearance_cm']=metric(old);row['new_contact_clearance_cm']=metric(row.get('new',{}))
    row['image']='data:image/jpeg;base64,'+base64.b64encode((path.parent/'clean.jpg').read_bytes()).decode()
    row['input_image']=next(p['image_url'] for m in body['input'] for p in m['content'] if p['type']=='input_image')
    (dest/'result.json').write_text(json.dumps({k:v for k,v in row.items() if k not in ('image','input_image')}))
    print(json.dumps({k:v for k,v in row.items() if k not in ('image','input_image','old','new')}),flush=True)
    return row
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(run,enumerate(cases)))
(out/'results.json').write_text(json.dumps(rows))
html='''<!doctype html><meta charset="utf-8"><title>Clearance prompt replay</title><style>body{background:#111820;color:#eee;font:16px system-ui;margin:24px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}canvas,img{width:100%}article{border-top:1px solid #789;padding:15px 0}pre{white-space:pre-wrap}summary{cursor:pointer}</style><h1>Original vs larger-buffer prompt · 14 saved requests</h1><p>Identical input images, prior context, model and settings. Only instructions changed. Cyan: proposed route. Red: original obstacle contacts. Yellow: target. Output overlays use the same clean frame. Point-clearance estimates are not body/box-edge clearance or proof of safety.</p><main id="rows"></main><script>const rows=DATA;function draw(canvas,r,a){const im=new Image();im.onload=()=>{canvas.width=640;canvas.height=480;const c=canvas.getContext('2d');c.drawImage(im,0,0,640,480);c.strokeStyle='#00eaff';c.fillStyle='#00eaff';c.lineWidth=3;c.beginPath();(a.route_pixels||[]).forEach((p,i)=>{i?c.lineTo(p.x,p.y):c.moveTo(p.x,p.y)});c.stroke();(a.route_pixels||[]).forEach((p,i)=>{c.beginPath();c.arc(p.x,p.y,4,0,7);c.fill();c.fillText(i+1,p.x+6,p.y-6)});c.strokeStyle='red';for(const o of r.old.obstacles||[]){const p=o.contact_pixel;c.strokeRect(p.x-5,p.y-5,10,10)}if(a.contact_pixel){const p=a.contact_pixel;c.strokeStyle='yellow';c.beginPath();c.arc(p.x,p.y,9,0,7);c.stroke()}};im.src=r.image}for(const r of rows){const art=document.createElement('article');const h=document.createElement('h2');h.textContent='Call '+r.index;art.append(h);const grid=document.createElement('div');grid.className='grid';for(const [title,key]of [['Exact shared input','input'],['Original output','old'],['Larger-buffer output','new']]){const box=document.createElement('div');const t=document.createElement('h3');t.textContent=title;box.append(t);if(key==='input'){const im=document.createElement('img');im.src=r.input_image;box.append(im)}else{const c=document.createElement('canvas');box.append(c);draw(c,r,r[key]||{});const p=document.createElement('p');p.textContent=(r[key]||{}).motion+' · '+((r[key]||{}).note||r.error||'');box.append(p)}grid.append(box)}art.append(grid);const p=document.createElement('p');p.textContent='Minimum route-to-original-carton-contact distance (cm): '+r.old_contact_clearance_cm+' → '+r.new_contact_clearance_cm;art.append(p);const d=document.createElement('details');const s=document.createElement('summary');s.textContent='Original and new JSON';const pre=document.createElement('pre');pre.textContent=JSON.stringify({old:r.old,new:r.new,error:r.error},null,2);d.append(s,pre);art.append(d);document.getElementById('rows').append(art)}</script>'''
(out/'comparison.html').write_text(html.replace('DATA',json.dumps(rows).replace('</','<\\/')))
