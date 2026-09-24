"""Motor-free, exact-image three-arm prompt replay; all requests/responses saved."""
import copy,json,os,sys,time,urllib.request,concurrent.futures
from pathlib import Path
source=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=False)
original=json.loads(source.read_text());candidate=Path('local_nav/prompts/turn-clearance-candidate-20260923.txt').read_text()
context='Current operator observation: the nut-mix can on the right is about 20cm away, not touching the robot. Distance reference point was not specified; treat it as approximate. Assess a small left recentering turn, not forward travel.'
def run(spec):
 arm,i=spec;body=copy.deepcopy(original)
 if arm!='baseline':
  content=body['input'][0]['content'];entry=next(x for x in content if x['type']=='input_text');d=json.loads(entry['text']);d['operator_observation']=context;entry['text']=json.dumps(d)
 if arm=='revised':body['instructions']+='\n\n'+candidate
 dest=out/(arm+'-'+str(i));dest.mkdir();(dest/'request.json').write_text(json.dumps(body))
 row=dict(arm=arm,repeat=i);start=time.monotonic()
 try:
  req=urllib.request.Request('https://api.openai.com/v1/responses',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+os.environ['OPENAI_API_KEY']})
  raw=urllib.request.urlopen(req,timeout=90).read();(dest/'response.json').write_bytes(raw);r=json.loads(raw)
  row['answer']=json.loads(''.join(c.get('text','') for o in r.get('output',[]) for c in o.get('content',[]) if c.get('type')=='output_text'));row['usage']=r.get('usage')
 except Exception as e:row['error']=str(e)
 row['seconds']=time.monotonic()-start;(dest/'result.json').write_text(json.dumps(row));print(json.dumps(row),flush=True);return row
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(run,[(a,i) for a in ['baseline','context_only','revised'] for i in range(2)]))
(out/'results.json').write_text(json.dumps(rows,indent=2))
