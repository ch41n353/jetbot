"""Offline matched model comparison with usage-based standard-price estimates."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import random
import statistics
from benchmark_vlm_planning import ROOT, request_one, summarize

RATES={
    'gpt-5.4': [2.5,.25,15,None],
    'gpt-5.5': [5,.5,30,None],
    'gpt-5.6-sol': [4,.4,20,5],
    'gpt-5.6-terra': [2,.2,12,2.5],
    'gpt-5.6-luna': [.2,.02,1.2,.25],
    'gpt-6-astra': [10,1,50,12.5],
}


def estimate_cost(usage,rates):
    """USD; output_tokens already includes reasoning tokens: never add twice."""
    inp=usage['input_tokens'];out=usage['output_tokens']
    details=usage.get('input_tokens_details') or {}
    cached=details.get('cached_tokens',0);writes=details.get('cache_write_tokens',0)
    if any(type(v) is not int or v<0 for v in [inp,out,cached,writes]) or cached+writes>inp:
        raise ValueError('Inconsistent usage counts')
    if writes and rates[3] is None:raise ValueError('Cache-write rate not verified')
    if inp>272000:raise ValueError('Long-context rate not implemented')
    return ((inp-cached-writes)*rates[0]+cached*rates[1]+writes*(rates[3] or 0)+out*rates[2])/1e6


def freeze(directory):
    data=json.loads((ROOT/'reports/vlm-comparison-2026-09-14/expanded/manifest.json').read_text())
    tuning=json.loads((ROOT/'reports/vlm-prompt-tuning-2026-09-14/manifest.json').read_text())
    data.update(tuning['variants']['pixel_contract'])
    data['configs']=[[m,e] for m in ('gpt-5.5','gpt-5.6-sol','gpt-5.6-terra','gpt-5.6-luna') for e in ('none','low','medium')]
    data['configs'] += [['gpt-5.4','none'],['gpt-6-astra','high']]
    data.update(repeats=2,workers=4,prices_usd_per_million=RATES,price_date='2026-09-14',
        price_fields=['uncached_input','cached_input','output_including_reasoning','cache_write'],
        price_sources={m:'https://developers.openai.com/api/docs/models/'+m for m in RATES},
        pricing_caveat='Standard short-context list-price estimate from reported usage; before credits, taxes, account discounts or regional uplifts. No tool-call charges. Sol promotional pricing is documented through at least 2026-11-21.',
        design='All configurations interleaved with identical tuned pixel prompt, image detail high, schema, 4500 max output tokens, four concurrent calls. New matching GPT-5.4 and Astra controls. No motors or live-plan execution.')
    p=directory/'manifest.json'
    if p.exists():raise ValueError('Manifest already frozen')
    p.write_text(json.dumps(data,indent=2)+'\n')


def aggregate(rows,data):
    groups=summarize(rows)
    cases={c['id']:c for c in data['cases']}
    for g in groups:
        rr=[r for r in rows if (r['model'],r['effort'])==(g['model'],g['effort'])]
        present=[r for r in rr if cases[r['case']]['reference']['target_visible']]
        g['present']=len(present)
        g['box_pass']=sum(r.get('score',{}).get('target_iou',0)>=.5 for r in present)
        g['contact_pass']=sum(r.get('score',{}).get('target_pass',False) for r in present)
        g['mean_iou']=statistics.mean(r.get('score',{}).get('target_iou',0) for r in present) if present else None
        g['obstacles_expected']=sum(len(cases[r['case']]['reference']['obstacle_boxes']) for r in rr)
        costs=[r['estimated_cost_usd'] for r in rr if 'estimated_cost_usd' in r]
        g['costed_requests']=len(costs)
        g['mean_cost_usd']=statistics.mean(costs) if costs else None
        g['min_cost_usd']=min(costs) if costs else None
        g['max_cost_usd']=max(costs) if costs else None
        g['total_cost_usd']=sum(costs)
        rate=data['prices_usd_per_million'][g['model']]
        cold=[(r['usage']['input_tokens']*(rate[3] if rate[3] is not None else rate[0])+
               r['usage']['output_tokens']*rate[2])/1e6 for r in rr if r.get('usage')]
        g['mean_same_usage_fresh_input_cost_usd']=statistics.mean(cold) if cold else None
        g['mean_output_tokens']=statistics.mean(r['usage']['output_tokens'] for r in rr if r.get('usage')) if any(r.get('usage') for r in rr) else None
        g['errors']=sum('error' in r for r in rr)
        g['resolved_models']=sorted(set(r.get('resolved_model','unknown') for r in rr))
    return groups


def report(directory,rows,data):
    groups=aggregate(rows,data)
    (directory/'summary.json').write_text(json.dumps(groups,indent=2)+'\n')
    lines=['# GPT-5.5 / GPT-5.6 planning and cost comparison','',
        '{} requests recorded. {}'.format(len(rows),data['design']),'',
        '| Model | Reasoning | Calls | Median time | Boxes | Box + contact | Obstacles | Unsafe approvals | Mean USD/request | Errors |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for g in sorted(groups,key=lambda x:(x['model'],{'none':0,'low':1,'medium':2,'high':3}[x['effort']])):
        price='unknown' if g['mean_cost_usd'] is None else '${:.5f}'.format(g['mean_cost_usd'])
        lines.append('| {model} | {effort} | {calls} | {median_seconds:.2f}s | {box_pass}/{present} | {contact_pass}/{present} | {obstacles_found}/{obstacles_expected} | {unsafe_approvals} | PRICE | {errors} |'.format(**g).replace('PRICE',price))
    lines+=['','## Cost calculation','',
        'Estimated USD = ((input − cached − cache-write) × input rate + cached × cached rate + cache-write × write rate + output × output rate) / 1,000,000. Output usage already includes reasoning tokens. Input usage includes the image. Costs are calculated for each response, then averaged; they are not based on a hypothetical fixed prompt length.',
        '', data['pricing_caveat'],
        '', 'Repeated photos may produce cache hits. `mean_same_usage_fresh_input_cost_usd` separately prices the same measured output usage with all input fresh (including the documented cache-write rate where applicable). Future requests can still use different token counts.',
        '', 'Only responses reporting usage can be priced; failed requests without usage are not assigned a fictitious zero cost. See `costed_requests` and min/max per-request estimates in `summary.json`. The response service tier is recorded; this calculation rejects an unrecognized non-default tier.',
        '', '| Model | Input / million | Cached / million | Output / million |', '|---|---:|---:|---:|']
    for m,rate in data['prices_usd_per_million'].items():
        lines.append('| [{}]({}) | ${:g} | ${:g} | ${:g} |'.format(m,data['price_sources'][m],*rate[:3]))
    lines+=['','## Limits','',
        '- Six correlated photos from one room, plus one absent-target task. Two repeats per configuration. References are frozen assistant annotations, not independent physical ground truth.',
        '- Boxes: IoU ≥0.5. Box + contact: also within 5 pixels. Obstacles: one-to-one IoU ≥0.3. Synthetic geometry uses the same five explicit swept-rectangle checks, repeated across requests; these are not 70 independent environments.',
        '- All configurations use the improved pixel prompt selected in earlier tuning. This is a matched new comparison; its four-request concurrency and prompt differ from the first benchmark. Do not splice old timings into this table.',
        '- Models returning an image location are not demonstrated safe autonomous planners. No model output was used to move the robot.',
        '- No automatic retries, hidden prompt fixes, coordinate offsets, or discarded failures. Reasoning effort changes output-token cost. Schema warmup/caching and provider latency are not separately controlled.',
        '', 'Raw responses: [responses.jsonl](responses.jsonl). Frozen protocol and rates: [manifest.json](manifest.json). Full metrics and price ranges: [summary.json](summary.json).']
    (directory/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['freeze','run','report'],required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if a.stage=='freeze':freeze(a.output);return
    data=json.loads((a.output/'manifest.json').read_text())
    path=a.output/'responses.jsonl'
    rows=list(map(json.loads,path.read_text().splitlines())) if path.exists() else []
    if a.stage=='report':report(a.output,rows,data);return
    for case in data['cases']:
        assert hashlib.sha256(Path(case['image']).read_bytes()).hexdigest()==case['image_sha256']
    done={(r['model'],r['effort'],r['case'],r['repeat']) for r in rows}
    jobs=[(tuple(config),case,rep) for config in data['configs'] for case in data['cases'] for rep in range(data['repeats'])
          if (config[0],config[1],case['id'],rep) not in done]
    random.Random(20260916).shuffle(jobs)
    def run(job):
        config,case,rep=job;r=request_one(config,case,data,rep,90)
        if r.get('usage'):
            try:
                tier=r.get('response',{}).get('service_tier','default')
                if tier not in ('default','auto',None):raise ValueError('Unpriced service tier: '+str(tier))
                r['estimated_cost_usd']=estimate_cost(r['usage'],data['prices_usd_per_million'][config[0]])
            except ValueError as e:r['cost_error']=str(e)
        return r
    with path.open('a') as out,concurrent.futures.ThreadPoolExecutor(max_workers=data['workers']) as pool:
        for f in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):
            r=f.result();rows.append(r);out.write(json.dumps(r)+'\n');out.flush()
            report(a.output,rows,data)
            print(json.dumps({k:r[k] for k in ('model','effort','case','total_seconds','estimated_cost_usd','error') if k in r}),flush=True)


if __name__=='__main__':main()
