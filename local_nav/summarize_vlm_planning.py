"""Write descriptive benchmark results; never calls a model or the robot."""
import argparse
import json
from pathlib import Path
import statistics
from benchmark_vlm_planning import summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    a=p.parse_args()
    records=[json.loads(s) for s in (a.directory/'responses.jsonl').read_text().splitlines()]
    summary=summarize(records)
    for group in summary:
        rows=[r for r in records if (r['model'],r['effort'])==(group['model'],group['effort'])]
        present=[r for r in rows if r['case']!='coke_absent']
        group['present_trials']=len(present)
        group['target_box_pass']=sum(r.get('score',{}).get('target_iou',0)>=.5 for r in present)
        group['present_target_pass']=sum(r.get('score',{}).get('target_pass',False) for r in present)
        group['mean_target_iou']=statistics.mean(r.get('score',{}).get('target_iou',0) for r in present)
        absent=[r for r in rows if r['case']=='coke_absent']
        group['absent_correct']=sum(r.get('score',{}).get('target_pass',False) for r in absent)
        group['absent_trials']=len(absent)
        group['missed_present_targets']=sum(not r.get('answer',{}).get('target_visible',False) for r in present)
        group['reasoning_tokens']=sum((r.get('usage') or {}).get('output_tokens_details',{}).get('reasoning_tokens',0) for r in rows)
    (a.directory/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Measured VLM planning comparison — 2026-09-14','',
           '{} completed requests. Six physical room photographs, seven tasks, two repeats per configuration.'.format(len(records)),
           'The robot stayed stopped during API trials. Results are offline and do not authorize candidate plans for motion.','',
           '| Model | Reasoning | Calls | Median response | Target box matches | Critical obstacles found | Synthetic checks correct | Unsafe approvals |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for g in sorted(summary,key=lambda x:(x['model'],{'none':0,'low':1,'medium':2,'high':3}.get(x['effort'],4))):
        lines.append('| {model} | {effort} | {calls} | {median_seconds:.2f}s | {target_box_pass}/{present_trials} | {obstacles_found}/{obstacles_expected} | {spatial_correct}/{spatial_total} | {unsafe_approvals} |'.format(**g))
    lines+=['','## Contact-point accuracy','',
            '| Model | Reasoning | Median error on reported targets | Present targets passing box + contact checks | Missed present targets | Correct absence decisions |',
            '|---|---|---:|---:|---:|---:|']
    for g in summary:
        error='n/a' if g['median_contact_error_px'] is None else '{:.1f}px'.format(g['median_contact_error_px'])
        lines.append('| {} | {} | {} | {}/{} | {} | {}/{} |'.format(g['model'],g['effort'],error,
                     g['present_target_pass'],g['present_trials'],g['missed_present_targets'],g['absent_correct'],g['absent_trials']))
    lines+=['','## Conclusion for this frozen dataset','',
            'Among the requested GPT-5.4 configurations, full GPT-5.4 with reasoning disabled had the best observed combination of response time, target-box agreement and obstacle recall. It also answered all supplied synthetic spatial checks correctly. More reasoning did not improve the visual localization results in this set.',
            '',
            'None of the six GPT-5.4 configurations met the complete target box + contact criterion on any of the 12 present-target trials. They are not demonstrated replacements for the current visual grounding step. Astra high passed 11/12 such trials; its remaining contact error was 10.3 pixels despite a closely matching bottle box.',
            '',
            'The next candidate architecture is full GPT-5.4 without reasoning for semantic decisions, with a validated local detector/tracker supplying image coordinates and deterministic code checking map geometry. That combination has not been benchmarked or deployed here. The results do not establish that more prompting, another image representation, or another model could not close the gap.',
            '', '## Interpretation limits','',
            '- Box matches use IoU ≥0.5 against frozen assistant annotations. Contact pass additionally requires ≤5 pixels of error. Obstacle recall uses one-to-one IoU ≥0.3 matches.',
            '- Contact-error medians exclude targets the model declined to locate. Read them alongside missed-target counts; refusing difficult images can improve this median.',
            '- The photographs are correlated views from one room. Synthetic scores repeat the same five maps across requests; 70 checks are not 70 independent navigation situations.',
            '- The synthetic maps contain explicit swept rectangles. They test containment, obstacle memory and route approval, not image-derived mapping or free-form trajectory generation.',
            '- All models receive identical photographs, prompts and schema, without prior conversation or tool access. Astra high is a matched API baseline, not the full interactive assistant workflow.',
            '- References are manual assistant annotations, not independent physical ground truth. Small contact-point differences can reflect annotation uncertainty.',
            '- Latency is complete request latency with two concurrent requests, including network overhead; schema warmup and caching were not isolated. The Astra block ran after the other configurations.',
            '- Errors and incomplete outputs remain failures. No automatic retries or candidate-specific prompt tuning were used.','',
            'Raw responses: [responses.jsonl](responses.jsonl). Frozen inputs and references: [manifest.json](manifest.json). Detailed counts: [analysis.json](analysis.json).','',
            'Model configuration documentation: [GPT-5.4](https://developers.openai.com/api/docs/models/gpt-5.4), [GPT-5.4 mini](https://developers.openai.com/api/docs/models/gpt-5.4-mini), [Astra](https://developers.openai.com/api/docs/models/gpt-6-astra).']
    (a.directory/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('Wrote',a.directory/'RESULTS.md')


if __name__=='__main__':main()
