"""Export frozen prompt variants and report paired held-out measurements."""
import argparse
import json
from pathlib import Path
from tune_vlm_prompt import groups


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
    manifest=json.loads((a.directory/'manifest.json').read_text())
    extra=a.directory/'fewshot-variant.json'
    if extra.exists():manifest['variants']['fewshot_pixels']=json.loads(extra.read_text())
    prompts=a.directory/'prompts';prompts.mkdir(exist_ok=True)
    for name,variant in manifest['variants'].items():
        (prompts/(name+'.txt')).write_text(variant['prompt']+'\n')
    lines=['# Prompt tuning results — 2026-09-14','',
        'Both models use reasoning `none`. Images, model settings, scoring thresholds and two-request concurrency stay fixed; only prompt/output-coordinate convention and the optional labeled example vary.',
        '', 'Development: four tasks, two repeats, five variants per model (80 requests). Held-out: four different tasks, three repeats, original versus the selected tuned variant for each model (48 requests).',
        '', 'The extra labeled-example variant was added after early development results. Each example comes from the development split and is never the query image. No held-out reference was sent to a model or used to select a prompt.',
        '', 'Selections were frozen in `selection.json` before held-out API calls. Selection ranks API/scoring errors, unsafe approvals, target/contact passes, mean IoU, obstacle matches, then latency. It chooses a tuned candidate; validation may still favor the original.', '']
    for split in ['dev','held']:
        rows=list(map(json.loads,(a.directory/(split+'.jsonl')).read_text().splitlines()))
        assert len(rows)==(80 if split=='dev' else 48)
        refs={c['id']:c['reference'] for c in manifest[split]}
        result=groups(rows)
        lines+=['## '+('Development' if split=='dev' else 'Held-out comparison'),'',
            '| Model | Prompt | Median response | Target boxes | Box + contact | Obstacles | Unsafe approvals | Errors |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
        for g in result:
            rr=[r for r in rows if (r['model'],r['variant'])==(g['model'],g['variant'])]
            g['obstacle_total']=sum(len(refs[r['case']]['obstacle_boxes']) for r in rr)
            lines.append('| {model} | {variant} | {median_seconds:.2f}s | {target_box_pass}/{present} | {target_pass}/{present} | {obstacles}/{obstacle_total} | {unsafe} | {errors} |'.format(**g))
        (a.directory/(split+'-analysis.json')).write_text(json.dumps(result,indent=2)+'\n')
        lines+=['']
    lines+=['## Conclusion for this run','',
        'The explicit pixel contract improved full GPT-5.4 on the held-out frames: target-box matches rose from 0/9 to 4/9 and critical obstacle matches from 6/36 to 24/36. Median complete response time stayed around four seconds (4.15s original, 3.80s tuned); this small run does not establish a latency improvement.',
        '',
        'The improved prompt spells out the original 640×480 image axes, quarter-image anchors, independent horizontal/vertical scaling, bottle appearance, cap-inclusive boxes and bottom-rim contact. Exact text: [pixel_contract.txt](prompts/pixel_contract.txt).',
        '',
        'Only 1/9 present-target trials passed both box and ≤5px contact checks after tuning. The prompt is a useful candidate for further evaluation, not a demonstrated replacement for precise local detection/tracking. Mini’s labeled example did not materially close the localization gap and still produced seven unsafe synthetic approvals. Normalized-coordinate prompts underperformed during development.',
        '', '## Limits','',
        '- Target-box pass is IoU ≥0.5; the stricter target pass also requires contact error ≤5 pixels. Obstacles use one-to-one IoU ≥0.3. These thresholds were retained from the original benchmark.',
        '- Held-out photos were not used in the earlier model benchmark or this tuning search. They still show the same room and objects, so this is a small frame-held-out check, not room-level generalization.',
        '- There are three present-target photos and one absent-Coke task per split. Repeats measure output variability; they are not independent scenes.',
        '- The five synthetic spatial maps are repeated from development, so their held-out scores are regression checks, not unseen planning problems. People visible in some held-out images are not comprehensively annotated; these tests do not establish safe human avoidance.',
        '- References are hand annotations by the current assistant, not independent ground truth. All results stay offline; no candidate output was passed to motors and no production prompt was replaced.',
        '- The original pixel answers and normalized native answers are retained. Normalization scales x by 0.64 and y by 0.48; raw images are unchanged. No candidate-specific post-hoc coordinate offsets were fitted.',
        '- One additional image and labeled answer are included for the few-shot condition. Its measured latency includes that extra context. The examples contain no synthetic spatial answers.',
        '', 'Reproduce with `local_nav/tune_vlm_prompt.py`; inspect exact prompts under `prompts/`, native outputs in `dev.jsonl` / `held.jsonl`, and split definitions in `manifest.json`.',
        '', 'Prompting guidance consulted: [GPT-5.4 model guidance](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.4).']
    (a.directory/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(a.directory/'RESULTS.md')


if __name__=='__main__':main()
