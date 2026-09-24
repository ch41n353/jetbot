# Prompt tuning results — 2026-09-14

Both models use reasoning `none`. Images, model settings, scoring thresholds and two-request concurrency stay fixed; only prompt/output-coordinate convention and the optional labeled example vary.

Development: four tasks, two repeats, five variants per model (80 requests). Held-out: four different tasks, three repeats, original versus the selected tuned variant for each model (48 requests).

The extra labeled-example variant was added after early development results. Each example comes from the development split and is never the query image. No held-out reference was sent to a model or used to select a prompt.

Selections were frozen in `selection.json` before held-out API calls. Selection ranks API/scoring errors, unsafe approvals, target/contact passes, mean IoU, obstacle matches, then latency. It chooses a tuned candidate; validation may still favor the original.

## Development

| Model | Prompt | Median response | Target boxes | Box + contact | Obstacles | Unsafe approvals | Errors |
|---|---|---:|---:|---:|---:|---:|---:|
| gpt-5.4 | baseline | 4.01s | 3/6 | 0/6 | 17/28 | 0 | 0 |
| gpt-5.4 | compact_normalized | 3.24s | 1/6 | 0/6 | 9/28 | 3 | 0 |
| gpt-5.4 | fewshot_pixels | 3.85s | 3/6 | 0/6 | 10/28 | 0 | 0 |
| gpt-5.4 | normalized | 3.71s | 0/6 | 0/6 | 6/28 | 0 | 0 |
| gpt-5.4 | pixel_contract | 3.71s | 3/6 | 1/6 | 13/28 | 0 | 0 |
| gpt-5.4-mini | baseline | 2.72s | 1/6 | 0/6 | 6/28 | 2 | 0 |
| gpt-5.4-mini | compact_normalized | 2.97s | 0/6 | 0/6 | 1/28 | 6 | 1 |
| gpt-5.4-mini | fewshot_pixels | 2.80s | 0/6 | 0/6 | 4/28 | 2 | 0 |
| gpt-5.4-mini | normalized | 2.65s | 0/6 | 0/6 | 0/28 | 6 | 0 |
| gpt-5.4-mini | pixel_contract | 2.42s | 0/6 | 0/6 | 4/28 | 6 | 0 |

## Held-out comparison

| Model | Prompt | Median response | Target boxes | Box + contact | Obstacles | Unsafe approvals | Errors |
|---|---|---:|---:|---:|---:|---:|---:|
| gpt-5.4 | baseline | 4.15s | 0/9 | 0/9 | 6/36 | 0 | 0 |
| gpt-5.4 | pixel_contract | 3.80s | 4/9 | 1/9 | 24/36 | 0 | 0 |
| gpt-5.4-mini | baseline | 2.46s | 0/9 | 0/9 | 5/36 | 8 | 0 |
| gpt-5.4-mini | fewshot_pixels | 2.47s | 0/9 | 0/9 | 7/36 | 7 | 0 |

## Conclusion for this run

The explicit pixel contract improved full GPT-5.4 on the held-out frames: target-box matches rose from 0/9 to 4/9 and critical obstacle matches from 6/36 to 24/36. Median complete response time stayed around four seconds (4.15s original, 3.80s tuned); this small run does not establish a latency improvement.

The improved prompt spells out the original 640×480 image axes, quarter-image anchors, independent horizontal/vertical scaling, bottle appearance, cap-inclusive boxes and bottom-rim contact. Exact text: [pixel_contract.txt](prompts/pixel_contract.txt).

Only 1/9 present-target trials passed both box and ≤5px contact checks after tuning. The prompt is a useful candidate for further evaluation, not a demonstrated replacement for precise local detection/tracking. Mini’s labeled example did not materially close the localization gap and still produced seven unsafe synthetic approvals. Normalized-coordinate prompts underperformed during development.

## Limits

- Target-box pass is IoU ≥0.5; the stricter target pass also requires contact error ≤5 pixels. Obstacles use one-to-one IoU ≥0.3. These thresholds were retained from the original benchmark.
- Held-out photos were not used in the earlier model benchmark or this tuning search. They still show the same room and objects, so this is a small frame-held-out check, not room-level generalization.
- There are three present-target photos and one absent-Coke task per split. Repeats measure output variability; they are not independent scenes.
- The five synthetic spatial maps are repeated from development, so their held-out scores are regression checks, not unseen planning problems. People visible in some held-out images are not comprehensively annotated; these tests do not establish safe human avoidance.
- References are hand annotations by the current assistant, not independent ground truth. All results stay offline; no candidate output was passed to motors and no production prompt was replaced.
- The original pixel answers and normalized native answers are retained. Normalization scales x by 0.64 and y by 0.48; raw images are unchanged. No candidate-specific post-hoc coordinate offsets were fitted.
- One additional image and labeled answer are included for the few-shot condition. Its measured latency includes that extra context. The examples contain no synthetic spatial answers.

Reproduce with `local_nav/tune_vlm_prompt.py`; inspect exact prompts under `prompts/`, native outputs in `dev.jsonl` / `held.jsonl`, and split definitions in `manifest.json`.

Prompting guidance consulted: [GPT-5.4 model guidance](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.4).
