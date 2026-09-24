# Measured VLM planning comparison — 2026-09-14

98 completed requests. Six physical room photographs, seven tasks, two repeats per configuration.
The robot stayed stopped during API trials. Results are offline and do not authorize candidate plans for motion.

| Model | Reasoning | Calls | Median response | Target box matches | Critical obstacles found | Synthetic checks correct | Unsafe approvals |
|---|---|---:|---:|---:|---:|---:|---:|
| gpt-5.4 | none | 14 | 3.88s | 5/12 | 21/46 | 70/70 | 0 |
| gpt-5.4 | low | 14 | 8.94s | 2/12 | 16/46 | 70/70 | 0 |
| gpt-5.4 | medium | 14 | 18.10s | 1/12 | 11/46 | 70/70 | 0 |
| gpt-5.4-mini | none | 14 | 2.73s | 1/12 | 8/46 | 48/70 | 8 |
| gpt-5.4-mini | low | 14 | 5.72s | 0/12 | 8/46 | 65/70 | 5 |
| gpt-5.4-mini | medium | 14 | 13.89s | 1/12 | 9/46 | 70/70 | 0 |
| gpt-6-astra | high | 14 | 28.27s | 12/12 | 46/46 | 70/70 | 0 |

## Contact-point accuracy

| Model | Reasoning | Median error on reported targets | Present targets passing box + contact checks | Missed present targets | Correct absence decisions |
|---|---|---:|---:|---:|---:|
| gpt-5.4 | low | 78.8px | 0/12 | 2 | 2/2 |
| gpt-5.4 | medium | 90.0px | 0/12 | 1 | 2/2 |
| gpt-5.4 | none | 39.3px | 0/12 | 6 | 2/2 |
| gpt-5.4-mini | low | 74.1px | 0/12 | 0 | 2/2 |
| gpt-5.4-mini | medium | 69.3px | 0/12 | 0 | 2/2 |
| gpt-5.4-mini | none | 62.4px | 0/12 | 0 | 2/2 |
| gpt-6-astra | high | 1.7px | 11/12 | 0 | 2/2 |

## Conclusion for this frozen dataset

Among the requested GPT-5.4 configurations, full GPT-5.4 with reasoning disabled had the best observed combination of response time, target-box agreement and obstacle recall. It also answered all supplied synthetic spatial checks correctly. More reasoning did not improve the visual localization results in this set.

None of the six GPT-5.4 configurations met the complete target box + contact criterion on any of the 12 present-target trials. They are not demonstrated replacements for the current visual grounding step. Astra high passed 11/12 such trials; its remaining contact error was 10.3 pixels despite a closely matching bottle box.

The next candidate architecture is full GPT-5.4 without reasoning for semantic decisions, with a validated local detector/tracker supplying image coordinates and deterministic code checking map geometry. That combination has not been benchmarked or deployed here. The results do not establish that more prompting, another image representation, or another model could not close the gap.

## Interpretation limits

- Box matches use IoU ≥0.5 against frozen assistant annotations. Contact pass additionally requires ≤5 pixels of error. Obstacle recall uses one-to-one IoU ≥0.3 matches.
- Contact-error medians exclude targets the model declined to locate. Read them alongside missed-target counts; refusing difficult images can improve this median.
- The photographs are correlated views from one room. Synthetic scores repeat the same five maps across requests; 70 checks are not 70 independent navigation situations.
- The synthetic maps contain explicit swept rectangles. They test containment, obstacle memory and route approval, not image-derived mapping or free-form trajectory generation.
- All models receive identical photographs, prompts and schema, without prior conversation or tool access. Astra high is a matched API baseline, not the full interactive assistant workflow.
- References are manual assistant annotations, not independent physical ground truth. Small contact-point differences can reflect annotation uncertainty.
- Latency is complete request latency with two concurrent requests, including network overhead; schema warmup and caching were not isolated. The Astra block ran after the other configurations.
- Errors and incomplete outputs remain failures. No automatic retries or candidate-specific prompt tuning were used.

Raw responses: [responses.jsonl](responses.jsonl). Frozen inputs and references: [manifest.json](manifest.json). Detailed counts: [analysis.json](analysis.json).

Model configuration documentation: [GPT-5.4](https://developers.openai.com/api/docs/models/gpt-5.4), [GPT-5.4 mini](https://developers.openai.com/api/docs/models/gpt-5.4-mini), [Astra](https://developers.openai.com/api/docs/models/gpt-6-astra).
