# GPT-5.5 / GPT-5.6 planning and cost comparison

196 requests recorded. All configurations interleaved with identical tuned pixel prompt, image detail high, schema, 4500 max output tokens, four concurrent calls. New matching GPT-5.4 and Astra controls. No motors or live-plan execution.

| Model | Reasoning | Calls | Median time | Boxes | Box + contact | Obstacles | Unsafe approvals | Mean USD/request | Errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-5.4 | none | 14 | 4.02s | 4/12 | 1/12 | 21/46 | 0 | $0.00812 | 0 |
| gpt-5.5 | none | 14 | 4.48s | 11/12 | 8/12 | 45/46 | 11 | $0.01667 | 0 |
| gpt-5.5 | low | 14 | 9.63s | 12/12 | 9/12 | 46/46 | 0 | $0.03001 | 0 |
| gpt-5.5 | medium | 14 | 20.84s | 12/12 | 10/12 | 46/46 | 0 | $0.05566 | 0 |
| gpt-5.6-luna | none | 14 | 3.46s | 8/12 | 7/12 | 41/46 | 10 | $0.00055 | 0 |
| gpt-5.6-luna | low | 14 | 9.45s | 12/12 | 12/12 | 42/46 | 0 | $0.00110 | 0 |
| gpt-5.6-luna | medium | 14 | 15.66s | 12/12 | 12/12 | 44/46 | 1 | $0.00152 | 0 |
| gpt-5.6-sol | none | 14 | 4.24s | 12/12 | 12/12 | 45/46 | 2 | $0.00964 | 0 |
| gpt-5.6-sol | low | 14 | 10.36s | 12/12 | 12/12 | 45/46 | 0 | $0.01771 | 0 |
| gpt-5.6-sol | medium | 14 | 16.79s | 12/12 | 12/12 | 44/46 | 0 | $0.02569 | 0 |
| gpt-5.6-terra | none | 14 | 3.78s | 8/12 | 8/12 | 40/46 | 0 | $0.00565 | 0 |
| gpt-5.6-terra | low | 14 | 8.26s | 12/12 | 12/12 | 37/46 | 2 | $0.00977 | 0 |
| gpt-5.6-terra | medium | 14 | 8.82s | 10/12 | 10/12 | 41/46 | 1 | $0.01118 | 0 |
| gpt-6-astra | high | 14 | 34.14s | 11/12 | 9/12 | 46/46 | 0 | $0.09033 | 0 |

## Cost calculation

Estimated USD = ((input − cached − cache-write) × input rate + cached × cached rate + cache-write × write rate + output × output rate) / 1,000,000. Output usage already includes reasoning tokens. Input usage includes the image. Costs are calculated for each response, then averaged; they are not based on a hypothetical fixed prompt length.

Standard short-context list-price estimate from reported usage; before credits, taxes, account discounts or regional uplifts. No tool-call charges. Sol promotional pricing is documented through at least 2026-11-21.

Repeated photos may produce cache hits. `mean_same_usage_fresh_input_cost_usd` separately prices the same measured output usage with all input fresh (including the documented cache-write rate where applicable). Future requests can still use different token counts.

Only responses reporting usage can be priced; failed requests without usage are not assigned a fictitious zero cost. See `costed_requests` and min/max per-request estimates in `summary.json`. The response service tier is recorded; this calculation rejects an unrecognized non-default tier.

| Model | Input / million | Cached / million | Output / million |
|---|---:|---:|---:|
| [gpt-5.4](https://developers.openai.com/api/docs/models/gpt-5.4) | $2.5 | $0.25 | $15 |
| [gpt-5.5](https://developers.openai.com/api/docs/models/gpt-5.5) | $5 | $0.5 | $30 |
| [gpt-5.6-sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol) | $4 | $0.4 | $20 |
| [gpt-5.6-terra](https://developers.openai.com/api/docs/models/gpt-5.6-terra) | $2 | $0.2 | $12 |
| [gpt-5.6-luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna) | $0.2 | $0.02 | $1.2 |
| [gpt-6-astra](https://developers.openai.com/api/docs/models/gpt-6-astra) | $10 | $1 | $50 |

## Limits

- Six correlated photos from one room, plus one absent-target task. Two repeats per configuration. References are frozen assistant annotations, not independent physical ground truth.
- Boxes: IoU ≥0.5. Box + contact: also within 5 pixels. Obstacles: one-to-one IoU ≥0.3. Synthetic geometry uses the same five explicit swept-rectangle checks, repeated across requests; these are not 70 independent environments.
- All configurations use the improved pixel prompt selected in earlier tuning. This is a matched new comparison; its four-request concurrency and prompt differ from the first benchmark. Do not splice old timings into this table.
- Models returning an image location are not demonstrated safe autonomous planners. No model output was used to move the robot.
- No automatic retries, hidden prompt fixes, coordinate offsets, or discarded failures. Reasoning effort changes output-token cost. Schema warmup/caching and provider latency are not separately controlled.

Raw responses: [responses.jsonl](responses.jsonl). Frozen protocol and rates: [manifest.json](manifest.json). Full metrics and price ranges: [summary.json](summary.json).
