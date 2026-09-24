# GPT-5.4 planning comparison — 2026-09-14

Status: complete. [Measured results](expanded/RESULTS.md) cover 84 candidate
requests plus 14 matched Astra-high baseline requests. All 98 completed without
API or scoring errors. The user explicitly authorized image uploads for this
work; that authorization is retained in the local collaboration preferences.

## Comparison

- Models: `gpt-5.4`, `gpt-5.4-mini`.
- Reasoning: `none`, `low`, `medium` for each model.
- Expanded set: six actual 640×480 room photographs, plus one absent Coca-Cola
  target task, each repeated twice: 84 requests total, two concurrently.
- Same prompt, high-detail image input, JSON schema, and 4,500-token output cap.
- Fixed shuffle seed; no automatic retries. HTTP failures, incomplete responses,
  malformed geometry, and scoring errors remain failures in the results.
- Record resolved model, usage, response, complete-request latency, and scores.

## Reference and scoring

Root assistant hand annotations were frozen before candidate responses. These
measure agreement with the current assistant, not independent physical ground
truth or general equivalence to Astra. The photos are correlated views of the
same room. No candidate will command motors in this test.

Target pass requires visibility agreement, box IoU at least 0.5 and floor-contact
error at most 5 pixels. An absent target requires false visibility and null box
and contact. Critical obstacle recall uses one-to-one box matches at IoU 0.3.
Not all visible objects are annotated; obstacle precision is not measured.

Every request also includes five explicitly independent synthetic spatial
checks: clear drive, retained offscreen obstacle during a turn, narrow gap,
uninspected rear extent, and clear detour. Their answers come from deterministic
rectangle containment/intersection. These test constraints, not visual map
generation or free-form trajectory planning. Report unsafe approvals separately.
Do not equate success on these questions with safe autonomous navigation.

Aggregate all-pass requires target, all annotated obstacles, and all spatial
checks to pass. Report individual metrics as well: a composite can obscure why
a model failed. Latency includes request preparation, connection and complete
response; it is not pure model inference latency. A matched `gpt-6-astra` / high-reasoning baseline ran after the candidate
requests, using the same 14 scene/repeat combinations and two-request concurrency.
This API baseline does not include the conversation history or tools used by the
interactive assistant.

## Work completed

Text-only API connectivity probe: `gpt-5.4-mini-2026-03-17`, reasoning none,
completed in 2.11 seconds for a trivial `OK` response. This is not a vision or
planning result and must not be used to rank models.

Five offline scoring tests passed. `expanded/manifest.json` holds the additional
fresh centered views and image hashes. The original `manifest.json` remains
unchanged as the initial four-photo reference set.

Additional authorized data-collection approach:
`local_nav/goals/session-20260914-225729-037635-route-01.json`.
Controller reported arrival after 6.96 seconds, 84.58cm estimated travel,
23.29cm estimated final range, zero intermediate model calls, one local floor
recovery during braking. Final image visibly contains the Advil ahead. Pack
minimum 11.46V; no power warning/stop. Session closed after capture.

## Reproduce or resume

```bash
/usr/bin/python3 local_nav/benchmark_vlm_planning.py \
  --output reports/vlm-comparison-2026-09-14/expanded \
  --run --repeats 2 --workers 2
```

Requires `OPENAI_API_KEY` in the environment. Never put its value in logs or files.
The script only sends requests to `https://api.openai.com/v1/responses` and does
not import a robot service. It saves resumable responses and a running summary.
Measured results identify full GPT-5.4 without reasoning as the strongest fast candidate in this set, but no GPT-5.4 configuration met the complete present-target localization criterion. See the results for the visual-grounding gap and limitations.
