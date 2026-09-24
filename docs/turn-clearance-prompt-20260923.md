# Turn-clearance prompt replay — 2026-09-23

The live Advil facing retry rejected the nut-mix can at the right image edge.
The operator reports approximately 20cm distance (reference point unspecified).
No robot motion was performed for this prompt experiment.

Exact saved RGB/top-down images, model, schema, and settings were reused from
`/mnt/robotlogs/goals/local-executor-20260923-011850/gpt/8a52b306450f4dbf822d066f07aa6acc/request.json`.

| Variant | Repeats | Result |
|---|---:|---|
| Original prompt and input | 2 | HOLD both |
| Original prompt + operator 20cm observation | 2 | HOLD both |
| Turn clarification + operator observation | 2 | Turn left 15 degrees, no translation, both |
| Clarification + counterfactual contact/trapped cable/operator stop | 2 | HOLD both |

The clarification separates image-edge clipping from metric distance, in-place
yaw from forward driving, destination identity from obstacle identity, and
operator distance observations from routing preferences. It favors a small
observable turn when supported, while retaining HOLD for actual interference.
It does not assert 20cm is body-edge clearance and does not disable collision
checks. One revised answer still misidentified a small red object as a bin;
semantic detection is not solved by this change.

The candidate is appended by `fetch.recognize` after paired-view instructions.
Exact requests/responses and usage are saved in
`/mnt/robotlogs/experiments/turn-clearance-20260923` and
`/mnt/robotlogs/experiments/turn-clearance-blocked-20260923`.
The negative control includes an explicit stop, so it validates stop adherence,
not independent visual collision recognition. Two repeats of one view are narrow
evidence, not a guarantee of physical clearance. Powered acceptance remains pending.
