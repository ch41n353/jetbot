# Revision 3: measured prompt and usage corrections

Date: 2026-09-20. Model unchanged: gpt-5.6-sol, reasoning none.

## Findings from actual runs and code

- Long instructions mixed target identity, search strategy, imagined floor points,
  and arrival conditions. A target named “carpet 25 cm before the bin” made the
  model choose a synthetic contact instead of consistently measuring the bin.
- The original prompt contradicted itself about judging gaps, made old routes
  mandatory after loss of view, and encouraged large escape turns for close
  objects. That includes a destination already approached.
- `all_done` was described as arrival-dependent even though the controller uses
  it to distinguish a final destination from a multi-step instruction.
- Empty route fallbacks in the runtime can override a model's refusal. Prompt
  wording alone cannot implement a stop.

## Changes

The shorter prompt separates identity, physical floor contact, trajectory and
motion intent. It does not ask GPT to estimate metric standoff or invent search
goals. It preserves useful route memory only when current observations support
it. Close/occluded objects can produce `motion=hold`. The response adapter raises
`PlannerHold` before the dashboard can revive a stale route. Follow and turn
responses cannot simultaneously request both types of movement.

The skill now uses concise visible-target briefs, leaves metric stopping to
configured local controls, and treats holds as exceptions for high-level review.
GPT DRIVE remains the default: seconds=8 is a ceiling, not a mandatory pause.

## Evidence and scope

24 paired API calls: six saved scenes, two repeats, old versus initial candidate.
No camera acquisition or motor commands were used for these comparisons.

| Check | Original | Initial candidate | Refined candidate |
|---|---|---|---|
| Single destination marked final (`all_done=true`) | 5/12 | 12/12 | 6/6 on targeted retest |
| Bin filling view: declines further motion | 0/2 | 2/2 holds | 2/2 holds |
| Negative image: does not hallucinate Advil | 2/2 | **0/2** | 2/2 |
| Visible Advil: identified and approach proposed | 2/2 | 2/2 | 2/2 |

The initial candidate regressed on target identification. A second wording change
explicitly separated the requested target from evidence that it is present and
required distinguishing appearance/label evidence. Six additional calls retested
visible Advil, absent Advil, and the occluded bin twice each. Other scenes were
not rerun after this last wording change. These are small samples, not reliability
estimates or broad proof of better navigation.

Median API latency: original 4.01 s (12 calls), initial candidate 3.12 s (12),
refined 4.06 s (six targeted calls). Scene mixes differ; **no speed improvement is
claimed for the final prompt**.

Actual operator-path replay with fake hardware used `/api/mission` to verify a
hold does not invoke `follow`, turn, or nonzero motor outputs, and reports
`mission ended`, not `mission complete`. Saved responses were also replayed
through `Bench.mission`, projection, obstacle adjustment, and actual dry-run
`fetch.follow`: both original bin responses falsely completed on inferred contact;
both revised bin responses held without claiming arrival. Visible-target replay
called the follower; it was a bounded one-look replay, not an arrival demonstration.

Tests: 76 fetch tests, 27 planner-page tests, four motion-contract tests passed.
Obsolete prompt-substring tests requiring blind continuation/automatic close-object
turns were removed and replaced with behavioral motion-contract coverage.

Raw evidence:

- `/mnt/robotlogs/goals/prompt-contract-20260920/results.json`
- `/mnt/robotlogs/goals/prompt-contract-20260920/refinement.jsonl`
- `/mnt/robotlogs/goals/prompt-contract-20260920/operator-replay.json`

## Remaining limitations

No powered acceptance run of the new prompt has been performed. Numeric standoff,
contact projection uncertainty, goal-adjacent obstacle exclusions, clearance
margin, and safe-route fallbacks for non-hold answers remain controller issues.
The prompt is not a collision detector. A model-derived range watcher is useful
for reducing assistant latency but is not independent ranging or guaranteed
timely cancellation. High-level visual identification remains necessary.

Method reference: [OpenAI evaluation guidance](https://developers.openai.com/api/docs/guides/evaluation-best-practices).
