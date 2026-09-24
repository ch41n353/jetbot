# Autonomous search and navigation: motor-free evaluation

This evaluation exercises the intended GPT-5.6 Sol recognition contract,
persistent search graph, deterministic local search policy, safety gates, and
production final-approach controller without contacting an API or robot service.
It does not import the OpenAI client or issue motor I/O.

## Result

All nine deterministic scenarios passed:

| Scenario | Expected result | Observed result |
| --- | --- | --- |
| Target visible | Grounded recognition ends search without authorizing graph motion | `object_found`; model contract `gpt-5.6-sol` |
| Target absent | Negative recognition advances bounded coverage | `known_map_searched` |
| Cable blocks turn/route | Fresh local sweep prevents motion | 0 turn candidates; graph falls back to `map_required` |
| API latency and failure | Robot remains stopped through timeout and bounded retry | 17 simulated seconds; 0 motor commands |
| Stale API reply | Reply from prior target/mission is discarded | `stale_reply_accepted: false` |
| Cancellation during inference | Late reply cannot revive cancelled work | Reply rejected; 0 motor commands |
| Low battery | Planned navigation cannot reach the executor | Motion dispatch false with latched `battery_pack_low` |
| Multi-room frontier/checkpoint | Search expands four places and resumes its checkpoint | Target found after 40 views and 3 certified edges; 0 unsafe traversals |
| Final approach | Local controller reaches standoff and stops | 18.66 cm range; 0 intermediate model calls; 0 clearance violations; motors `[0,0]` |

The machine-readable evidence is in `evaluation.json`. Reproduce it with:

```bash
/usr/bin/python3 local_nav/evaluate_autonomous_stack.py \
  --output reports/autonomous-stack-2026-09-15/evaluation.json
```

Run the focused assertions with:

```bash
/usr/bin/python3 -m unittest tests.test_autonomous_stack_simulation
```

## What this proves

The current component contracts can keep GPT-5.6 Sol outside the motion-control
loop. The model supplies only target visibility and image grounding. Local code
selects coverage actions and graph routes, checks fresh clearance and power,
rejects replies bound to an old mission, and performs the final tracked approach.
Cancellation, inference delay, and API failure do not create a motor heartbeat.

## Limits

The graph's mapping, localization, doorway discovery, and API timing are
scheduled events. They do not validate live SLAM, real network behavior, or a
physical cable detector. The approach case uses rendered camera/IMU/wheel
dynamics and does not establish hardware accuracy. A live multi-room stack still
needs a mapping/localization adapter that supplies versioned submaps, fresh pose,
and corridor evidence to these tested interfaces.
