# Simulation report — 2026-09-15

All work in this pass was offline. The physical robot and its service remained
off. No new VLM/API calls, GitHub pushes, or unattended physical experiments.

## Changes made

1. **Removed unnecessary recognition handoff.** A motion-uncertainty stop no
   longer prevents Sol from inspecting the final image. Search-only detection
   can complete without pretending the global pose is accurate. Further movement
   and object approaches remain blocked until localization is restored.
2. **Added persistent multi-room scheduling.** Places, doorway connections,
   frontiers, coverage, and versioned route evidence live in `search_graph.py`.
   Unknown doors are not traversable by default; graph plans need fresh local
   clearance checks. The scheduler has no fixed room-count limit.
3. **Fixed stalled exploration.** A blocked frontier can be deferred so another
   reachable place is searched. A map revision makes it eligible for retry.
4. **Reject stale route and recognition results.** Door changes invalidate a
   planned graph route before handoff. Recognition tickets bind replies to the
   exact mission and image, bound the pending queue to two, and reject replies
   from an earlier target search. New searches reuse maps but not another
   object's negative coverage. Invalid localization timestamps request recovery.
5. **Made turn previews faster.** Vectorized and cached the existing sweep
   calculation without changing its pivots, samples, or clearance allowance.

## Measured results

| Experiment | Result |
| --- | --- |
| Focused regressions |42 tests passed;0 failures/errors |
| Multi-room graph scenarios |10/10 expected outcomes |
| Largest graph scenario |1,000 rooms;999 graph-edge traversals; simulated target found |
| Graph planning on this Jetson,1,000-room scenario |Median0.042ms across all scheduler actions; maximum75.75ms, including route planning |
|1,000-room checkpoint size in measured run |977,277 bytes before later recognition-ticket metadata fields |
| Random directed graphs |200/200 matched an independent Bellman–Ford distance oracle |
| Duplicate simulated coverage |0 across the10 scenarios |
| Traversal of invalid graph edges |0 across the10 scenarios |
| Rendered controller experiments |8 cases;0 simulated clearance violations; motors zero at every finish |
| Turn-preview hull computation |3.26–3.53s before;49–62ms after;57–68× faster |
| Preview geometry equivalence |Maximum hull-vertex difference0cm for−30°,30°,180° |

Median graph latency includes cheap observation-selection actions; it is not
the median route-search time. These numbers do not measure camera processing,
API latency, physical search duration, or real SLAM performance.

Graph scenarios: two rooms;100 rooms;1,000 rooms; closed-door detour; lost
localization with simulated reacquisition; checkpoint/resume; absent target;
100 incrementally discovered rooms; blocked frontier; combined door/localization/
restart faults. The absent-target case returned `known_map_searched`, not a
claim of absence from all unknown space. The blocked frontier returned
`map_required`, rather than driving through unknown space or repeatedly asking
the same question.

## Rendered controller results

These exercise the production object controller and tracker against synthetic
carpet, target, IMU, and wheel dynamics. Target standoff was20cm.

| Case | Outcome | Simulated final target range | Clearance violations |
| --- | --- | --- | --- |
| Baseline |Reached |19.76cm |0 |
| Brief occlusion |Reached after one local recovery |18.74cm |0 |
| Cancellation |Stopped |75.62cm |0 |
| Stale camera |Stopped |75.20cm |0 |
| Faster drive |Reached |17.45cm |0 |
| Steering drift |Reached |18.14cm |0 |
| Longer coasting |Reached |17.96cm |0 |
| Insufficient texture |Stopped before motion |80.00cm |0 |

The faster/coasting cases show a2–3cm standoff error, even while staying clear
in these scenes. These are hypothesized dynamics, not a newly calibrated robot.

## What still prevents general multi-room deployment

**Reliable mapping and relocalization are not implemented yet.** The graph tests
receive simulated places and corridor evidence; simulated localization recovery
does not demonstrate camera/IMU relocalization. The current live search harness
still uses an inspected local rectangle and cannot autonomously build a whole
building map or cross unknown doorways.

There is also no general dynamic-obstacle detector. Sol's object boxes do not
establish free space, and sparse visual landmarks would not establish a safe
occupancy map by themselves. The successful1,000-room graph experiment is a
scalability check on scheduling, not proof of1,000-room robot autonomy.

Next hardware-independent milestone: add a mapping-backend adapter and evaluate
it on a recorded two-room camera/IMU traversal with return/loop closure and
tracking-loss recovery. Measure frame/IMU synchronization, full extrinsics,
metric drift, relocalization correctness and processing budget. Keep motor
operation disabled until those outputs can supply the planner's clearance and
localization contracts. Architecture detail is in
`reports/search-architecture-2026-09-15/README.md`.

## Artifacts and reproduction

- `search-graph.json`:10 scenario results and event prefixes.
- `random-graphs.json`:200 seeds/results and independent-oracle verification.
- `controller-summary.json`, `stress-summary.json`: rendered experiment summaries.
- `controller-*.json`, `stress-*.json`: complete rendered controller evidence.
- `preview-performance.json`: paired old/new geometry and timings.
- `regression-summary.json`, `regression-tests.txt`:42-test result.

Run graph scenarios without camera, motors, or API access:

```
/usr/bin/python3 local_nav/simulate_search_graph.py --output /tmp/search-graph.json
```

Run the rendered production controller simulation:

```
/usr/bin/python3 local_nav/simulate_object_mission.py --output /tmp/object-mission.json
```

All changes remain local. No background job is required to read these results.
