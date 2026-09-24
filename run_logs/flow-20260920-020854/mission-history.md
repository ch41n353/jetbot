# FLOW search and approach — SD storage authorized for this run

## Command issued 2026-09-20T09:09:22.996377+00:00

Fresh scan from doorway orientation; initial clear footprint authorized. Recording verified on SD for this run.

POST http://127.0.0.1:8770/api/turn

```json
{
  "degrees": -90
}
```


## Command issued 2026-09-20T09:09:52.505523+00:00

FLOW viewpoint approach to reveal floor hidden by the bin; target Advil not yet visible. One-second request interval with overlapping requests.

POST http://127.0.0.1:8770/api/flow

```json
{
  "target": "Approach the visible green nut-mix container just right of the pink bin. Pass to the right of the cyan block ahead and left of the yellow blocks, leaving room for the whole robot. Avoid the bin and all blocks. The container is the only destination. Stop short.",
  "every": 1
}
```


## 2026-09-20T02:10:09.266507-07:00 — FLOW observation

flowing to Approach the visible green nut-mix container just right of the pink bin. Pass to the right of the cyan block ahead and left of the yellow blocks, leaving room for the whole robot. Avoid the bin and all blocks. The container is the only destination. Stop short.: a look every 1.0 s, wheels never stop
look 1 seen, unplaceable | rebased 0 cm | 3 in flight
look 3 target 93 cm | rebased 13 cm | 2 in flight, 1 stale dropped
look 4 seen, unplaceable | rebased 25 cm | 2 in flight, 1 stale dropped
look 5 seen, unplaceable | rebased 26 cm | 2 in flight, 1 stale dropped
  blocked by "cyan block"
  blocked by "cyan block"
  blocked by "cyan block"
look 7 seen, unplaceable | rebased 14 cm | 1 in flight, 2 stale dropped

## 2026-09-20T02:10:15.360633-07:00 — FLOW observation


look 8 target 74 cm | rebased 12 cm | 1 in flight, 2 stale dropped

## 2026-09-20T02:10:23.501389-07:00 — FLOW observation


PLANNER HOLD: Target not identifiable; visible object right of bin lacks a readable distinguishing label.

ended after 30 s: 10 looks issued, 8 applied, 2 stale dropped

## 2026-09-20T02:13:12.121266-07:00 — Stationary debugging result

Confirmed and fixed duplicate obstacle-memory transform. Original-code replay fails; fixed replay and 103 existing tests pass. No powered validation. Full report: debug-report.md.
