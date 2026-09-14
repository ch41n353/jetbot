# Persistent preview-and-maneuver session — 2026-09-14

Start one terminal session. Sensing stays warm while the planner inspects the
scene and prepares a route. The default is disarmed; physical movement requires
existing user authorization and `--enable-motion`.

```bash
/usr/bin/python3 /home/jetbot/jetbot/local_nav/maneuver_session.py \
  --enable-motion \
  --preview-directory /home/jetbot/.codex/visualizations/2026/09/10/01a088e8-29c5-7473-92d9-59b23900f569
```

Enter one JSON command per line:

```json
{"command":"capture","preview_distance_cm":15}
{"command":"status"}
{"command":"execute","plan":"/absolute/path/to/checked-plan.json","feature_budget":125,"predictive_braking":true}
{"command":"stop"}
{"command":"shutdown"}
```

Capture saves an immutable image/metadata pair and optionally produces a standard
floor-point and full-chassis preview. Its geometry is a **candidate**, not an
obstacle-clearance assertion. Inspect the fresh image and supply the ordinary
checked route plan before executing. The route schema and its 15 cm / two-second
powered limits are unchanged. Each preview gets a distinct path.

The supervisor handles stdin while a route subprocess runs, so `stop` can revoke
its motor lease and terminate the worker. It parses buffered byte lines explicitly
so a stop sent in the same input chunk as an execute request is not hidden in a
text-input buffer. Shutdown, EOF or Ctrl+C cleans up the owned service and route.
It refuses to attach to or replace an existing service. Do not run a second
session against the same hardware.

Timing records separate startup, command-to-result duration and the plan image's
age at dispatch. Captures and routes share one service/session identity; the
usual stale-plan and cancellation-token checks remain active.

First live trial: 15 cm requested, 16.377 cm settled estimate (`overshot_goal`),
1.380 seconds powered, 3.021 seconds from execute request to returned result.
There were no model calls or intentional intermediate stops during the route.
Initial capture-to-first-controller-frame delay was 72.619 seconds, including
one-time preview/code preparation; this is not a steady-state planning benchmark.
The automatic capture/preview helper subsequently took 0.050–0.053 seconds locally,
excluding interpreter startup, tool transport and model image inspection. The
next candidate was not executed because a person entered/approached the corridor.

The final supervisor's automatic-preview integration and stop behavior have
code checks; the initial session lifecycle and route launch were exercised live.
Do not conflate these partial measurements into an end-to-end speedup claim.
