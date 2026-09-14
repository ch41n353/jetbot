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

## Prepare an object approach — revision 6, 2026-09-14

After `capture`, select the object's **carpet-contact pixel** in that exact image.
Send `plan_approach` with `image_path`, `target_base_pixel: [u, v]`,
`inspected_free_rectangle_cm: [left, back, right, front]`, and
`obstacle_rectangles_cm: [[left, back, right, front], ...]`.
`standoff_cm` defaults to 15 and accepts 15–25 cm. Explicitly inspect the map;
the command does not discover obstacles or certify side/rear clearance.

The command projects the point, reserves a 3 cm projection allowance, caps the
next segment at 15 cm, runs the existing swept-corridor check, and writes a plan
and preview together. Show the returned preview before `execute`. Existing
session/generation, image age/registration, two-second and watchdog limits apply.
It makes no motion request itself. Off-axis targets (>6 cm lateral) return
`alignment_required`; nearby targets return `within_standoff_band_estimate`
without a plan. This is a camera-derived standoff assessment, not verified arrival.
It does not track a target across frames or autonomously chain approaches.

This replaces separate projection calculations, handwritten plan JSON, and
preview generation. It does not yet remove high-level inspection between segments.

### One planner call for a whole straight approach

Set `batch: true` on `plan_approach` to prepare up to 30 cm of inspected travel.
This reserves 5 cm of projection allowance beyond the selected standoff and
checks a wider whole-route corridor: x ±20 cm, rear −28 cm, front goal +15 cm.
Pass `batch: true` on the subsequent `execute` command as well. Batch execution
always enables post-stop verification regardless of `predictive_braking`.

The batch uses `approach_batch.py`, with at most three bounded local segments
and a 15-second supervisory deadline. Each segment retains the 15 cm / two-second
powered cap and all existing tracking, lease, progress and tilt guards. It stops
and observes locally between segments; there is no VLM call at those boundaries.
It transforms each prospective full swept rectangle into the original map and
rejects conflicts, excessive accumulated uncertainty, lateral drift or heading
change. An external stop between segments cancels the batch rather than being
adopted as a new authorization token. An unverified/failed segment ends the batch.

A successful result is `approach_distance_reached_estimate`. A failure returns
`requires_planner: true` with a reason. These are tool results; this code does not
invoke a model API. There is no target reacquisition, automatic turning, dynamic
obstacle segmentation, or autonomous rerouting. Use it only for an inspected
static corridor. A complete approach may still require an earlier alignment turn.

Example command sequence (replace image path and all geometry with inspected
values; these numbers are illustrative):

```json
{"command":"capture"}
{"command":"plan_approach","image_path":"/exact/returned/image.jpg","target_base_pixel":[345,238],"inspected_free_rectangle_cm":[-22,-30,22,60],"obstacle_rectangles_cm":[[-4,45,10,55]],"batch":true}
```

Show the returned whole-route preview, then:

```json
{"command":"execute","plan":"/exact/returned/plan.json","batch":true,"feature_budget":125}
```

Use `stop` to cancel the whole worker, or `shutdown` to close the session.
