# JetBot search and navigation stack

**Code snapshot: 2026-09-15, America/Los_Angeles.** This document describes the
repository implementation, including configurable scan increments introduced
after skill revision 20. It is an experimental, locally supervised system.
Successful simulation does not establish reliable autonomous room coverage.

## Ownership and data flow

```mermaid
flowchart TD
    U[Operator / Astra<br/>target, authorization, inspected map] --> S[maneuver_session.py<br/>persistent supervisor]
    S --> Q[sol_search.py<br/>deterministic coverage policy]
    Q --> V[GPT-5.6 Sol API<br/>stationary image recognition]
    V -->|target absent| Q
    Q --> T[spatial_turn.py<br/>measured scan turn]
    Q --> P[spatial_planner.py<br/>bounded SE(2) A*]
    P --> E[spatial_executor.py<br/>measured relocation]
    V -->|grounded target| O[object_mission.py<br/>local tracking and approach]
    O -. optional advisory images .-> A[AsyncScene worker]
    A -. Sol shadow replies .-> O

    T --> C[service.py control socket]
    E --> C
    O --> C
    C --> L[expiring motor lease]
    L --> M[independent motor worker]
    CAM[camera worker] --> C
    IMU[IMU worker] --> C
    BAT[input and pack monitors] --> C
    BAT -->|independent motion gate| M
```

Only one supervised execution child runs at a time. Search invokes controllers
sequentially; Sol receives images and returns scene JSON, never motor access.
The object controller owns its motor renewals while an optional background
perception worker performs HTTP. Parallel development agents must not compete
for hardware or run heavy simulation during powered experiments.

Per [AGENTS.md](../AGENTS.md), a new user request to navigate/search supplies
roughly 35 cm of initial surrounding clearance. Use this for the initial scan
without asking again. Fresh observations that contradict it take precedence.
It does **not** supply free space along subsequent travel routes or measured room
bounds. Neither the model nor the search process automatically constructs a map.

## Coordinates, geometry and plans

Map coordinates are centimetres, `x` right and `z` forward at the original camera
anchor. Pose is `[x, z, yaw_degrees]`; positive yaw turns right. The camera is at
the front centre of a 12 × 15 cm chassis, 9.5 cm above the carpet. At zero pose,
the body occupies `x=[-6,6]`, `z=[-15,0]`. Rectangles use
`[xmin,zmin,xmax,zmax]`. Obstacle contact/touching is rejected.

The map is one explicitly inspected free rectangle with retained obstacle
rectangles. Actual footprint checks add 5 cm clearance and 2 cm position
allowance. Drive checks include braking; turn sweeps include unknown pivot
positions inside the chassis and angular overrun/uncertainty. Predictions using
a nominal pivot are search seeds, never measured motion.

### Search plan

`sol_search.py` consumes JSON with these fields. There is no `plan_search`
supervisor command; build the file from the latest `capture` event and inspected
geometry. Do not copy an old session, epoch or monotonic timestamp.

| Field | Meaning / default |
|---|---|
| `session_id`, `control_epoch` | Exact current service generation and cancellation epoch. |
| `image_path` | Absolute path to the fresh 640 × 480 anchor JPEG. |
| `captured_monotonic` | Camera acquisition timestamp; initial anchor age must be 0–120 seconds. |
| `target_label` | Requested object description. |
| `preview_path` | Absolute writable HTML destination, updated before each action. |
| `inspected_free_rectangle_cm` | Original-anchor free rectangle, not merely current camera field of view. |
| `obstacle_rectangles_cm` | Retained obstacles in that same coordinate frame; use an empty array only when justified. |
| `room_bounds_cm` | Optional measured rectangular room used to generate perimeter stations; does not expand free space. |
| `search_stations_cm` | Optional explicit ordered `[[x,z], ...]` stations; overrides generated perimeter stations, including when empty. |
| `scan_step_degrees` | Default `30`; allowed 1–30 degrees. Requested increment, not guaranteed measured rotation. |
| `scan_direction` | Integer `1` for right, `-1` for left; default `1`. Changes scan direction, not perimeter order. |
| `max_recognition_calls` | Integer 1–72, default `20`; includes initial and subsequent stationary views. |
| `recognition_every_actions` | Local actions between Sol calls, 1–12; default `1`. Host policy and safety checks continue on skipped-recognition actions. |
| `max_search_actions` | Bounded local action ceiling, 1–144; default `72`. |
| `scan_turn_controller` | `continuous` by default; `pulse` retains the older camera-heavy scan turn. |
| `checkpoint_path`, `resume_checkpoint_path` | Atomic mission state output and explicitly requested eligible action-complete resume input. |
| `search_only` | Boolean, default false. Report `object_found` without approach when true. |
| `search_turn_power` | Default `0.14`; validated range 0.12–0.16. Do not treat it as measured torque. |
| `standoff_cm` | Search approach default `25`; downstream object mission permits 18–30. |
| `target_radius_cm` | Search approach default `7`; downstream permits 5–15. |
| `goal_tolerance_cm` | Inherited planner option, default `3`, range 1–5. |
| `floor_alignment_rotation`, `floor_alignment_evidence` | Optional evidenced local camera-frame floor correction. Its heading scope is limited to ±10 degrees from the search origin. |

A validation-only invocation instantiates search state and checks initial
candidates without motors or API calls. It is **not exhaustive schema validation**:
`max_recognition_calls` is checked in `execute`, and some downstream geometry,
image, power and API requirements are checked only when used. Supply integer
fields as integers; current parsing uses numeric conversion, not a standalone
strict JSON Schema validator.

### Other execution plans

- Spatial navigation uses the same anchor and map, plus `goals_cm` (1–4 floor
  goals), optionally `initial_pose_cm_degrees`, accumulated position/yaw
  variances, `measured_map_drive`, `precise_turn_sweep`, `coalesce_drives`, and
  `goal_heading_turns`. Use `plan_navigation` to generate a previewed plan.
- Object missions use `target_box=[left,top,right,bottom]`, `target_label`, map,
  standoff/radius and optional `projective_contact`, `sol_shadow`, and floor
  alignment. `plan_mission` reconstructs the capture's IMU attitude from its
  saved metadata and freezes `capture_attitude` and `target_ground_cm`.
- Default straight-route plans use increasing `waypoints_cm` and optional
  `travel_direction="forward"|"reverse"`. The supervisor flags `search`,
  `mission`, `spatial`, and `batch` are mutually exclusive.

## Deterministic search policy

1. Register the fresh anchor against the live camera; reject movement exceeding
   0.3 cm or 1 degree since inspection. Start pose is `[0,0,0]`.
2. At a stopped, healthy view, record coverage and call Sol for recognition.
3. If the target is confidently grounded, either report `object_found` in
   search-only mode or prepare, preview and run the local approach.
4. Otherwise turn by the configured signed increment, subject to the complete
   checked turn sweep. Accumulate **measured absolute rotation**, not requested
   angle. Continue until accumulated rotation is at least 358.5 degrees.
5. After a full station scan, select a reachable unvisited observation station.
   Initially choose least path cost; thereafter follow the station list's cyclic
   order, skipping unreachable stations. Relocate with local navigation and
   begin a fresh scan there.
6. Stop with a specific blocked, localization or budget result if continuation
   cannot be established. This does not prove that the object is absent.

Generated stations are 35 cm inside supplied room bounds and at most 40 cm apart,
in clockwise perimeter order. Each side is limited to 100 subdivisions. A room
must be wider and longer than twice the inset. Routes and stations still need
free-space/obstacle checks. With no room bounds or explicit stations, search can
scan only its starting location.

Coverage headings are deduplicated within 10 degrees; fine 5-degree scans still
make recognition calls at each step, although not every view adds a distinct
coverage heading. The internal candidate ID remains `scan_right` even for a
negative configured turn; use its signed `degrees` field as authoritative.

The recognition budget includes the initial view. A nominal 30-degree full scan
needs about 13 views to recognize the final orientation; smaller increments may
exhaust the call budget before a full scan. In particular, 5-degree turns need
72 turn actions plus the initial view, exceeding 72 calls for a final recognition
at exactly 360 degrees. Measured turn tolerance changes these counts. Reducing
step size does not make the overall scan faster or guarantee completion.

Search checks 300 seconds and 120 cm accumulated measured travel at loop
boundaries. Turn-induced camera translation counts as travel. These are not
hard deadlines preempting an in-flight API call or downstream controller.
Accumulated pose variance above 4 cm² or 16 deg² blocks further motion. One
stopped recognition is still allowed: search-only can report a found object with
`found_pose_valid=false`; otherwise the result is `localization_required`.
A camera image does not reset global odometry uncertainty.

## Sol recognition and asynchronous shadow mode

The active request implementation is [async_scene.py](../local_nav/async_scene.py):
`gpt-5.6-sol`, reasoning `none`, Responses API, `store=false`, high-detail JPEG,
strict JSON output, 4,500 output-token ceiling, and a 12-second HTTP timeout.
The worker inherits `OPENAI_API_KEY`; keep credentials out of plans and logs.
Prompts are `local_nav/prompts/sol_scene.txt` and, for search,
`local_nav/prompts/sol_search.txt`.

Response fields are `target_visible`, nullable `target_box` with
`x0,y0,x1,y1`, nullable `contact_pixel` with `x,y`, obstacle labels/boxes,
`spatial_answers`, and `approach_side`. Search validates the target box within
640 × 480, contact inside the box and within 8 pixels of its bottom. Absent or
uncertain targets continue deterministic search; model-authored motion choices
cannot replace the policy. Returned obstacle boxes are **not** automatically
converted into trusted map geometry or new free space.

Current search validates Sol's contact pixel but passes the target box and
`projective_contact=true` to mission preparation; it does not pass that explicit
contact pixel as a mission anchor. The object's initial floor projection uses
box-bottom centre, followed by local projective contact tracking. This distinction
matters for partly occluded objects and inaccurate boxes.

Search schedules one bounded `AsyncScene` request when recognition is due. Local
geometry and candidate calculation can proceed while the request is in flight,
but semantic output is consumed only at a stopped decision boundary. Recognition
cadence can skip configured intermediate views without consulting Astra.

Separately, an object mission with `sol_shadow=true` uses `AsyncScene`: one
in-flight request, no image backlog, at least 4 seconds between starts, at most
30 requests. Frame, pose, attitude and generation are frozen at capture. Responses
older than 8 seconds, with changed generation, more than 60 cm displacement or
15 degrees heading change are rejected. Accepted observations are rebased into
the current frame for **advisory logging only**: no route, target or obstacle-map
mutation. Cleanup does not wait for HTTP; a sent request may finish and incur
cost, but cannot command motors.

## Local planning and execution

`spatial_planner.py` is a bounded weighted A* / best-first SE(2) search. Visited
keys quantize position to 5 cm and heading to 2 degrees. Actions are forward
15/10/5 cm, reverse 10/5 cm, and ±30-degree turns, with optional goal-bearing and
small heading corrections. Default limits are 10,000 nodes and one second.
The heuristic weight is 2.5: the planner seeks feasible routes quickly and does
not claim optimal length or travel time. Every primitive checks a chassis sweep.

`spatial_executor.py` executes measured actions and replans when needed. Limits:
60 seconds checked locally, 12 primitive budget, 90 cm travel, accumulated
2 cm position / 4 degree yaw uncertainty. With measured-map driving and
`coalesce_drives=true`, adjacent same-direction primitives may become one
continuous drive up to 30 cm / four powered seconds. Defaults retain 15 cm /
two seconds. Turns, reversals and final arrivals still require stops.

`route_executor.py` uses visual/IMU feedback and yaw-rate-damped steering. Lateral
correction requests at most ±1.5 degrees desired heading; differential duty stays
within ±0.025 and the 5-degree route heading guard remains. Predictive braking
is followed by a measured stationary window; crossing distance alone is not a
verified arrival. `spatial_turn.py` currently uses bounded powered pulses and
rests, records measured translation as well as yaw, and checks the world sweep.

`object_mission.py` tracks an immutable target identity locally, follows checked
waypoints without scheduled 15/30 cm stops, and has bounded target/floor recovery.
Its 180-second and 120-cm ceilings do not demonstrate that duration of reliable
autonomy. Target recovery stops motors, requires consecutive valid observations,
and retains map consistency checks. Floor recovery retains the last accepted
pose, collects motor-off observations, verifies complete IMU coast history,
quiet frames, unchanged floor-fit checks and a conservative coast envelope,
then adds the bridge and uncertainty without resetting odometry. Missing history,
persistent scale change or insufficient clearance fails closed. Arrival uses
stable post-stop observations and standoff tolerance, not model recognition.

Autonomous scan turns use `continuous_turn.py` by default. It keeps one expiring
differential command renewed from fresh IMU/status samples, predicts braking from
measured yaw rate, and verifies a quiet zero-motor finish. The search harness
still checks the full retained-map sweep before motion and uses stopped VIO plus
camera/IMU yaw agreement before accepting the new pose.

## Browser dashboard

`local_nav.dashboard.server` is a loopback-only console at
`http://127.0.0.1:8765`. It owns one supervisor child and exposes target entry,
stop, stopped-only reset, RGB, the current trajectory preview, live Sol/policy/
controller events, telemetry, result, and checkpoint. It incrementally tails the
worker event journal, so the timeline updates before route completion. The
browser has no API credential or motor-socket access. See
`local_nav/dashboard/README.md` for motor-free and explicitly armed launch
commands.

## Estimation, previews and safety

The estimator combines tracked carpet motion with calibrated camera intrinsics,
height, IMU mounting and gyro-relative yaw. It does not infer translation by
unbounded accelerometer integration or substitute uncalibrated magnetic heading.
Quiet startup can account for a carpet slope within its calibrated-offset limit;
the runtime relative tilt guard remains 5 degrees. Camera/IMU disagreement,
frame/history gaps, excessive rotation, bad feature geometry and uncertainty
can all terminate motion. A local plane correction requires evidence and cannot
be carried through arbitrary search rotations.

Search writes `trajectory_preview` before every turn, relocation and approach.
`turn_sweep_preview.py` and `spatial_preview.py` draw the chassis corridor over
the original camera image and provide geometric context. A preview is a plan,
not an obstacle detector or proof that unknown floor is clear. The same HTML
path is overwritten; display the latest event's preview before a new high-level
route. Revision 20 accelerated turn-preview hull calculation without changing
its sampled geometry. A search relocation preview can fail conservatively when
an inner local free rectangle cannot represent a world-map route.

`service.py` owns independent camera, IMU, battery and motor processes. Motor
commands are expiring leases, typically 200 ms; the motor worker checks about
every 10 ms. Controllers normally require updates within 180 ms. Power monitoring
runs roughly five times per second and independently gates motion:

| Supply | Warning | Latched stop |
|---|---:|---:|
| Regulated input | Below 4.9 V | Below 4.8 V or above 5.25 V |
| Inferred three-cell pack profile | Below 11.4 V | Below 10.8 V or above 12.9 V |
| Telemetry | — | Missing or older than 0.6 seconds |

Voltage is not battery percentage. Resolve a latched power fault and start a fresh
service; recovery of voltage alone must not restart motion.

`session_id` and `control_epoch` reject stale commands. Search rechecks generation
after inference. It advances expected epochs only for its own known action stops;
it must not adopt an arbitrary newer service epoch after cancellation. Supervisor
`stop` revokes motion and terminates the child; `shutdown` closes the service too.
Inspect `stop_error` even if another result field appears successful. Software
leases do not constitute a hardware-guaranteed cutoff against a frozen motor
driver or system.

## Invocation example

These are operator instructions, not commands executed while writing this file.
Use the existing authorization and fresh sensor/power checks. One persistent
terminal can host the supervisor; do not start a second motor owner.

```bash
# OPENAI_API_KEY must already be set securely in this process's environment.
mkdir -p /tmp/jetbot-previews
/usr/bin/python3 /home/jetbot/jetbot/local_nav/maneuver_session.py \
  --enable-motion --preview-directory /tmp/jetbot-previews
```

Send one JSON object per line on its stdin:

```json
{"command":"status"}
{"command":"capture"}
```

Copy the returned `image_path`, `captured_monotonic`, `session_id` and
`control_epoch` into a plan. This valid JSON illustrates field shape only;
replace its anchor values and derive its map from actual evidence. Its map allows
only the supplied starting area and has no perimeter travel stations.

```json
{
  "session_id": "REPLACE_WITH_CAPTURE_SESSION",
  "control_epoch": 0,
  "image_path": "/absolute/fresh-capture.jpg",
  "captured_monotonic": 0.0,
  "target_label": "Advil bottle",
  "preview_path": "/tmp/jetbot-previews/search-current.html",
  "inspected_free_rectangle_cm": [-35, -35, 35, 35],
  "obstacle_rectangles_cm": [],
  "search_stations_cm": [],
  "scan_step_degrees": 30,
  "scan_direction": 1,
  "max_recognition_calls": 20,
  "search_only": true,
  "standoff_cm": 25,
  "target_radius_cm": 7
}
```

Check the plan without accessing motors or the API:

```bash
/usr/bin/python3 /home/jetbot/jetbot/local_nav/sol_search.py /absolute/search-plan.json
```

Then dispatch through the running supervisor:

```json
{"command":"execute","plan":"/absolute/search-plan.json","search":true}
{"command":"status"}
{"command":"observe"}
```

Use `observe` during an active worker; `capture` is reserved for stopped planning
and rejects while a worker is active. To approach after recognition, set
`search_only=false` **and supply enough inspected free space for the approach**.
Changing that boolean cannot turn the initial-clearance assumption into a route
map. To stop or finish:

```json
{"command":"stop"}
{"command":"shutdown"}
```

Direct `sol_search.py PLAN --execute --log LOG` exists, but requires an already
running guarded service and lacks the supervisor's child lifecycle management.

## Results, logs and diagnosis

Supervisor outputs include `captured`, `route_started`, completion, status and
power events. Session files are under `local_nav/goals` by default, with service
logs and per-route stdout logs. Search child stdout is redirected there; it is
not all streamed directly through the supervisor terminal.

For search log `LOG`:

- `LOG`: final result, event history, `sol_calls`, `requires_astra`, elapsed time
  and searched headings.
- `LOG.events.jsonl`: `sol_called`, `search_decision`, `trajectory_preview`,
  `local_action_complete`, `motion_paused_for_localization`, `object_found`,
  `localization_required` or `search_stopped` events.
- `LOG.step-NN.jpg` and `.capture.json`: exact recognition image and sensor,
  generation and power metadata.
- `.turn.json`, `.navigation.json` or `.approach.json`: delegated action results,
  with nested observation, event, status and settled-image artifacts where
  supported.

Search outcomes include `object_found`, `object_reached_estimate`,
`localization_required`, `search_blocked`, `search_budget_exhausted`, `stopped`,
and a propagated approach outcome. `requires_astra=true` is a handoff indicator;
it does **not** itself invoke a remote model. Search-only `object_found` means
recognition, not physical arrival. Preserve failed trials and actual termination
reasons. Inspect actual image, rejected pose, power state and cancellation epoch
before attributing a failure to the recognizer.

## Validation and boundaries

Relevant regression sources include `tests/test_sol_search.py`,
`test_maneuver_session.py`, `test_async_scene.py`, `test_search_graph.py`,
`test_object_mission.py`, `test_mission_floor_recovery.py`, `test_route_executor.py`,
`test_route_steering.py`, `test_mapped_drive.py`, and `test_battery.py`. Search tests
cover absent→scan→visible→approach, model-motion rejection, cancellation during
inference, perimeter ordering, configurable scan step/direction, and recognition
after uncertainty blocks motion. Read the current test code before assuming it
covers every new option; the recognition-call ceiling is validated in execution.

Skill revisions [17](../skills/jetbot-navigation/references/revision-2026-09-15-r17.md),
[18](../skills/jetbot-navigation/references/revision-2026-09-15-r18.md),
[19](../skills/jetbot-navigation/references/revision-2026-09-15-r19.md), and
[20](../skills/jetbot-navigation/references/revision-2026-09-15-r20.md) distinguish
API replay, mocked search, rendered controllers, recorded robot evidence and
actual live runs. Their test counts are dated results, not a claim about this
working tree. Repository skill revisions may be newer than the installed skill.

`search_graph.py` and `simulate_search_graph.py` provide a separate motor-free
persistent graph of places, doorways, frontiers and coverage, with versioned
route evidence, checkpoints and relocalization requests. Revision 20's 1,000-room
simulation and independent shortest-path comparisons validate that abstraction,
not live mapping, doorway clearance or multi-room robot autonomy. The live
`sol_search.py` harness still uses its static rectangle and local pose.

Remaining limits include synchronous search recognition latency, accumulated
odometry uncertainty, unverified far/off-axis floor projection, target-contact
and floor-plane errors, limited initial clearance, conservative map transforms,
no general moving-obstacle detection, and no automatic room-map expansion.
Never turn absent recognition or a blocked route into a declaration that a room
has been exhaustively searched.
