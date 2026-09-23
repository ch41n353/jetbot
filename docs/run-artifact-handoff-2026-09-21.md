# JetBot run artifact handoff — 2026-09-21

This document is an evidence map for another AI describing how the high-level search, GPT mid-level planner, local planner/controller, recording, audit logs, and visualizer worked during the Advil and toy-car tasks. Times in audited JSON are UTC. Run directory names and image file modification times use America/Los_Angeles (PDT, UTC-7).

## Outcomes

- **Advil:** found behind the pink bin, approached, then aligned in place so the label faced the camera. The final distance was not independently measured.
- **Toy car:** remembered from the earlier room scan as being beside the computer desk and wall cables; later reacquired in a fresh image. The robot approached it across open carpet, but the mid-level planner held at a floor cable. The final follow-up approach was interrupted before dispatch, so the car was **found but not fully approached**.
- At handoff, the toy-car mission was not running, its recording was finalized, and no pending follow-up motion command had been dispatched.

## Primary run roots

| Task | Run root | Recording | Duration | Size |
|---|---|---|---:|---:|
| Advil | `/mnt/robotlogs/goals/advil-20260921-013247/` | `/mnt/robotlogs/recordings/advil-20260921-013247.mkv` | 342.96 s | 76,080,912 bytes |
| Toy car | `/mnt/robotlogs/goals/toy-car-20260921-013926/` | `/mnt/robotlogs/recordings/toy-car-20260921-013926.mkv` | 706.12 s | 166,226,810 bytes |

Browser replay URLs:

- `http://192.168.86.158:8771/play/advil-20260921-013247.mkv`
- `http://192.168.86.158:8771/play/toy-car-20260921-013926.mkv`

The control visualizer at `http://192.168.86.158:8772/` follows `/mnt/robotlogs/current-search.json`; it currently points at the toy-car run.

## Key evidence and chronology

### Advil run

1. **Initial view, 01:32:53 PDT:** `initial.jpg`. Advil was not visible.
2. **Four-direction scan, 01:33:36 PDT:** `scan.jpg` and the complete structured result `scan.json`. The scan did not reveal Advil.
3. **Viewpoint plan, starting 08:34:22 UTC:** the assistant asked GPT DRIVE to approach the green nut-mix can beside the pink bin so the hidden floor behind the bin would become visible. The exact instruction is in `commands.jsonl` and `mission-history.md`. Seven GPT calls for this handoff occupy these request directories:
   - `gpt/795b1bc79f604e819cd77594d9e4bce9/`
   - `gpt/0b8fb7da286d46c58b4edffe7a642f64/`
   - `gpt/2d2fafc1b61e4e1d98c1e7c0e625cee2/`
   - `gpt/5b4b8cbc61244bb8b44dd72c8f840244/`
   - `gpt/fb3561eea1584cd59e486c534b6a7e56/`
   - `gpt/af570c41befd4842bed632a535417387/`
   - `gpt/e75020f6da0547a08123207fd69fb508/`
4. **Nut-can viewpoint:** `viewpoint-final.jpg` shows the reached viewpoint.
5. **Advil found, 01:36:08 PDT / identification handoff at 08:36:48 UTC:** `behind-bin.jpg` is the clearest evidence of discovery: the Advil bottle is visible beyond the pink bin. The exact assistant rationale says, “Fresh left view positively identifies Advil behind the pink bin.”
6. **Advil approach GPT calls:**
   - `gpt/e6882076b2c546c5b962d45c8926e6bd/`
   - `gpt/fefec4d87f364ba29de0515b630e0acc/`
   - `gpt/1d63c76a2e4a4f94b8c3360d0ee3cd09/`
7. **Approach result:** `advil-final.jpg` shows Advil close at the left edge after the planner held because the close bin blocked the prior route.
8. **Final alignment, command at 08:37:49 UTC:** `align.json` records a requested -45° turn and measured -44.7°. `aligned.jpg` shows Advil centered and facing the camera.
9. **Run completion note, 08:38:32 UTC:** `mission-history.md`. It explicitly says the exact final distance was not independently measured.

### Toy-car run

1. **Remembered location, run creation at 01:39:26 PDT:** the first lines of `mission-history.md` record the search lead: “earlier scan showed car beside computer desk and wall cables.” This was a prior-memory search lead, not proof that the car was currently visible.
2. **Initial state:** `initial.jpg` shows the robot starting near Advil.
3. **Occlusion investigation:**
   - `search.jpg`: the pink bin fills the view.
   - `search2.jpg`: the nut-mix can and toy ramp are visible; the car is still occluded.
   - `around-bin.jpg`, `room.jpg`, and `corridor.jpg`: views used to test paths around the bin.
4. **Rejected narrow corridor, 08:43:02 UTC:** GPT held rather than use the power-adapter landmark because the bin and protruding toys did not leave supported full-robot clearance. The exact GPT call is `gpt/942d76b3181c414ea99a6b5e7427d7a3/`.
5. **Wide detour via the doorway package, starting 08:43:37 UTC:** six GPT calls are stored in:
   - `gpt/863db0fd25b84705aa423bcbaefaf2a1/`
   - `gpt/a9b6cd19277a4da7a60e51b5d398e2b9/`
   - `gpt/b2ce282dbfc544e39af6baeb4de3bb53/`
   - `gpt/bba79195ff50453aaea0cf912f412728/`
   - `gpt/d1e6eb236966489ca973494e1ce6272e/`
   - `gpt/b7025c39ad184a099057bbc3737a92b3/`
   `wide-final.jpg` shows the reached doorway-package viewpoint.
6. **Toy car reacquired, 01:45:23 PDT:** `car-view.jpg` clearly shows the small wheeled model beside the computer desk, between the wall power adapters and desk leg. The exact approach instruction and rationale were issued at 08:45:54 UTC in `commands.jsonl` and `mission-history.md`.
7. **Toy-car approach GPT calls:**
   - `gpt/f38669aa73de4738a6a4d6fe37b66228/`
   - `gpt/99a8d36f76f442a9b0f56eaa7f70ae12/`
   - `gpt/7475dc2fefad4eadbce722c6f345d1e8/`
   - `gpt/62908862204640f7b7c3de34b1363ca8/`
   - `gpt/dd2bd285eb2b41f783fae85f7cb39f2c/`
8. **Approach stopped at cable:** `approach.jpg` shows the car ahead-right and the floor cable between/left of the approach. The planner log ends with `PLANNER HOLD: Floor cable blocks the visible approach; hold rather than cross it.`
9. A proposed front-right follow-up around the visible free cable end was interrupted before the audited command was dispatched. There are no request/response artifacts for that proposed follow-up. Do not describe the toy-car task as completed.

## Per-run artifact schema

Each run root contains:

- `commands.jsonl`: paired assistant command issuance and HTTP response records, including exact JSON bodies and concise observable rationales.
- `api-commands.jsonl`: commands received at the planner API boundary.
- `mission-history.md`: append-only chronological narrative with command bodies and links to GPT request directories.
- `mission.md`: newest-first rendered operator log.
- `events.jsonl`: sampled planner/controller states with observation timestamps. These are observation times, not guaranteed exact motor-command times.
- Named JPEGs such as `behind-bin.jpg`, `aligned.jpg`, `car-view.jpg`, and `approach.jpg`: high-value search/verification views.
- `gpt/<request-id>/request.json`: exact OpenAI Responses API request, including the encoded image payload.
- `gpt/<request-id>/image-0.jpg`: the exact image supplied to GPT, possibly including the previous-plan annotation.
- `gpt/<request-id>/clean.jpg`: the corresponding clean camera image used by the visualizer to avoid overlapping old and new trajectories.
- `gpt/<request-id>/response.json`: complete raw OpenAI API response before planner normalization.
- `gpt/<request-id>/events.jsonl`: dispatch, response, and—when the plan was applied—`controller_route` events with projected route/obstacle data.

Completeness checks performed for this handoff:

- Advil: 10 GPT request directories; all 10 have request, response, and exact input image files.
- Toy car: 15 GPT request directories; all 15 have request, response, and exact input image files.
- Advil: 12 `commands.jsonl` rows and 6 planner-boundary API rows.
- Toy car: 20 `commands.jsonl` rows and 10 planner-boundary API rows.
- Both MKV files were finalized and successfully read by `ffprobe`.

## Implementation map

The main source files are:

- `/home/jetbot/jetbot/local_nav/planner_demo.py`: HTTP planner bench; recurrent GPT DRIVE orchestration; mission/turn/inspection endpoints; synchronized planner snapshots; camera and top-down rendering.
- `/home/jetbot/jetbot/local_nav/fetch.py`: GPT request construction, prompt invocation, camera-to-floor projection, obstacle adjustment, odometry/IMU route following, expiring motor commands, and safety guards.
- `/home/jetbot/jetbot/local_nav/prompts/trajectory-20260920.txt`: current mid-level planner instructions and response contract.
- `/home/jetbot/jetbot/local_nav/prompts/trajectory-20260920-baseline.txt`: preserved earlier prompt.
- `/home/jetbot/jetbot/local_nav/gpt_audit.py`: exact GPT request/image/response and planner-application event capture.
- `/home/jetbot/jetbot/local_nav/inspection_sequence.py`: bounded maneuver-and-capture sequences such as the four-direction scan.
- `/home/jetbot/jetbot/scripts/planner_audit_command.py`: records each assistant intervention before dispatch, sends it to the planner, and records the response.
- `/home/jetbot/jetbot/scripts/control_visualizer.py`: reads GPT calls, interventions, and controller state for the visualizer API.
- `/home/jetbot/jetbot/scripts/mission_log_server.py`: serves the control visualizer on port 8772 and follows the current run pointer.
- `/home/jetbot/jetbot/scripts/static/control_visualizer.html`: operator UI showing GPT calls, assistant interventions, and paired planner snapshots.
- `/home/jetbot/jetbot/scripts/camera_recording_server.py`: recording viewer on port 8771.
- `/home/jetbot/jetbot/scripts/render_mission_log.py`: creates newest-first Markdown from the append-only history.
- `/home/jetbot/jetbot/skills/jetbot-navigation/SKILL.md`: operating workflow, ownership boundaries, audit requirements, and known limitations.
- `/home/jetbot/.codex/skills/jetbot-navigation/references/operations.md`: installed operational interface and limitations used during these runs.

## How the stack worked in these runs

1. The high-level assistant inspected fresh images, selected search viewpoints, and issued a complete instruction for one visible physical destination.
2. `planner_audit_command.py` wrote that instruction and rationale to USB before sending it to `/api/mission`, `/api/turn`, `/api/inspection`, or `/api/halt`.
3. GPT DRIVE repeatedly captured a view, called the GPT mid-level planner, saved the full request/response artifacts, projected returned image-space waypoints onto the floor, adjusted the route for reported obstacles, and handed bounded motion slices to the local controller.
4. The local controller used camera motion and IMU heading feedback, retained collision/power/stall/generation guards, and stopped on planner `hold` rather than reviving an old route.
5. The high-level assistant re-entered only at viewpoint completion, explicit planner hold, target identification, or final alignment.
6. The visualizer read USB audit artifacts and showed paired camera/top-down planner snapshots from the same planning boundary. The recorder independently saved the camera stream to MKV on `/mnt/robotlogs`.

## Important limitations for any generated narrative

- Model-reported object range is an estimate, not independent ground truth.
- The Advil task required multiple high-level handoffs: scan, nut-can viewpoint, target identification, target approach, and final alignment.
- The toy-car task also required multiple viewpoint handoffs. It ended at a cable-related planner hold; it did not finish the last approach.
- A `mission ended` message is not equivalent to arrival. A `mission complete` claim still needs a fresh-image and stopped-motor check.
- `events.jsonl` status timestamps are poll/observation times unless the underlying event explicitly records dispatch or response time.
- The local collision corridor is narrower than the desired five-centimeter chassis margin documented in the task-control skill; do not describe obstacle avoidance as guaranteed.
