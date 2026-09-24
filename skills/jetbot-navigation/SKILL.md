---
name: jetbot-navigation
description: Find and approach objects with the local JetBot using camera-guided search, downstream navigation planners, local motion controllers, USB recording, and live progress logs. Also use for short drives, turns, trajectory previews, and controller diagnosis.
metadata:
  baseline-date: "2026-09-13"
  timezone: America/Los_Angeles
  revision: "22"
---

# JetBot navigation

Revision 22 — 2026-09-22. Workspace: `/home/jetbot/jetbot`.
Read [the operating interface and limitations](references/operations.md) before operating. Use `/usr/bin/python3`.
This is the single skill for finding and navigating to objects. Use the workflow
below for ordinary requests; read [controller details and historical evidence](references/controller-guide.md)
and [manual commands](references/operation.md) only for manual primitives or
controller diagnosis. Current workflow instructions override historical defaults.
See [merge evidence](references/revision-2026-09-22-r22.md) for preserved sources.

## Health, geometry, and stop behavior

Use fresh camera and IMU data; never substitute an old frame after a failed capture.
Check `status.power`: input warning below 4.9 V, latched stop below 4.8 V or above
5.25 V; pack warning below 11.4 V, stop below 10.8 V or above 12.9 V. Missing/stale
telemetry (over 0.6 s) gates motion. These voltages are not battery percentages.
Resolve the cause before restarting an idle latched service; retain all guards.
Supply downstream planners the 12 × 15 cm chassis, 5 cm margin, front-center lens
9.5 cm above carpet, and current pivot calibration/uncertainty. Keep observed
obstacles in context even when they leave the frame; planners own clearance.
On stop/hold, cancel the controller and verify zero motors. Shut down the sensor/
motor service at run end to prevent renewed commands; leave the log viewer up.
A failed stop request is not confirmation. See the manual commands for shutdown.


## Required start of every navigation run

These steps apply in a new chat as well as an ongoing one. Do not rely on conversation history to supply them.

1. **Send the user the clickable visualizer link before any motion:** [Live log, camera and local plan](http://192.168.86.158:8772/). Include it in the first progress message, not only in the final answer. Also offer [all GPT inputs and outputs](http://192.168.86.158:8772/gallery). Verify the viewer is responding and points to this run; report an unavailable viewer honestly.
2. Create a unique USB run directory and set `/mnt/robotlogs/current-search.json`. Start and verify a growing recording before search motion, and verify GPT audit is active. Append commands and observations to `mission-history.md`; the viewer regenerates the newest-first view from it. Save the first inspected frame and its timestamp immediately for end-to-end timing.
3. Inspect a **fresh** camera image and sensor/power health. If the target is visible and positively identified, delegate its complete approach. If it is not visible but there is a useful remembered location, recent bearing, or visible clue, inspect that direction or reach a justified visible viewpoint first. Memory guides search; it does not establish current target visibility.
4. **Scan whenever you have no useful idea where to look for the target**, at any stage of the run—not automatically at the start and not every time it leaves the frame. Call the audited `inspection` endpoint with `{"preset":"four_way"}`, poll `/api/inspection`, and inspect the returned four images/contact sheet together. This captures the starting view and views after three 90-degree turns; it ends about 270 degrees from the start, not facing the original direction. Use returned measured headings. If nearby obstacles make a full scan unsafe, log the concrete obstruction and use bounded safe views instead.
5. If the scan does not reveal the target, choose a **visible** landmark or carpet patch that reveals an uninspected or occluded area; delegate that viewpoint approach, then inspect again. Keep track of inspected headings and viewpoints to avoid repeating the same scan or blind pivots. If clues run out later, scan again from a useful new viewpoint. Do not send an unseen object to GPT DRIVE/FLOW and expect it to search.

## Follow the user's task

- Follow the latest target, route preferences, stop instructions, and recording instructions. Existing task authorization covers routine bounded continuation; do not repeatedly ask about the charger or API access without new contradictory evidence.
- An explicit stop/hold/cancel stops motion immediately. A request to forget the layout clears search assumptions and planner memory; use fresh observations, not previous target coordinates. Do not discard obstacles observed during the current run.
- Distinguish **find** from **find and approach**. Finding requires visual identification; approaching also requires verified arrival. Do not invent an additional destination once the task is complete.
- Keep changes local. Push only when requested. This skill describes supervised operation, not permission for unattended indefinite driving.

## High-level planning latency — user preference, 2026-09-22

Target at most **5 seconds per high-level navigation planning iteration**, from
receipt of a usable fresh observation/result to dispatch of the next actionable
handoff or hold decision. This is a best-effort wall-clock target, not an enforced
runtime deadline. Include assistant reasoning and intermediate tool overhead;
log observation-receipt and dispatch times and disclose overruns, separating
external tool/API waits rather than hiding them. Initial service/recording setup
and actual downstream motion duration are separate lifecycle timings.

Use one concise observation review and one complete visible-destination handoff.
Trust downstream clearance decisions. Avoid repeated stationary ASK calls,
custom preview construction, redundant status checks, and per-waypoint assistant
approvals; use existing downstream previews and local monitoring. Re-engage at
arrival, HOLD/failure, target identification, or changed user instructions.
Do not interrupt a valid executing downstream plan merely because high-level
review takes longer than 5 seconds. If a required new decision is unavailable,
retain the controller's safe stop/lease behavior; never force motion or bypass
power, sensor, collision, or cancellation guards to meet the target.

## Clearance ownership — user instruction, 2026-09-22

For navigation delegated to the GPT mid-level planner and local controllers,
trust those downstream planners to assess clearance, choose routes, and evaluate
the full chassis and turning sweep. The high-level assistant supplies the visible
destination, observed obstacles, retained offscreen context, and user constraints;
it does not independently certify clearance or veto an accepted route because
its own visual estimate is uncertain. Do not require a second high-level clearance
approval or stop solely because an obstacle is clipped or offscreen. Pass that
uncertainty downstream and let the planner decide whether to proceed or HOLD.

The high-level role is search strategy, target identification, destination/intent
checking, and final verification. Show downstream trajectory previews for user
visibility; these are not an additional assistant clearance gate. Respect actual
planner HOLD/blockage, controller faults, power/sensor guards, user stops, and
concrete evidence of imminent collision. Never disable guards or bypass a HOLD
with raw motor commands. Manual motion without a downstream clearance planner
still requires the applicable geometry and clearance checks.

The user corrected the final stop in run `advil-20260922-015713`: clearance was
sufficient. Record that as user-supplied evidence; do not describe the assistant's
uncertainty as a proven obstruction. This changes supervision policy, not the
controller implementation or its validation status. This section takes precedence
over older references that assign delegated-route clearance approval to the
high-level assistant.

## Responsibilities

**Planning priorities (user instruction, 2026-09-22):** (1) avoid collisions with
the entire chassis and turning sweep; (2) safely reach the object of interest;
(3) follow high-level routing preferences; (4) preserve the reference trajectory.
Keep providing the reference for continuity, but allow the mid-level planner to
replace unsafe segments and choose a safer side than the high-level suggestion.
Allow useful partial paths rather than forcing a final segment through clutter.
Stop/cancel commands and explicit no-go boundaries remain hard constraints.
Do not override a safer planner detour merely to enforce your left/right preference.

**High-level assistant:** owns search strategy, reviews fresh images, identifies targets, chooses visible viewpoint landmarks, checks destination/intent and responds to downstream exceptions, and verifies completion. When the target is absent, use current clues to choose a directed look or visible viewpoint; scan whenever no useful target-location clue remains. Never hand the mid-level planner an unseen object and expect it to conduct the search.

**GPT mid-level planner:** receives a concrete destination it can currently see, an approach side, a stopping condition, and relevant obstacles. It proposes image-space trajectories. Example: “Approach the visible pink bin's left-front corner across clear carpet, stop short, avoid chair legs; this is a viewpoint move.” Once the target is identified, hand it that specific visible object.

**Local stack:** projects image points to the floor, adjusts paths around reported obstacles, follows waypoints using camera/IMU feedback, and controls expiring motor commands. Keep power, sensor, cancellation, stall, and collision checks active. Only one controller owns motion.

**Latency and intervention budget:** the GPT mid-level planner has approximately **4 seconds of request latency** (an observed typical value, not a deadline guarantee). The high-level assistant should be involved as little as possible: issue a complete visible-target approach, then let GPT and the local controller execute it without assistant approval or replanning at every waypoint. Prefer local progress monitoring and termination checks over frequent assistant calls. Re-engage for a new search viewpoint, target identification, meaningful failure, changed instructions, or final verification. Do not add stationary waits solely to obtain high-level review when the existing checked plan can continue; do not extend motion into unobserved space to hide model latency.

Minimize assistant calls by handing off a complete inspected approach. Let the local stack perform its internal steps. Review at viewpoint arrival, target identification, completion, or failure; do not remotely steer each motor update. This does not make the stack a global mapper or guarantee obstacle avoidance.

## Run lifecycle

1. **Observe:** check fresh camera and service health, zero/unexpected motor state, battery telemetry, and current calibration. For a new scan, retain the user's roughly 35 cm initial-clearance assumption unless observations contradict it; it does not apply to later routes. Account for the 12 × 15 cm chassis, 5 cm margin, measured pivot uncertainty, and offscreen obstacles.
2. **Record and log:** unless the user explicitly overrides it, start camera recording before the first search movement. Verify the process and growing file on mounted `/mnt/robotlogs`. Create a unique run log. Storage failure must be reported; never silently fall back to SD or claim recording started. For storage failures, retry an actual small read/write check three times with short delays; verify recording growth after recovery. If recording cannot start, resolve it or ask whether to proceed without it.
3. **Search:** follow the clue-driven search procedure above; when location is unknown, scan and inspect the returned views together. Select a justified visible landmark; show the intended trajectory and clearance in the demo before execution. Label tentative areas and offscreen uncertainty. Do not treat a centerline overlay as verified clearance.
4. **Delegate:** send the mid-level planner the visible destination and complete approach intent. Prefer the obstacle-aware mission path. Check whether it understood the destination and intent; correct a mismatched destination. Supply obstacle context and trust downstream clearance decisions. Do not use the legacy autonomous search endpoint for high-level search.
5. **Monitor locally:** keep recording, battery monitoring, and status logging running across handoffs. Check background processes actually survive tool/turn boundaries. A VLM call or assistant response may be stale before it arrives; stopping must not depend solely on assistant latency.
6. **Verify and finish:** inspect a fresh image and measured/reported result. Confirm identity, final range with its uncertainty, and zero motor output. Stop and finalize the recording at task completion or cancellation. By the standing recording preference, stop and finalize search recording at target identification; the subsequent approach is unrecorded unless the user requests it. Continue navigation until approach is verified. On an unrecoverable failure, stop safely and finalize with a failure result. Continue recoverable work within the authorization, never by disabling guards.

## Audit and results

Read [audit and result requirements](references/audit-and-results.md) before a run.
Persist every complete GPT request, exact input image, response, error, and applied/
discarded result on USB. Verify capture is active before powered planning. Keep
newest-first mission logs plus chronological events, exact commands, and timestamps.
Report visual identity/arrival honestly, estimated range and motion, battery and
motor state, recording path, and end-to-end timing; disclose gaps and failures.

## How to brief the mid-level planner

Use a short, stable brief: **physical visible target + distinguishing appearance + approach side if needed + important obstacles + one final destination**. Example: “Approach the visible Advil bottle with blue cap and yellow label, left of center. Use the clear carpet between bin and can; avoid the blocks beside the bottle. Stop short; this is the only destination.” Pixels may seed identification in the first image, but are not persistent coordinates after movement.

- Do not describe a made-up floor point “25 cm before the object” as a second destination. The model cannot reliably place an invisible metric point; its contact must describe the physical target. A genuine visible floor patch can be a target when explicitly selected.
- Do not ask the model to subtract centimeters from its contact estimate. Spoken standoff does not configure the controller. Configure a supported numeric controller limit, or explicitly use the local stop observer while acknowledging its model-derived range and observation delay.
- Keep search rationale out of the driving instruction. Choose visible landmarks yourself, then let the mid-level planner finish the bounded approach. A four-second request is still preferable to another high-level assistant intervention at each leg.
- Preserve the user-selected mode. Latest live configuration (2026-09-22): **FLOW**, `POST /api/flow {"target":"<visible destination>","every":2}`, with the stronger reference-preserving prompt. `every` is the staggered request interval, not a per-move duration or guaranteed response rate. For GPT DRIVE use `/api/mission` with `seconds:3`; the next request follows completion/interruption of that bounded motion slice. Do not mix these parameter names.
- When the fresh camera frame is clear enough to plan directly, seed `/api/mission` with `initial_plan.route_pixels` and that frame's `seed_frame_token`. The local stack executes this checked, bounded first segment immediately and starts recurrent GPT planning after it. This saves one model call and its latency. This seed interface is implemented for GPT DRIVE; do not claim FLOW accepts or executed a high-level seed without checking its implementation. Never seed pixels from memory, an older frame, or a frame captured before robot motion; let the server reject a stale token rather than removing the check.
- For **GPT DRIVE only**, always include `"seconds":3` for a visible-destination handoff. Omitting it from the raw HTTP request selects one look only, forcing unnecessary high-level reissues. Do not deliberately break a reviewed approach into repeated one-look missions. The audit command helper defaults omitted seconds to 3; explicit null is reserved for deliberate diagnostic moves. Re-engage on arrival, hold, failure, or a newly observed search target, not after each local segment.
- Respect explicit `motion=hold`. It ends the local handoff without declaring arrival. Inspect one fresh view and the reason, then choose a justified new viewpoint; do not immediately repeat the same blocked request or revive its old route.
- A close, frame-filling object or hidden base is an observation limit, not an instruction to orbit it. Never rely on automatic reacquisition turns at close range. Pass recent traversed clearance and obstacle context to the downstream planner for recovery decisions.
- Confirm target identity yourself before handing it off. Prompt evaluation found that the model can mistake distant clutter for the requested product. A target name in an instruction is not evidence it is visible.

## Current FLOW configuration and observed limitations

- Live stronger-prompt profile: `JETBOT_CLEARANCE_PROFILE=extreme_reference` in the planner process. `fetch.py` appends `local_nav/prompts/clearance-candidate-20260922.txt` to `trajectory-20260920.txt`. Retain the annotated reference trajectory and prior context; reference remains the lowest priority. Verify actual saved request instructions, not merely a shell variable. Set `JETBOT_GPT_AUDIT=1` and confirm saved images, full requests/responses, and applied/discarded events.
- The candidate asks for wide body clearance (20 cm spare carpet as a planning intention, not a measured guarantee), checks silhouettes/corners, and permits useful safe partial paths. Two-frame offline replay had no scored carton crossings in six stronger/reference samples. This is limited evidence, not general collision safety.
- 2026-09-22 powered runs: Advil was found after a four-way scan and nut-can viewpoint. FLOW stalled around 21–24 cm and returned HOLD, so the earlier 10 cm goal was **not** achieved. The other wheeled robot was subsequently found and approached; final FLOW range was about 13 cm, visually verified nearby, not independently measured.
- `MISSION_STANDOFF_CM` is currently 20 cm. Natural-language “10 cm” does not change it. Repeated near-identical 21–24 cm estimates with zero displacement indicate no progress, not arrival or justification for endless calls. Inspect the reason, stop redundant execution, and report the achieved distance honestly; do not disable guards to force closeness.
- A wide detour can put a visible target at the frame edge and cause identification HOLD. Inspect a fresh image; if identity is still confirmed, request downstream-planned bounded recentering with retained obstacle context, then issue an updated visible-target handoff. Do not blindly repeat a held command or reuse its stale route.
- Let each complete handoff run locally; re-engage at arrival, HOLD/failure, new viewpoint, or final verification. FLOW's “wheels never stop” log text is not a guarantee of continuous motion.
- On startup, use a bounded readiness poll: camera initialization may take longer than five seconds. A stale-power latch can remain after telemetry recovers; inspect it and restart only an idle service with current healthy power, retaining guards. Do not assume a latch was caused by an operator stop. Preserve the configured API environment before an idle planner restart; use normal inherited configuration, never print keys or search unrelated process environments. Keep the motion planner bound to `127.0.0.1`; the read-only visualizer serves the LAN.

## Current evidence

Read [revision 3 evidence and limitations](references/prompt-revision-20260920.md) before claiming this improved prompt is validated. Saved-image API tests and a motor-free dashboard replay are distinct from live driving acceptance. The new response contract uses explicit follow/turn/hold; `fetch.recognize` raises `PlannerHold` before a hold can reach old-route fallback. An already-running Python server must reload before it uses the new prompt and contract.

Earlier powered trials found and approached Advil, but also overshot standoff and drove too close to a bin. Prompt changes do not fix calibration error, all stale-path fallbacks, goal-adjacent obstacle exclusions, or the mismatch between the local checker and the five-cm margin. Keep those limits visible. Recording/logging are lifecycle responsibilities, not automatic hooks installed by loading this skill.

## High-level reviewed-image contract

Update the high-level planner panel only when the high-level assistant has actually reviewed an image. After each review, publish the exact reviewed RGB image with `scripts/publish_highlevel_review.py`, the exact prompt that will be sent to the GPT mid-level planner, and the high-level `route_pixels` and `goal_pixel`. The panel must remain frozen until the next explicit high-level image review. Camera polling, mid-level GPT calls, local-controller snapshots, reset captures, and command handoffs must not replace it. A new/reset run with no reviewed image should clearly say that no high-level review exists; it must not substitute an unreviewed readiness frame. Publish the review before dispatching its associated handoff, then verify `/api/highlevel-view` shows that prompt and trajectory.

## Live control visualizer

Share http://192.168.86.158:8772/ before motion. It displays the current run, every saved GPT input image and response, proposed waypoints, controller output, and timestamped assistant instructions. For every intervention use `scripts/planner_audit_command.py ENDPOINT JSON REASON` with a concise observable rationale; it resolves the active run from `/mnt/robotlogs/current-search.json`. Verify the instruction appears in `/api/control` before continuing. An emergency halt must never wait for logging. Historical calls without execution events must remain labelled unverified; do not infer arrival from a model response or range estimate alone.
