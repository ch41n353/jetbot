---
name: jetbot-task-control
description: Supervise JetBot search and approach tasks through the GPT mid-level trajectory planner and local controllers, with camera recording, trajectory review, and timestamped newest-first mission logs.
metadata:
  baseline-date: "2026-09-20"
  revision: "5"
---

# JetBot task control

Revision 5 — 2026-09-20. Workspace: `/home/jetbot/jetbot`.
Read [the operating interface and limitations](references/operations.md) before operating.

## Follow the user's task

- Follow the latest target, route preferences, stop instructions, and recording instructions. Existing task authorization covers routine bounded continuation; do not repeatedly ask about the charger or API access without new contradictory evidence.
- An explicit stop/hold/cancel stops motion immediately. A request to forget the layout clears search assumptions and planner memory; use fresh observations, not previous target coordinates. Do not discard obstacles observed during the current run.
- Distinguish **find** from **find and approach**. Finding requires visual identification; approaching also requires verified arrival. Do not invent an additional destination once the task is complete.
- Keep changes local. Push only when requested. This skill describes supervised operation, not permission for unattended indefinite driving.

## Responsibilities

**High-level assistant:** owns search strategy, reviews fresh images, identifies targets, chooses visible viewpoint landmarks, reviews proposed paths and exceptions, and verifies completion. When the target is absent, choose a scan or a visible destination that will reveal another part of the room. Never hand the mid-level planner an unseen object and expect it to conduct the search.

**GPT mid-level planner:** receives a concrete destination it can currently see, an approach side, a stopping condition, and relevant obstacles. It proposes image-space trajectories. Example: “Approach the visible pink bin's left-front corner across clear carpet, stop short, avoid chair legs; this is a viewpoint move.” Once the target is identified, hand it that specific visible object.

**Local stack:** projects image points to the floor, adjusts paths around reported obstacles, follows waypoints using camera/IMU feedback, and controls expiring motor commands. Keep power, sensor, cancellation, stall, and collision checks active. Only one controller owns motion.

**Latency and intervention budget:** the GPT mid-level planner has approximately **4 seconds of request latency** (an observed typical value, not a deadline guarantee). The high-level assistant should be involved as little as possible: issue a complete visible-target approach, then let GPT and the local controller execute it without assistant approval or replanning at every waypoint. Prefer local progress monitoring and termination checks over frequent assistant calls. Re-engage for a new search viewpoint, target identification, meaningful failure, changed instructions, or final verification. Do not add stationary waits solely to obtain high-level review when the existing checked plan can continue; do not extend motion into unobserved space to hide model latency.

Minimize assistant calls by handing off a complete inspected approach. Let the local stack perform its internal steps. Review at viewpoint arrival, target identification, completion, or failure; do not remotely steer each motor update. This does not make the stack a global mapper or guarantee obstacle avoidance.

## Run lifecycle

1. **Observe:** check fresh camera and service health, zero/unexpected motor state, battery telemetry, and current calibration. For a new scan, retain the user's roughly 35 cm initial-clearance assumption unless observations contradict it; it does not apply to later routes. Account for the 12 × 15 cm chassis, 5 cm margin, measured pivot uncertainty, and offscreen obstacles.
2. **Record and log:** unless the user explicitly overrides it, start camera recording before the first search movement. Verify the process and growing file on mounted `/mnt/robotlogs`. Create a unique run log. Storage failure must be reported; never silently fall back to SD or claim recording started. If recording cannot start, resolve it or ask whether to proceed without it.
3. **Search:** inspect new views yourself. Select a bounded scan or visible landmark; show the intended trajectory and clearance in the demo before execution. Label tentative areas and offscreen uncertainty. Do not treat a centerline overlay as verified clearance.
4. **Delegate:** send the mid-level planner the visible destination and complete approach intent. Prefer the obstacle-aware mission path. Review whether it understood the destination and obstacles; reject a mismatched route. Do not use the legacy autonomous search endpoint for high-level search.
5. **Monitor locally:** keep recording, battery monitoring, and status logging running across handoffs. Check background processes actually survive tool/turn boundaries. A VLM call or assistant response may be stale before it arrives; stopping must not depend solely on assistant latency.
6. **Verify and finish:** inspect a fresh image and measured/reported result. Confirm identity, final range with its uncertainty, and zero motor output. Stop and finalize the recording at task completion or cancellation. For find-only, stop at identification; for find-and-approach, include the approach. On an unrecoverable failure, stop safely and finalize with a failure result. Continue recoverable work within the authorization, never by disabling guards.

## Deliverables

### Complete GPT request and response records

User requirement, 2026-09-20: persist **every prompt, every image actually sent to GPT, and every full response** for every planner call (including ASK, GPT DRIVE, FLOW, retries, and results discarded as stale). Dashboard summaries and temporary filmstrip images do not satisfy this requirement.

- Use a unique run ID and per-request ID. Save the complete outbound request body before dispatch: model, instructions, user text, prior context, schema, reasoning settings, and image payloads. Preserve the exact encoded image bytes sent, including trajectory annotations, as separately viewable files linked to the request. A raw camera recording alone cannot reconstruct the actual prompt.
- Save the complete returned API response before parsing, normalizing, or modifying it. Preserve original waypoint pixels, obstacle detections, notes, contact points, and any returned usage/request identifiers. Record HTTP errors, timeouts, incomplete responses, and parsing failures; do not invent a response for a failed call. Never save authorization headers or API keys.
- Record capture time, dispatch time, response time, request sequence, and captured robot pose when available. For overlapping FLOW calls, retain all responses and record which were applied or discarded. Link the chosen response to the projected/rebased/adjusted route delivered to the controller so planned and executed paths can be compared.
- Store artifacts under the current USB run directory, for example `gpt/<request-id>/request.json`, `image-0.jpg`, `response.json`, and `events.jsonl`. Follow the recording-storage policy and bounded retries; SD storage requires an explicit applicable user override. Do not overwrite previous calls.
- Before the next powered planner run, verify this capture is actually implemented and active at the API request boundary. Loading this skill does not install logging hooks. If records cannot be written, report the failure and resolve it or obtain an explicit exception rather than silently running without them. Share the live log link before movement and link each call's artifacts from the run log.
- Verify persisted requests, images, and responses are readable and paired at the end of the run. Report any missing artifacts honestly. When asked what GPT suggested, quote the saved response; never substitute the high-level instruction, stale dashboard state, or inferred waypoints.

- A **newest-first Markdown log** with timestamps, observations, concise decision rationales, exact commands and JSON arguments, planner/controller responses, failures and interventions. Preserve raw chronological events separately. Log observable rationale, not private internal reasoning.
- Distinguish command issuance time, observation time, and actual execution time; do not fabricate unavailable timestamps or motor telemetry. Disclose gaps, terminated recorders, and uncertain estimates.
- Provide the recording link, final camera image when useful, outcome, commanded versus measured motion, battery status, and motor state. “Mission ended” is not arrival. Report multiple high-level handoffs honestly; they are not one-shot search success.

## How to brief the mid-level planner

Use a short, stable brief: **physical visible target + distinguishing appearance + approach side if needed + important obstacles + one final destination**. Example: “Approach the visible Advil bottle with blue cap and yellow label, left of center. Use the clear carpet between bin and can; avoid the blocks beside the bottle. Stop short; this is the only destination.” Pixels may seed identification in the first image, but are not persistent coordinates after movement.

- Do not describe a made-up floor point “25 cm before the object” as a second destination. The model cannot reliably place an invisible metric point; its contact must describe the physical target. A genuine visible floor patch can be a target when explicitly selected.
- Do not ask the model to subtract centimeters from its contact estimate. Spoken standoff does not configure the controller. Configure a supported numeric controller limit, or explicitly use the local stop observer while acknowledging its model-derived range and observation delay.
- Keep search rationale out of the driving instruction. Choose visible landmarks yourself, then let the mid-level planner finish the bounded approach. A four-second request is still preferable to another high-level assistant intervention at each leg.
- Use `/api/mission` (GPT DRIVE). Eight seconds is an upper motion bound: the next GPT call follows completion or interruption immediately. FLOW overlaps calls and motion but remains a distinct experimental mode.
- Always include `"seconds":3` for a visible-destination handoff. Omitting it from the raw HTTP request selects one look only, forcing unnecessary high-level reissues. Do not deliberately break a reviewed approach into repeated one-look missions. The audit command helper defaults omitted seconds to 3; explicit null is reserved for deliberate diagnostic moves. Re-engage on arrival, hold, failure, or a newly observed search target, not after each local segment.
- Respect explicit `motion=hold`. It ends the local handoff without declaring arrival. Inspect one fresh view and the reason, then choose a justified new viewpoint; do not immediately repeat the same blocked request or revive its old route.
- A close, frame-filling object or hidden base is an observation limit, not an instruction to orbit it. Never rely on automatic reacquisition turns at close range. Verify turn sweep and recent traversed clearance before recovery.
- Confirm target identity yourself before handing it off. Prompt evaluation found that the model can mistake distant clutter for the requested product. A target name in an instruction is not evidence it is visible.

## Current evidence

Read [revision 3 evidence and limitations](references/prompt-revision-20260920.md) before claiming this improved prompt is validated. Saved-image API tests and a motor-free dashboard replay are distinct from live driving acceptance. The new response contract uses explicit follow/turn/hold; `fetch.recognize` raises `PlannerHold` before a hold can reach old-route fallback. An already-running Python server must reload before it uses the new prompt and contract.

Earlier powered trials found and approached Advil, but also overshot standoff and drove too close to a bin. Prompt changes do not fix calibration error, all stale-path fallbacks, goal-adjacent obstacle exclusions, or the mismatch between the local checker and the five-cm margin. Keep those limits visible. Recording/logging are lifecycle responsibilities, not automatic hooks installed by loading this skill.

## Live control visualizer

Share http://192.168.86.158:8772/ before motion. It displays the current run, every saved GPT input image and response, proposed waypoints, controller output, and timestamped assistant instructions. For every intervention use `scripts/planner_audit_command.py ENDPOINT JSON REASON` with a concise observable rationale; it resolves the active run from `/mnt/robotlogs/current-search.json`. Verify the instruction appears in `/api/control` before continuing. An emergency halt must never wait for logging. Historical calls without execution events must remain labelled unverified; do not infer arrival from a model response or range estimate alone.
