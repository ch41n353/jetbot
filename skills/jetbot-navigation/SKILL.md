---
name: jetbot-navigation
description: Operate and iterate the local JetBot camera/IMU navigation tools for object approaches, short drives, and turns. Use for this robot's navigation, trajectory previews, and controller diagnosis; this is an experimental, locally supervised system, not general autonomous obstacle avoidance.
metadata:
  baseline-date: "2026-09-13"
  timezone: America/Los_Angeles
  revision: "11"
---

# JetBot navigation

Baseline **2026-09-13**, current revision **11**, America/Los_Angeles.
Workspace: `/home/jetbot/jetbot`. Use `/usr/bin/python3` for the local tools.

Read [the dated algorithm and evidence](references/baseline-2026-09-13.md)
before operating or changing these controllers. Read [operating commands](references/operation.md)
when running the robot. `baseline-manifest.json` fingerprints the implementation
and calibration files at capture time; inspect changed files rather than assuming
the dated instructions describe a newer implementation.

Read [revision 2](references/revision-2026-09-13-r2.md) for the new straight
route executor and its validation limits. The original baseline is preserved. Read [revision 3](references/revision-2026-09-14-r3.md)
for opt-in predictive braking and post-stop arrival measurement. Read
[revision 4](references/revision-2026-09-14-r4.md) for offline simulation tools,
feature-budget experiments and the slow-camera settling fix. Read
[revision 5](references/revision-2026-09-14-r5.md) for persistent sessions and
automatic previews when optimizing end-to-end maneuver time. Read
[revision 6](references/revision-2026-09-14-r6.md) for object approach planning
and opt-in local batches that avoid VLM calls at internal segment boundaries.
Read [revision 7](references/revision-2026-09-14-r7.md) for live return evidence,
persistent IMU state across segments, and supervised reverse repositioning.
Read [revision 8](references/revision-2026-09-14-r8.md) for local spatial planning,
measured turns, target queues, live results, and mandatory input-voltage monitoring.
Read [revision9](references/revision-2026-09-14-r9.md) for the subsequently found
INA219 battery-pack monitor, current power thresholds, and paired simulation results.
Read [revision10](references/revision-2026-09-14-r10.md) for opt-in joining of
adjacent drives into up to30cm/four-second runs and its simulation-only evidence.
Read [revision11](references/revision-2026-09-14-r11.md) for the experimental
object mission controller, local target recovery, continuous waypoint following,
asynchronous observation, and the actual validation limits.

## Scope and control division

- The high-level assistant identifies the requested object from fresh RGB images,
  chooses short routes, assesses obstacles, shows trajectory previews, and
  evaluates results between actions. It does not steer on each model call.
- Local processes acquire camera/IMU data, estimate motion, issue expiring motor
  leases, and stop on health, timing, tracking, tilt, or motion limits.
- There is no implemented global SLAM map, object tracker or automatic obstacle
  segmentation. Revision 8 adds local static-map search and bounded replanning;
  revision 2 adds a conservative
  straight swept-rectangle checker for explicitly inspected static maps.
  Do not represent a sketched route as machine-verified free space.

## Before movement

1. Use the current task's authorization. Creating/loading this skill is not
   authorization to move. Existing authorization can cover bounded continuation;
   avoid routine reconfirmations. A user stop/hold cancels motion immediately.
2. Read current service status and a fresh camera image. Sensor startup needs
   approximately three seconds; `READY` alone does not mean healthy sensors.
   Never use an old `latest.jpg` after a failed/stale snapshot request.
   Check `status.power`: the service now samples input voltage about five times
   per second and independently gates motors. Warning below4.9V; latched stop
   below4.8V, above5.25V, missing telemetry or sample age>0.6s. The pack monitor
   additionally warns below11.4V and latches off below10.8V or above12.9V
   for the inferred three-cell Waveshare-compatible profile; see revision9. A stopped power
   guard needs a fresh service after resolving the supply problem. Both pack and regulated input voltages are monitored; neither is
   **battery charge percentage**. A reported empty
   battery invalidates subsequent powered benchmarking until power is restored.
   Never infer state of charge from the 5V rail.
3. Verify calibration and geometry against the actual setup. The robot is
   **12 cm wide × 15 cm long including wheels**, with the lens at **front center**,
   9.5 cm above the carpet. Relative to the lens's ground projection, the chassis
   occupies `x = [-6, 6]`, `z = [-15, 0]` cm (right and forward positive).
4. Include a **5 cm clearance margin**, making a straight corridor 22 cm wide.
   Model front/rear extent, braking overrun, estimation uncertainty, and the
   entire turn sweep. The turn pivot/axle position is **not measured**; do not
   silently assume that rotation occurs about the lens or chassis center.
5. Preserve nearby obstacles outside the camera field of view. Transform their
   last observations with measured motion and retain uncertainty, or mark their
   positions unknown. **Out of view does not mean behind, passed, or clear.**
   If side/rear clearance or the turn sweep is unresolved, hold position rather
   than guessing or making a blind turn just to look.

## Plan, preview, execute, inspect

Before each new high-level route segment, show the proposed trajectory on a
fresh camera image. This is an explicit user preference. Show the **swept
corridor**, not only a centerline: chassis width/length, five-cm margin, next
waypoint/standoff, known obstacles, and uncertain/offscreen regions. Distinguish
the next executable segment from tentative later segments. Do not label a
translation-only overlay as a verified turning envelope. Side/rear footprint
outside the image must be stated; use a top-down inset when helpful.

Native HTML/SVG overlays preserve the original photograph; follow the available
visualization skill when using that surface. Projection uses calibrated floor
rays, but large off-axis and near-field projections are not independently
validated. A preview explains a plan; it is not a collision sensor.

Prefer a whole inspected route over model calls between small movements:

- **Object mission (experimental, opt-in):** `plan_mission` seeds a selected
  target box and inspected static map; `execute` with `mission:true` runs local
  tracking, forward waypoint following and bounded recovery. No scheduled stop
  at15/30cm or2/4seconds. Limits are180seconds/120cm plus unchanged uncertainty
  guards; these ceilings do not establish minutes of reliable autonomy. This
  mode also changes turn execution and has no powered hardware validation yet.
  `observe` and `mission_status` can inspect without interrupting it. See revision11.
  Keep simulation/test CPU load separate from live controller runs.

- **Spatial target or target queue:** revision-8 `plan_navigation` and `execute`
  with `spatial: true`, up to four targets in one inspected static map. Local
  code chooses drives/turns and rechecks a retained route using measured pose.
  Limits remain12 search primitives,60seconds and90cm total. Default drives retain
  15cm/two-second limits; revision10 opt-in joined drives permit30cm/four seconds
  with continuous visual checks. Stops remain at target, turn and direction changes.
  Turns have measured VIO translation and cancellation tokens. This is
  experimental; live straight continuation and turn-to-drive continuation
  succeeded, while full live out-and-back remained blocked by map clearance.
  The opt-in `measured_map_drive` adapter is still under validation; do not
  describe it as proven obstacle avoidance.

Other bounded options:

- **Whole straight approach:** revision-6 `plan_approach` and `execute` with
  `batch: true`, up to 30 cm in an inspected static corridor. Preview the entire
  route once. Internal segments retain 15 cm / two-second limits and verified
  stops. One live two-segment return exercised continuation without planner calls;
  final overshoot was correctly reported. See revision 7 for limits.

- **Straight waypoint batch:** `route_executor.py`, total 1–15 cm, no model calls
  or intermediate waypoint stops. Requires a fresh preview and inspected static
  map including side/rear clearance; see revisions 2 and 3. Revision 2 has one successful powered 5 cm trial;
  predictive braking has separate validation requirements.

- **Reverse repositioning:** ordinary route executor with `travel_direction:
  "reverse"`, up to 15 cm / two seconds, only with inspected or retained rear
  clearance. Results use negative physical z. See revision 7.
- **Straight travel:** `smooth_drive_probe.py`, 1–15 cm requested per invocation,
  at most two seconds powered. Hold commands between valid sensor updates.
  Check the frame and remaining full-chassis clearance before the next segment.
- **General turns:** `turn_controller.py`, at most 30° per invocation, pulse and
  settle. Power 0.14 has worked; 0.16 has also worked but sometimes trips the
  rotation-rate guard. Do not repeatedly raise power or retry stalled motors.
- **Steady rotation experiment:** `smooth_turn_probe.py` is hard-coded to a
  15° right turn at 0.16. It is not a general arbitrary-angle tool, and runs
  immediately when invoked. Prefer the general turn tool for other angles.
- **Floor-point experiment:** `point_controller.py` is the slower validated
  stop-and-observe option, restricted to near-center initial targets.

Do not revive archived continuous scripts or remove stop guards just because
an action fails. On a failure, inspect the log and camera, identify a justified
change, and keep the next test bounded. If motor command produces little motion,
stop rather than sustaining a stall. Motor power is a duty command, not a torque
measurement. The user previously reported concern about motor damage.

## Stop and report

Shut down the service on stop/hold or at the end of a motion batch. A standalone
`stop` cancels revision-2 route tokens. For older scripts it
can be overwritten by a controller that is still running; terminate that
controller and/or use `shutdown` to prevent renewal. Ctrl+C in the service
terminal invokes its stop/cleanup path. A hardware power cut remains the fallback
if software cannot communicate; do not claim a failed stop request succeeded.

Report commanded versus measured motion, actual termination reason, and the
motor/service state. `bounded_probe_completed` may mean a time limit, not arrival.
Measured threshold crossings can overshoot by one observation plus braking.
Camera/IMU estimates are **not independent ground truth**. Reaching a waypoint
does not establish reaching the object. Name uncertainty without claiming an
unseen obstacle's location as fact.

## Iterate this baseline

Preserve `references/baseline-2026-09-13.md` and its manifest as the dated starting
point. For each substantive revision, add a new dated reference (use an explicit
revision suffix for same-day changes), update this entrypoint's revision and
links, and record the concrete change, live/offline evidence, and remaining
limits. Retain failed trials as well as successes. Do not silently upgrade an
experimental capability to validated autonomy.
