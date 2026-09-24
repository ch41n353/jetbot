# Local trajectory interface — 2026-09-22

The new `/api/local-execution` endpoint isolates execution from GPT planning.
Existing GPT DRIVE and FLOW are unchanged; migrating them is the next stage.

POST body:
```json
{"frame_token":42,"trajectory":{"initial_turn_deg":20,"waypoints_cm":[[12,25],[15,40]]},"visualization":{"rgb_jpeg_base64":"...","topdown_jpeg_base64":"..."}}
```

`trajectory` is the only motion input. Points are [right, forward] centimeters
in the camera frame at submission, **before** the optional initial turn.
Positive angles turn right. Waypoints retain their order and fixed frame;
the executor transforms them using its estimated pose, including camera pivot
translation. No target label, target position or obstacle list is accepted.
The caller owns collision-free route selection. Hardware health, expiring motor
leases, cancellation, waypoint timeout and no-progress limits remain active.
This endpoint has no semantic obstacle avoidance and must not be used on an
uninspected route. It does not call GPT, adjust a route, or skip a blocked point.

`frame_token` ties submission to the latest staged observation. Visualization
is optional, has no effect on control, and accepts base64 JPEGs. The supplied
top-down image is 640×640, covering 4×4 meters centered on the submission camera,
forward up. The current RGB image is projected onto the visible floor using the same fisheye
calibration as the existing top-down view. Unseen floor remains blank unless an
upstream top-down image is supplied as context. Supplied pixels are warped as pose
changes, then the current visible-floor projection is overlaid. This is a floor-plane
projection: upright objects distort and it is not a 3D reconstruction.
The current camera remains centered and current heading points up. Blue shows
the command; green shows estimated motion. Camera overlays use the same pose.

GET `/api/local-execution` returns execution ID, phase, error, current pose,
ordered estimated trajectory, completed waypoint count and artifact directory.
POST `/api/halt` cancels through the existing stop path. A failed or timed-out
execution is never reported as completed. The downstream mid-level planner can
use the returned trajectory and final pose to rebase its reference.

Translation uses calibrated carpet feature tracking (the inverse static-floor
SE(2) transform); heading uses IMU changes. During explicit turns, IMU rotation
plus calibrated pivot shift supplies motion. Visual tracking failure stops;
there is no timed-distance fallback. This is planar visual/inertial odometry,
not full six-degree-of-freedom VIO, SLAM, or ground-truth accuracy. Completion is
within a 3 cm *estimated* waypoint tolerance. Max trajectory length 4m, waypoint
coordinates within +/-2m, max 64 points, execution timeout 90s.

Every run saves command.json, initial.jpg, states.jsonl and result.json under
its USB local-executions directory. The shared visualizer reads paired snapshots
through /api/planner-snapshot and updates during execution. No FLOW threading
redesign or GPT DRIVE handoff migration is included in this first local stage.

## Acceptance evidence

129 motor-free regression tests passed, including actual HTTP submission,
turn-first execution, simulated translation, image generation and tracking-loss
failure. Powered test under `/mnt/robotlogs/goals/local-executor-20260922-220336`:
20° right requested, 19.9375° measured; 15 cm post-turn segment commanded;
final odometry endpoint error 2.99 cm (3 cm completion tolerance). The robot
completed one waypoint; zero motors verified and sensor/motor service shut down.
No independent external position measurement was made. Headless Chromium verified
both camera/grid panels displayed the same completed snapshot (ID 10), with pose
and completed-waypoint metadata. Recording:
`/mnt/robotlogs/recordings/local-executor-20260922-220336.mkv`.

## Path history and viewport

The local executor retains a run-frame history across successive instructions
on the same bench and sensor session. Each instruction gets a new origin at the
last estimated pose, and its local trajectory remains separate. Result metadata
includes `run_pose_cm_deg`, `run_estimated_trajectory`, `instruction_origin_cm_deg`,
`instruction_history`, and `remaining_trajectory_cm` in the current camera frame.
Completed waypoints are removed from the remaining path; its drawing starts at
the current camera. Full history is green, latest-instruction history orange,
and remaining waypoints blue. The orange line is thinner so the overlapping green
history remains visible. All positions are estimates. This history covers the
new local executor, not unrelated legacy motion; reset the run after outside
movement. Service/bench restart starts a new odometry history, not a falsely
stitched continuation. Integrating continuous session state with GPT DRIVE is
still a later migration step.

The top-down panel supports 1–8× zoom (buttons, slider or wheel), drag panning,
native scrolling in both axes, Center robot and Fit. Zoom/pan stay unchanged when
new snapshots arrive. The underlying floor remains 4×4 m; zoom changes display
scale, not metric coordinates. Paths outside that square are retained in result
metadata but are outside the current image.

### Mid-level reference visualization (2026-09-22)

The control-room page now includes paired RGB and 4 m × 4 m top-down reference
views. Magenta denotes the reference; cyan denotes a saved GPT proposal. The
read-only `/api/midlevel-view` endpoint renders the most recent local command's
full supplied trajectory on the matching executor's clean RGB observation,
rebased by its estimated pose. If that observation is unavailable it explicitly
uses the original instruction image/frame. It never overlays on an already
annotated image or pairs an unrelated execution's pose with the reference.

Selecting a GPT call uses `/api/midlevel-view?call=<audit-id>` and its saved clean
image, request `last_time.route_pixels`, and response waypoints. Existing GPT
Drive audits retain a previous-plan reference, **not a separate persistent
high-level trajectory**; the viewer labels this distinction. A missing reference
or clean image is reported rather than fabricated. These views do not change
GPT inputs, controller behavior, or the pending mid-level interface redesign.

Validation: saved local-pose reprojection and historical-request fixture replay;
headless Chromium against the actual port-8772 dashboard verified both reference
images, provenance labels, and existing local zoom/pan controls. No powered motion
was performed for this visualization change.

The mid-level display is now a single merged panel. It renders three distinct
trajectory layers in the saved observation frame: magenta top-level reference,
orange previous GPT proposal after estimated motion, cyan newly returned proposal.
Current reported obstacle contacts are red on RGB and top-down. Remembered
contacts outside the projected camera field are gray crosses/rings on top-down
only; they are not painted as fresh RGB evidence. Yellow marks the target.
Request `local_map` supplies already-rebased metric layers for paired GPT Drive
calls. Legacy `last_time.route_pixels` supplies only the previous proposal;
it is never relabeled as the top-level reference. Exact request images (all of
them), prompt, response, timing and call selection remain in the merged panel.
Saved-call browser replay and metric-layer fixture tests passed without motion.
Full paired-input GPT Drive powered validation remains outstanding.

### Fresh observation precedence and reference aging

GPT Drive paired inputs suppress remembered target/obstacle coordinates when
the estimated contact projects inside RGB. The prompt requires fresh localization
there, and offscreen memory alone gets metric map markers. FOV is a geometric
prediction, not proof of visibility: an occluded/missing detection must not be
interpreted as free space. Controller obstacle safeguards remain independent.
The target's legacy `last_time.target_was` text is also omitted when in FOV.

A reference's advisory weight is `2**(-distance_cm/100 - yaw_degrees/180)`.
Distance is accumulated inter-planning estimated displacement, not ground-truth
arc length; yaw is accumulated absolute estimated yaw. Neither is a calibrated
uncertainty model. Replanning preserves this age; a new mission resets it.
The magenta overlay dims as weight falls. RGB evidence always takes priority,
regardless of weight. Offscreen obstacle memory does not expire into free space.
Tests exercise paired request construction with a mocked API, in/out-of-FOV
memory, accumulated aging and rendering. No new API call or powered validation
was performed for this policy change; it loads on the next planner restart.

### GPT Drive final facing (2026-09-22)

Arrival requires a fresh visible floor-contact estimate within the configured
standoff and an absolute target bearing no greater than 8 degrees. A close but
off-axis target triggers a pivot-aware turn (at most 30 degrees per observation)
through the same `Execution` class, with no translational waypoints. The upstream
handoff checks the swept 12x15 cm body with 5 cm margin plus 2 cm pivot uncertainty
against observed obstacle contacts and the target. A new GPT/camera observation
must confirm the range and bearing before arrival. Up to three facing attempts
are allowed; a blocked turn or nonconvergence reports HOLD, not arrival.
An empty standoff-trimmed route no longer independently declares success.
One-look operation can turn but reports arrival unverified until reobserved.

Validated without motors: real executor turn-only replay, blocked-sweep rejection,
and full GPT Drive replay with an off-axis observation followed by a centered
observation. No real-robot facing acceptance run has yet been performed.

### Previous-plan anchoring correction

Inspection of calls 3/4 in `local-executor-20260922-234910` showed the
confidence decay affected an absent top-level reference, while the complete
previous GPT route remained brightly drawn and repeated in text. GPT Drive
now sends only the first 15 cm of the forward previous-route polyline as a
faint, optional continuity hint. Text and both request images use that same
prefix. Prompt instructions require a fresh floor-based approach; an old side
or bend is not a reason to detour. The full previous route remains in the audit
sidecar for the orange inspection overlay, but is excluded from the API JSON.
Tests verify the prefix bound and absence of the private full-route field in
constructed requests. This correction has not yet had a live-model comparison.

User refinement: the 15 cm previous-plan truncation above is superseded. Keep
its full geometry available with faint rendering, but preserve it only when the
current route to the goal is uncertain and memory remains consistent with
obstacle evidence. Clear current visual evidence takes precedence. No motor
behavior changed with this prompt adjustment.

### Continuous local turns — 2026-09-23

The trajectory-only executor now uses `Robot.turn_continuous`: one regulated
IMU sweep with rate taper and predicted-coast cutoff, then a final stop and
coast measurement. It does not invoke fine burst corrections and no longer
splits a requested heading into 60-degree segments. Final angle error still
fails execution if outside its tolerance. Cancellation, power/sensor health,
yaw-rate, stall, motor-lease and bounded-duration checks remain active.

Hardware test `local-executor-20260922-235831`: commanded +30 degrees, measured
+29.5625 degrees; completed without controller error, no translational command,
zero motors verified and recording finalized. Motor-free tests check nonzero
commands between feedback updates and final stop on fault. This validates one
small turn, not all angles or surfaces. Driving waypoint pauses and GPT
replanning pauses are separate and unchanged.

### Threaded snapshots and forward continuity — 2026-09-23

While driving, a single background renderer receives an immutable copy of the
image, pose and trace. Intermediate visualization updates are skipped when it
is busy; the motor/control loop does not wait for JPEG encoding or USB writes.
Final snapshots flush after motor stop. Forward waypoint transitions no longer
issue zero motor commands, and the visual update interval was shortened from
100 ms sleep to 30 ms sleep (processing time is additional). Turn transitions,
faults, cancellation and GPT replanning intervals still stop as appropriate.

Hardware test `local-executor-20260923-001052` executed waypoints (0,6),(0,10) cm,
ending at estimated (−0.44,7.68) cm under the existing 3 cm waypoint tolerance.
Control updates were 87–132 ms apart, below the unchanged 200 ms motor lease.
Motor status sampled during the test showed zero intermediate zero-output
samples. Both waypoints completed with no controller error. Motors were stopped
and recording finalized. This is a short straight-run validation, not a guarantee
for arbitrary curves or load. A motor-free blocked-render test confirms control
continues across waypoints while the final stopped state waits for rendering.

### Forward continuity rollback — 2026-09-23

At the user's request, forward execution again permits stops between local
iterations: command a 100 ms motion interval, command zero, wait 50 ms, then
capture/estimate the resulting motion. Waypoint boundaries also command zero.
Background rendering/USB writes and continuous IMU turns are retained. This
supersedes the continuous forward behavior above; it does not fix the missing
per-frame pitch/roll integration or prove that carpet scale faults are resolved.
Motor-free real-executor and HTTP regression tests validate the stop commands.
No powered motion was performed for this rollback; next planner startup loads it.

### 2026-09-23 — longer stop-and-measure segments

Forward execution now holds a segment for up to 0.3 seconds (previously 0.1),
shortening near the 3cm waypoint tolerance and the GPT Drive slice deadline.
It checks cancellation and robot health and renews the existing 200ms motor lease
every 75ms while moving. It does not stop between those renewals. Each segment
still ends with zero motors, 50ms settling, and a fresh visual displacement
estimate. The 8cm visual-displacement sanity limit remains unchanged. This is
not continuous navigation or a threefold end-to-end speed guarantee; camera,
settling and GPT latency remain. Larger segments also mean less frequent visual
correction, so powered acceptance is required.

### 2026-09-23 — restore measurement during motion

The 500ms and 300ms stop-then-measure experiments failed powered acceptance
with the floor-displacement guard; they are superseded. The executor now
captures and estimates displacement while motors remain on, with a 60ms wait
before each observation. Stops occur at segment boundaries (0.8 seconds or
8cm estimated travel, checked per iteration), near waypoints, before turns,
and on completion/cancellation/fault. The motor lease is unchanged; no blind
background renewal is used. Rendering remains asynchronous. Segment bounds
are checked per observation, not exact hard distance limits.

Motor-free HTTP, cancellation, motion-sampling and blocked-render tests passed.
Powered acceptance was deferred: the fresh camera view showed the robot on
its side. No movement was issued in that acceptance run; service was disarmed.

### 2026-09-23 — continuous-motion hardware retest

At the user's request after hardware improvements, removed periodic 0.8s/8cm
stops. Vision/IMU feedback runs while driving; stops remain at waypoint/turn,
mission deadline, requested travel limit, cancellation and fault boundaries.
No blind motor renewal worker or relaxed odometry thresholds were added.
Floor-displacement failures now include residual and displacement values.

Powered acceptance: `/mnt/robotlogs/goals/local-executor-20260923-010426`.
30cm forward reference, requested pause after 10cm; estimated travel 10.22cm,
yaw change 2.44 degrees, no error. Six visual increments, motor renewal intervals
126–178ms; 30ms status sampling detected zero intermediate zero-output samples.
This is one short successful trial, not validation of long routes or calibration
following the hardware change. Motor-free segment/cancellation tests also passed.
