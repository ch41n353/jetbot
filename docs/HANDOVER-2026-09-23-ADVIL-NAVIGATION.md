# JetBot navigation handover — 2026-09-23

## Current state

- Robot is stopped. Last verified motor output: `[0.0, 0.0]`.
- Sensor service remains motion-enabled and healthy under session
  `5256e03c4d1249cca5e8b8e21dfa3bc4`; do not assume this session survives a
  terminal or machine restart.
- Planner is running on `127.0.0.1:8770`. The read-only dashboard follows the
  current run at <http://192.168.86.158:8772/>; GPT gallery is `/gallery`.
- The Advil task is unfinished. Two scans and several viewpoint moves did not
  produce a positively identifiable Advil bottle.
- Current run root:
  `/mnt/robotlogs/goals/advil-20260923-153127`
- Current-run pointer: `/mnt/robotlogs/current-search.json`
- Search recording was finalized in three fragments because planner reloads
  close the MJPEG stream:
  - `/mnt/robotlogs/recordings/advil-20260923-153127.mkv` (104 MB)
  - `/mnt/robotlogs/recordings/advil-20260923-153127-part2.mkv` (151 MB)
  - `/mnt/robotlogs/recordings/advil-20260923-153127-part3.mkv` (110 MB)

## User requirements that matter

- Share the dashboard link before motion.
- Start and verify USB recording for every find/search request; stop and
  finalize it at identification or cancellation. Never fall back to SD.
- Preserve navigation history across target requests unless the user says
  reset/start over. A reset invalidates old plans and images using a new,
  monotonically increasing frame token.
- Use the high-level assistant as little as possible. The mid-level GPT planner
  should finish a visible-target handoff; the local executor owns motor control.
- Scan only when there is no useful clue where to look, not automatically at
  every start.
- Robot chassis is 12 cm wide by 15 cm long and can pivot in place.
- The user formally approved recurrent GPT DRIVE with three-second local slices
  during this run.
- The user asked to assume powered motion is fine after the readiness probe
  passed; do not repeat the wheel probe unless the service changes or a fault
  appears.

## Stack and interfaces

1. High level reviews fresh images, chooses a visible destination/viewpoint,
   publishes its prompt and reference using
   `scripts/publish_highlevel_review.py`, and dispatches through
   `scripts/planner_audit_command.py`.
2. Mid level (`local_nav/planner_demo.py` plus GPT in `local_nav/fetch.py`)
   receives RGB, top-down memory/reference, and target text. GPT DRIVE uses
   `POST /api/mission` with `seconds: 3`; FLOW uses `/api/flow`.
3. Local level (`local_nav/trajectory_executor.py`) accepts only a metric SE(2)
   trajectory, uses camera floor translation plus IMU yaw, controls motors, and
   publishes camera/top-down execution snapshots.

Useful endpoints:

- `GET /api/progress`
- `POST /api/capture {}`
- `POST /api/inspection {"preset":"four_way"}`
- `POST /api/ask {"target":"..."}` (stationary GPT plan)
- `POST /api/mission {"target":"...","seconds":3,...}`
- `POST /api/halt {}`
- `GET /api/inspection`

Always use the audit helper for interventions. It saves commands, handoff
images, rationales, and responses in the current USB run.

## Changes made in this session

### Fixed 20% drive power

`local_nav/trajectory_executor.py` now uses:

```python
DRIVE_MIN_DUTY = .20
DRIVE_MAX_DUTY = .20
```

The previous dynamic 20–32% profile measured about 10 cm/s on this run with
individual segments near 16 cm/s. Earlier fixed-power runs were about 6–7 cm/s.

### Recover isolated visual-odometry rejections

The local executor previously aborted an entire segment on any of:

- `Lost carpet tracking`
- `Floor motion is inconsistent`
- `Implausible floor displacement`

It now treats an isolated occurrence as a rejected camera pair:

- retain the motor lease without inserting a zero command;
- apply measured IMU yaw;
- record zero translation for that interval rather than inventing travel;
- anchor the next comparison on the newest camera frame;
- resume normal visual translation when the next pair is valid.

The fallback remains bounded: six consecutive rejected pairs or 0.8 seconds
without valid visual translation stops the segment. Power, camera/IMU health,
cancellation, execution deadline, and no-progress guards remain active.
Telemetry now includes `tracking_rejections` and
`tracking_rejection_streak`.

Test command:

```bash
/usr/bin/python3 -m unittest \
  tests.test_trajectory_executor \
  tests.test_executor_segment \
  tests.test_executor_async_publish
```

Result: 17 tests passed through the actual local HTTP execution path. The first
sandboxed run could not bind localhost; the same tests passed with localhost
binding enabled.

Live validation after reload completed accepted segments of approximately
17.9 cm, 26.8 cm, and 15.0 cm without the former floor-tracking aborts. One
earlier segment failed because a turn exceeded its separate four-second bound.
The final stop came from mid-level remembered-obstacle clearance, not local
optical flow.

### Other active uncommitted work

This worktree contains extensive prior changes in `fetch.py`,
`planner_demo.py`, `point_controller.py`, `service.py`, dashboard scripts,
skills, tests, and new untracked files. Do not commit only the two files above
without auditing the complete worktree. `local_nav/trajectory_executor.py` and
several tests are currently untracked by Git even though the live planner uses
them.

## Collision diagnosis from the preceding run

Run: `/mnt/robotlogs/goals/advil-reset-20260923-150918`

The robot hit the red obstacle because:

1. The high-level seed executed before GPT with `obstacles: []`.
2. GPT's first call did identify the close red dispenser at pixel `(556,478)`.
3. `fetch.project_obstacles` discarded it through the lower fisheye-corner
   filter, so the local clearance layer never received it.
4. There is no physical collision sensor; successful waypoint tracking cannot
   itself detect contact.

Do not use an executable high-level seed without explicit obstacle contacts.
The corner rule should eventually retain a conservative image-region hazard
while marking metric range uncertain, rather than deleting the observation.

There is also an audit-helper identity bug: if both `initial_plan` and
`high_level_visual` are supplied as separate but equal JSON objects,
`planner_audit_command.py` may omit `seed_frame_token`, causing
`initial plan rejected: its frame token is stale`. For now, supply only
`high_level_visual`; the helper promotes that exact object to `initial_plan`
and attaches the frame token.

## Current search evidence

- Initial four-way scan found no identifiable Advil.
- A second scan at a moved viewpoint captured three scheduled images; the
  third 90-degree turn measured about 80 degrees and failed the strict 8-degree
  acceptance threshold. The already-reached fourth direction was then captured
  manually without more motion.
- No scan image positively identified Advil.
- Important reviewed images:
  - `first-reviewed.jpg`
  - `scan-contact-sheet.jpg`
  - `after-viewpoint-failure.jpg`
  - `after-slower-viewpoint.jpg`
  - `look-behind-pink.jpg`
  - `scan2-sheet.jpg`
  - `scan2-fourth-manual.jpg`
  - `after-recovery-live-test.jpg`
- Exact GPT inputs, output JSON, event logs, and images are under the run's
  `gpt/` directory and visible in the dashboard gallery.

The last fresh image was `after-recovery-live-test.jpg`. It showed the pink bin
left of center, a yellow character block in the foreground, the green nut-mix
can near center-right, cartons farther right, and no identifiable Advil.

## Remaining problems

1. **Advil is still not found.** It is likely hidden by an object, absent from
   the current room, or visually indistinguishable at available viewpoints.
   Treat this only as a search hypothesis.
2. **Remembered obstacle placement stopped progress.** The last mission carried
   a remembered red dispenser at roughly 10–13 cm although the fresh image did
   not clearly show that object at the claimed position. Do not simply erase
   memory; reconcile it with fresh visibility and pose uncertainty.
3. **Turn timeout:** one local segment failed with
   `turn exceeded its 4.0 s bound`. This is independent of optical-flow recovery.
4. **Scan tolerance:** the second scan aborted after an approximately 80-degree
   turn because it missed the requested 90 degrees by more than 8 degrees.
5. **No IMU-only translation:** the BNO055 provides useful yaw, but double
   integrating acceleration is too drift-prone, especially with calibration
   `(system, gyro, accel, mag) = (0,3,0,0)`. A future short bridge should use a
   calibrated motor-speed model plus IMU yaw and increasing uncertainty, then
   correct from vision. Wheel encoders are the preferred translation fallback.
6. The three-second GPT DRIVE architecture deliberately stops between local
   slices while waiting for the next GPT response. FLOW is the overlapping-call
   alternative; this session fixed continuity only within a local execution.

## Recommended next actions

1. Re-run readiness after any service/process change; otherwise reuse the
   successful motion check per the user's instruction.
2. Start a new USB recording fragment and verify growth before search motion.
3. Capture and review a fresh frame; do not reuse stored pixel trajectories.
4. Reconcile the remembered near dispenser against the fresh RGB/top-down view.
   If fresh evidence disproves its metric placement, retire or inflate its
   uncertainty through the mid-level memory interface rather than deleting all
   obstacle history.
5. Choose a visible floor/viewpoint around the pink bin that exposes its hidden
   rear area. Do not send unseen Advil as the mid-level destination.
6. Continue with fixed 20% power and inspect `tracking_rejections` in each local
   execution artifact.
7. After finding Advil, stop/finalize recording, approach the positively
   identified bottle, verify final image/range/facing, and confirm motors zero.

## Process and safety notes

- `/api/halt` was issued for handover and motors were verified zero.
- Pack voltage at handover was approximately 11.48 V, above the 11.4 V warning
  threshold but close enough to monitor during the next run.
- Recording fragment 3 was stopped with ffmpeg's normal quit path. Fragment 2
  and fragment 1 were closed when their planner streams ended.
- Do not broadly kill Python or ffmpeg processes. Identify the exact service,
  planner, or recorder PID/session and verify motor output afterward.
