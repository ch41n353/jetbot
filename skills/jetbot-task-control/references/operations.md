# Operating interface — 2026-09-20

Use `/usr/bin/python3` (Python 3.6 compatibility). Inspect current code if changed.

## Services and tools

| Component | Interface | Purpose |
|---|---|---|
| Planner bench | `http://127.0.0.1:8770` | GPT trajectories, local execution, camera, preview |
| LAN planner UI | `http://192.168.86.158:8770/` | User-visible trajectory and progress |
| Recording viewer | `http://192.168.86.158:8771/` | Live camera, replay, original download |
| Sensor/motor service | `python3 local_nav/client.py status` | Fresh health, power and motor outputs |
| Bench implementation | `local_nav/planner_demo.py` | HTTP handlers and orchestration |
| Trajectory/controller | `local_nav/fetch.py` | `recognize`, `avoid`, `follow`, `Robot.turn`, `drive_leg` |
| Viewer implementation | `scripts/camera_recording_server.py` | Read-only LAN recording viewer |

The user authorized token-free access on this trusted LAN. Do not expose externally. Do not restart a live service or create a second motor owner just to use this skill.

### Tool boundary and latency

The HTTP bench endpoints are the callable tools exposed to the high-level assistant. `/api/ask` wraps a GPT mid-level planning call; `/api/mission` wraps GPT planning plus local trajectory execution. GPT itself is the model inside those tools, not the low-level motor controller. Typical GPT planning latency is about 4 seconds; the high-level assistant should not be on the per-waypoint execution path. Actual latency varies, and the existing recurrent mission can pause while awaiting GPT. Do not describe it as uninterrupted asynchronous control merely because it is exposed as one tool.

### HTTP requests

Send JSON with `Content-Type: application/json`. Record exact body and timestamp before issuing motion.

- `GET /api/progress`: `running`, `log`, `power`, `points`, `hazards`, `note`, frame token. Log lines have no original timestamps; timestamps added by a poller are observation times.
- `GET /api/frame?view=camera&live=1`: fresh camera when idle; during a run may return the most recent staged frame. Verify freshness. Save only to the mounted USB volume.
- `GET /api/stream.mjpg`: live MJPEG stream, possibly trajectory overlays. Resolution and frame rate depend on current server settings; do not call it raw full-rate camera capture.
- `POST /api/project {"points":[]}`: clears route and remembered plan. Use when explicitly resetting a task/layout; it does not replace a fresh camera observation.
- `POST /api/capture {}`: refresh/reset the staged view when idle.
- `POST /api/ask {"target":"<visible destination and constraints>"}`: stationary GPT planning and preview, no driving.
- `POST /api/mission {"target":"<visible destination>","seconds":8}`: recurrent GPT approach; `seconds` bounds each local motion slice, not total mission duration. Without `seconds`, makes one look-and-drive attempt. Internal 30 cm commitment may end a one-look attempt short of the destination.
- `POST /api/turn {"degrees":90}`: local IMU turn; right positive, left negative. Review the swept footprint first. Handler permits up to 180 degrees; that is not an assurance of clearance.
- `POST /api/halt {}`: cancels the bench worker and stops motors. Verify `running:false` and service motor outputs `[0,0]`; an HTTP response alone is not confirmation.
- `GET /api/step?i=4`: retained frame from the current mission. Useful for finding a target noticed during an approach, but reacquire it in a fresh view before a new handoff.

**Avoid `/api/search`:** it assigns search strategy to the mid-level stack, contrary to this user's required division of responsibility. **Do not drive `/api/run` unreviewed:** this revision's manual path omits displayed obstacles when calling `fetch.follow`. `/api/flow` exists but its overlapping in-flight plans are experimental; it is not the default for this workflow.

### Known limitations to check

- `MISSION_STANDOFF_CM` is 20 cm. A number in the natural-language instruction does not configure this constant. Repeated model contact estimates can vary; do not promise exact stopping distance.
- The implementation contains `stop_short(...) or previous_route` fallbacks. An empty safe route can therefore revive an earlier route; inspect this before relying on close-range autonomous termination.
- `fetch.avoid` excludes some goal-adjacent obstacles; it is not a guarantee that blocks beside the target remain protected. Independently inspect these hazards.
- Current `CORRIDOR_HALF_CM` is 9 cm (3 cm margin beyond half-width), below the stored 5 cm user footprint margin. Account for the full required margin in route review; do not claim the current collision checker enforces it.
- Pivot calibration is `calibration/turn_pivot.json` (approximately x=1.12, z=-7.28 cm, uncertainty 2 cm). Older footprint/skill text says unmeasured; use the current calibration with uncertainty, not an assumed lens pivot.
- Optical-flow distance can fall back to timing. Report the visual sample fraction and distinguish estimated pose from independent ground truth.

## Recording lifecycle

Check `mountpoint /mnt/robotlogs`, actual write success, and available space. A mounted volume can still fail with I/O errors. Keep all runtime output on USB; no per-record `fsync`. Use unique filenames.

Start ffmpeg detached with stdin disabled and stdout/stderr redirected to a USB log:

```text
ffmpeg -nostdin -hide_banner -loglevel warning \
  -use_wallclock_as_timestamps 1 \
  -i http://127.0.0.1:8770/api/stream.mjpg \
  -map 0:v:0 -c:v copy -cluster_time_limit 2000 \
  /mnt/robotlogs/recordings/jetbot-<run-id>.mkv
```

Use Python `subprocess.Popen(..., start_new_session=True)` to detach. Save PID, exact path, source, and start time to `/mnt/robotlogs/recordings/active-recording.json`. Before reusing an existing recording, verify `/proc/<pid>/cmdline` matches ffmpeg AND that exact path. Verify process liveness and file growth over several seconds. Do not start duplicates or stop unrelated recorders.

At completion/cancellation, validate PID and path again, send `SIGINT` to that ffmpeg process, and wait for exit so Matroska finalizes. Confirm final size and use `ffprobe` or decode a short clip. Log stop timestamp and result. Do not use broad `pkill ffmpeg`. If the process does not stop, report and resolve rather than claiming finalization. Leave the viewer available.

MKV is the original; the port-8771 viewer transcodes replay into browser-compatible fragmented MP4 without modifying it. Its Replay endpoint is `/play/<filename>` and original download is `/download/<filename>`.

## Revised planner response

The prompt is maintained at `local_nav/prompts/trajectory-20260920.txt`; the previous version is preserved beside it as `trajectory-20260920-baseline.txt`. The response includes `motion: follow|turn|hold`. A hold raises `fetch.PlannerHold` before any route fallback; the dashboard reports PLANNER HOLD and ends without claiming arrival. Follow suppresses simultaneous turn instructions; turn suppresses simultaneous drive points. `all_done` means the current destination is the final step, not that the robot has already arrived. The controller still verifies range.

## Per-run audit

Create `/mnt/robotlogs/goals/<unique-run-id>/` with `commands.jsonl`, `events.jsonl`, append-only `mission-history.md`, and newest-first `mission.md`. Store photographs there, plus links to the recording. Log commands before dispatch and responses afterward; include errors and stop outcomes. Record meaningful new status, not unlimited duplicate sensor snapshots.

The existing `scripts/planner_audit_command.py`, `record_planner_mission.py`, `render_mission_log.py`, and `watch_approach_range.py` default to the old 2026-09-20 Advil run. `planner_audit_command.py` and `watch_approach_range.py` now accept `JETBOT_AUDIT_ROOT`; the observer also accepts `JETBOT_APPROACH_MARKER` and `JETBOT_STOP_RANGE_CM`. Always specify the current run. `mission_log_server.py` on port 8772 follows `/mnt/robotlogs/current-search.json` and regenerates newest-first Markdown every two seconds. It only displays events that are actually written; it does not itself capture every planner response. Preserve old evidence. The range observer reads model text, not an independent range sensor, and can miss changes before a controller acts.

Keep newest-first output current after each event with atomic replacement; retain the chronological source. Check any background logger after tool or conversation interruptions: earlier sessions ended unexpectedly. Do not claim complete logs when only snapshots are available.

## Verification before handoff

For a skill-only edit, validate frontmatter and referenced paths; do not move the robot to test documentation. For changed execution/recording code, test actual operator paths at the highest safe level available, including stop/failure reporting. Separate replay/simulation evidence from powered hardware evidence. The 2026-09-20 session demonstrated an Advil approach and the recording viewer, not a guaranteed unattended end-to-end search system.
