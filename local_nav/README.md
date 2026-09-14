# Local JetBot sensing and motor watchdog

Run with the system Python; the motor worker loads the existing JetBot virtual
environment's motor dependencies. Camera and IMU have separate persistent workers.
No network camera server is started: the API is a Unix socket in a private 0700
directory. Do not run old camera or motor scripts concurrently with this service.

```bash
/usr/bin/python3 /home/jetbot/jetbot/local_nav/service.py
/usr/bin/python3 /home/jetbot/jetbot/local_nav/client.py status
/usr/bin/python3 /home/jetbot/jetbot/local_nav/client.py snapshot
/usr/bin/python3 /home/jetbot/jetbot/local_nav/client.py stop
/usr/bin/python3 /home/jetbot/jetbot/local_nav/client.py shutdown
```

Default mode rejects ALL motor movement. This is the mode used for live testing.
Camera capture settles for three seconds, then continues at 640x480. It uses the
existing color profile if installed. Snapshot requests reuse the latest frame.
The BNO055 at bus 0 / address 0x28 runs in NDOF mode; graceful shutdown restores its
original mode. Status includes calibration (system, gyro, accel, mag), errors,
sequence counters, monotonic sample times, and sample ages. Timestamps represent
host acquisition completion, not hardware-synchronized exposure/sample times.

## Motor lease interface (for the future local planner)

`--enable-motion` explicitly permits movement. A local client may then send one
newline-terminated JSON request per socket connection:

```json
{"action":"motors","left":0.2,"right":0.2}
```

Each accepted request grants only a 200 ms lease. Speeds are bounded to +/-0.3.
The independent motor process rechecks deadlines every 10 ms; no commands means
stop. Camera older than 500 ms or IMU older than 200 ms blocks movement; stale
samples also invalidate an existing lease. Sensor errors block new commands.
Invalid requests revoke the existing lease. `stop` revokes it immediately; the
response acknowledges submission, and the status output is updated asynchronously.
Sensor updates alone never renew motion. A main-process stall/crash lets the
lease expire. SIGTERM/SIGINT and normal watchdog exit stop the motors.

This is a software watchdog, not a hardware motor cutoff: killing the motor
process with SIGKILL, OS failure, or a wedged I2C bus cannot be guaranteed safe.
Other programs can bypass this watchdog if they directly drive the motor board.
No collision detection, autonomous planner, heading controller, floor calibration,
or reboot-start service is included yet. Do not treat fresh sensors as proof of
an obstacle-free path. Compass calibration is currently incomplete.

## Validation

```bash
/usr/bin/python3 -m unittest discover -s /home/jetbot/jetbot/tests
```

Tests cover malformed/expired leases, stale sensors, disabled motion and an actual
watchdog subprocess with a fake motor: it stops after commands cease. Live testing
uses disabled movement only. Initial measurement: about 30 camera frames/s and
42 IMU samples/s. Physical motor stopping under motion remains to be tested in a
controlled test with the wheels lifted before enabling driving.

## Experimental floor-point controller

`point_controller.py` projects a near-center pixel through the calibrated fisheye
model and measured floor geometry. It tracks carpet features using forward/backward
optical flow and fits floor motion with RANSAC. The IMU guards against excessive
rotation or gross tilt; steering and distance use visual estimates. Initial goals
are limited to 30–40 cm ahead and within 5 cm of the centerline, on an operator-
inspected clear path. It has no obstacle detector and must not be used unattended.

```bash
# Stationary tracking preflight, no movement:
/usr/bin/python3 /home/jetbot/jetbot/local_nav/point_controller.py
# Requires motion-enabled service and a freshly inspected clear path:
/usr/bin/python3 /home/jetbot/jetbot/local_nav/point_controller.py --execute
```

The experiment uses 100 ms movement pulses followed by stopping and settling.
Guard limits include tracking inlier quality, apparent floor scale, tilt, maximum
travel, heading deviation, timeout, and stale sensors. Completion means estimated
camera position within 10 cm of the goal, not exact chassis-center arrival.
Logs include starting frame, tracked goal, estimated motion, and stop reason.

Live test: initial continuous attempt aborted on apparent floor scale change.
Stop-and-observe run progressed approximately 15 cm toward a 35 cm goal, then
stopped with approximately 20 cm remaining because the floor scale changed to
0.962 (outside the retained 0.97–1.03 limit). Goal arrival has NOT been validated.
Do not remove this guard to force completion. Next work is distinguishing chassis
pitch changes from optical tracking error, with recorded synchronized observations.

## IMU + camera estimator

The point controller now requires `calibration/imu_mount.json` with a verified
sensor-to-camera rotation. It uses gyro propagation of gravity direction and yaw,
slow accelerometer tilt correction during quiet periods, frame-time interpolation,
and IMU-predicted feature motion. Floor rays use the current measured gravity plane
instead of a fixed pitch. Visual yaw and gyro yaw are fused by variance weighting;
translation still comes from floor tracking. The estimator reports planar position,
velocity, heading and accumulated uncertainty. It does not integrate acceleration
into position, use uncalibrated compass heading, or constitute a full SLAM/VIO EKF.

The `observation` Unix-socket request returns the selected frame (base64 JPEG), its
acquisition timestamp, and recent IMU samples. These are coherent payloads, not a
file that another client may replace. Timing uses monotonic host acquisition times,
not hardware synchronization; camera/IMU latency offset remains uncalibrated.
Interpolation requires samples around the frame time; extrapolation is limited to
35 ms, and gaps over 80 ms stop execution.

### Mount alignment (motors must remain disabled)

Keep the IMU rigidly attached throughout. With the chassis flat on the carpet:

```bash
/usr/bin/python3 /home/jetbot/jetbot/local_nav/calibrate_imu_mount.py level
```

Lift only the front 15–25 degrees, with no sideways tilt. Hold still, then run:

```bash
/usr/bin/python3 /home/jetbot/jetbot/local_nav/calibrate_imu_mount.py nose-up
```

Set the robot back on its original level surface, hold still, then run:

```bash
/usr/bin/python3 /home/jetbot/jetbot/local_nav/calibrate_imu_mount.py verify-level
```

This identifies up and forward in IMU coordinates and measures stationary gyro
bias. It assumes that the instructed movement really is nose-up without roll,
that the camera has the previously measured mounting pitch and zero nominal roll,
and that both mounts are rigid. Return-level validation checks repeatability,
not all extrinsic errors. Do not reuse the profile after remounting either sensor.
The unverified/incomplete profile blocks the point controller.

### Verification and limits

Tests include known gyro rotations, frame-time interpolation, missing history,
invalid mounting calibration, visual/gyro disagreement, planar position updates,
and synthetic forward movement combined with a two-degree pitch change. The
original visual scale, inlier, residual, speed, distance and motor watchdog checks
remain in place. Compensation does not account for changes in camera HEIGHT from
chassis bounce or IMU/camera translational lever arms. Full obstacle avoidance is
still absent. First run stationary preflight after mount calibration, then a short
supervised test; successful simulated compensation alone is not a physical result.

### Continued physical experiments

Runs `goals/fused-live-04.json` through `fused-live-06.json` tested stronger
bearing correction (gain 0.30, differential capped at 0.05), attitude-based initial
goal projection, and settling intervals. The initial measured target is checked
against the same 30–40 cm corridor. Commands and raw observations are recorded.
`--settle-seconds` accepts 0.25–0.5 seconds; the default remains 0.25.

With 0.5 seconds settling, run 05 tracked approximately 19 cm and stopped with
11.8 cm remaining on lost floor features. A global exposure offset caused the
failure: subtracting each floor region's mean brightness recovered 233 inliers
in offline replay, scale 0.9998 and residual 0.034 cm. Tracking now applies that
normalization; synthetic known-motion verification also covers brightness changes.
Run 06 tracked approximately 18.6 cm but stopped with 14.1 cm remaining on the
unchanged 0.9-second observation interval limit. Longer settling leaves little
timing margin; next evaluate a 0.4-second interval on a fresh clear corridor.
Neither run establishes arrival within 10 cm. Accumulated uncertainty estimates
exclude systematic calibration error. Physical repositioning is needed before
further forward tests because nearby objects now border the corridor.

After repositioning, run `goals/fused-live-07.json` with 0.4-second settling
completed in 16.74 seconds: initial target 32.23 cm forward, final target
[-0.36, 9.36] cm. The controller reported `within_10cm_visual_estimate` and
motor control was shut down afterward. This is an estimated arrival from the
camera/IMU controller, not an independently measured position accuracy result.

The attitude estimator skips transient acceleration outside 7–13 m/s² as a
gravity reference, but stops for sustained excursions (80 ms) or severe values
outside 3–20 m/s². Gyro propagation continues during rejected gravity updates.
This accommodates recorded brief braking spikes without treating them as tilt.

### Settled floor attitude correction

The stop-and-observe controller uses `settled_at` only for observations taken
after motors have stopped. A preceding 180 ms IMU window must contain at least
six samples spanning 120 ms, gyro rates below 1.5 degrees/s, acceleration axis
standard deviations below 0.12 m/s², and gravity magnitude within 0.35 m/s² of
9.80665. Its averaged gravity replaces the frame's pitch/roll direction, leaving
integrated yaw unchanged. A discrepancy over two degrees stops the controller.
The point controller requires quiet estimates on both sides of every motion
pair, waiting up to another 180 ms with motors stopped for a quiet window. It
stops if none arrives, and retains the original 900 ms frame interval limit.
This avoids mixing gravity-corrected and uncorrected floor planes. Do not use this method
for a continuous-drive controller without a verified stationary interval.

The failed final pair from `advil-approach-03.json.observations` replays at scale
1.0146 (previously 1.031), with 238 inliers and 0.098 cm residual. The original
0.97–1.03 scale guard is unchanged. Tests cover quiet tilt correction, rejection
of moving windows, and stopping on large gravity disagreement. This addresses
settled tilt error; hardware camera/IMU latency remains uncalibrated.

Live validation `advil-settled-fix-03b.json` completed about 18 cm of forward
progress in 10.65 seconds. All 15 accepted observations used quiet gravity;
scale stayed between 1.0014 and 1.0208. The requested conservative waypoint
standoff was 15 cm; the final estimate was 14.98 cm. A subsequent image placed
the Advil base approximately 13.6 cm ahead of the camera. Neither estimate is an
independent ground-truth measurement. `--stop-distance-cm` permits 10–20 cm only.
An earlier test stopped when a person's foot crossed the tracked floor region;
this was a valid tracking stop, not a reason to weaken feature checks.

### Bounded scan turns

`turn_controller.py --degrees -15 --execute --log PATH` turns left, positive
angles turn right. Commands are limited to 30 degrees per invocation and a
12-second deadline. Inspect robot footprint clearance before each turn and the
new camera view afterward. The controller uses 60 ms differential pulses at
0.12 power with 180 ms stops, IMU yaw feedback, a 90 degrees/s rotation limit,
five-degree tilt guard, wrong-direction and overshoot guards, and sensor leases.
Omit `--execute` for stationary preflight. This is not obstacle avoidance.

Initial 0.18-power testing tripped the rotation-rate guard; lower-power testing
completed a requested 15-degree left pivot at -15.45 degrees estimated yaw.
All turn observations and stop reasons are recorded. The camera is used for
high-level inspection, not independent closed-loop turn-angle measurement.

Six completed low-power turns were checked afterward using camera-only floor
tracking (no IMU yaw supplied to the visual fit). Camera/IMU angle differences
were 0.06–1.81 degrees for commanded turns of 12, 15, and 30 degrees. This is a
cross-sensor consistency check, not external ground truth.

After scanning, three waypoint runs approached a second small wheeled robot:
`rover-approach-01` stopped at the retained 20-second timeout; runs 02 and 03
completed their conservative waypoint standoffs. The final visible wheel contact
was estimated at 18.3 cm forward, and a loose component at 17.4 cm forward.
Further approach was stopped due to that component and a visible wire. Motors
were disarmed. See `goals/experiment-report.md` and arrival images for evidence.

### Continuous experiment rolled back

At user request, continuous turning and drive experimentation were withdrawn.
The turn controller again uses 60 ms pulses and 180 ms settling, capped at 30
 degrees per invocation. The service again revokes the prior lease before each
motor command. The continuous probe is archived as `continuous_probe.py.disabled`;
its historical experiment logs are retained. `--continuous` is no longer accepted.
