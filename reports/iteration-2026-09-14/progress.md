# Three-hour navigation session — 2026-09-14

Requested interval: 20:38:28–23:38:28 UTC (13:38:28–16:38:28 Pacific).
This session is active; do not mark the three-hour request complete early.

Initial movement backed away from Advil but stopped at a lateral guard. A 90°
right turn initially failed stationary IMU startup, then completed at 90.358°
after a tested quiet-window initializer. The robot approached the white foam
blocks and stopped approximately 21–22 cm from their corner (image projection).
Controller runs included tracking/pose failures, so proximity is not a clean
controller-arrival claim.

Turned 60.312° left, then 8° and 5° left to approach the long white box while
clearing the right-side blocks/door panel. A drive stopped at estimated lateral
error 1.018 cm and forward progress 12.294 cm. Replay confirmed the footprint
still fitted its checked corridor. Replaced independent lateral thresholds with
an analytic expanded-body sweep check; retained heading, fore/aft overrun,
clearance, uncertainty, motor and sensor limits. Added rejected-pose diagnostics.

A quiet stationary initialization path requires >=150 ms of samples, >=6
measurements, per-axis acceleration std <=0.12 m/s², mean norm within0.35m/s²
of gravity, gyro <=1.5°/s, and initial inclination <=5°. Direct single-sample
feed retains the old4°cap. Added initialization metadata and preserved it and
older IMU history when duplicate camera-frame reads overwrite observation logs.
82 tests passed at this checkpoint. No published skill update or Git commit yet.

Latest fresh scene: a person appeared on the left and Advil was moved from its
retained offscreen location into the ahead-left area. Old obstacle position is
invalid; holding and reassessing before another drive. Current target is the
long white box; Coke remains a later candidate.

## 14:30 Pacific — model-call reduction

White-box approach completed at estimated 16.048 cm travel on a 16 cm batch goal,
in 3.024 seconds command-to-result. User then correctly identified continued
manual route decisions as the bottleneck. Suspended manual tiny-step navigation.

Implemented spatial_planner.py, spatial_executor.py, spatial_turn.py and
simulate_spatial.py. Search chooses forward/reverse drives and turns in an
inspected static map. Executor retains and rechecks route suffixes using actual
VIO pose, replanning locally on conflict or endpoint drift. Initial fresh-search
every-step policy oscillated forward/backward in simulation; preserved failure
in spatial-first.json and fixed route retention. Nominal turn pivot is a search
seed only; measured translation carries the map across turns. Turn collision
envelopes cover fixed pivots anywhere inside the chassis and angular overrun.

Simulation spatial-second.json reached [30,30] cm using production search, drive,
turn and visual estimator: true error 0.812 cm, about14 simulated seconds, no
intermediate model calls. IMU attitude is simulated directly and obstacles are
not rendered; this is not dynamic obstacle avoidance or hardware IMU validation.
93 tests passed before adding multi-target queues.

Live new supervisor session-20260914-142321-17f02c, route01, autonomously executed
two reverse10 cm actions toward [0,-20]. Final estimated pose
[1.294,-19.704,-1.676deg], target error1.328 cm. 4.422 seconds command-to-result,
3.566 seconds inside executor, 0.0024 seconds search, zero intermediate model
requests. Service remained available, motors stopped. Robot is now back from
white box, roughly40cm away. Fresh capture1134068.915251.

Adding an optional queue of up to four targets in one retained map, so an
approach-and-return mission need not ask the high-level model at either end.
All actions retain existing powered limits; mission max12 actions/60sec/90cm.

## Battery depletion and protection

The user reported an empty battery. The service was no longer running, and the
robot's monotonic clock had reset after a reboot. Sent an explicit hardware
motor stop. All earlier map anchors/control generations are invalid for a new
live mission. No further powered navigation was performed after this report.

Added read-only INA3221 input-voltage monitoring in battery.py, sampled about5Hz
in a separate worker and checked by the motor watchdog independently of the
planner. Thresholds are experiment policy: warn4.9V, latched stop below4.8V or
above5.5V, invalid/missing telemetry, or sample age>0.6s. Three good startup
samples required. Sensor shared-lock acquisition is bounded so a failed reader
cannot block the motor watchdog. The supervisor reports power state changes,
refuses new routes when blocked, and terminates an active child on a power stop.

Power is read at bus6/address0x40, channel1, confirmed POM_5V_IN from device tree,
with INA3221 chip IDs and continuous conversion verified. No sensor configuration
or hardware power thresholds are written. The root-protected sysfs values were
not readable without a password; the already-authorized I2C register interface
works without changing permissions. Actual supply readings were approximately
5.00–5.08V during stationary/disarmed checks. Service power status and raw samples
are saved in power-service-status.json and power-service-disarmed.jsonl. Hardware
service showed healthy camera/IMU, motion_enabled=false, motor output[0,0], and
shut down cleanly. Fake-motor process tests prove stopping and no restart after
voltage recovery; the full suite passed101 tests.

This measures the regulated Jetson input, not remaining battery charge. A power
bank can maintain5V until cutoff; a separate motor battery would be unobserved.
Asked the user for the power-bank/battery-board model; actual fuel-gauge telemetry
is still unavailable. Earlier motor experiments have no voltage history, so
low battery cannot be retrospectively separated from software/geometry effects.

Offline work continues within the requested three-hour window. Live navigation
is paused following the reported empty battery and reboot.
