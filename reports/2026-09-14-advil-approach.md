# Advil approach and fewer planner calls — 2026-09-14

The robot approached the Advil bottle and stopped with approximately 14–16 cm
camera-to-base clearance from calibrated image projection. This is not independent
metric ground truth; near-field projection remains less validated than 30–50 cm.
The robot was explicitly confirmed unplugged and free to move. Both the motion
session and subsequent disarmed validation session shut down normally.

The approach used three 15 cm requests and a 17° left alignment turn. First drive:
braked at estimated 13.40 cm, then post-stop floor tracking rejected scale 1.038;
final odometry was unverified. A fresh image was used to replan. The turn reported
−17.57° in 2.57 seconds. Subsequent drives settled at estimated 15.42 cm and 15.95 cm,
with 1.21 and 1.11 seconds powered. Each new high-level segment was previewed.
The turn preview conservatively enumerated pivots throughout the chassis and
angles through 22° (requested turn plus guard allowance), adding 7 cm clearance
and uncertainty. This is not an axle calibration.

First capture to final capture was **219.02 seconds**, while the three drive
commands together took 7.95 seconds and the turn controller 2.57 seconds. This
includes one-time preview preparation and planner/tool latency, not a steady-state
benchmark. It identifies high-level orchestration as the dominant delay in this
run; no claim that the robot completed the whole approach quickly.

Revision 6 therefore adds object approach preparation directly to the persistent
session and an opt-in local approach batch. The planner chooses one object base
point and inspects the whole static corridor once. The local tool can execute up
to 30 cm using existing bounded segments without model calls between them. It
retains the initial obstacle map, verifies each transformed swept rectangle,
accumulates uncertainty, and returns on completion or exception. It does not
recognize new obstacles or autonomously turn/recover. See `local_nav/SESSIONS.md`.

The single-plan helper was exercised on the actual final bottle image with motors
disarmed and returned `within_standoff_band_estimate`, issuing no movement.
Batch validation uses the production controller, rendered fisheye carpet and
independent simulated motor dynamics. Full batch physical validation is pending;
the robot was already at the bottle when the batch was implemented. No live
batch speedup has been measured. Structured simulation results include success
and failure cases in `2026-09-14-batch-validation.json`.

Live source logs: `local_nav/goals/session-20260914-125122-236639*` and
`local_nav/goals/advil-approach-20260914-turn-01.json`. Summary metrics:
`2026-09-14-advil-approach.json`.

Final validation: 75 tests passed. Nine additional production-controller batch
simulations exercised five motor settings, external cancellation between segments,
sensor failure, stall, and a one-centimeter uncommanded repositioning between
segments. Four normal settings completed; the high-coasting setting and all four
fault cases stopped. Continuation uses the exact verified settled frame so
registration can reject a moved robot. These simulations do not model dynamic
obstacle recognition or prove real-world 30 cm batch safety.
