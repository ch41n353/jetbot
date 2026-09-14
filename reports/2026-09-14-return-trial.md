# Live reposition-and-return test — 2026-09-14

The robot reversed away from the Advil and returned using the local approach
batch. The final return executed two segments with **zero intermediate planner
calls** in **4.023 seconds command-to-result** (3.247 seconds inside the batch
controller, 1.615 seconds powered). It measured 21.309 cm against a 19 cm plan
and correctly reported **stopped: Batch overshot final distance**. This proves
local segment continuation on hardware, not accurate waypoint arrival.
Final camera projection placed the bottle base roughly 17–20 cm ahead, compared
with roughly 14–16 cm before repositioning. It returned near the bottle, not to
an independently measured original pose. Motion service shut down normally.

## Actual trials and fixes

1. Added a bounded reverse option to the existing route executor to reposition
   without turning near the bottle. Retained clearance came from the preceding
   traversed carpet corridor; original/current floor-image registration differed
   by only 0.0066 cm. Front imagery alone does not prove rear clearance.
   Reverse requests of 15 and 10 cm measured 15.462 and 10.519 cm, respectively.
2. First 19 cm batch completed its first segment at 15.729 cm, then stopped
   before powering the second segment: `Starting pose differs from calibrated
   floor pose`. Reinitialization had selected a recent braking acceleration
   sample with apparent inclination 4.919°, beyond the 4° initialization guard.
   Replaying the same recorded history through a continuous estimator passed.
3. Preserved one attitude timeline and original tilt reference across the batch.
   No calibration/tilt guards were relaxed. Reversed 15.736 cm for a retry.
4. The retry stopped after its first segment at 16.438 cm because it treated an
   internal 15 cm marker as the final goal. Changed batch handling to accept a
   verified stopped position at an internal marker and recompute remaining
   distance, while retaining all corridor, uncertainty and final-goal checks.
5. Reversed 15.555 cm for the final retry. The batch then executed 15.970 cm and
   a measured 5.294 cm against its remaining 3.030 cm request. Transforming both
   displacements gives x=1.013 cm, z=21.309 cm and yaw=−3.277°. Final overshoot
   2.309 cm caused the expected stop and planner-return result. No further motor
   commands were issued to chase stopping precision.

The camera/IMU distances are estimates, not external measurements. This is one
completed two-segment physical return after two failed attempts. It does not
establish statistical reliability, exact return-to-start localization, general
autonomous obstacle avoidance, or continuous motion through segment boundaries.
Static obstacle maps still require planner inspection; dynamic recognition is
absent. The local tool stops/observes between segments, but does not ask the VLM.

Evidence: `local_nav/goals/session-20260914-131650-3b2978*`; final batch is
`...-route-07.json`, with segment logs alongside it. Compact metrics:
`reports/2026-09-14-return-trial.json`. Reverse simulation:
`reports/2026-09-14-reverse-validation.json`. Tests cover reverse rear clearance,
wrong-direction rejection, one IMU timeline per batch, internal overshoot
continuation, final overshoot rejection, cancellation and external repositioning.

Final verification: 77 tests passed, skill validation passed, and the control
socket was absent after normal session shutdown. Updated code, failed trials,
final return and dated skill revision are preserved together in Git.
