# Straight route batches — 2026-09-13, revision 2

`route_executor.py` consumes a planner-supplied static map and passes increasing
straight waypoints without stopping or making model calls between them. It uses
camera/IMU odometry for forward displacement (rather than accumulated path length),
heading correction and corridor checks. The service retains motor output between
valid updates. No automatic obstacle detection or turn execution is implemented.

A JSON plan uses centimetres in a fixed frame anchored at the initial camera lens,
x right and z forward:

```json
{
  "waypoints_cm": [5, 10, 15],
  "inspected_free_rectangle_cm": [-20, -30, 20, 30],
  "obstacle_rectangles_cm": [],
  "session_id": "REPLACE_FROM_LIVE_STATUS",
  "control_epoch": 0,
  "captured_monotonic": 0,
  "image_path": "/absolute/path/to/saved-fresh-preview-image.jpg"
}
```

**The example free rectangle is illustrative, not an observation of this room.**
The planner must inspect the full region, including offscreen side/rear space,
retain all known obstacles as bounding rectangles (with their location uncertainty),
and show the swept route over a fresh image before execution. Never fill unknown
space as free. Save a separate copy of a successful snapshot, its `time`,
`session_id` and `control_epoch`; `latest.jpg` is overwritten by subsequent frames.
A plan expires after 120 seconds, any intervening motor command/stop from legacy
clients, or a service restart. Floor-image registration also rejects changed starts.
Neither registration nor a timestamp establishes that objects remained stationary.

```bash
/usr/bin/python3 local_nav/route_executor.py /absolute/path/plan.json
/usr/bin/python3 local_nav/route_executor.py /absolute/path/plan.json \
  --execute --log /absolute/path/new-run.json
```

Default only checks static geometry and never opens the robot socket. Live execution
requires an already enabled service. Keep the existing two-terminal service/command
workflow. Use a fresh log path. Total route remains 1–15 cm and at most two seconds
powered. This removes intermediate planning stops; it does not increase motor power
or establish a measured speed improvement yet.

The conservative swept rectangle includes the 12x15 cm body, 5 cm clearance,
2 cm position allowance, 2 cm heading allowance, 1 cm lateral tracking allowance
and 4 cm braking allowance. Measured lateral error >1 cm, yaw >5 degrees, bad
tracking/IMU, slow updates, loss of progress, or deadline stops execution. The
braking allowance and estimator uncertainty are experimental, not physical
collision guarantees. No new obstacles are detected during execution.

The service returns cancellation tokens on responses. The new executor includes
them on every renewal. `stop`, errors and legacy motor commands invalidate the
token; stale renewals are rejected. Older scripts do not participate and can still
renew after `stop`; use `shutdown` to cancel those. Independent 200 ms motor leases
remain in force.

Validation: 48 offline tests, including a simulated three-waypoint run with no
intermediate stop, obstacle/unknown-space rejection, stalled motion and token
cancellation. Live stationary preflight: estimated drift 0.0274 cm; no powered
trial of this executor yet. Turn batches await a measured axle offset and a
separately tested turning envelope/controller.
