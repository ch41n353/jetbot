# Offline controller validation

All commands here read recorded files or generate synthetic frames. They never
start the camera/IMU service or issue hardware motor commands. Keep real robot
operation separate; the current user instruction prohibits movement.

## Production-controller simulation

```bash
/usr/bin/python3 local_nav/simulate_route.py --cases 60 \
  --output local_nav/goals/route-simulation-2026-09-14.json
/usr/bin/python3 local_nav/evaluate_braking_candidate.py --cases 16 \
  --output local_nav/goals/braking-candidate-holdout-2026-09-14.json
```

The simulator executes `route_executor.execute`, `FloorTracker.motion` and
`PlanarState.update`. Only socket calls, clock, camera generation and IMU attitude
are replaced in scoped mocks. Carpet is rendered through the calibrated fisheye
model, then processed by the real tracker. The simulated motor plant has delayed
acceleration/coasting and an independently expiring 200 ms command lease.
Reported ground truth comes from the plant, not the tracker's position estimate.

The main paired comparison varies speed, coasting time constant, frame frequency,
acquisition delay and image noise under a fixed seed. The candidate uses a separate
seed and an 80 ms coast hypothesis rather than 40 ms. It remains an offline
hypothesis; it does not alter the production braking function.

Baseline simulation assumptions: flat static floor, fixed height/tilt, perfect
synthetic IMU yaw, no wheel slip/contact/cable, no unexpected obstacles. The random
parameter ranges are hypotheses, not measured physical distributions. A 5 cm goal
is the only target in the current paired campaign. Do not extrapolate success
rates to longer routes or physical navigation.

## Recorded-frame timing

```bash
/usr/bin/python3 local_nav/benchmark_tracker.py OBSERVATION_DIRECTORY \
  --threads 1 2 4 --output /tmp/tracker-times.json
/usr/bin/python3 local_nav/evaluate_feature_budget.py OBSERVATION_DIRECTORY \
  --output /tmp/feature-budget-times.json
```

These load frames and IMU attitude before timing. They benchmark the tracker,
not camera acquisition, service communication, logging, motor control, or model
planning latency. Do not run timing comparisons alongside other CPU-heavy work.
The feature-budget experiment changes the corner count inside a scoped mock;
production still uses 250 features unless explicitly changed after validation.

`route-simulation-2026-09-14.manifest.json` records the hashes and parameter ranges
for the main comparison. Preserve completed output and failed trials. JSON outputs
are replaced atomically as paired cases complete; `complete: false` is partial
progress, not a finished campaign.

## Stress scenarios and faster tracking option

```bash
/usr/bin/python3 local_nav/simulation_stress.py \
  --output local_nav/goals/simulation-stress-after-2026-09-14.json
```

This adds 10/15 cm goals, slow processing, camera freezes, read noise, exposure
steps, wrong camera height, IMU drift and explicit fault cases. Each runs with
250 and 125 features. `run_case(..., feature_budget=125, target_cm=10)` is also
available from Python. The simulated compute delay is kept equal between feature
budgets to isolate tracking quality; real timing is measured separately.

`route_executor.py --feature-budget 125` exposes the reduced budget for later
supervised physical validation. The default remains 250. Do not treat a simulated
clear rectangle or generated carpet image as authorization or evidence for a
physical route.
