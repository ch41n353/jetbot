# Camera/IMU navigation experiment results

## Changes and evidence

The repeated 3.1–3.8% floor-scale failures were reduced by using averaged gravity
after a verified stationary interval. The controller now requires quiet gravity
on both sides of a floor-motion pair rather than mixing corrected and integrated
attitudes. It waits briefly with motors stopped when the IMU is still noisy.
The original 3% floor-scale check remains unchanged.

In `advil-settled-fix-03b.json`, all 15 accepted frames used quiet gravity,
the scale range was 1.0014–1.0208, and the robot completed about 18 cm of forward
progress in 10.65 seconds. It stopped at its requested conservative waypoint
distance. The subsequent manually selected bottle-base pixel gives approximately
13.6 cm forward clearance from the camera, as recorded in `advil-arrival.json`.

An earlier test lost tracking when a person's foot crossed the floor region.
That stop was preserved. Another test showed that falling back to integrated
attitude for a noisy frame still caused scale failure; quiet-frame selection
addresses that case. Service-disconnection errors are now retained in logs
without obscuring the original failure with a cleanup traceback.

## Turns

A new bounded turn controller uses IMU yaw and 60 ms motor pulses. Each invocation
is limited to 30 degrees and 12 seconds. Initial 0.18 motor power exceeded the
90 degrees/s guard and stopped; 0.12 power completed six subsequent turns.

| Run | IMU yaw | Camera-only yaw |
| --- | ---: | ---: |
| turn-left-02 | -15.45° | -15.78° |
| turn-scan-03b | -29.79° | -29.36° |
| turn-scan-04 | -28.97° | -27.16° |
| turn-scan-05 | -29.55° | -28.35° |
| turn-scan-06 | -28.64° | -28.70° |
| turn-rover-align | -11.65° | -11.49° |

Camera-only yaw was computed offline with `FloorTracker.motion` without IMU
attitudes, using the calibrated static floor plane. Every pair passed the
existing visual checks. These estimates provide cross-sensor consistency,
not an external ground-truth angle measurement.

## Second object

The scan exposed a clear frontal approach to a small wheeled robot. The first
waypoint timed out after about 16.6 cm of progress, with no scale failure. A
second run completed its requested 15 cm waypoint standoff in 19.65 seconds;
a third completed its 20 cm waypoint standoff in 10.57 seconds. These are
distances to intermediate floor waypoints, not distances to the object.

The final camera image (`rover-arrival.jpg`) shows a loose component and wire
in front of the wheeled robot. The selected wheel-contact pixel projects to
approximately [5.5, 18.3] cm (right, forward), and the component to [-4.2, 17.4]
cm. The robot stopped there and motor control was shut down.

## Validation and remaining limits

37 unit tests pass, including quiet-gravity correction, rejection of nonquiet
frames, excessive disagreement, turn direction and overshoot, motor leases,
known visual translation, exposure changes, and tilt compensation.

No external ruler or motion-capture measurement was available. Near-object
distances use manually selected image pixels and calibrated geometry; the
original measured floor-distance validation covered the 30–50 cm range.
Camera/IMU time offset and systematic mounting errors remain uncalibrated.
Obstacle selection and clearance inspection were performed by the high-level
assistant between runs; these controllers do not implement general obstacle
avoidance. A tracking stop when someone entered the scene does not validate
reliable person detection or collision avoidance.

The new correction resolved the observed failure in a completed live approach
and subsequent waypoint runs, but does not guarantee that scale errors can
never recur on other surfaces or with different mounting or lighting.
