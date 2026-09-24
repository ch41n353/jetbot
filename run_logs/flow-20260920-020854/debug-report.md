# FLOW cyan-block collision investigation — 2026-09-20

Robot motion is prohibited during this investigation. No powered verification or server reload was performed after the user went to sleep.

## Confirmed defect and fix

FLOW rebases stored route and obstacles into the current robot frame after every motion slice. When a delayed GPT answer arrives, build() correctly rebases that answer from its captured pose. FLOW then incorrectly applied that same captured-pose displacement to already-current obstacle memory. The fix merges current memory with current detections using zero additional displacement.

Motor-free regression executes the actual Bench.flow loop with delayed synthetic responses and a fake controller. An obstacle initially 30 cm ahead must remain 20 cm ahead after a 10 cm movement. The original implementation fails, producing 10 cm; the corrected implementation passes. Repeated duplicate transforms can eventually place an obstacle behind the robot, where merge_obstacles drops it.

## Evidence from the failed run

collision-progress.json contains the final live log. It reports cyan-block controller refusals, later plans without the cyan block in the final obstacle list, and a terminal PLANNER HOLD. The user reported physical contact. Ten retained request-view images, look-00.jpg through look-09.jpg, and camera.mkv were preserved on SD with authorization for this run. Motor output was verified [0,0] when evidence was collected.

The coordinate bug is confirmed, but its contribution to this specific contact is not conclusively reconstructed. The original run did not persist raw GPT responses, exact request timestamps, or complete FLOW controller pose events. Contact timing is not established by the text log alone.

## Other limitations found

- One-second FLOW setting is not an independent 1 Hz scheduler: request issuance shares a loop with synchronous follow() execution. This run issued 10 looks in approximately 30 seconds, about 0.33 requests/second overall. Four-in-flight cap and call latency also constrain scheduling.
- The log's applied count is latest sequence plus one, not the number of responses actually applied; it can overcount when answers are discarded. The wording “wheels never stop” is also not a verified continuity guarantee.
- Current collision corridor gives 3 cm margin beyond the half-width, below the requested 5 cm margin. Turn sweep and goal-adjacent obstacle exclusions remain separate concerns.
- Earlier FLOW hold compatibility fix is retained: PlannerHold interrupts following and ends the handoff instead of preserving the old route.

## Validation

Original-code regression: fails as expected (10 cm vs expected 20 cm).
Fixed actual FLOW-loop motor-free replay: passes.
Existing fetch tests: 76 passed.
Existing planner-page tests: 27 passed.
No claim of collision-free hardware operation. The newly corrected obstacle-memory code has not been loaded into the live server. No GitHub push performed.

## Next acceptance test

Before powered use, validate delayed-answer obstacle retention through additional translations/turns, capture request/response/pose traces, and verify a bounded obstacle course with the full footprint margin. Do not resume motion until the user authorizes it.
