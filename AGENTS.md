# JetBot user preferences

- On 2026-09-15, the user instructed that future requests to start robot
  navigation/search imply roughly 35 cm of clear floor all around the robot
  for the initial scan. Treat this as user-supplied starting clearance; do not
  repeatedly ask for confirmation or ask the user to locate the search target.
- This starting-clearance assumption does not establish clearance along later
  travel routes. Inspect fresh vision and sensor health, retain observed
  obstacles, and stop if observations contradict the assumption. Preserve
  controller, power, and collision checks.
- Before handing a navigation feature or dashboard change to the user, test the
  actual operator path end to end at the highest safe level available. Reproduce
  recorded failures, exercise dashboard-generated plans through the same planner
  and controller entry points, and verify that success and failure are reported
  accurately. Unit tests alone are insufficient when an offline replay or bounded
  powered acceptance test is available. State clearly which parts were tested in
  simulation/replay and which were tested on hardware.
