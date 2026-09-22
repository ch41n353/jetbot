# JetBot user preferences

- Before the first robot movement of every run, give the user the live progress
  log URL (normally http://192.168.86.158:8772/). Verify it follows the current
  run; if unavailable, state that clearly and provide the available log path.
  Do not wait until the run finishes to share the link.

- On 2026-09-20, the user requested a few bounded retries for recording-storage
  failures before reporting a blocker. Retry an actual small read/write check
  three times with a short delay; verify recording growth after recovery. If
  failures persist, report the current error without claiming recording works.

- On 2026-09-20, the user requested automatic camera recording for every
  find/search task. Start recording before the first search motion; keep it
  running across viewpoint changes and planner handoffs. Stop and finalize the
  recording when the requested object is found or the user cancels the search.
  A request to approach after identification does not extend recording unless
  the user asks. Save recordings on the mounted USB volume /mnt/robotlogs,
  never fall back to the SD card. Verify recording actually starts and report
  storage/recording failures honestly. Include the recording path in the result.

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
