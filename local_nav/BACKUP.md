# Navigation backup — 2026-09-13

This checkpoint preserves the experimental robot controllers, sensor/motor
service, camera color calibration, calibration profiles, tests, dated skill,
existing basic-motion notebook edits, and compact experiment result logs.
Raw image captures, per-frame observation directories, Python caches, and
generated native build outputs remain on the robot and are excluded from Git.

The current scripts depend on this JetBot's system Python/OpenCV/GStreamer and
motor dependencies described in `README.md`; this is not an OS image backup.
No navigation or speed optimization is performed by this checkpoint.

The original lens calibration resides outside this repository at
`/home/jetbot/Projects/car/calibration/fisheye_calibration.json`. Its contents are
also backed up as `calibration/fisheye_calibration.json`. The live floor profile
still refers to the original absolute path. When restoring on another system,
restore that path or deliberately update `intrinsics_path` to the backed-up
file before running preflight. Revalidate mounts and floor geometry after
hardware changes; saved calibration does not establish a new robot's safety.

The editable skill is `skills/jetbot-navigation`. The installed copy under
`~/.codex/skills/jetbot-navigation` can be restored from it. Its dated manifest
records the earlier baseline and is intentionally not rewritten at backup time.
The conversation's route-overlay images are outside this repository; the
algorithm and preview requirements are documented in the skill.
