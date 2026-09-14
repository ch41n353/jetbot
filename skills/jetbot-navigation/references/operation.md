# Operating the local tools

All paths below are on this JetBot. These commands are instructions, not
permission to move. Start disarmed and inspect before enabling motors.

## Service and shutdown

```bash
/usr/bin/python3 /home/jetbot/jetbot/local_nav/service.py
```

In another terminal, after camera warmup:

```bash
/usr/bin/python3 /home/jetbot/jetbot/local_nav/client.py status
/usr/bin/python3 /home/jetbot/jetbot/local_nav/client.py snapshot
```

Only view the returned snapshot path when the request succeeds. To enable motion,
shut the disarmed instance down, wait for exit/lock release, then restart with
`--enable-motion`:

```bash
/usr/bin/python3 /home/jetbot/jetbot/local_nav/client.py shutdown
/usr/bin/python3 /home/jetbot/jetbot/local_nav/service.py --enable-motion
```

The service takes several seconds to release workers on shutdown. A duplicate
instance fails its lock. Do not kill an unidentified process to take the lock.
Sessions have sometimes ended between calls; recheck state, do not automatically
restart into motion after a user stop or unexplained handling.

Two terminals suffice: service in one, commands in another. **Ctrl+C in the
service terminal stops and shuts down the service.** The shutdown command is
another way to disarm. A `stop` request alone does not cancel a live controller.

## Straight continuous step

```bash
/usr/bin/python3 /home/jetbot/jetbot/local_nav/smooth_drive_probe.py \
  --execute --distance-cm 5 \
  --log /home/jetbot/jetbot/local_nav/goals/UNIQUE-RUN.json
```

Use a new log path each run. Requested distance accepts 1–15 cm. Omit `--execute`
for stationary preflight. The tool does not steer toward an object or compute
obstacle clearance; it holds its starting heading with visual/IMU correction.

## Pulse-and-settle turn

```bash
/usr/bin/python3 /home/jetbot/jetbot/local_nav/turn_controller.py \
  --execute --degrees 15 --power .14 \
  --log /home/jetbot/jetbot/local_nav/goals/UNIQUE-TURN.json
```

Positive angles turn right, negative left. Maximum magnitude 30°. Power accepts
0.12–0.16; default 0.12 has sometimes failed to move the robot. Omit `--execute`
for preflight. `--continuous` is **not supported** in this controller.

## Fixed smooth-turn experiment

```bash
/usr/bin/python3 /home/jetbot/jetbot/local_nav/smooth_turn_probe.py
```

**This immediately requests movement** when service motion is enabled. It always
commands a 15° right turn at 0.16 and overwrites
`local_nav/goals/smooth-turn-016.json`; preserve an existing log before an authorized
new trial. It has no angle, direction, or execute/preflight arguments.

## Slower point approach

```bash
/usr/bin/python3 /home/jetbot/jetbot/local_nav/point_controller.py \
  --execute --pixel 320 249 --settle-seconds .4 --stop-distance-cm 15 \
  --log /home/jetbot/jetbot/local_nav/goals/UNIQUE-POINT.json
```

The pixel is from the current image, not a reusable object location. Omit
`--execute` for stationary preflight. Initial target must pass both static and
measured-attitude checks: 30–40 cm forward and under 5 cm lateral. Stop-distance
accepts 10–20 cm to the waypoint, **not necessarily the object**.

## Offline validation

```bash
cd /home/jetbot/jetbot
/usr/bin/python3 -m unittest discover -s tests
/usr/bin/python3 local_nav/replay_estimator.py local_nav/goals/RUN.json.observations
```

The replay uses the **current** calibration and estimator, not a frozen historical
implementation. Use the dated manifest to notice differences. Replays do not
move the robot. Tests passed 37 cases in the last recorded full suite; tests do
not prove hardware health, clearance, or ground-truth position accuracy.
