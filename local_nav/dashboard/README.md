# JetBot Search Console

This local dashboard enters a target, starts/stops the existing GPT-5.6 Sol
search, shows the latest camera frame and trajectory preview, and follows model,
policy, planner, controller, power, sensor, checkpoint, and result events.

By default the HTTP server binds to `127.0.0.1` and owns exactly one
`maneuver_session.py` child. The browser cannot address the Unix motor socket,
choose raw motor values, read arbitrary files, or receive `OPENAI_API_KEY`.
Reset is rejected until the run is stopped. The supervisor remains the only
motor owner and retains its camera, IMU, voltage, lease, collision, and
cancellation checks.

## Operator use

Start in motor-free mode for UI inspection:

```bash
cd /home/jetbot/jetbot
/usr/bin/python3 -m local_nav.dashboard.server
```

Open <http://127.0.0.1:8765>. The interface and saved results remain viewable,
but **Start search** is disabled because motion is disarmed.

For an authorized physical search, first make sure no other local navigation
service owns `/tmp/jetbot-local-nav/control.sock`, then run:

```bash
cd /home/jetbot/jetbot
OPENAI_API_KEY='your-key' /usr/bin/python3 -m local_nav.dashboard.server --enable-motion
```

Open <http://127.0.0.1:8765>, enter the object description, and select **Start
search**. **Stop** cancels the route through `maneuver_session`; **Reset** closes
the stopped supervisor and clears the console for a new run. The configured
initial scan uses the user-provided 35 cm clearance assumption. Later movement
still requires retained inspected space and all normal controller guards.

Press Ctrl+C in the dashboard terminal to stop and shut down its owned session.

## Access from another computer without port forwarding

Bind the dashboard to the JetBot's LAN interfaces:

```bash
cd /home/jetbot/jetbot
/usr/bin/python3 -m local_nav.dashboard.server --host 0.0.0.0
```

The command prints a URL such as `http://192.168.86.158:8765/`. Open that URL
on the other computer. No port forwarding or token is required.

Add `--enable-motion` to the command when the robot is ready for a physical
search. LAN mode uses plain HTTP and is visible to every device on the local
network. Add `--access-token YOUR_SECRET_TOKEN` if authentication is desired.

Run the focused motor-free tests with:

```bash
/usr/bin/python3 -m unittest tests.test_navigation_dashboard
```
