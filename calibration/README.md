# Camera color calibration

This workflow measures a global BGR correction from a neutral reference. It does
not calibrate lens geometry, repair focus, or measure full color-chart accuracy.
The old guessed gains have been replaced by identity gains until a measured
profile is installed. Existing camera consumers must restart to load changes.

## Prepare

Use a matte neutral gray card (preferred) or plain white paper without glare.
Place it in the camera view under the lighting used for operation. Make it large
enough to select a uniform rectangle at least 16x16 pixels. Avoid shadows,
reflections, and overexposure. Keep the camera stationary and focused. All captures
here bypass the software correction, use auto white balance, and settle for three
seconds. Stop other camera consumers first; these commands never move motors.

## Capture and fit

Run from `/home/jetbot/jetbot` using system Python (OpenCV/GStreamer installed):

```bash
/usr/bin/python3 scripts/calibrate_camera_color.py capture --output calibration/reference.png
```

Inspect the image and choose X Y WIDTH HEIGHT entirely inside the neutral card.
The following rectangle is an example, not an automatically detected reference:

```bash
/usr/bin/python3 scripts/calibrate_camera_color.py fit \
  --image calibration/reference.png --roi 280 200 80 60 \
  --camera original-imx219 --lighting indoor-daylight \
  --output calibration/candidate.json
```

The fit uses median BGR values and equalizes them to their mean. It rejects
clipped/dark references, large intensity variations and extreme gains. A colored
surface can still pass these checks: the operator must identify a neutral target.

## Independent validation and installation

Keep the same lighting and capture again. Slightly reposition the card and select
its new rectangle; do not reuse the fitting image.

```bash
/usr/bin/python3 scripts/calibrate_camera_color.py capture --output calibration/validation.png
/usr/bin/python3 scripts/calibrate_camera_color.py validate \
  --image calibration/validation.png --roi 280 200 80 60 \
  --profile calibration/candidate.json \
  --preview calibration/validated.png --report calibration/validation.json
```

Inspect the preview and the report. The neutral channel spread must be at most
5% of their mean and must not materially worsen. Then repeat the validation
command with `--install` to activate `calibration/color_profile.json`. Restart
camera consumers, including notebook camera objects. OpenCV, ZMQ publisher,
local snapshot helpers and the previously updated HTTP server use this profile.

For a stronger check, repeat validation with the card at the center and four
corners, and after restarting capture. If only the center passes, a global gain
is insufficient: investigate spatial shading rather than increasing the gains.
Compare several familiar colored objects visually for unwanted changes.

Repeat separately under indirect daylight with room lamps off and under normal
indoor light. Record camera identity and lighting in each candidate. Auto white
balance and lighting can change between sessions; revalidate after a camera swap
or substantial lighting change. These images are ISP-processed BGR, not sensor RAW.

## Disable / override

`JETBOT_COLOR_GAINS=1,1,1` disables correction for a process.
`JETBOT_COLOR_PROFILE=/absolute/path/profile.json` selects another profile.
Explicit gains take precedence over a profile. Missing default profiles use
identity; invalid profiles or explicit missing paths fail visibly.
