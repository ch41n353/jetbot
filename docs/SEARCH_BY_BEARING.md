# Searching for something the robot cannot see

**Code: `local_nav/explore.py`. Measurements below taken 2026-09-17 against
stored frames only; nothing in this document was validated by driving.**

`planner_demo.mission()` drives to a named object it can already see. It looks,
asks the model for a route, drives about four seconds, looks again, and carries
its unreached waypoints and remembered obstacles forward between looks. It
arrives reliably, and it has exactly one gap:

```python
if not route:
    self.say('nothing to drive and nothing remembered; stopping')
```

If the object has never been in frame there is nothing to carry forward, so the
run ends before it starts. This document describes the layer above: how the
robot finds a direction to hand that controller, and — just as important — what
it cannot promise while doing so.

## The one measurement the whole design rests on

A pixel becomes a **direction** under much weaker conditions than it becomes a
**distance**.

`fetch.Robot.ground()` intersects the camera ray with the floor. It needs the
ray to point below the horizon, and it is only trusted to about 140 cm, because
past that a pixel of error in the contact point is worth tens of centimetres.
Measured on this calibration, straight ahead:

| pixel row | floor distance |
|---|---|
| y = 215 | 74 cm |
| y = 200 | 120 cm |
| y = 195 | 151 cm (past the trusted range) |
| y ≤ 190 | refused: at or above the horizon |

`explore.bearing()` takes the same undistorted ray and returns its azimuth about
the gravity axis. It needs nothing. It works on a pixel at the top of the frame,
on a pixel inside a doorway three rooms deep, on anything the model can point
at. The search therefore spends its whole life in angles and never asks for a
range until the reach controller takes over, by which time the object is close
enough for the projection to mean something.

The second half of the same argument is that **heading is the one thing this
robot measures well**. Turns land at 0.7 degrees mean error against the IMU;
visual odometry lands anywhere from one sample in eight to five in five,
depending on the carpet. So a search built on rotation composes accurately and a
search built on translation does not. That is why there is no map here, why
stations are labels rather than coordinates, and why the journal handed to the
model contains no positions at all — a model given numbers will reason about
them as if they were true.

## The shape of the thing

```
sweep from where you stand
   |
   +-- sure sighting, away from the frame edge ---> turn to it, hand over
   |
   +-- nothing ---> write the journal
                       |
                       +-- ask the model for a few steps
                       +-- check every step against what the robot can do
                       +-- run them, abandoning the rest the moment one
                           of them says something the plan did not know
```

The agent proposes a multi-step plan up front and the executor revises it. It is
not a per-step conversation — that would pay a planning call for every turn —
and it is not a geometric frontier explorer, because a frontier needs a map and
this robot cannot build one.

## The cheap question

`explore.look()` asks one question per direction, with a strict schema:

```
visible, target_pixel, confidence (sure | unsure | no),
worth_a_closer_look, open_pixel, obstacles[], scene
```

It is deliberately much smaller than `fetch.PROMPT`: roughly a third of the
instructions and half the output bound, because it is asked six to ten times per
station while the route question is asked once, after the search is over.

Everything the model must judge for itself is expressed in **pixels of its own
image**, never in centimetres. This was established repeatedly on this robot:
"turn if the obstacle is too close" changed nothing, "turn if its contact pixel
is in the bottom third of the picture" worked at once. So:

* an obstacle is *blocking* if its contact pixel is below y = 320 and within the
  middle half of the width;
* `open_pixel` must be below y = 200 and within a sixth of the frame width of
  the centre line — the first because the floor is not measurable above it, the
  second because the robot drives forward and will otherwise have to pivot
  before it moves at all;
* `scene` is capped at ten words because it is the *only* thing the journal
  keeps about a direction the robot has turned away from.

`confidence` exists because the two mistakes cost different amounts. A missed
sighting costs one more look. A false "sure" hands the reach controller a
bearing to something that is not there, and it drives. The prompt says so, and
tells the model that "unsure" is cheap: the robot will turn and photograph that
direction from the middle of the frame, where the lens is sharp and the object
is biggest.

## Sweep, and where the second look goes

A coarse pass of six looks closes the circle. The step is 60 degrees because the
lens covers about ±63 degrees at the height in the frame where distant objects
sit (measured: `bearing(640, 240) = +62`, `bearing(0, 240) = -63`), so 60 degree
cells overlap by half a frame rather than leaving seams an object can live in.

Then, and only where the coarse pass flagged something, it looks again. Two
kinds of flag want different second looks:

* **An unsure sighting has a pixel, and a pixel is a bearing.** The second look
  turns straight at the candidate, so it lands in the centre of the frame. If it
  is already within 12 degrees of centre the refinement is skipped: the second
  photograph would contain the same pixels and produce the same answer.
* **A direction that merely could hide something has nothing to aim at**, so the
  second look goes to the two flanks of the cell, ±20 degrees, which is where a
  coarse sweep is weakest.

A confident sighting within 70 pixels of the left or right edge is treated as a
lead, not a sighting: the fisheye squeezes more than 120 degrees into 640
pixels, so an object on the rim is a handful of smeared pixels whose apparent
position moves fast with heading. It gets the same aimed second look. Height in
the frame is not treated this way — high means far, not distorted sideways, and
a bearing taken from it is as good as any other.

The refinement budget is four looks per sweep, taken one flank from each flagged
direction before any second flank. That interleaving is a direct consequence of
a measurement below: in a cluttered room the flag comes back true on most
directions, and taking them in order spent the entire budget on the first two
cells.

## The journal

One station is a name, the bearing and rough distance that reached it, and a
list of looks. It renders as a dozen short lines:

```
station start (where the search started)   <- the robot is here, facing 0
  +000  open carpet, storage bins, and partial could hide it
  +060  open carpet, toys, and dark doorway wo could hide it
  -180  open carpet, loose cables, and model v
  -020  office carpet; shadows under desk and  could hide it, close look

budget left: 30 looks, 600 cm of driving, 4 more place(s) to stand
```

Text rather than JSON because it is read once and never parsed, and a table of
short lines costs a fraction of the tokens. The header states plainly that the
robot has no map and no position, and that distances are what the wheels were
asked for rather than what they did.

## The plan, and what happens to a bad one

The model reviews that journal and returns up to four steps from a closed set:

| action | fields | what it means |
|---|---|---|
| `sweep` | heading, arc | turn on the spot and photograph an arc of the circle |
| `go` | heading, distance (20–150 cm) | turn to a heading and drive roughly that far, to see the room from somewhere else |
| `give_up` | — | nothing here is worth trying |

Headings are always relative to the way the robot faces at the start of that
step, which for a step after a `go` is the direction it drove. There is no
"return to station A": without position the robot cannot do it, so it is not in
the vocabulary. There is no waypoint, coordinate or map field either.

`validate_plan()` then checks the plan and, if anything is wrong, **rejects the
whole thing and asks again with the reasons attached** — up to three times,
after which a dull built-in fallback runs instead. It is never repaired. A
quietly patched plan is a journey nobody chose, and the one thing this layer
must not do is invent its own reason to move. Rejections include: an action that
is not in the set, a heading outside ±180, a hop shorter than 20 or longer than
150 cm, a missing distance, more steps than the bound, `give_up` anywhere but
last, driving further than the remaining budget, a first hop along a heading the
journal has never photographed, and a first hop into a direction the journal
records as blocked.

Only the *first* hop is checked against the floor. Steps after it start from
somewhere the robot has never stood, so there is nothing to check them against
and rejecting them would be a guess dressed as a rule.

`go` is executed as the same look-drive-look cadence the reach controller uses:
each slice takes a photograph, projects the model's `open_pixel` through the
calibration, widens the leg to the clearance the wheels need with `fetch.avoid`,
and hands it to `fetch.follow` with the obstacle list from that same photograph.
Travel is therefore never time spent not looking — in practice this is where the
target often turns up, the plan having only said where to stand.

The plan is abandoned early whenever a sweep produces a sighting or a lead, or a
hop is stopped short. Carrying on through steps chosen before that would be
spending the budget on a question that has changed.

## What this cannot do

* **There is no coverage guarantee and no proof of absence.** Without position
  the robot cannot say which parts of a room it has stood in, cannot return to a
  station it has left, and cannot tell a second sweep of the same corner from a
  sweep of a new one. `give_up` therefore reports what ran out — looks, travel,
  stations — and never that the object is not there.
* **A hop is open loop in position.** The distance driven is measured by carpet
  optical flow when the carpet allows and by timing when it does not. The
  journal says "drove about 100 cm" because that is the honest phrasing.
* **Occlusion is not modelled.** "Could hide it" is the model's opinion about
  one photograph, and the measurements below show that opinion is weakly
  discriminating in a cluttered room.
* **Two stations can be the same place.** Nothing detects a loop.
* **The reach controller, not this, decides whether the thing found is the thing
  wanted.** All this layer produces is a direction and a claim.

## Measured, offline

All numbers below come from stored frames in `local_nav/goals/dashboard/` via
`local_nav/evaluate_explore.py`. No motors were commanded. The frames are one
real rotation on the spot — 37 photographs of a cluttered room, taken by the
earlier stack on 2026-09-16.

### Does the cheap question answer correctly?

52 looks over 13 frames spanning the full rotation, four targets, two
independent runs:

| target | present | sure and right | unsure and right | missed | **sure and wrong** | correctly silent |
|---|---|---|---|---|---|---|
| pink storage bin | 4 frames | 4 | 0 | 0 | **0** | 9 / 9 |
| black remote-control car | 6 frames | 4–5 | 0–1 | 1 | **0** | 7 / 7 |
| a red fire extinguisher (not in the room) | 0 | — | — | — | **0** | 13 / 13 |
| the Advil bottle (label uncertain, see below) | ? | 0 | 2–3 | — | **0** | 10–11 |

A third run of the two unambiguous targets scored 10 of 10 positives found
(9 sure, 1 unsure), 0 missed, 0 false alarms in 16 negatives.

**Across 81 looks there was not one confident false positive.** The single
repeated miss was the remote-control car reduced to one wheel at the extreme
left edge of the frame — which is exactly the case the edge rule treats as a
lead rather than a sighting.

### Is a bearing taken from a pixel any good?

Yes, and this is the strongest result here.

* **Repeatability.** Across 11 sightings reported in both independent runs, the
  same frame gave bearings agreeing to a mean of **0.52 degrees** (worst 1.7).
* **Consistency.** The robot turned by a fixed amount between consecutive
  frames, so a truthful bearing must be linear in frame index. Fitting the four
  away-from-the-rim sightings of the car: **−9.25 deg/frame, worst residual 0.47
  deg** in run 1, and **−9.04 deg/frame, worst residual 0.76 deg** in run 2 —
  two independent estimates of the robot's own turn agreeing to 0.2 degrees.

That is the same order as the turn controller's own 0.7 degree error, so the
bearing is not the weak link in a turn-to-face.

### Does detection really outrun placement?

Directly tested, and yes. Asked for a cardboard box visible through a doorway
into the next room, in three frames taken 27 degrees apart:

| frame | confidence | pixel | `ground()` | bearing |
|---|---|---|---|---|
| step-09 | sure | (466, 190) | **refused: above the horizon** | +26.7 |
| step-12 | sure | (325, 184) | **refused** | +0.3 |
| step-15 | sure | (178, 188) | **refused** | −27.2 |

Every one of those sightings is invisible to a stack that needs a range, and all
three bearings fall on the same 9.0 deg/frame line the near objects produced. A
black microphone stand in the same far room came back the same way. This is the
whole argument for the layer in one table.

One honest correction to the brief, though: **the confident sightings in the
near room were all between 37 and 131 cm**, comfortably inside the projection's
range. This data set does not demonstrate confident identification of a *small
specific product* at two metres; it demonstrates confident identification of
large or distinctive things at any range, and of a cardboard box at several
metres. See the next section.

### Does it propose sane plans?

Four live planning calls against real and synthetic journals. **All four
validated on the first attempt; none was rejected.** Given a journal whose only
promising direction was recorded as blocked by boxes, it swept that direction
instead of driving into it, then proposed `give_up`. Given a journal with only
three directions photographed, it swept the unseen half rather than hopping into
floor nobody had looked at. Given 40 cm of travel budget left, it asked for a
40 cm hop, not 100. From the real 10-look sweep of the cluttered room it
proposed: re-sweep the closet arc, hop 100 cm towards the doorway along a
heading it had photographed, sweep 360 there, then 50 cm further.

The rejection path is therefore covered by unit tests rather than by live
evidence — it has not yet been provoked by a real model answer.

### End to end, on the stored rotation

Replaying the real sweep loop against all 37 frames with the wheels stubbed:

* **"the black remote control car"** — found in 5 looks. It saw the car at
  heading −180 with a bearing of +49, *declined to hand over* because the
  sighting was on the frame rim, kept sweeping, saw it again at −120 with a
  bearing of −9, and handed over facing −129. An independent estimate from the
  frame geometry puts the car at −132.
* **"a red fire extinguisher"** — 6 coarse looks and 4 refinements, all "no",
  then a valid four-step plan. Correctly never claimed the room was empty of it,
  only that the looks were spent.

### The measurement that changed the design

`worth_a_closer_look` came back **true on 52 of 52 looks** with the first
wording. A flag that is always true carries no information and hands the
refinement budget to whichever cell happens to be swept first. Sharpening the
wording — name the hiding place or say false; a wall of boxes you can see all of
is *answered*, not hidden; you can afford two or three closer looks in a circuit
— brought it to **29 of 52**, with no loss of detection (the same four bin
frames, one more car frame). It is still true on more than half the directions
of a genuinely cluttered room, so the refinement order was changed to interleave
across flagged directions rather than trust the flag to be selective. Both the
before and after numbers are here because the flag is the weakest part of the
design and should be read as weak.

### The label I could not settle

The room contains a white bottle with a dark blue cap. Measured through the
calibration from two viewpoints it is **12.0 and 12.6 cm tall at 65 and 74 cm**,
which is the size of a large Advil bottle — and the session that recorded these
frames was searching for an Advil bottle. The earlier stack's recognizer called
it a "white medicine bottle" while reporting the target as not visible; the
sweep question calls it "possible Advil bottle" (unsure) in some frames, "white
bottle" while answering "no" in others, and "sure" twice under one prompt
wording.

I could not decide from the photographs whether it is the target, so the Advil
rows are reported separately above and excluded from the false-positive count.
What can be said without settling it is that **identification of a specific
small product at three quarters of a metre is not reliable**, which is precisely
why `unsure` exists and why the refinement aims a second photograph at the
candidate instead of acting on the first.

## What is untested

Everything that moves. Specifically:

* No sweep, hop or handover has been run on the robot. `Search.face`,
  `Search.hop`, the interaction with `fetch.follow`, and the heading bookkeeping
  across a drive are code, not measurements.
* The offline replay assumes the stored frames are evenly spaced around the
  rotation. They are not exactly: the last recorded turn asked for 30 degrees
  where the others asked for 10, and the fitted spacing (9.0–9.3 deg/frame)
  disagrees with the per-turn IMU figures recorded at the time (≈3.5 deg). The
  replay only uses the assumption to decide which photograph a heading gets, and
  is stated in the harness.
* Nothing has been measured about a second station, because stepping to one
  requires driving.
* The plan-rejection and re-ask loop has never been triggered by a live answer.
