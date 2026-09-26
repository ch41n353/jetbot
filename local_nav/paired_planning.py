"""GPT Drive observation pair. Metric memory is explicit, never painted as RGB evidence."""
import math
import cv2
import numpy as np
import fetch
from trajectory_executor import CameraFloorProjection,ZonedFloorProjection,camera_path

INSTRUCTIONS='''
GPT DRIVE PAIRED OBSERVATION CONTRACT (overrides conflicting single-image rules):
Trajectory-first execution: the local executor turns toward an off-axis waypoint
before driving. A follow route may therefore start well to the left or right;
the single-image rule requiring its first waypoint within 20 degrees of forward
does NOT apply here. Do not insert a dummy forward waypoint or artificial wide
curve merely to encode the initial turn. When a supported floor route exists,
return motion=follow, turn_degrees=null, and place the first useful waypoint on
that route, even if strongly lateral. The local executor owns turn angle and
turn-before-translation sequencing, including measured camera pivot motion.
local_map.target_cm is the remembered metric position of the target, carried
on odometry. When target_requires_fresh_rgb is true the target should be
visible in image 1: identify it there and use the fresh pixel, because the
remembered position is an estimate and the image is evidence. But if you
cannot identify the target in the image, the remembered position is still the
best information available -- use it to turn toward the target or to continue
the approach. Do not return hold citing an absent target belief when
target_cm is present.
Use current RGB floor pixels, not a point on the object's side or an imaginary
future frame. Check the swept body and route clearance; a lateral waypoint is
not permission to collide. Do not invent lateral translation just to face an
already-reached object. When no supported floor waypoint is visible and only
a viewpoint change is justified, turn-only remains the fallback.
TURN BEFORE HOLDING. This is a differential-drive robot: it rotates about a
point 7.3 cm behind the lens and its swept radius is 9.6 cm, so it can almost
always turn in place. Before returning motion=hold because an obstacle blocks
the forward corridor, you MUST consider rotating. Sweep the whole range, not
just small angles -- a large turn of 60 to 90 degrees often clears an obstacle
that 30 degrees does not, because clearance grows with the sine of the angle.
If any rotation would put the obstacle further from your new heading than half
of required_clear_px at its row, return motion=turn with that turn_degrees and
an empty route so the next look is taken from the new viewpoint. Reserve hold
for when no rotation helps either -- typically only when the obstacle is so
close that rotating cannot move it out of the corridor at all. State in the
note which angles you considered and why none worked.
Image 1 is current RGB. Image 2 is a 4m x 4m egocentric ground-plane map,
640x640, camera at (320,320), forward up, right right; 1.6 pixels/cm.
Magenta is the top-level reference, cyan is the previous mid-level plan,
yellow star is the remembered target, red rings are previously observed obstacle
contacts with conservative uncertainty radii. Gray unknown space is NOT free.
The warped RGB depicts only the currently observed floor, not obstacle geometry.
Memory positions are odometry estimates, not fresh detections or exact boundaries.
SCALE -- CHECK THIS BEFORE RETURNING ANY ROUTE.
local_map.scale_px_by_row lists, per image row: robot_width_px (the 12 cm
chassis at that row) and min_px_from_route (the distance every obstacle must
keep from your route at that row). min_px_from_route is the number to compare
against. Do not halve it, scale it, or convert anything to centimetres.
For each obstacle you report, take the table row nearest its own pixel row and
measure, in pixels, how far your route passes from it. If that is less than
min_px_from_route, the route drives the chassis through the obstacle -- redraw
it further away or route on the other side. This is not a tight fit to be
judged by eye; it is a measurement you can take from the pixels you are already
emitting. A route that fails this is refused by the controller and the robot
does not move, so returning one wastes the look.
The same applies to a gap: at the row where you see it, a gap narrower than
robot_width_px does not admit the robot and no route through it exists.
Measured 2026-09-24: a route was returned leaving 44 px where min_px_from_route
was 125, with the note "green block remains well right of approach". It passed
5.9 cm from the block against a 6 cm chassis half-width -- through it. State in
note the pixel figure you measured for the nearest obstacle.
Avoid the whole chassis (12x15cm) and turning sweep, then reach the target,
then follow instructions, then preserve reference. Retain unseen obstacles.
Fresh RGB judgment overrides projected memory for targets and obstacles in view.
Identify and localize them anew; do not snap detections onto old coordinates.
Only offscreen target/obstacle positions are supplied as projected memory.
A projected point inside the image is only potentially visible: occlusion or
non-detection is not proof that an obstacle disappeared or that space is free.
Reference reliability is a decreasing heuristic based on accumulated estimated
translation and rotation, NOT a calibrated probability. As it drops, prefer
fresh visual evidence and revise the route freely; never force alignment with
an old path. A newly returned plan does not reset the top-level reference age.
Preserve or revise the previous trajectory when the target is outside current
RGB and the top-down belief still contains the target. The top-down target,
route, and obstacle memory are deliberately supplied so navigation can continue
while the target is temporarily outside the camera field. Target loss by itself
is NEVER a reason to hold. Continue a plausible remembered route, or return a
turn toward target_bearing_deg to reacquire it. When current RGB clearly supports
a route, plan from that evidence rather than retaining an old bend or approach
side. The previous plan need not be traced exactly, but do not discard it merely
because its target is offscreen. Do not hallucinate an RGB contact: for an
offscreen remembered target return visible=false and contact_pixel=null while
using the metric top-down belief. Hold only when a concrete observed or remembered
obstacle blocks every immediate safe continuation/turn, or when the top-down
belief has no target. State that concrete blocker in note. A target behind can
require a large turn; choose a bounded view-changing turn if needed. After a
turn-only result, obtain a fresh observation before translation.
Return the existing schema: for turn use signed turn_degrees (right positive),
route_pixels=[], visible=false and contact_pixel=null for an offscreen target.
All returned pixels refer ONLY to image 1, never the map.
contact_pixel is the nearest visible base where the TARGET MEETS THE FLOOR,
never its face, logo, center, top, or a point behind it. Every route waypoint and
connecting segment must lie on visible traversable FLOOR, outside every object's
silhouette including the TARGET. The target is not drivable. Approach its base
from the near side; do not overshoot beside/behind it then hook back to the base.
Close on the target ALONG ITS BEARING, never past its side: the final segment must
run from the robot toward the target base, so that stopping at the standoff leaves
the target ahead rather than abeam. Do not park parallel to it. Facing is a pivot
about the wheel axis and the chassis sweeps a circle of roughly 10 cm radius, so a
target inside that circle cannot be turned toward at all -- an approach that ends
abeam can strand the robot within reach and unable to face what it reached.
Measured 2026-09-24: a carton reached at 3 cm and 75 degrees off-axis left no legal
pivot and the mission held. Prefer a longer, straighter approach that arrives
pointing at the target over a shorter one that arrives beside it.
Stop the geometric route at the floor contact, or earlier on safe floor if the
final connection is unsupported. Never append a contact point after a segment
that crosses the object. The controller alone applies its configured standoff.
Discard stale reference segments that project onto the target or past its base.
For a single target all_done=true means reaching it completes the instruction;
it does NOT mean arrival has occurred. The controller determines arrival. No known target position means
request a view change/hold, not an invented target. Old observations never expire
into free space solely because time passed.
'''


# The model is told its footprint in centimetres and forbidden to estimate
# centimetres from an image, so it has had no way to judge whether a gap admits
# the robot. Measured 2026-09-24: it twice planned past a box leaving 8.9 cm of
# lateral room where 10 cm is the minimum, drove it, and hit the box on the
# third look. This gives the same judgement in the units it already works in --
# pixels -- at the row where a gap is seen.
SCALE_ROWS = (400, 340, 290, 250, 215, 190, 172, 158, 146, 136)
# Near field only. This is a fisheye, and the row-to-range mapping stops being
# usable well before the horizon: rows compress, so a row a few pixels higher
# is metres further out, and the width figure derived from it is meaningless.
# Inside half a metre the rows are still spread far enough apart that a pixel
# gap means something. Beyond it the table would be a precise-looking number
# the lens cannot support, which is worse than no number at all.
SCALE_MAX_RANGE_CM = 50.


def scale_table(lens):
    """Robot width and required clearance in pixels, per image row."""
    half = fetch.CORRIDOR_HALF_CM
    need = fetch.CORRIDOR_HALF_CM + fetch.OBSTACLE_RADIUS_CM
    table = []
    for row in SCALE_ROWS:
        try:
            forward = lens.ground(320., float(row))[1]
            if not (0. < forward <= SCALE_MAX_RANGE_CM):
                continue
            body = [fetch.Robot.pixel(lens, s * half, forward) for s in (-1, 1)]
            clear = [fetch.Robot.pixel(lens, s * need, forward) for s in (-1, 1)]
            if None in body or None in clear:
                continue
            width = abs(clear[1][0] - clear[0][0])
            table.append(dict(image_row=int(row), range_cm=round(forward, 1),
                              robot_width_px=round(abs(body[1][0] - body[0][0]), 1),
                              required_clear_px=round(width, 1),
                              # The number to compare against directly. The rule
                              # used to be "at least half of required_clear_px",
                              # and the model got it wrong by a factor of three
                              # on 2026-09-24 (44 px where 125 were needed).
                              # Halving is arithmetic it does not need to do.
                              min_px_from_route=round(width / 2., 1)))
        except Exception:
            continue
    return table

# Margin from the frame border before a point counts as usably in view. The
# metric target belief is withheld when the goal is in the field of view, so
# that the model identifies it from fresh RGB instead of trusting odometry --
# sound in the middle of the frame, wrong at its edge, where the object is
# clipped and fisheye-distorted and identification is least reliable.
#
# Measured 2026-09-24 on call f71a6cdc: the red block sat at the extreme right
# border, counted as in_fov, so target_cm was nulled AND the TARGET star was
# left off the top-down. Re-issuing that identical request 5 times gave
# visible=true 2/5 and three holds that each named the missing target belief
# ("top-down belief has no target"). Re-issuing it with target_cm restored and
# nothing else changed gave visible=true 5/5.
FOV_EDGE_MARGIN_PX = 60.



STANDOFF_CM = 20.                 # what counts as arrived; the model is told
DEFAULT_OBSTACLE_RADIUS_CM = 6.   # only for memories recorded before radius existed


def vec(point):
    return dict(right=round(float(point[0]),1),forward=round(float(point[1]),1))


def pixel_of(lens,point):
    """Image-1 pixel for a floor position, or None when it is not in frame."""
    if point is None or point[1]<=0:return None
    try:q=lens.pixel(float(point[0]),float(point[1]))
    except Exception:return None
    if not q or not all(math.isfinite(v) for v in q):return None
    if not (0<=q[0]<640 and 0<=q[1]<480):return None
    return dict(x=round(float(q[0]),1),y=round(float(q[1]),1))


def pixels_of(lens,points):
    """The same list in image-1 pixels, dropping what falls outside the frame."""
    out=[]
    for p in points or []:
        q=pixel_of(lens,p)
        if q:out.append(q)
    return out


def in_fov(lens,point,margin=FOV_EDGE_MARGIN_PX):
    if point[1]<=0:return False
    pixel=lens.pixel(*point)
    return bool(pixel and all(math.isfinite(v) for v in pixel)
                and margin<=pixel[0]<640-margin and margin<=pixel[1]<480-margin)


def reference_reliability(distance_cm,turn_deg):
    # Advisory only: half weight at 1 m translation OR 180 degrees cumulative yaw.
    return 2.**(-max(0.,distance_cm)/100.-max(0.,turn_deg)/180.)



def observation(lens,image,memory=None,pose=(0,0,0),reference=None,
                obstacles_in_text=True):
    memory=memory or {}
    route=fetch.rebase(memory.get('route',[]),pose)
    ref=fetch.rebase(memory.get('top_reference',reference or []),pose)
    goal=fetch.rebase([memory['goal']],pose)[0] if memory.get('goal') is not None else None
    # Fresh RGB should take precedence over remembered geometry, but that is a
    # preference and it was being enforced by deleting the number: whenever the
    # goal was judged in view, target_cm was nulled AND the TARGET star was left
    # off the top-down. When identification then failed for any other reason --
    # a small object, clutter, colour -- no fallback remained, and the planner
    # held citing the very belief that had been withheld. Measured 2026-09-24:
    # an offline replay of one such call returned visible=true 2/5 as sent and
    # 5/5 with target_cm restored and nothing else changed. Keep the belief and
    # state the preference; target_requires_fresh_rgb still carries it.
    goal_in_fov=goal is not None and in_fov(lens,goal)
    distance=float(memory.get('reference_distance_cm',0))+math.hypot(pose[0],pose[1])
    rotation=float(memory.get('reference_turn_deg',0))+abs(pose[2])
    reliability=reference_reliability(distance,rotation)
    obstacles=[]
    for item in memory.get('obstacles',[]):
        name,point=item[:2];p=fetch.rebase([point],pose)[0]
        if in_fov(lens,p):continue
        radius=float(item[2]) if len(item)>2 and item[2] else DEFAULT_OBSTACLE_RADIUS_CM
        obstacles.append(dict(label=name,position_cm=vec(p),radius_cm=round(radius,1),
                              source='seen earlier, position carried on odometry'))

    # Nothing is drawn on either picture. Everything remembered is supplied as
    # numbers instead, in two matching sets: metric for the frame the robot
    # drives in, and pixels for image 1 so the model can relate a number to
    # what it sees. Ink was worse than useless -- the model has no way to tell
    # a drawn belief from an observed thing, and measured 2026-09-24 it followed
    # an inked route into an obstacle it had itself reported (6.1 cm median
    # clearance with the ink present, 14.2 cm with it removed).
    context=dict(
        frame='centimetres in the robot frame: [right, forward]. '
              'Right positive, forward positive. The robot is at [0,0] facing [0,+1]. '
              'This is the frame of image 2 and the frame you answer in.',
        target_cm=vec(goal) if goal is not None else None,
        # target_px is deliberately NOT supplied. The model reports a fresh
        # sighting as a pixel, and handing it the pixel the remembered position
        # projects to would let that report be a copy rather than an
        # observation -- which is exactly what happened on 2026-09-24, when
        # every obstacle and the target came back with the supplied
        # coordinates unchanged and visible=true. reference_px and previous_px
        # stay: those are never reported back, so they cannot be echoed.
        target_bearing_deg=(round(math.degrees(math.atan2(goal[0],goal[1])),1)
                            if goal is not None else None),
        reference_cm=[vec(p) for p in ref],
        reference_px=pixels_of(lens,ref),
        previous_cm=[vec(p) for p in route],
        previous_px=pixels_of(lens,route),
        obstacles=obstacles if obstacles_in_text else [],
        footprint_cm=dict(width=12,length=15),
        standoff_cm=STANDOFF_CM,
        # Near field only -- see SCALE_MAX_RANGE_CM.
        scale_px_by_row=scale_table(lens),
        reference_reliability=round(reliability,3),
        reference_note='the operator\'s intended direction of travel, not a path to '
                       'trace; accuracy falls off with distance from the robot',
        # Plumbing, not prompt content. target_requires_fresh_rgb is what
        # recognize() keys on to drop target_was from last_time on the legacy
        # path, and the dashboard reports the two accumulators. Dropping them in
        # the metric rewrite broke both silently: the visible-target memory
        # leaked into last_time and midlevel_view raised KeyError.
        target_requires_fresh_rgb=goal_in_fov,
        reference_distance_cm=round(distance,1),
        reference_turn_deg=round(rotation,1))
    zones=ZonedFloorProjection(lens);floor=zones.apply(image)
    Z=ZonedFloorProjection
    hub=zones.place(0.,0.)                       # the robot, after cropping
    def ring_px(cm):return int(round(Z.radius_px(cm)*zones.zoom))
    # Tint everything past the boundary, so which scale you are reading is
    # visible before any label is read. A piecewise scale only works if the
    # break cannot be missed.
    shade=floor.copy()
    cv2.circle(shade,hub,ring_px(Z.OUTER_CM),(70,55,55),-1)
    cv2.circle(shade,hub,ring_px(Z.INNER_CM),(0,0,0),-1)
    floor[:]=cv2.addWeighted(floor,1.,shade,.22,0)
    for degrees in range(-90,91,15):                 # bearing spokes
        a=math.radians(degrees);reach=ring_px(Z.OUTER_CM)
        cv2.line(floor,hub,(int(hub[0]+reach*math.sin(a)),
                            int(hub[1]-reach*math.cos(a))),(60,60,60),1)
    for cm,tone in [(20,0),(40,0),(60,0),(80,0),(125,1),(150,1),(175,1)]:
        r=ring_px(cm)
        cv2.circle(floor,hub,r,(86,74,74) if tone else (86,86,86),1)
        cv2.putText(floor,'%d'%cm,(hub[0]+2,hub[1]-r+13),0,.36,
                    (150,135,135) if tone else (150,150,150),1)
    r=ring_px(Z.INNER_CM)
    cv2.circle(floor,hub,r,(90,200,255),2)
    cv2.putText(floor,'100cm  SCALE HALVES OUTSIDE THIS RING',
                (hub[0]+2,hub[1]-r+15),0,.40,(90,200,255),1)
    cv2.circle(floor,hub,ring_px(Z.OUTER_CM),(120,120,160),1)
    body=[zones.place(x,z) for x,z in ((-6.,0.),(6.,0.),(6.,-15.),(-6.,-15.))]
    cv2.polylines(floor,[np.array(body,np.int32)],True,(220,220,220),1)
    cv2.putText(floor,'RADIAL MAP | rings = range | spokes = bearing, every 15 deg',
                (8,18),0,.40,(230,230,230),1)
    cv2.putText(floor,'inside blue ring %.1f px/cm | outside HALF that (tinted) | DARK IS NOT KNOWN FREE'
                %(Z.INNER_PX_PER_CM*zones.zoom),(8,36),0,.38,(230,230,230),1)
    return image.copy(),floor,context

