"""GPT Drive observation pair. Metric memory is explicit, never painted as RGB evidence."""
import math
import cv2
import numpy as np
import fetch
from trajectory_executor import CameraFloorProjection,camera_path

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


def scale_table(lens):
    """Robot width and required clearance in pixels, per image row."""
    half = fetch.CORRIDOR_HALF_CM
    need = fetch.CORRIDOR_HALF_CM + fetch.OBSTACLE_RADIUS_CM
    table = []
    for row in SCALE_ROWS:
        try:
            forward = lens.ground(320., float(row))[1]
            if not (0. < forward < 400.):
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
        obstacles.append(dict(label=name,position_cm=list(p),uncertainty_cm=5.,source='previous observation; estimated pose',age='not timestamped'))
    # Remembered obstacles can be shown on the map without being asserted as
    # a list. The map says "something was seen here, with this uncertainty";
    # the text list reads as a fact about the present. In the goal-only run the
    # drawing is kept and the list is dropped, so memory informs the picture
    # without competing with what the camera can see now.
    context=dict(frame='current camera, cm right/forward; clockwise degrees',reference_cm=ref,
                 previous_route_cm=route,target_cm=goal,
                 obstacles=obstacles if obstacles_in_text else [],
                 previous_route_role='preserve only when current route to goal is uncertain; fresh evidence takes precedence',
                 target_requires_fresh_rgb=goal_in_fov,reference_reliability=reliability,
                 reference_distance_cm=distance,reference_turn_deg=rotation,
                 reference_weight_kind='heuristic, not calibrated probability',
                 unknown_is_free=False,pose_uncertainty='not statistically calibrated',footprint_cm=[12,15],
                 scale_px_by_row=scale_table(lens),
                 scale_note='robot_width_px is the 12 cm chassis at that image row; '
                            'required_clear_px is the full width a route needs there '
                            '(chassis half-width plus obstacle radius, both sides). '
                            'Measure gaps in pixels at the row where you see them and '
                            'compare against these; do not convert to centimetres.')
    if goal is not None:context['target_bearing_deg']=math.degrees(math.atan2(goal[0],goal[1]))
    floor=CameraFloorProjection(lens).apply(image)
    for v in range(0,640,40):
        cv2.line(floor,(v,0),(v,639),(65,65,65),1);cv2.line(floor,(0,v),(639,v),(65,65,65),1)
    def gp(p):return tuple(int(round(v)) for v in (320+p[0]*1.6,320-p[1]*1.6))
    rgb=image.copy()
    for points,color,width in [(ref,tuple(int(v*(.25+.75*reliability)) for v in (220,100,255)),3),(route,(100,85,40),1)]:
        if len(points)>1:cv2.polylines(floor,[np.array([gp(p) for p in points],np.int32)],False,color,width)
        # Project only visible metric segments. Do not project remembered obstacles into RGB.
        for a,b in zip(points,points[1:]):
            last=None
            for t in np.linspace(0,1,120):
                p=[a[i]+t*(b[i]-a[i]) for i in (0,1)]
                q=lens.pixel(*p) if p[1]>0 else None
                q=tuple(int(round(v)) for v in q) if q and all(math.isfinite(v) and abs(v)<1e6 for v in q) else None
                if q and last:
                    ok,start,end=cv2.clipLine((0,0,640,480),last,q)
                    if ok:cv2.line(rgb,start,end,color,width)
                last=q
    for o in obstacles:
        p=gp(o['position_cm']);cv2.circle(floor,p,int((6+o['uncertainty_cm'])*1.6),(80,80,255),2)
        cv2.putText(floor,o['label'][:24],p,0,.35,(100,100,255),1)
    if goal is not None:
        cv2.drawMarker(floor,gp(goal),(60,220,220),cv2.MARKER_STAR,22,2)
        cv2.putText(floor,'TARGET',gp(goal),0,.45,(60,220,220),1)
    cv2.rectangle(floor,(310,320),(330,344),(220,220,220),1)
    cv2.arrowedLine(floor,(320,320),(320,296),(0,220,255),2)
    cv2.putText(floor,'4m x 4m | 25cm grid | UNKNOWN IS NOT FREE',(8,20),0,.45,(230,230,230),1)
    return rgb,floor,context
