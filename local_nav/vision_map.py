#!/usr/bin/env python3
"""Turn recognizer image geometry into metric floor geometry (centimetres).

The recognizer supplies semantics and pixels only. Every centimetre here comes
from the calibrated fisheye intrinsics and the measured camera height, never
from the model: a model cannot know scale from one monocular frame.

Two rules keep this conservative:

* A box's bottom edge is the nearest point where the object meets the floor.
  Its extent away from the camera is unobservable, so an obstacle footprint is
  projected from that contact line to the far edge of the mapped region. You
  cannot see behind a thing.
* Free space is only the floor actually seen in this frame, inside the
  calibrated projection limit, with every obstacle footprint removed.

Coordinates match the rest of the stack: x right, z forward, in the frame of
the pose supplied by the caller.
"""
import math

import numpy as np

from route_geometry import finite, overlap, rectangle


#: Beyond this range one pixel of box-bottom error becomes several centimetres
#: of floor error, and the floor-plane assumption is not independently
#: validated. Projections past it are treated as unknown, not as free.
MAX_PROJECTION_CM = 140.

#: Padding on a projected obstacle footprint, for projection noise only.
#: This is deliberately small: every swept envelope already carries the full
#: 5 cm chassis clearance plus 2 cm of pose allowance, so padding obstacles by
#: another clearance-sized margin counts the same budget twice. At 4 cm it did
#: exactly that, pulling an object whose real contact was 18.1 cm from the turn
#: pivot to an apparent 12.8 cm and refusing turns the robot had room for.
OBSTACLE_MARGIN_CM = 1.

#: How far past its floor contact an obstacle is assumed to extend. A box gives
#: no depth information, so this is an assumption, deliberately bounded: floor
#: beyond it is unknown rather than occupied, and is never certified free.
OBSTACLE_DEPTH_CM = 20.


def _pixels(values):
    out = []
    for x, y in values:
        x, y = finite(x), finite(y)
        if not (0. <= x <= 640. and 0. <= y <= 480.):
            raise ValueError('Pixel outside the 640x480 frame')
        out.append([x, y])
    if not out:
        raise ValueError('No pixels supplied')
    return np.asarray(out, dtype=np.float32).reshape(-1, 1, 2)


def project(tracker, pixels, attitude=None):
    """Project image pixels onto the floor plane, in camera-local centimetres.

    Returns an (N,2) array of (right, forward). Raises when a point projects
    beyond the validated range, at or above the horizon, or behind the camera.
    """
    ground = tracker.ground(_pixels(pixels), attitude)
    for right, forward in ground:
        if not math.isfinite(right) or not math.isfinite(forward):
            raise ValueError('Pixel does not project onto the floor plane')
        if forward <= 0.:
            raise ValueError('Pixel projects behind the camera, not onto floor')
        if math.hypot(right, forward) > MAX_PROJECTION_CM:
            raise ValueError('Floor projection beyond %.0f cm is not trusted'
                             % MAX_PROJECTION_CM)
    return np.asarray(ground, dtype=float)


def obstacle_footprint(tracker, box, attitude=None, margin=OBSTACLE_MARGIN_CM,
                       limit=MAX_PROJECTION_CM, depth=OBSTACLE_DEPTH_CM):
    """Floor rectangle occupied by a recognized box, as [x0,z0,x1,z1].

    Only the bottom edge touches the floor, so depth away from the camera is
    unobservable and is filled by a bounded `depth`, not to the range limit.
    Filling to the limit looks conservative but is not usable: an axis-aligned
    box around a long diagonal object -- a power cable snaking over carpet --
    covers mostly empty floor, and projecting its bottom edge puts the near
    contact at the robot's nose. Extending that to the horizon turned one cable
    into a wall that blocked every turn, which stops the robot without making
    it safer.

    Floor past an obstacle is therefore unknown, not occupied. Nothing here
    certifies it as free either: free space comes only from `fan_rectangles`.
    """
    x0, y0, x1, y1 = [finite(box[k]) for k in ('x0', 'y0', 'x1', 'y1')]
    if not (0. <= x0 < x1 <= 640. and 0. <= y0 < y1 <= 480.):
        raise ValueError('Obstacle box outside the frame or reversed')
    contact = project(tracker, [(x0, y1), (x1, y1)], attitude)
    left, right = float(min(contact[:, 0])), float(max(contact[:, 0]))
    near = float(min(contact[:, 1]))
    far = min(limit, max(contact[:, 1]) + depth)
    return rectangle([left - margin, max(.1, near - margin),
                      right + margin, max(near + depth, far)])


def fan_rectangles(tracker, attitude=None, limit=MAX_PROJECTION_CM, bands=8,
                   left_px=40., right_px=600., bottom_px=472., top_px=225.):
    """Tile the floor this frame actually shows, as camera-local rectangles.

    The visible floor is a fan: narrow close to the robot and wide far away
    (measured on this camera, roughly +/-11 cm at 3 cm and +/-70 cm at 50 cm).
    A single axis-aligned rectangle would either waste most of the view or
    claim unseen floor, so the fan is returned as a staircase of rectangles.
    Each band is inscribed -- its half width is the narrowest the fan reaches
    anywhere in that band -- so the union stays inside what was really seen.

    Obstacles are deliberately not subtracted here. The planner already rejects
    envelopes overlapping obstacle rectangles, and mixing the two would hide
    which evidence certified free floor and which merely hid it.
    """
    rows = np.linspace(bottom_px, top_px, max(3, int(bands) + 1))
    centre = .5 * (left_px + right_px)
    edges = []
    for y in rows:
        # Near the horizon the outer columns stop projecting onto trusted
        # floor, so step inward and keep the widest pair that still does.
        # Fixing the columns at the image edges would cut the fan short.
        for reach in np.linspace(.5 * (right_px - left_px), 20., 15):
            try:
                side = project(tracker, [(centre - reach, float(y)),
                                         (centre + reach, float(y))], attitude)
            except ValueError:
                continue
            half = float(min(abs(side[0, 0]), abs(side[1, 0])))
            forward = float(min(side[:, 1]))
            if 0. < forward <= limit and half >= 5.:
                edges.append((forward, half))
                break
    edges.sort()
    if len(edges) < 2:
        raise ValueError('The current frame shows no usable floor')
    out = []
    for (z0, half0), (z1, _) in zip(edges, edges[1:]):
        if z1 - z0 < .5 or half0 < 5.:
            continue
        # half0 is the fan width at the NEAR edge of this band, the narrowest
        # it gets across the band, so the rectangle stays inside the fan.
        out.append(rectangle([-half0, z0, half0, min(z1, limit)]))
    if not out:
        raise ValueError('No usable floor band in this frame')
    return out


def to_world(pose, points):
    """Rotate camera-local floor points into the map frame of `pose`."""
    angle = math.radians(finite(pose[2]))
    c, s = math.cos(angle), math.sin(angle)
    out = []
    for right, forward in points:
        out.append([pose[0] + c * right + s * forward,
                    pose[1] - s * right + c * forward])
    return out


def world_rectangle(pose, local):
    """Axis-aligned map-frame bounds of a camera-local rectangle."""
    x0, z0, x1, z1 = rectangle(local)
    corners = to_world(pose, [(x, z) for x in (x0, x1) for z in (z0, z1)])
    xs = [p[0] for p in corners]
    zs = [p[1] for p in corners]
    return rectangle([min(xs), min(zs), max(xs), max(zs)])
