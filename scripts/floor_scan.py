#!/usr/bin/env python3
"""How far the carpet runs, per bearing, from one camera frame.

The local executor is not a collision checker, so something has to decide where
a randomised trajectory is allowed to go. This is that something: a free-range
profile measured off the floor itself, using the same calibrated lens the
executor uses to turn pixels into centimetres.

Carpet is separated from paint by texture, against an ABSOLUTE threshold. An
earlier version scaled the threshold to a reference patch at the bottom of the
frame -- "what the robot is standing on" -- which inverts the moment it matters:
nose to a wall, the bottom of the frame IS the wall, so the scan calibrated
itself to the obstacle and then reported 120 cm clear in every direction. It
drove into walls. Measured lower-third median texture: 6-7 on real carpet, 2.0
on the frame taken against a wall, 0.5-1.2 on bare paint.

Ranges are reported from the camera. Near the horizon a pixel carries a huge
range bracket, so a long reading means "clear for a while", never a measurement
of where the wall is -- callers must keep their own margin on top.
"""
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, 'local_nav'))

# Absolute, and used ONLY to decide whether the near field is a floor at all.
# A self-calibrated threshold fails against a wall, where the reference becomes
# the obstacle. Measured near-field texture: carpet 4-5, wooden floor 3.0,
# painted wall 1.0 -- so this separates "on a surface" from "nose to paint"
# without assuming which surface the robot is on.
FLOOR_TEXTURE_MIN = 2.0

# The two cues are applied separately, and asymmetrically, because they fail
# differently. Neither works alone across surfaces: texture separates carpet
# from paint but not wood from paint, and lightness separates wood from paint
# but not carpet from paint.
#
# Colour is compared as a distance -- a wall or a box is a different colour in
# either direction. Texture is compared as a ONE-SIDED ratio: a surface much
# smoother than the floor is paint, but rougher is just clutter, which colour
# already catches. Folding texture into a symmetric distance made distant floor
# fail the test on its own, because carpet roughness legitimately falls with
# range (measured 5 near, 2 at the horizon) and that drift alone exhausted the
# budget: open carpet that reads 92 cm clear collapsed to 26.
COLOUR_TOLERANCE = 18.
TEXTURE_RATIO_MIN = .35
TEXTURE_WEIGHT = 4.
NEAR_CHECK_CM = 22.          # the robot cannot be standing on an obstacle
PATCH = 4                    # half-width of the patch sampled at each step
STEP_CM = 2.
NEAR_CM = 14.                # closer than this the camera sees mostly chassis
CONSECUTIVE = 2              # steps that must agree before calling it blocked
HALF_WIDTH_CM = 7.           # chassis half-width plus a little; a ray is not a robot


def _lens():
    from evaluate_gpt_routes import Lens
    import fetch
    return Lens(), fetch


def texture_map(image):
    return np.abs(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                                cv2.CV_32F))


def _rough(texture, x, y):
    h, w = texture.shape[:2]
    x0, x1 = max(0, x - PATCH), min(w, x + PATCH + 1)
    y0, y1 = max(0, y - PATCH), min(h, y + PATCH + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return float(np.median(texture[y0:y1, x0:x1]))


def _feature(lab, texture, x, y):
    """(L, a, b, texture*weight) for the patch at a pixel, or None if off-frame."""
    h, w = texture.shape[:2]
    x0, x1 = max(0, x - PATCH), min(w, x + PATCH + 1)
    y0, y1 = max(0, y - PATCH), min(h, y + PATCH + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    block = lab[y0:y1, x0:x1].reshape(-1, 3)
    return np.array([np.median(block[:, 0]), np.median(block[:, 1]),
                     np.median(block[:, 2]),
                     np.median(texture[y0:y1, x0:x1]) * TEXTURE_WEIGHT])


def surface_reference(image):
    """What the floor under the robot looks like, or None if it is not floor.

    Sampled over an arc of the near field rather than one patch, so a single
    cable or seam under the nose cannot define the surface.
    """
    lens, fetch = _lens()
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    texture = texture_map(image)
    height, width = image.shape[:2]
    features, roughness = [], []
    for bearing in (-30, -15, 0, 15, 30):
        angle = math.radians(bearing)
        for r in (NEAR_CM, NEAR_CM + 4., NEAR_CHECK_CM):
            spot = fetch.Robot.pixel(lens, r * math.sin(angle), r * math.cos(angle))
            if spot is None:
                continue
            x, y = int(round(spot[0])), int(round(spot[1]))
            if not (0 <= x < width and 0 <= y < height):
                continue
            value = _feature(lab, texture, x, y)
            if value is not None:
                features.append(value)
                roughness.append(value[3] / TEXTURE_WEIGHT)
    if not features:
        return None
    if float(np.median(roughness)) < FLOOR_TEXTURE_MIN:
        return None                       # nose to paint: no usable reference
    return np.median(np.asarray(features), axis=0)


def standing_on_floor(image):
    """Is the robot's immediate surround a floor at all?

    False means the camera is filled by something that is not floor -- against a
    wall, nosed into furniture -- and no bearing should be called clear.
    """
    return surface_reference(image) is not None


def _floor_at(state, right, forward):
    lab, texture, lens, fetch, reference, width, height = state
    spot = fetch.Robot.pixel(lens, right, forward)
    if spot is None:
        return None
    x, y = int(round(spot[0])), int(round(spot[1]))
    if not (0 <= x < width and 0 <= y < height):
        return None
    value = _feature(lab, texture, x, y)
    if value is None:
        return None
    colour = float(np.linalg.norm(value[:3] - reference[:3]))
    smooth = value[3] < reference[3] * TEXTURE_RATIO_MIN
    return colour <= COLOUR_TOLERANCE and not smooth


def _state(image):
    reference = surface_reference(image)
    if reference is None:
        return None
    lens, fetch = _lens()
    return (cv2.cvtColor(image, cv2.COLOR_BGR2Lab), texture_map(image),
            lens, fetch, reference, image.shape[1], image.shape[0])


def free_range(image, bearing_deg, max_cm=120.):
    """Centimetres of floor along `bearing_deg` before something is not floor.

    A corridor, not a ray: each step also samples the chassis half-width either
    side, because a 12 cm robot does not fit through a gap its centreline
    happens to clear. Anything that cannot be seen -- off the edge of the frame,
    behind the horizon -- ends the range there. Unseen is never clear.
    """
    state = _state(image)
    if state is None:
        return 0.

    angle = math.radians(bearing_deg)
    across = (math.cos(angle), -math.sin(angle))
    strikes = 0
    reach = 0.
    r = NEAR_CM
    while r <= max_cm:
        centre = (r * math.sin(angle), r * math.cos(angle))
        verdicts = []
        for offset in (-HALF_WIDTH_CM, 0., HALF_WIDTH_CM):
            verdicts.append(_floor_at(
                state, centre[0] + across[0] * offset,
                centre[1] + across[1] * offset))
        if any(v is None for v in verdicts):
            return reach                      # cannot see the whole corridor
        if not all(verdicts):
            strikes += 1
            if strikes >= CONSECUTIVE:
                return max(0., r - CONSECUTIVE * STEP_CM)
        else:
            strikes = 0
            reach = r
        r += STEP_CM
    return max_cm


def profile(image, bearings=None, max_cm=120.):
    """Free range for each bearing, as a list of (bearing_deg, range_cm).

    Everything is zero when the robot is not standing on visible floor, so a
    caller that only looks at this profile still cannot be told to drive on.
    """
    if bearings is None:
        bearings = list(range(-75, 76, 5))
    if not standing_on_floor(image):
        return [(b, 0.) for b in bearings]
    return [(b, free_range(image, b, max_cm)) for b in bearings]


def clear_path(image, waypoints, margin_cm=12.):
    """Is every leg of a metric path over floor, including between waypoints?

    Checking only the straight line from the origin to each waypoint misses the
    corridor a dogleg actually drives -- the second leg runs from the first
    waypoint, in a direction nothing ever scanned.
    """
    state = _state(image)
    if state is None:
        return False
    previous = (0., 0.)
    for point in waypoints:
        span = math.hypot(point[0] - previous[0], point[1] - previous[1])
        steps = max(1, int(span / STEP_CM))
        for i in range(steps + 1):
            t = float(i) / steps
            x = previous[0] + (point[0] - previous[0]) * t
            z = previous[1] + (point[1] - previous[1]) * t
            if math.hypot(x, z) < NEAR_CM:
                continue
            direction = math.atan2(point[0] - previous[0], point[1] - previous[1])
            across = (math.cos(direction), -math.sin(direction))
            for offset in (-margin_cm, 0., margin_cm):
                if not _floor_at(state, x + across[0] * offset,
                                 z + across[1] * offset):
                    return False
        previous = point
    return True


if __name__ == '__main__':
    for path in sys.argv[1:]:
        image = cv2.imread(path)
        if image is None:
            print('%s: unreadable' % path)
            continue
        standing = standing_on_floor(image)
        rows = profile(image)
        clear = [r for _, r in rows]
        print('%-24s on-floor=%-5s  min %3.0f  median %3.0f  max %3.0f cm'
              % (os.path.basename(path), standing, min(clear),
                 float(np.median(clear)), max(clear)))
        print('   ' + '  '.join('%+d:%.0f' % (b, r) for b, r in rows[::3]))
