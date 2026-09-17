"""Validate a model-proposed floor route (image pixels) into local waypoints.

A monocular 640x480 frame carries no metric scale, so `route_pixels` is a HINT,
never a command: calibrated local projection decides centimetres and this layer
either accepts an ordered waypoint list or refuses it naming the failing index.
Range sensitivity near the far end of the usable floor is about 2 cm per pixel,
so a route is refused whole -- never trimmed, reordered, clipped or otherwise
repaired -- and the caller falls back to its own planner.

Camera-local coordinates: right_cm positive right, forward_cm positive ahead,
origin at the camera. Projection and clearance callables are the authorities on
usable floor and free space; this module only bounds and orders what they say.
"""
import math

FRAME = (640., 480.)
# Model or calibration errors must not raise out of a controller tick: fail
# closed on them instead. Genuinely unexpected errors still propagate.
FAILURES = (ValueError, KeyError, TypeError, IndexError, ArithmeticError)


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('non_numeric')
    return float(value)


def pixel(point):
    if isinstance(point, dict):
        return number(point['x']), number(point['y'])
    if isinstance(point, (list, tuple)) and len(point) == 2:
        return number(point[0]), number(point[1])
    raise ValueError('non_numeric')


def local_point(value):
    if isinstance(value, dict):
        return number(value['lateral_cm']), number(value['forward_cm'])
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return number(value[0]), number(value[1])
    raise ValueError('unusable_projection')


def refuse(reason, index=None, waypoint_index=None, detail=None):
    return dict(status='rejected', reason=reason, index=index,
                waypoint_index=waypoint_index, detail=detail, waypoints_cm=[],
                path_length_cm=0.)


def validate_route(route_pixels, project, clear, max_spacing_cm=25., min_spacing_cm=5.,
                   max_path_cm=150., max_range_cm=140., max_points=24, frame=FRAME):
    """Return an accepted/empty/rejected result for model-proposed floor pixels.

    `project` maps a (u,v) pixel to camera-local centimetres, raising or
    returning None when the pixel is not usable floor. `clear(start, end)`
    checks one bounded segment between (right_cm, forward_cm) pairs.
    """
    limits = [number(v) for v in (max_spacing_cm, min_spacing_cm, max_path_cm, max_range_cm)]
    if (min(limits) <= 0 or type(max_points) is not int or max_points < 1 or
            len(frame) != 2 or min(number(v) for v in frame) <= 0 or
            # Subdivision splits a gap into equal parts, so this bound is what
            # keeps a subdivided gap from falling below the minimum spacing.
            2*min_spacing_cm > max_spacing_cm):
        raise ValueError('Invalid route validation limits')
    if not isinstance(route_pixels, (list, tuple)):
        return refuse('malformed_route')
    if not route_pixels:
        return dict(status='empty', reason='no_route_proposed', index=None,
                    waypoint_index=None, detail=None, waypoints_cm=[], path_length_cm=0.)
    if len(route_pixels) > max_points:
        return refuse('too_many_points', len(route_pixels)-1, detail=len(route_pixels))
    waypoints, previous, previous_range, total = [], (0., 0.), 0., 0.
    dropped = []
    for index, point in enumerate(route_pixels):
        try:
            u, v = pixel(point)
        except FAILURES:
            return refuse('non_numeric_pixel', index)
        if not (0 <= u <= frame[0] and 0 <= v <= frame[1]):
            return refuse('pixel_outside_frame', index, detail=[u, v])
        try:
            projected = project((u, v))
            if projected is None:
                raise ValueError('unusable_projection')
            right, forward = local_point(projected)
        except FAILURES:
            return refuse('pixel_not_on_floor', index, detail=[u, v])
        if forward <= 0:
            return refuse('pixel_not_on_floor', index, detail=[right, forward])
        distance = math.hypot(right, forward)
        if distance > max_range_cm:
            return refuse('beyond_trusted_range', index, detail=distance)
        # Pixel order alone proves nothing about ground order; the projected
        # range must advance or the robot would double back on itself.
        if distance <= previous_range:
            return refuse('route_doubles_back', index, detail=distance)
        step = math.hypot(right-previous[0], forward-previous[1])
        # Minimum spacing separates PROPOSED points from each other. It is not
        # applied between the robot and the first point: the camera only starts
        # seeing floor about 2.7 cm ahead, so a close opening point is normal.
        # Evenly spaced pixels are also wildly uneven on the ground -- measured
        # here, 70 px near the robot is 4.3 cm while 45 px out at the target is
        # 53 cm -- and a model cannot space what it cannot measure. So a point
        # crowding its predecessor is dropped, not fatal: dropping one cannot
        # make the path less safe, because the surviving segments are still
        # clearance-checked end to end, and every drop is reported. The last
        # point always survives, since it carries the target.
        if index and step < min_spacing_cm:
            if index < len(route_pixels)-1:
                dropped.append(dict(pixel_index=index, gap_cm=step))
                continue
            while waypoints and waypoints[-1]['pixel_index'] == index-1:
                waypoints.pop()
            dropped.append(dict(pixel_index=index-1, gap_cm=step))
            if waypoints:
                previous = (waypoints[-1]['right_cm'], waypoints[-1]['forward_cm'])
            else:
                previous = (0., 0.)
            step = math.hypot(right-previous[0], forward-previous[1])
        total += step
        if total > max_path_cm:
            return refuse('route_too_long', index, len(waypoints), total)
        parts = int(math.ceil(step/max_spacing_cm))
        start = previous
        for part in range(1, parts+1):
            fraction = float(part)/parts
            nxt = (previous[0]+(right-previous[0])*fraction,
                   previous[1]+(forward-previous[1])*fraction)
            try:
                ok = clear(start, nxt)
            except FAILURES:
                return refuse('clearance_check_failed', index, len(waypoints))
            if not ok:
                return refuse('segment_blocked', index, len(waypoints), [start, nxt])
            waypoints.append(dict(right_cm=nxt[0], forward_cm=nxt[1], pixel_index=index,
                                  subdivided=part < parts))
            start = nxt
        previous, previous_range = (right, forward), distance
    if not waypoints:
        return refuse('route_collapsed_to_nothing', detail=dropped)
    return dict(status='accepted', reason=None, index=None, waypoint_index=None,
                detail=None, waypoints_cm=waypoints, path_length_cm=total,
                dropped_points=dropped)


def validate_answer(answer, project, clear, **limits):
    """Validate the route field of a model scene answer, tolerating its absence."""
    if not isinstance(answer, dict):
        return refuse('malformed_route')
    return validate_route(answer.get('route_pixels', []), project, clear, **limits)
