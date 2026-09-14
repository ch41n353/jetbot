"""Prepare a bounded straight object approach from a planner-selected floor pixel.

No hardware access or automatic free-space inference. The pixel must mark the
object's carpet contact; obstacle bounds and inspected space come from the planner.
"""
import math
from route_geometry import StraightRoute, finite


def prepare(capture, request, tracker):
    if request['image_path'] != capture['image_path']:
        raise ValueError('Target pixel must refer to the latest captured image')
    pixel = request['target_base_pixel']
    if len(pixel) != 2:
        raise ValueError('Expected target carpet-contact pixel [u, v]')
    u, v = map(finite, pixel)
    if not (0 <= u < 640 and 0 <= v < 480):
        raise ValueError('Target pixel outside camera image')
    standoff = finite(request.get('standoff_cm', 15))
    batch = request.get('batch', False)
    if type(batch) is not bool:
        raise ValueError('batch must be boolean')
    if not 15 <= standoff <= 25:
        raise ValueError('Object standoff must be 15 to 25 cm')
    x, z = map(float, tracker.ground([[u, v]])[0])
    if not all(math.isfinite(a) for a in (x, z)) or not 5 <= z <= 80:
        raise ValueError('Target floor projection outside usable approach range')
    allowance = 5 if batch else 3
    assessment = dict(target_ground_cm=[x, z], standoff_cm=standoff, batch=batch,
                      bearing_degrees=math.degrees(math.atan2(x, z)),
                      projection_allowance_cm=allowance, independent_ground_truth=False)
    # A nearby off-axis object must not be called reached merely because z is small.
    if abs(x) > 6:
        return dict(assessment, outcome='alignment_required')
    distance = min(30 if batch else 15, math.floor(z - standoff - allowance))
    if distance < 1:
        return dict(assessment, outcome='within_standoff_band_estimate')
    plan = {k: capture[k] for k in ('session_id', 'control_epoch',
                                  'captured_monotonic', 'image_path')}
    plan.update(waypoints_cm=[distance],
                inspected_free_rectangle_cm=request['inspected_free_rectangle_cm'],
                obstacle_rectangles_cm=request['obstacle_rectangles_cm'],
                target_base_pixel=[u, v], target_ground_cm=[x, z],
                standoff_cm=standoff, projection_allowance_cm=allowance)
    if batch:
        from approach_batch import BatchRoute
        plan.pop('waypoints_cm')
        plan['approach_distance_cm'] = distance
        route = BatchRoute(plan)
    else:
        route = StraightRoute(plan)
    return dict(assessment, outcome='approach_plan_prepared', plan=plan,
                approach_distance_cm=distance, swept_rectangle_cm=route.corridor)
