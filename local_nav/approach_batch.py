#!/usr/bin/env python3
"""Execute one inspected straight approach as bounded local segments.

No model calls. Static obstacles are retained in the initial frame. Each segment
keeps the existing 15 cm / two-second limits. Any unverified result stops the batch.
This does not detect newly arriving obstacles or perform turns.
"""
import argparse
import json
import math
import os
import time

from route_geometry import StraightRoute, finite, rectangle, contains, overlap


class BatchRoute:
    def __init__(self, plan):
        self.distance = finite(plan['approach_distance_cm'])
        if not 1 <= self.distance <= 30:
            raise ValueError('Approach batch distance must be 1 to 30 cm')
        self.free = rectangle(plan['inspected_free_rectangle_cm'])
        self.obstacles = [rectangle(r) for r in plan['obstacle_rectangles_cm']]
        self.corridor = [-20, -28, 20, self.distance + 15]
        if not contains(self.free, self.corridor):
            raise ValueError('Whole approach enters uninspected space')
        if any(overlap(self.corridor, r) for r in self.obstacles):
            raise ValueError('Whole approach intersects a retained obstacle')

    def segment(self, capture, position, yaw):
        x, z = map(finite, position)
        yaw = finite(yaw)
        if abs(x) > 2 or abs(yaw) > 5 or z < -.5 or z > self.distance + 4:
            raise RuntimeError('Batch pose left its checked envelope')
        remaining = self.distance - z
        if remaining <= 1:
            return None
        distance = min(15, remaining)
        # Transform all corners of the local executor's full swept rectangle.
        # AABB containment is conservative, including offscreen body and obstacles.
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        points = [(x+c*u+s*v, z-s*u+c*v)
                  for u in (-16, 16) for v in (-24, distance+13)]
        bounds = [min(p[0] for p in points), min(p[1] for p in points),
                  max(p[0] for p in points), max(p[1] for p in points)]
        if not contains(self.corridor, bounds):
            raise RuntimeError('Next segment leaves the previewed batch corridor')
        if not contains(self.free, bounds) or any(overlap(bounds, r) for r in self.obstacles):
            raise RuntimeError('Next segment conflicts with retained map')
        local = {k: capture[k] for k in ('session_id', 'control_epoch',
                                        'captured_monotonic', 'image_path')}
        # This local rectangle is backed by the transformed-corner proof above;
        # it is not inferred free space from an absence of detections.
        local.update(waypoints_cm=[distance],
                     inspected_free_rectangle_cm=[-16, -24, 16, distance+13],
                     obstacle_rectangles_cm=[], map_proof_bounds_cm=bounds)
        return local


def execute_batch(plan, log, feature_budget=125):
    from point_controller import call
    from route_executor import execute
    result = dict(outcome='stopped', segments=[], intermediate_planner_calls=0,
                  requires_planner=False)
    started = time.monotonic()
    try:
        batch = BatchRoute(plan)
        capture = {k: plan[k] for k in ('session_id', 'control_epoch',
                                        'captured_monotonic', 'image_path')}
        position, yaw = [0., 0.], 0.
        position_variance = yaw_variance = 0.
        for index in range(4):
            if time.monotonic()-started > 15:
                raise RuntimeError('Batch deadline exceeded')
            local = batch.segment(capture, position, yaw)
            if local is None:
                result.update(outcome='approach_distance_reached_estimate',
                              final_position_cm=position, final_yaw_degrees=yaw)
                break
            if index == 3:
                raise RuntimeError('Batch segment limit reached')
            segment_log = log+'.segment-%02d.json' % (index+1)
            segment = execute(local, StraightRoute(local), segment_log,
                              predictive_braking=True, feature_budget=feature_budget)
            result['segments'].append(dict(log_path=segment_log, outcome=segment['outcome']))
            if segment['outcome'] != 'goal_reached' or 'stop_error' in segment:
                raise RuntimeError('Segment failed or final position unverified: '+segment['outcome'])
            final = segment['settling_samples'][-1]
            position_variance += final['position_sigma_cm']**2
            yaw_variance += final['yaw_sigma_degrees']**2
            if position_variance > 4 or yaw_variance > 16:
                raise RuntimeError('Accumulated batch uncertainty exceeded limits')
            dx, dz = segment['final_position_cm']
            c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            position = [position[0]+c*dx+s*dz, position[1]-s*dx+c*dz]
            yaw += final['yaw_degrees']
            # Successful predictive execution has exactly two owned stop calls.
            # Never adopt an arbitrary newer generation: an external stop cancels
            # continuation even when it lands between two segments.
            expected = dict(session_id=capture['session_id'], control_epoch=capture['control_epoch']+2)
            observation = call('observation')
            if any(observation[k] != v for k, v in expected.items()):
                raise RuntimeError('Batch cancelled between segments')
            # Anchor continuation to the actual settled pose, not an unregistered
            # newer photograph. The next executor rejects intervening movement.
            capture = dict(expected, **segment['final_anchor'])
            # Re-evaluate the original map and entire chassis at the next iteration.
        else:
            raise RuntimeError('Batch segment limit reached')
    except Exception as exc:
        result.update(outcome='stopped', reason=str(exc), requires_planner=True)
    finally:
        try:
            call('stop')
        except Exception as exc:
            result.update(outcome='stopped', stop_error=str(exc), requires_planner=True)
        result['elapsed_seconds'] = time.monotonic()-started
        with open(log, 'w') as out:
            json.dump(result, out, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--log')
    parser.add_argument('--feature-budget', type=int, choices=(80,125,250), default=125)
    args = parser.parse_args()
    with open(args.plan) as source:
        plan = json.load(source)
    batch = BatchRoute(plan)
    if not args.execute:
        print(json.dumps(dict(outcome='static_batch_check_passed', corridor_cm=batch.corridor)))
        return 0
    if not args.log:
        parser.error('--execute requires --log')
    result = execute_batch(plan, args.log, args.feature_budget)
    print(json.dumps(result))
    return 0 if result['outcome']=='approach_distance_reached_estimate' else 1


if __name__ == '__main__':
    raise SystemExit(main())
