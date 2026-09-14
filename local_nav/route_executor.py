#!/usr/bin/env python3
"""Run a checked straight waypoint batch without model calls or waypoint stops.

Default is offline check only. Static map must be inspected by the planner;
this does not detect new obstacles. Turns remain unsupported in this revision.
"""
import argparse
import json
import math
import os
import time

from route_geometry import StraightRoute, ProgressGuard
from braking import braking_distance, SettlingCheck


def load_json(path):
    with open(path) as source:
        return json.load(source)


def execute(plan, route, log, predictive_braking=False):
    import cv2
    import numpy as np
    from point_controller import ROOT, FloorTracker, call, frame
    from state_estimator import AttitudeTimeline, PlanarState
    result = dict(outcome='stopped', samples=[], passed_waypoints_cm=[])
    started = None
    powered_seconds = None
    try:
        footprint = load_json(os.path.join(ROOT, 'calibration/robot_footprint.json'))
        if (footprint['width_cm'], footprint['length_cm'], footprint['camera_location']) != (12, 15, 'front_center'):
            raise RuntimeError('Footprint differs from checked geometry')
        status = call('status')
        token = {k: status[k] for k in ('session_id', 'control_epoch')}
        if any(plan.get(k) != v for k, v in token.items()):
            raise RuntimeError('Plan belongs to an old service/control generation')
        if status['motor']['output'] != [0, 0]:
            raise RuntimeError('Robot must be stopped before route execution')
        if not status['healthy'] or not status['motion_enabled']:
            raise RuntimeError('Service is not ready for motion')
        age = time.monotonic() - float(plan['captured_monotonic'])
        if not math.isfinite(age) or not 0 <= age <= 120:
            raise RuntimeError('Plan image expired; inspect and preview a fresh route')
        profile = load_json(os.path.join(ROOT, 'calibration/floor_geometry.json'))
        tracker = FloorTracker(profile, load_json(profile['intrinsics_path']))
        timeline = AttitudeTimeline(load_json(os.path.join(ROOT, 'calibration/imu_mount.json')))
        if predictive_braking:
            timeline.record_directory = log + '.observations'
            os.makedirs(timeline.record_directory, exist_ok=True)
        old, t0, a0 = frame(timeline, settled=False)
        anchor = cv2.imread(plan['image_path'])
        if anchor is None or anchor.shape != old.shape:
            raise RuntimeError('Missing or incompatible plan image')
        r, t, _ = tracker.motion(anchor, old)
        if float(np.linalg.norm(t)) > .3 or abs(math.atan2(r[1, 0], r[0, 0])) > math.radians(1):
            raise RuntimeError('Robot moved since preview; replan')
        state = PlanarState()
        progress = ProgressGuard()
        last_command = None
        initial_up = timeline.up.copy()
        while True:
            current, t1, a1 = frame(timeline, settled=False)
            now = time.monotonic()
            if last_command is not None and now - last_command > .18:
                raise RuntimeError('Control update exceeded lease budget')
            if t1 <= t0:
                time.sleep(.005)
                continue
            dt = t1 - t0
            if not 0 < dt <= .18:
                raise RuntimeError('Camera interval exceeded 180 ms')
            r, t, q = tracker.motion(old, current, a0, a1)
            pose = state.update(r, t, dt, q)
            x, z = pose['position_cm']
            route.check_pose(x, z, pose['yaw_degrees'])
            if timeline.up.dot(initial_up) < math.cos(math.radians(5)):
                raise RuntimeError('Tilt guard')
            elapsed = 0 if started is None else time.monotonic() - started
            result['samples'].append(dict(time=t1, elapsed=elapsed, **pose))
            result['passed_waypoints_cm'] = [v for v in route.waypoints if z >= v]
            lead = braking_distance(pose['velocity_cm_s'][1], time.monotonic() - t1, dt) if predictive_braking else 0.
            result['samples'][-1]['braking_lead_cm'] = lead
            if z >= route.waypoints[-1] or (predictive_braking and started is not None
                                           and route.waypoints[-1] - z <= lead):
                result['outcome'] = 'distance_threshold_reached'
                if predictive_braking:
                    call('stop')
                    stopped_at = time.monotonic()
                    powered_seconds = None if started is None else stopped_at - started
                    result.update(outcome='final_position_unverified', brake_position_cm=[x, z],
                                  brake_lead_cm=lead, settling_samples=[])
                    checker = SettlingCheck(route.waypoints[-1])
                    old, t0, a0 = current, t1, a1
                    deadline = stopped_at + 1.
                    while time.monotonic() < deadline:
                        current, t1, a1 = frame(timeline, settled=False)
                        if t1 <= t0:
                            time.sleep(.005)
                            continue
                        if t1 - t0 > .18:
                            raise RuntimeError('Tracking gap after stop; final position unknown')
                        r, t, q = tracker.motion(old, current, a0, a1)
                        final_pose = state.update(r, t, t1 - t0, q)
                        x, z = final_pose['position_cm']
                        route.check_pose(x, z, final_pose['yaw_degrees'])
                        result['settling_samples'].append(dict(time=t1, **final_pose))
                        # Only frames acquired after the stop can establish settling.
                        assessment = checker.update(t1, x, z) if t1 >= stopped_at else None
                        old, t0, a0 = current, t1, a1
                        if assessment:
                            status = call('status')
                            if status['motor']['output'] != [0, 0]:
                                raise RuntimeError('Motor output nonzero after stop')
                            result.update(assessment)
                            result['passed_waypoints_cm'] = [v for v in route.waypoints if z >= v]
                            break
                break
            if started is not None:
                if elapsed >= 2:
                    raise RuntimeError('Two-second powered limit; target not reached')
                progress.check(elapsed, z)
            # Vision gates every renewal. No background heartbeat can mask stale tracking.
            if last_command is not None and time.monotonic() - last_command > .18:
                raise RuntimeError('Processing exceeded lease budget')
            steer = float(np.clip(-.3 * state.yaw - .02 * x, -.025, .025))
            call('motors_hold', left=.16 + steer, right=.16 - steer, **token)
            last_command = time.monotonic()
            if started is None:
                started = last_command
            old, t0, a0 = current, t1, a1
    except Exception as exc:
        result.update(outcome='stopped', reason=str(exc))
    finally:
        try:
            call('stop')
        except Exception as exc:
            result.update(outcome='stopped', stop_error=str(exc))
        result['powered_seconds'] = powered_seconds if powered_seconds is not None else (None if started is None else time.monotonic() - started)
        with open(log, 'w') as out:
            json.dump(result, out, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--log')
    parser.add_argument('--predictive-braking', action='store_true',
                        help='Experimental early stop and post-stop arrival measurement')
    args = parser.parse_args()
    if args.execute and not args.log:
        parser.error('--execute requires --log')
    with open(args.plan) as source:
        plan = json.load(source)
    route = StraightRoute(plan)
    if args.execute:
        result = execute(plan, route, args.log, args.predictive_braking)
        print(json.dumps({k: v for k, v in result.items() if k not in ('samples', 'settling_samples')}, indent=2))
        return 0 if result['outcome'] in ('distance_threshold_reached', 'goal_reached') else 1
    print(json.dumps(dict(outcome='static_map_check_passed', swept_rectangle_cm=route.corridor,
                         waypoints_cm=route.waypoints, requires_fresh_preview=True), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
