"""Bounded maneuver/capture batches; no model calls or implicit disk writes."""
import base64
import datetime
import math
import threading
import cv2
import numpy as np
import fetch


def validate(body):
    steps = body.get('steps')
    if body.get('preset') == 'four_way':
        steps = [{'action': 'capture'}]
        for _ in range(3):
            steps += [{'action': 'turn', 'degrees': 90}, {'action': 'capture'}]
    if not isinstance(steps, list) or not 1 <= len(steps) <= 16:
        raise ValueError('steps must contain 1 to 16 actions')
    result = []
    turns = 0
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError('each step must be an object')
        action = step.get('action')
        if action == 'capture':
            result.append({'action': action})
        elif action == 'turn':
            degrees = float(step.get('degrees', 0))
            if not math.isfinite(degrees) or not 0 < abs(degrees) <= 180:
                raise ValueError('turn must be nonzero and within +/-180 degrees')
            turns += abs(degrees)
            result.append({'action': action, 'degrees': degrees})
        elif action == 'route':
            route = step.get('points_cm')
            obstacles = step.get('obstacles_cm')
            if not isinstance(route, list) or not 1 <= len(route) <= 12 or not isinstance(obstacles, list):
                raise ValueError('route needs 1-12 points_cm and explicit obstacles_cm')
            def point(p):
                if not isinstance(p, (list, tuple)) or len(p) != 2:
                    raise ValueError('floor point must be [x, forward]')
                values = [float(v) for v in p]
                if not all(math.isfinite(v) and abs(v) <= 300 for v in values):
                    raise ValueError('invalid floor coordinates')
                return values
            route = [point(p) for p in route]
            previous, length = [0, 0], 0
            for p in route:
                length += math.hypot(p[0]-previous[0], p[1]-previous[1]); previous = p
            if length > 100:
                raise ValueError('each route is limited to 100 cm')
            found = []
            for obstacle in obstacles:
                if not isinstance(obstacle, dict) or not isinstance(obstacle.get('label'), str):
                    raise ValueError('obstacles need label and point_cm')
                found.append((obstacle['label'], point(obstacle.get('point_cm'))))
            result.append(dict(action=action, points_cm=route, obstacles_cm=found))
        else:
            raise ValueError('supported actions: turn, route, capture')
    if turns > 360 or not any(s['action'] == 'capture' for s in result):
        raise ValueError('batch requires a capture and at most 360 degrees of turns')
    return result


def jpeg(image):
    ok, data = cv2.imencode('.jpg', image)
    if not ok:
        raise RuntimeError('JPEG encoding failed')
    return base64.b64encode(data).decode('ascii')


def run(bench, steps, result):
    tiles = []
    try:
        # Prevent concurrent frame acquisition or another manual controller.
        with bench.lock:
            for index, step in enumerate(steps):
                if bench.abort.is_set():
                    raise fetch.Stop('cancelled by operator')
                bench.robot.check()
                event = dict(step=index, action=step['action'],
                             started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
                if step['action'] == 'turn':
                    turned = bench.robot.turn(step['degrees'])
                    bench.heading += turned
                    shift = fetch.pivot_shift(turned)
                    bench.advance([shift[0], shift[1], turned])
                    event.update(requested_degrees=step['degrees'], measured_degrees=turned)
                    if abs(turned - step['degrees']) > 8:
                        result['events'].append(event)
                        raise fetch.Stop('turn outside 8 degree acceptance tolerance')
                elif step['action'] == 'route':
                    route, _ = fetch.avoid(step['points_cm'], step['obstacles_cm'])
                    if not route:
                        raise fetch.Stop('no usable route')
                    motion_events = []
                    def record(kind, **fields):
                        motion_events.append(dict(event=kind, **fields))
                    pose = fetch.follow(bench.robot, bench.odometer, route, record,
                                        step['obstacles_cm'], interrupt=bench.abort.is_set)
                    event.update(pose=pose, controller_events=motion_events)
                    bench.heading += pose[2]
                    bench.advance(pose)
                    result['events'].append(event)
                    if bench.abort.is_set():
                        raise fetch.Stop('cancelled during route')
                    if math.hypot(pose[0]-route[-1][0], pose[1]-route[-1][1]) > 5:
                        raise fetch.Stop('route stopped short of endpoint')
                    continue
                else:
                    # A fresh stopped frame, not the last dashboard thumbnail.
                    bench.robot.hold(0., 0.)
                    image, _ = bench.robot.frame()
                    if bench.abort.is_set():
                        raise fetch.Stop('cancelled during capture')
                    bench.restage(image)
                    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    shot = dict(index=len(result['images']), step=index, captured_at=stamp,
                                relative_heading_degrees=bench.heading-result['start_heading'],
                                jpeg_base64=jpeg(image))
                    result['images'].append(shot)
                    tile = cv2.resize(image, (640, 480))
                    cv2.putText(tile, '%d | heading %+.1f deg | %s' % (shot['index'], shot['relative_heading_degrees'], stamp[11:19]),
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
                    tiles.append(tile)
                    canvas = np.zeros((((len(tiles)+1)//2)*480, 1280, 3), dtype=np.uint8)
                    for n, tile in enumerate(tiles):
                        canvas[(n//2)*480:(n//2+1)*480, (n%2)*640:(n%2+1)*640] = tile
                    result['contact_sheet_base64'] = jpeg(canvas)
                    event.update(captured_at=stamp, image_index=shot['index'])
                result['events'].append(event)
                bench.say('inspection step %d: %s' % (index+1, step['action']))
            result['status'] = 'complete'
    except Exception as exc:
        result['status'] = 'cancelled' if bench.abort.is_set() else 'failed'
        result['error'] = str(exc)
        bench.say('inspection %s: %s' % (result['status'], exc))
    finally:
        try:
            bench.robot.hold(0., 0.)
        except Exception as exc:
            result.update(status='failed', error='stop failed: '+str(exc))
        result['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        bench.live_route = []
        bench.live_obstacles = []
        bench.memory = None


def launch(bench, body):
    steps = validate(body)
    if bench.running():
        return {'error': 'robot busy; stop the active task first'}
    result = dict(status='running', steps=steps, images=[], events=[], start_heading=bench.heading)
    bench.inspection_result = result
    bench.abort.clear()
    bench.flight = threading.Thread(target=run, args=(bench, steps, result), daemon=True)
    bench.flight.start()
    return {'inspection': True, 'result_url': '/api/inspection', 'status': 'running'}
