"""Opt-in Sol shadow perception. No motor access and no route mutation.

One request in flight, no backlog. Capture pose/attitude belong to the image,
not the response. Controller callers only copy a frame and poll a mailbox.
"""
import base64
import copy
import json
import math
import os
from pathlib import Path
import queue
import threading
import time
import urllib.request

from benchmark_vlm_planning import SCHEMA
from spatial_planner import transform


def request_scene(image, target, mission_context=None):
    import cv2
    ok, encoded = cv2.imencode('.jpg', image)
    if not ok:
        raise ValueError('JPEG encoding failed')
    body = dict(model='gpt-5.6-sol', reasoning=dict(effort='none'), store=False,
                max_output_tokens=4500,
                instructions=(Path(__file__).parent/'prompts/sol_scene.txt').read_text(),
                input=[dict(role='user', content=[
                    dict(type='input_text', text=json.dumps(dict(target=target, spatial_checks=[]))),
                    dict(type='input_image', detail='high', image_url='data:image/jpeg;base64,'+
                         base64.b64encode(encoded).decode('ascii'))])],
                text=dict(format=dict(type='json_schema', name='robot_scene_plan', strict=True, schema=SCHEMA)))
    if mission_context is not None:
        body['instructions']+='\n'+(Path(__file__).parent/'prompts/sol_search.txt').read_text()
        body['input'][0]['content'][0]['text']=json.dumps(dict(target=target,spatial_checks=[],mission=mission_context))
    request = urllib.request.Request('https://api.openai.com/v1/responses',
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json',
        'Authorization': 'Bearer '+os.environ['OPENAI_API_KEY']})
    with urllib.request.urlopen(request, timeout=12) as response:
        raw = json.load(response)
    if raw.get('status') != 'completed':
        raise ValueError('Incomplete scene response')
    answer = json.loads(''.join(c.get('text', '') for item in raw.get('output', [])
        if item.get('type') == 'message' for c in item.get('content', [])
        if c.get('type') == 'output_text'))
    return dict(answer=answer, usage=raw.get('usage'), model=raw.get('model'))


def assess(snapshot, response, now, pose, generation, ground):
    """Rebase a static floor contact using measured capture and current SE(2).

This is an advisory comparison, never clearance authorization. ground must use
the frozen capture attitude, including any mission-specific floor alignment.
"""
    age = float(now)-snapshot['time']
    result = dict(age_seconds=age, applied_to_control=False)
    try:
        values = list(pose)+list(snapshot['pose'])+[age]
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError('nonfinite_pose')
        if generation != snapshot['generation']:
            raise ValueError('generation_changed')
        if not 0 <= age <= 8:
            raise ValueError('stale_response')
        yaw = (pose[2]-snapshot['pose'][2]+180) % 360-180
        travel = math.hypot(pose[0]-snapshot['pose'][0], pose[1]-snapshot['pose'][1])
        if abs(yaw) > 15 or travel > 60:
            raise ValueError('motion_scope_exceeded')
        answer = response['answer']
        if answer['target_visible'] is not True:
            raise ValueError('target_absent')
        box, pixel = answer['target_box'], answer['contact_pixel']
        x0,y0,x1,y1 = [float(box[k]) for k in ('x0','y0','x1','y1')]
        u,v = float(pixel['x']),float(pixel['y'])
        if not (0 <= x0 < x1 <= 640 and 0 <= y0 < y1 <= 480 and
                x0 <= u <= x1 and y0 <= v <= y1 and abs(v-y1) <= 8):
            raise ValueError('invalid_contact_or_box')
        local = ground([[u,v]], snapshot['attitude'])[0]
        if not all(math.isfinite(float(v)) for v in local) or not 10 <= math.hypot(*local) <= 150:
            raise ValueError('invalid_floor_projection')
        world = transform(snapshot['pose'], *local)
        dx,dz = world[0]-pose[0], world[1]-pose[1]
        a = math.radians(pose[2])
        current = [math.cos(a)*dx-math.sin(a)*dz, math.sin(a)*dx+math.cos(a)*dz]
        result.update(status='shadow_observation', target_world_cm=list(world),
                      target_current_cm=current, moved_cm=travel, turned_degrees=yaw)
    except (ValueError, KeyError, TypeError, OverflowError, RuntimeError) as exc:
        result.update(status='rejected', reason=str(exc))
    return result


class AsyncScene:
    def __init__(self, request=request_scene, clock=time.monotonic,
                 min_interval_seconds=4., max_requests=30):
        if (not math.isfinite(float(min_interval_seconds)) or min_interval_seconds < 0 or
                type(max_requests) is not int or not 1 <= max_requests <= 72):
            raise ValueError('Invalid asynchronous recognition limits')
        self.request, self.clock = request, clock
        self.mailbox = queue.Queue(maxsize=1)
        self.pending = False
        self.closed = False
        self.next_request = 0.
        self.sent = 0
        self.min_interval_seconds = float(min_interval_seconds)
        self.max_requests = max_requests

    def submit(self, image, timestamp, pose, attitude, generation, target,
               mission_context=None):
        now = self.clock()
        if (self.closed or self.pending or now < self.next_request or
                self.sent >= self.max_requests):
            return False
        if not 0 <= now-timestamp <= .18:
            return False
        if image.shape[:2] != (480,640):
            return False
        snapshot = dict(time=float(timestamp), pose=list(pose),
                        attitude=copy.deepcopy(attitude), generation=copy.deepcopy(generation))
        frozen = image.copy()
        self.pending = True
        self.sent += 1
        self.next_request = now+self.min_interval_seconds

        def run():
            try:
                if mission_context is None:
                    response = self.request(frozen, target)
                else:
                    response = self.request(frozen, target, copy.deepcopy(mission_context))
                result = dict(response=response)
            except Exception as exc:
                # Do not serialize request headers or credentials in errors.
                result = dict(error=type(exc).__name__)
            result.update(snapshot=snapshot, finished=self.clock())
            if not self.closed:
                self.mailbox.put_nowait(result)
        threading.Thread(target=run, daemon=True).start()
        return True

    def poll(self):
        if self.closed:
            return None
        try:
            result = self.mailbox.get_nowait()
        except queue.Empty:
            return None
        self.pending = False
        return result

    def close(self):
        # Never wait for HTTP from motor cleanup; late replies cannot actuate.
        self.closed = True
