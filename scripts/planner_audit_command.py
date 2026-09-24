#!/usr/bin/env python3
"""Send one explicit bench request and retain its timestamp and response on USB."""
import datetime
import json
import os
import sys
import urllib.request

root = os.environ.get('JETBOT_AUDIT_ROOT')
if not root:
    with open('/mnt/robotlogs/current-search.json') as pointer:
        root = json.load(pointer)['root']
if not os.path.ismount('/mnt/robotlogs'):
    raise RuntimeError('robot log volume not mounted')
path, body, reason = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3]
# A high-level destination handoff should finish locally. Explicit null still
# permits a deliberate one-look diagnostic; omission must not select it silently.
if path == 'mission' and 'seconds' not in body:
    body['seconds'] = 3
if path not in ('ask', 'mission', 'flow', 'turn', 'halt', 'capture', 'project', 'run', 'inspection', 'local-execution'):
    raise ValueError('unsupported endpoint')
url = 'http://127.0.0.1:8770/api/' + path
visual=body.pop('high_level_visual',None)
# A visible-destination handoff is an interface between the high- and mid-level
# planners.  Do not silently degrade it to prose: the operator must be able to
# inspect the exact goal and reference trajectory before/during execution.
if path in ('mission','flow'):
    visual = visual or body.get('initial_plan') or body.get('reference')
    if not isinstance(visual,dict):
        raise ValueError(path+' requires high_level_visual with route_pixels and goal_pixel')
    route = visual.get('route_pixels')
    goal = visual.get('goal_pixel') or visual.get('contact_pixel')
    if not isinstance(route,list) or not route:
        raise ValueError(path+' high_level_visual requires a non-empty route_pixels list')
    if not isinstance(goal,dict) or 'x' not in goal or 'y' not in goal:
        raise ValueError(path+' high_level_visual requires goal_pixel {x,y}')
    # GPT Drive consumes the same reference that the high-level panel displays.
    # This also makes the first local segment available without waiting for GPT.
    if path == 'mission' and 'initial_plan' not in body:
        body['initial_plan'] = visual
issued = datetime.datetime.now(datetime.timezone.utc).isoformat()
event = dict(issued_at=issued, method='POST', url=url, body=body, rationale=reason)
if path in ('mission','flow','ask','local-execution'):
    # Capture the staged submission frame before the command can start movement.
    import pathlib
    progress=json.load(urllib.request.urlopen('http://127.0.0.1:8770/api/progress',timeout=5))
    if progress.get('running'):raise RuntimeError('Cannot log a new handoff while controller is active')
    token=progress.get('frame_token')
    if visual and visual.get('frame_token')!=token:raise ValueError('High-level visual frame token is stale')
    if path == 'mission' and body.get('initial_plan') is visual:
        body['seed_frame_token'] = token
    data=urllib.request.urlopen('http://127.0.0.1:8770/api/frame?view=camera',timeout=5).read()
    after=json.load(urllib.request.urlopen('http://127.0.0.1:8770/api/progress',timeout=5))
    if after.get('frame_token')!=token:raise RuntimeError('Handoff frame changed during capture')
    name='handoff-'+issued.replace(':','-')+'.jpg'
    pathlib.Path(root,name).write_bytes(data)
    event.update(handoff_image=name,high_level_visual=visual,frame_token=token)
with open(root + '/commands.jsonl', 'a') as f:
    f.write(json.dumps(event) + '\n')
with open(root + '/mission-history.md', 'a') as f:
    f.write('\n## Command issued ' + issued + '\n\n' + reason + '\n\nPOST ' + url +
            '\n\n```json\n' + json.dumps(body, indent=2) + '\n```\n\n')
try:
    request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                    headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
except Exception as exc:
    result = {'request_error': str(exc)}
with open(root + '/commands.jsonl', 'a') as f:
    f.write(json.dumps(dict(response_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                           issued_at=issued, response=result)) + '\n')
print(json.dumps(result))
