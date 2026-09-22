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
if path not in ('ask', 'mission', 'flow', 'turn', 'halt', 'capture', 'project', 'run', 'inspection'):
    raise ValueError('unsupported endpoint')
url = 'http://127.0.0.1:8770/api/' + path
issued = datetime.datetime.now(datetime.timezone.utc).isoformat()
event = dict(issued_at=issued, method='POST', url=url, body=body, rationale=reason)
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
