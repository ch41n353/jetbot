#!/usr/bin/env python3
"""Stop a named demo approach on its reported visual standoff; no motor renewal."""
import datetime
import json
import os
import re
import time
import urllib.request

ROOT = os.environ.get('JETBOT_AUDIT_ROOT', '/mnt/robotlogs/goals/advil-audit-20260920')
BASE = 'http://127.0.0.1:8770/api/'
MARKER = os.environ.get('JETBOT_APPROACH_MARKER', 'Approach the visible ADVIL bottle with blue cap')
CUTOFF = float(os.environ.get('JETBOT_STOP_RANGE_CM', '25'))

def last_range(log):
    found = re.findall(r'look \d+: sees it at (\d+) cm', log)
    return float(found[-1]) if found else None

if __name__ == '__main__':
    started = False
    for _ in range(600):
        with urllib.request.urlopen(BASE + 'progress', timeout=4) as response:
            state = json.load(response)
        log = state.get('log', '')
        if MARKER in log:
            started = True
            distance = last_range(log)
            if distance is not None and distance <= CUTOFF and state.get('running'):
                now = datetime.datetime.now(datetime.timezone.utc).isoformat()
                event = dict(issued_at=now, method='POST', url=BASE+'halt', body={},
                             rationale='Local observer: latest reported target range %.0f cm <= %.0f cm standoff.' % (distance, CUTOFF))
                with open(ROOT+'/commands.jsonl','a') as f:
                    f.write(json.dumps(event)+'\n')
                with open(ROOT+'/mission-history.md','a') as f:
                    f.write('\n## Local stop '+now+'\n\n'+event['rationale']+
                            ' This is a model-derived range, not independent measurement.\n\nPOST /api/halt {}\n\n')
                with urllib.request.urlopen(urllib.request.Request(BASE+'halt',data=b'{}',
                       headers={'Content-Type':'application/json'}),timeout=4) as response:
                    print(response.read().decode(), flush=True)
                break
            if not state.get('running'):
                print('Approach ended; inspect outcome.', flush=True)
                break
        elif started:
            break
        time.sleep(.5)
