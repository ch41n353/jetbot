#!/usr/bin/env python3
"""Read-only audit of the demo's public progress; never commands motors."""
import datetime
import json
import os
import time
import urllib.request

ROOT = '/mnt/robotlogs/goals/advil-audit-20260920'
URL = 'http://127.0.0.1:8770/api/progress'

def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def main():
    if not os.path.ismount('/mnt/robotlogs'):
        raise RuntimeError('robot log volume not mounted')
    os.makedirs(ROOT, exist_ok=True)
    previous = None
    with open(ROOT + '/events.jsonl', 'a') as raw, open(ROOT + '/mission-history.md', 'a') as doc:
        doc.write('# Advil mission audit — 2026-09-20\n\n'
                  'This records observable actions and decision summaries, not private reasoning. '
                  'UTC timestamps below are observation times, not exact execution times. '
                  'The existing demo exposes untimestamped text and does not expose all motor commands '
                  'or complete GPT request/response payloads. Poll interval: 3 seconds.\n\n'
                  '## Initial high-level handoff (before logging began; exact time unavailable)\n\n'
                  '`POST http://192.168.86.158:8770/api/search`\n\n'
                  '```json\n' + json.dumps(dict(target=INSTRUCTION, seconds=8), indent=2) + '\n```\n\n'
                  'Rationale: delegate systematic search and trajectory planning to GPT; '
                  'use the local controller for execution. Astra issued no individual motor, '
                  'turn, or waypoint commands. Search hands off to recurrent approach automatically. '
                  'The 8-second field is the approach motion interval between model looks.\n\n'
                  'Preflight: fresh image inspected, sensors healthy, motor output zero, '
                  'battery 11.944 V, power guard clear. Initial plan: overlapping scan, '
                  'then observed clear-floor travel to inspect candidates.\n\n'
                  '## Observed planner/controller progress\n\n')
        doc.flush()
        while True:
            now = stamp()
            try:
                with urllib.request.urlopen(URL, timeout=5) as response:
                    state = json.load(response)
                if state != previous:
                    raw.write(json.dumps(dict(observed_at=now, state=state)) + '\n')
                    raw.flush()
                    oldlog = (previous or {}).get('log', '')
                    log = state.get('log', '')
                    new = log[len(oldlog):] if oldlog and log.startswith(oldlog) else log
                    if new.strip() or previous is None or state.get('running') != previous.get('running'):
                        doc.write('### Observed at ' + now + '\n\n')
                        doc.write('Running: %s. %s.\n\n' % (state.get('running'), state.get('power')))
                        if new.strip():
                            doc.write('```text\n' + new.strip() + '\n```\n\n')
                        doc.flush()
                    previous = state
            except Exception as exc:
                raw.write(json.dumps(dict(observed_at=now, error=str(exc))) + '\n')
                raw.flush()
            time.sleep(3)

INSTRUCTION = ('Find and approach the Advil medicine bottle in this room. '
               'Search systematically if unseen; inspect uncertain bottle candidates before identifying them. '
               'Stay in this room. Plan trajectories on clear carpet around all blocks, cables, furniture '
               'and other obstacles, leaving room for the entire robot. Preserve the goal across views '
               'and stop at a safe standoff from the bottle. Do not confuse the can of nuts with Advil.')

if __name__ == '__main__':
    main()
