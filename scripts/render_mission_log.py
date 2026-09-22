#!/usr/bin/env python3
"""Keep a newest-first Markdown view of the append-only mission history."""
import os
import re
import time

ROOT = '/mnt/robotlogs/goals/advil-audit-20260920'

def render(text):
    starts = [m.start() for m in re.finditer(r'(?m)^#{2,3} ', text)]
    sections = [text[:starts[0]]] if starts else [text]
    sections += [text[a:b] for a, b in zip(starts, starts[1:] + [len(text)])]
    return ('# Advil mission — newest first\n\n'
            'UTC timestamps; observations are distinguished from command issuance. '
            '[Original chronological history](mission-history.md) · '
            '[Exact command records](commands.jsonl)\n\n' +
            '\n'.join(reversed(sections[1:])) + '\n' + sections[0])

if __name__ == '__main__':
    if not os.path.ismount('/mnt/robotlogs'):
        raise RuntimeError('robot log volume not mounted')
    history = ROOT + '/mission-history.md'
    output = ROOT + '/mission.md'
    if not os.path.exists(history):
        os.rename(output, history)
    previous = None
    while True:
        with open(history) as f:
            text = f.read()
        if text != previous:
            with open(output + '.tmp', 'w') as f:
                f.write(render(text))
            os.replace(output + '.tmp', output)
            previous = text
        time.sleep(3)
