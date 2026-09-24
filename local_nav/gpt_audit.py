"""Persist exact model payloads; never include HTTP authorization headers."""
import base64
import datetime
import json
import os
import pathlib
import uuid


def enabled():
    """Audit by default on a configured robot run; permit explicit opt-out."""
    value = os.environ.get('JETBOT_GPT_AUDIT')
    if value is not None:
        return value.strip().lower() not in ('', '0', 'false', 'no', 'off')
    return pathlib.Path('/mnt/robotlogs/current-search.json').exists()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def command(endpoint, body):
    """Record actual incoming planner commands, including direct API callers."""
    if not enabled():
        return
    with open('/mnt/robotlogs/current-search.json') as f:
        root = pathlib.Path(json.load(f)['root']).resolve()
    if not str(root).startswith('/mnt/robotlogs/goals/'):
        raise RuntimeError('command audit requires a USB run directory')
    with (root / 'api-commands.jsonl').open('a') as f:
        f.write(json.dumps(dict(issued_at=now(), method='POST',
                               url=endpoint, body=body,
                               source='planner API received'))+'\n')


def plan_event(answer, event, **fields):
    path = answer.get('_audit_path')
    if path:
        with (pathlib.Path(path) / 'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(event=event, time=now(), **fields))+'\n')


class Capture:
    def __init__(self, body):
        self.path = None
        if not enabled():
            return
        with open('/mnt/robotlogs/current-search.json') as f:
            root = pathlib.Path(json.load(f)['root']).resolve()
        if not str(root).startswith('/mnt/robotlogs/goals/'):
            raise RuntimeError('GPT audit requires a USB run directory')
        self.path = root / 'gpt' / uuid.uuid4().hex
        self.path.mkdir(parents=True)
        (self.path / 'request.json').write_text(json.dumps(body))
        number = 0
        for message in body.get('input', []):
            for part in message.get('content', []):
                if part.get('type') == 'input_image':
                    data = part['image_url'].split(',', 1)[1]
                    (self.path / ('image-%d.jpg' % number)).write_bytes(base64.b64decode(data))
                    number += 1
        self.event('dispatch', model=body.get('model'))
        with (root / 'mission-history.md').open('a') as f:
            f.write('\n## GPT request '+now()+'\n\nArtifacts: '+str(self.path)+'\n')

    def event(self, kind, **fields):
        if self.path:
            with (self.path / 'events.jsonl').open('a') as f:
                f.write(json.dumps(dict(event=kind, time=now(), **fields))+'\n')

    def response(self, data):
        if self.path:
            (self.path / 'response.json').write_bytes(data)
            self.event('response')
