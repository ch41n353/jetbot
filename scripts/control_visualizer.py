"""Read-only views of persisted planner requests, interventions and controller state."""
import copy
import datetime
import json
import pathlib
import re
import urllib.request


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def rows(path):
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    result = []
    for line in lines:
        try: result.append(json.loads(line))
        except ValueError: pass  # writer may still be finishing its last record
    return result


def calls(root):
    result = []
    for directory in (root / 'gpt').glob('*'):
        if not re.fullmatch('[a-f0-9]{32}', directory.name): continue
        events = rows(directory / 'events.jsonl')
        request = read_json(directory / 'request.json')
        if request is None: continue
        response = read_json(directory / 'response.json')
        answer, error = None, None
        if response is not None:
            try:
                answer = json.loads(''.join(c.get('text', '') for x in response.get('output', []) for c in x.get('content', []) if c.get('type') == 'output_text'))
            except (ValueError, TypeError): error = 'Response could not be parsed as planner JSON'
        failures = [e.get('error') for e in events if e.get('event') == 'error']
        if failures: error = failures[-1]
        dispatched = next((e.get('time') for e in events if e.get('event') == 'dispatch'), '')
        received = next((e.get('time') for e in events if e.get('event') == 'response'), None)
        latency = None
        if dispatched and received:
            try:
                fmt = '%Y-%m-%dT%H:%M:%S.%f%z'
                latency = (datetime.datetime.strptime(received.replace('+00:00', '+0000'), fmt)-datetime.datetime.strptime(dispatched.replace('+00:00', '+0000'), fmt)).total_seconds()
            except ValueError: pass
        result.append(dict(id=directory.name, dispatched=dispatched, received=received,
                           model=request.get('model'), answer=answer, error=error,
                           events=events, event_count=len(events), latency=latency, status='error' if error else ('returned' if response is not None else 'awaiting response'),
                           image='/api/gpt/%s/image-0.jpg' % directory.name,
                           clean_image=('/api/gpt/%s/clean.jpg' % directory.name) if (directory/'clean.jpg').exists() else None,
                           usage=(response or {}).get('usage')))
    return sorted(result, key=lambda c:c['dispatched'], reverse=True)


def interventions(root):
    commands = rows(root / 'commands.jsonl')
    indexed = {}
    for row in commands:
        key = row.get('issued_at')
        if key: indexed.setdefault(key, {}).update(row)
    combined = list(indexed.values())
    for event in rows(root / 'api-commands.jsonl'):
        # Keep helper rationale and actual dispatch together, not duplicate cards.
        matches = [c for c in combined if c.get('body') == event.get('body')
                   and c.get('url','').split('/')[-1] == event.get('url','').split('/')[-1]
                   and c.get('issued_at','')[:16] == event.get('issued_at','')[:16]]
        if matches:
            matches[-1]['api_received_at'] = event.get('issued_at')
        else:
            combined.append(event)
    return sorted(combined, key=lambda c:c.get('issued_at',''), reverse=True)


def detail(root, ident):
    directory = root / 'gpt' / ident
    request = read_json(directory / 'request.json')
    if request is None: return None
    request = copy.deepcopy(request)
    index = 0
    for message in request.get('input', []):
        for part in message.get('content', []):
            if part.get('type') == 'input_image':
                part['image_url'] = '/api/gpt/%s/image-%d.jpg' % (ident, index)
                index += 1
    return dict(request=request, response=read_json(directory / 'response.json'), events=rows(directory / 'events.jsonl'))


def planner_state():
    try:
        with urllib.request.urlopen('http://127.0.0.1:8770/api/progress', timeout=1) as response:
            return json.load(response)
    except Exception as exc:
        return dict(error=str(exc), running=None)
