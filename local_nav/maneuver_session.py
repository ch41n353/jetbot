#!/usr/bin/env python3
"""Keep sensing warm across preview and execution; JSON commands on stdin.

Starts disarmed unless --enable-motion is explicit. Commands: capture, execute,
status, stop, shutdown. Stop remains responsive while the route subprocess runs.
This supervises the existing bounded route tool; it does not extend its limits.
"""
import argparse
import base64
import datetime
import json
import os
import select
import subprocess
import sys
import time
import uuid
from point_controller import ROOT, call


def route_command(plan, log, predictive=False, feature_budget=250, batch=False):
    if type(predictive) is not bool or type(feature_budget) is not int or feature_budget not in (80, 125, 250):
        raise ValueError('Invalid route options')
    if not isinstance(plan, str) or not plan:
        raise ValueError('Expected plan file path')
    if type(batch) is not bool:
        raise ValueError('batch must be boolean')
    executable = 'approach_batch.py' if batch else 'route_executor.py'
    command = [sys.executable, os.path.join(ROOT, 'local_nav', executable),
               os.path.abspath(plan), '--execute', '--log', log,
               '--feature-budget', str(feature_budget)]
    if predictive and not batch:
        command.append('--predictive-braking')
    return command


def emit(**fields):
    print(json.dumps(fields), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--enable-motion', action='store_true')
    parser.add_argument('--preview-directory')
    parser.add_argument('--output-directory', default=os.path.join(ROOT, 'local_nav/goals'))
    args = parser.parse_args()
    os.makedirs(args.output_directory, exist_ok=True)
    name = 'session-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6]
    prefix = os.path.join(os.path.abspath(args.output_directory), name)
    started = time.monotonic()
    service = worker = None
    service_output = open(prefix + '.service.log', 'w')
    worker_output = None
    route_started = None
    route_log = None
    events = []
    run_count = 0
    input_buffer = b''
    last_capture = None
    try:
        # Do not replace, attach to or shut down an unrelated service.
        if os.path.exists('/tmp/jetbot-local-nav/control.sock'):
            raise RuntimeError('Existing service detected; stop that service first')
        command = [sys.executable, os.path.join(ROOT, 'local_nav/service.py')]
        if args.enable_motion:
            command.append('--enable-motion')
        service = subprocess.Popen(command, stdout=service_output, stderr=service_output)
        deadline = started + 15
        while True:
            if service.poll() is not None:
                raise RuntimeError('Sensing service exited during startup')
            try:
                status = call('status')
                if status['healthy']:
                    break
            except (RuntimeError, OSError):
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError('Sensing startup timed out')
            time.sleep(.1)
        own_session = status['session_id']
        emit(event='ready', motion_enabled=status['motion_enabled'],
             startup_seconds=time.monotonic()-started, log_prefix=prefix)
        while True:
            if service.poll() is not None:
                raise RuntimeError('Sensing service exited; session stopped')
            if worker is not None and worker.poll() is not None:
                worker_output.close()
                worker_output = None
                event = dict(event='route_finished', started_monotonic=route_started, finished_monotonic=time.monotonic(),
                             elapsed_seconds=time.monotonic()-route_started,
                             exit_code=worker.returncode, result_path=route_log)
                if os.path.exists(route_log):
                    with open(route_log) as result:
                        payload = json.load(result)
                    event['result'] = {k:v for k,v in payload.items() if k not in ('samples', 'settling_samples')}
                events.append(event)
                emit(**event)
                worker = None
            if b'\n' not in input_buffer:
                ready, _, _ = select.select([sys.stdin], [], [], .1)
                if not ready:
                    continue
                chunk = os.read(sys.stdin.fileno(), 65536)
                if not chunk:
                    break
                input_buffer += chunk
                if len(input_buffer) > 65536:
                    raise RuntimeError('Session command input exceeded 64 KiB')
                if b'\n' not in input_buffer:
                    continue
            line, input_buffer = input_buffer.split(b'\n', 1)
            try:
                request = json.loads(line)
                command = request['command']
                if command == 'shutdown':
                    break
                if command == 'stop':
                    call('stop')
                    if worker is not None and worker.poll() is None:
                        worker.terminate()
                    event = dict(event='stopped', monotonic=time.monotonic())
                    events.append(event)
                    emit(**event)
                elif command == 'status':
                    emit(event='status', status=call('status'))
                elif command == 'capture':
                    if worker is not None:
                        raise ValueError('Wait for route completion before creating a new plan')
                    observation = call('observation')
                    path = prefix + '-%.6f' % observation['time']
                    with open(path+'.jpg', 'wb') as image:
                        image.write(base64.b64decode(observation['jpeg_base64']))
                    metadata = {k:v for k,v in observation.items() if k != 'jpeg_base64'}
                    with open(path+'.json', 'w') as out:
                        json.dump(metadata, out, indent=2)
                    event = dict(event='captured', image_path=path+'.jpg', metadata_path=path+'.json',
                                 captured_monotonic=observation['time'], session_id=observation['session_id'],
                                 control_epoch=observation['control_epoch'])
                    if 'preview_distance_cm' in request:
                        if not args.preview_directory:
                            raise ValueError('Session needs --preview-directory for automatic previews')
                        from route_preview import write_preview
                        destination = os.path.join(args.preview_directory, name+'-%.6f-preview.html' % observation['time'])
                        event['preview_path'] = write_preview(path+'.jpg', request['preview_distance_cm'], destination)
                        event['clearance_checked'] = False
                    events.append(event)
                    emit(**event)
                    last_capture = event
                elif command == 'plan_approach':
                    if worker is not None or last_capture is None:
                        raise ValueError('Capture a stationary scene before planning an approach')
                    if not args.preview_directory:
                        raise ValueError('Approach planning needs --preview-directory')
                    from approach_plan import prepare
                    from point_controller import FloorTracker
                    from route_preview import write_preview
                    with open(os.path.join(ROOT, 'calibration/floor_geometry.json')) as source:
                        profile = json.load(source)
                    with open(profile['intrinsics_path']) as source:
                        tracker = FloorTracker(profile, json.load(source))
                    assessment = prepare(last_capture, request, tracker)
                    if 'plan' in assessment:
                        path = last_capture['image_path'][:-4] + '-approach-' + uuid.uuid4().hex[:6]
                        with open(path+'.plan.json', 'w') as out:
                            json.dump(assessment.pop('plan'), out, indent=2)
                        assessment['plan_path'] = path+'.plan.json'
                        assessment['preview_path'] = write_preview(last_capture['image_path'],
                            assessment['approach_distance_cm'],
                            os.path.join(args.preview_directory, os.path.basename(path)+'.html'),
                            batch=assessment['batch'])
                    event = dict(event='approach_planned', **assessment)
                    events.append(event)
                    emit(**event)
                elif command == 'execute':
                    if not args.enable_motion:
                        raise ValueError('Session is disarmed')
                    if worker is not None:
                        raise ValueError('A route is already running')
                    run_count += 1
                    route_log = prefix + '-route-%02d.json' % run_count
                    argv = route_command(request['plan'], route_log, request.get('predictive_braking', False),
                                         request.get('feature_budget', 250), request.get('batch', False))
                    worker_output = open(route_log+'.stdout.log', 'w')
                    with open(os.path.abspath(request['plan'])) as source:
                        plan_metadata = json.load(source)
                    route_started = time.monotonic()
                    event = dict(event='route_started', result_path=route_log, started_monotonic=route_started,
                                 plan_image_age_seconds=route_started-float(plan_metadata['captured_monotonic']))
                    worker = subprocess.Popen(argv, stdout=worker_output, stderr=worker_output)
                    events.append(event)
                    emit(**event)
                else:
                    raise ValueError('Unknown session command')
            except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
                # Bad commands cannot silently leave an active motor command running.
                call('stop')
                emit(event='command_rejected', reason=str(exc))
    finally:
        if service is not None and service.poll() is None:
            try:
                status = call('status')
                if status['session_id'] == locals().get('own_session', status['session_id']):
                    call('shutdown')
            except (OSError, RuntimeError):
                service.terminate()
        if worker is not None and worker.poll() is None:
            worker.terminate()
            worker.wait(timeout=5)
        if service is not None:
            try:
                service.wait(timeout=8)
            except subprocess.TimeoutExpired:
                service.terminate()
                service.wait(timeout=5)
        if worker_output is not None:
            worker_output.close()
        service_output.close()
        with open(prefix+'.session.json', 'w') as out:
            json.dump(dict(elapsed_seconds=time.monotonic()-started, events=events), out, indent=2)
        emit(event='session_closed', log_path=prefix+'.session.json')


if __name__ == '__main__':
    main()
