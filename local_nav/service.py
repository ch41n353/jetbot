#!/usr/bin/env python3
"""Local-only persistent sensing with an independent motor lease watchdog."""
import argparse
import base64
from collections import deque
import fcntl
import json
import multiprocessing as mp
import os
import queue
import signal
import socket
import sys
import time

from core import allowed, ControlGeneration
from hardware import BNO055, ROOT, robot


def latest(q, value):
    try:
        q.put_nowait(value)
    except queue.Full:
        pass  # Bounded queue: samples retain acquisition timestamps even when delayed.


def camera_worker(q, stop):
    import cv2
    sys.path.insert(0, os.path.join(ROOT, 'jetbot', 'camera'))
    from color_balance import correct_bgr
    cap = None
    try:
        cap = cv2.VideoCapture('nvarguscamerasrc sensor-mode=3 wbmode=1 ! video/x-raw(memory:NVMM), width=1640, height=1232, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, width=640, height=480, format=(string)BGRx ! videoconvert ! appsink max-buffers=1 drop=true sync=false', cv2.CAP_GSTREAMER)
        start = time.monotonic()
        seq = 0
        while not stop.is_set():
            ok, frame = cap.read()
            timestamp = time.monotonic()
            if not ok:
                raise RuntimeError('Camera read failed')
            if timestamp-start < 3:
                continue
            ok, jpeg = cv2.imencode('.jpg', correct_bgr(frame))
            if not ok:
                raise RuntimeError('JPEG encoding failed')
            seq += 1
            latest(q, dict(time=timestamp, sequence=seq, jpeg=jpeg.tobytes()))
    except Exception as exc:
        latest(q, dict(fault=str(exc)))
    finally:
        if cap is not None:
            cap.release()


def imu_worker(q, stop):
    sensor = None
    try:
        sensor = BNO055()
        seq = 0
        while not stop.is_set():
            sample = sensor.sample()
            seq += 1
            sample.update(time=time.monotonic(), sequence=seq)
            latest(q, sample)
            stop.wait(.02)
    except Exception as exc:
        latest(q, dict(fault=str(exc)))
    finally:
        if sensor:
            sensor.close()


def motor_worker(pipe, enabled):
    # Separate process: parent stalls or dies -> lease expires -> stop.
    def terminate(*args):
        raise SystemExit()
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    r = robot()
    r.stop()
    current = {}
    output = (0, 0)
    try:
        pipe.send({'ready': True})
        while True:
            if pipe.poll(.01):
                try:
                    current = pipe.recv()
                except EOFError:
                    break
                if current.get('shutdown'):
                    break
            valid = allowed(current, time.monotonic(), enabled)
            desired = (current['left'], current['right']) if valid else (0, 0)
            if desired != output:
                if desired == (0, 0):
                    r.stop()
                else:
                    r.set_motors(*desired)
                output = desired
                pipe.send({'output': output, 'time': time.monotonic()})
    finally:
        r.stop()
        pipe.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-motion', action='store_true')
    parser.add_argument('--directory', default='/tmp/jetbot-local-nav')
    args = parser.parse_args()
    os.umask(0o077)
    os.makedirs(args.directory, mode=0o700, exist_ok=True)
    if os.stat(args.directory).st_uid != os.getuid() or os.stat(args.directory).st_mode & 0o077:
        raise RuntimeError('Runtime directory must be owned by this user with mode 0700')
    lock = open(os.path.join(args.directory, 'lock'), 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ctx = mp.get_context('spawn')
    stop = ctx.Event()
    signal.signal(signal.SIGTERM, lambda *a: stop.set())
    signal.signal(signal.SIGINT, lambda *a: stop.set())
    cameraq, imuq = ctx.Queue(2), ctx.Queue(4)
    parent, child = ctx.Pipe()
    motor = ctx.Process(target=motor_worker, args=(child, args.enable_motion))
    workers = [ctx.Process(target=camera_worker, args=(cameraq, stop)),
               ctx.Process(target=imu_worker, args=(imuq, stop))]
    path = os.path.join(args.directory, 'control.sock')
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    camera, imu = {}, {}
    imu_history = deque(maxlen=512)
    motor_status = {'output': [0,0]}
    generation = ControlGeneration()
    try:
        motor.start()
        child.close()
        if not parent.poll(5) or not parent.recv().get('ready'):
            raise RuntimeError('Motor watchdog did not start')
        for worker in workers:
            worker.start()
        if os.path.exists(path):
            os.unlink(path)
        server.bind(path)
        server.listen(4)
        server.settimeout(.02)
        print('READY '+path+' motion_enabled='+str(args.enable_motion), flush=True)
        while not stop.is_set():
            for q, name in ((cameraq, 'camera'), (imuq, 'imu')):
                while True:
                    try:
                        value = q.get_nowait()
                    except queue.Empty:
                        break
                    if name == 'camera': camera = value
                    else:
                        imu = value
                        if "time" in value:
                            imu_history.append(value)
            while parent.poll():
                motor_status.update(parent.recv())
            if not motor.is_alive():
                raise RuntimeError('Motor watchdog exited')
            now = time.monotonic()
            healthy = (all(w.is_alive() for w in workers) and now-camera.get('time', 0) < .5
                       and now-imu.get('time', 0) < .2 and imu.get('error') == 0
                       and imu.get('system_status') == 5)
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            with conn:
                conn.settimeout(.1)
                try:
                    data = b''
                    while b'\n' not in data and len(data) <= 4096:
                        chunk = conn.recv(4096)
                        if not chunk: break
                        data += chunk
                    if len(data) > 4096 or b'\n' not in data:
                        raise ValueError('Expected a newline-terminated JSON request <=4096 bytes')
                    request = json.loads(data.split(b'\n')[0])
                    action = request.get('action')
                    if action == 'status':
                        response = dict(healthy=healthy, motion_enabled=args.enable_motion,
                            camera={k:v for k,v in camera.items() if k != 'jpeg'}, imu=imu,
                            camera_age=now-camera.get('time',now), imu_age=now-imu.get('time',now), motor=motor_status)
                    elif action in ('snapshot', 'observation'):
                        if now-camera.get('time',0) >= .5: raise ValueError('Camera is stale')
                        target = os.path.join(args.directory, 'latest.jpg')
                        with open(target+'.tmp','wb') as f: f.write(camera['jpeg'])
                        os.replace(target+'.tmp',target)
                        response = {'path': target, 'time': camera['time']}
                        if action == 'observation':
                            if not healthy: raise ValueError('Sensors unhealthy')
                            since = float(request.get('since', camera['time']-2.0))
                            response.update(jpeg_base64=base64.b64encode(camera['jpeg']).decode('ascii'),
                                imu_samples=[s for s in imu_history if s['time'] >= since-.1],
                                healthy=healthy, motion_enabled=args.enable_motion,
                                timestamp_basis='host acquisition completion; hardware offset uncalibrated')
                    elif action in ('stop', 'motors', 'motors_hold'):
                        if action == 'stop':
                            generation.invalidate()
                        else:
                            generation.validate(request)
                        if action != 'motors_hold':
                            parent.send({})  # Original pulse behavior.
                        if action in ('motors', 'motors_hold'):
                            if not healthy or not args.enable_motion: raise ValueError('Motion disabled or sensors unhealthy')
                            issued = time.monotonic()
                            command = dict(left=request['left'],right=request['right'],issued=issued,
                                expires=issued+.2,camera_time=camera['time'],imu_time=imu['time'])
                            if not allowed(command, issued, True): raise ValueError('Invalid motor speeds; limit is +/-0.3')
                            parent.send(command)
                        response = {'accepted': True}
                    elif action == 'shutdown':
                        generation.invalidate()
                        parent.send({})
                        stop.set()
                        response = {'accepted':True}
                    else: raise ValueError('Unknown action')
                    response.update(generation.token())
                    conn.sendall(json.dumps(response).encode()+b'\n')
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    generation.invalidate()
                    parent.send({})
                    try: conn.sendall(json.dumps({'error':str(exc)}).encode()+b'\n')
                    except OSError: pass
    finally:
        stop.set()
        try: parent.send({'shutdown':True})
        except (OSError, BrokenPipeError): pass
        if motor.pid: motor.join(3)
        for worker in workers:
            if worker.pid:
                worker.join(3)
                if worker.is_alive(): worker.terminate(); worker.join(2)
        server.close()
        if os.path.exists(path): os.unlink(path)
        parent.close()


if __name__ == '__main__':
    main()
