#!/usr/bin/env python3
import argparse
import json
import socket

p=argparse.ArgumentParser()
p.add_argument('action', choices=['status','snapshot','stop','shutdown'])
p.add_argument('--socket',default='/tmp/jetbot-local-nav/control.sock')
a=p.parse_args()
with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as s:
    s.settimeout(2)
    s.connect(a.socket)
    s.sendall(json.dumps({'action':a.action}).encode()+b'\n')
    data=b''
    while b'\n' not in data:
        chunk=s.recv(65536)
        if not chunk: break
        data+=chunk
    print(json.dumps(json.loads(data),indent=2))
