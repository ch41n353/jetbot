#!/usr/bin/env python3
"""Replay recorded observations with no hardware access or motor commands."""
import argparse,glob,json,os
import cv2
from point_controller import FloorTracker,ROOT
from state_estimator import AttitudeTimeline

p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args()
profile=json.load(open(os.path.join(ROOT,'calibration','floor_geometry.json')))
i=json.load(open(profile['intrinsics_path']))
timeline=AttitudeTimeline(json.load(open(os.path.join(ROOT,'calibration','imu_mount.json'))))
tracker=FloorTracker(profile,i)
previous=None
for path in sorted(glob.glob(os.path.join(a.directory,'*.json'))):
    obs=json.load(open(path))
    if timeline.last is None and obs.get('attitude_initialization') in ('stationary_5deg','stationary_10deg'):
        timeline.initialize_stationary(obs['imu_samples'],obs['time'])
    else:
        timeline.feed(obs['imu_samples'])
    att=timeline.settled_at(obs['time']);image=cv2.imread(path[:-5]+'.jpg')
    if previous:
        rotation,translation,quality=tracker.motion(previous[0],image,previous[1],att)
        print(json.dumps(dict(time=obs['time'],translation_cm=translation.tolist(),**quality)))
    previous=image,att
print('Rejected acceleration updates:',timeline.rejected_acceleration_samples)
