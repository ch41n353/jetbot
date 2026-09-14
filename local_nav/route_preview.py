"""Deterministic straight-route illustration; does not assert obstacle clearance."""
import base64
import json
import math
import os
import cv2
import numpy as np
from point_controller import ROOT
from route_geometry import finite


def write_preview(image_path, distance_cm, destination, batch=False):
    distance_cm = finite(distance_cm)
    if type(batch) is not bool or not 1 <= distance_cm <= (30 if batch else 15):
        raise ValueError('Preview distance exceeds mode limit')
    with open(os.path.join(ROOT, 'calibration/floor_geometry.json')) as source:
        profile = json.load(source)
    with open(profile['intrinsics_path']) as source:
        intrinsics = json.load(source)
    pitch = math.radians(profile['pitch_degrees'])
    height = profile['camera_height_cm']

    def project(points):
        rays = np.array([[x, height*math.cos(pitch)-z*math.sin(pitch),
                          height*math.sin(pitch)+z*math.cos(pitch)] for x,z in points])
        uv, _ = cv2.fisheye.projectPoints(rays.reshape(-1,1,3), np.zeros(3), np.zeros(3),
                                        np.array(intrinsics['K']), np.array(intrinsics['D']))
        return uv.reshape(-1,2)

    front = distance_cm + (15 if batch else 13)
    half_width, rear = (20, 28) if batch else (16, 24)
    points = []
    corners = [(-half_width,0), (-half_width,front), (half_width,front), (half_width,0), (-half_width,0)]
    for a,b in zip(corners,corners[1:]):
        points.extend(np.array(a)*(1-f)+np.array(b)*f for f in np.linspace(0,1,30))
    boundary = project(points)
    goal = project([(0,distance_cm)])[0]
    path = project([(0,z) for z in np.linspace(1,distance_cm,20)])
    with open(os.path.join(ROOT, 'local_nav/assets/straight_preview.html')) as source:
        markup = source.read()
    with open(image_path, 'rb') as source:
        encoded = base64.b64encode(source.read()).decode('ascii')
    values = dict(STRAIGHT_DISTANCE='%g' % distance_cm, CAMERA_IMAGE='data:image/jpeg;base64,'+encoded,
                  FLOOR_POLYGON=' '.join('%.2f,%.2f' % tuple(v) for v in boundary),
                  FLOOR_LINE='M '+' L '.join('%.2f %.2f' % tuple(v) for v in path),
                  GOAL_X='%.2f' % goal[0], GOAL_Y='%.2f' % goal[1],
                  SWEEP_Y='%.2f' % (170-5*front), SWEEP_HEIGHT='%.2f' % (5*(front+rear)),
                  SWEEP_X=str(160-5*half_width), SWEEP_WIDTH=str(10*half_width),
                  TOP_VIEW_Y=str(min(0,160-5*front)), TOP_VIEW_HEIGHT=str(340-min(0,160-5*front)),
                  MANEUVER_LABEL='One inspected approach; local segments up to 15 cm' if batch else 'One local maneuver',
                  TARGET_Y='%.2f' % (170-5*distance_cm))
    for key,value in values.items():
        markup = markup.replace(key,value)
    with open(destination, 'w') as target:
        target.write(markup)
    return destination
