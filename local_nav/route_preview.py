"""Deterministic straight-route illustration; does not assert obstacle clearance."""
import base64
import json
import math
import os
import cv2
import numpy as np
from point_controller import ROOT
from route_geometry import finite


def write_preview(image_path, distance_cm, destination):
    distance_cm = finite(distance_cm)
    if not 1 <= distance_cm <= 15:
        raise ValueError('Preview distance must be 1 to 15 cm')
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

    front = distance_cm + 13
    points = []
    corners = [(-16,0), (-16,front), (16,front), (16,0), (-16,0)]
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
                  SWEEP_Y='%.2f' % (170-5*front), SWEEP_HEIGHT='%.2f' % (5*(front+24)),
                  TARGET_Y='%.2f' % (170-5*distance_cm))
    for key,value in values.items():
        markup = markup.replace(key,value)
    with open(destination, 'w') as target:
        target.write(markup)
    return destination
