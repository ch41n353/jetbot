"""Fisheye pixels to floor coordinates. Uses measured height and pitch.

Coordinates: lateral positive right, forward positive ahead; pitch positive down.
A single centerline floor reference estimates pitch, assuming camera roll is zero.
That assumption and metric distance need independent validation before driving.
"""
import json
import math
import cv2
import numpy as np


def ray_from_pixel(pixel, intrinsics):
    point = np.asarray(pixel, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError('Expected a finite (u,v) pixel')
    if not (0 <= point[0] < intrinsics['image_width'] and 0 <= point[1] < intrinsics['image_height']):
        raise ValueError('Pixel outside calibrated image')
    xy = cv2.fisheye.undistortPoints(point.reshape(1,1,2),
        np.asarray(intrinsics['K'], dtype=float), np.asarray(intrinsics['D'], dtype=float))[0,0]
    return np.array([xy[0],xy[1],1.0])


def estimate_pitch(pixel, height_cm, forward_cm, intrinsics):
    if not (math.isfinite(height_cm) and math.isfinite(forward_cm) and height_cm > 0 and forward_cm > 0):
        raise ValueError('Reference height and forward distance must be positive')
    ray = ray_from_pixel(pixel, intrinsics)
    return math.degrees(math.atan2(height_cm,forward_cm)-math.atan2(ray[1],ray[2]))


def project_floor(pixel, profile, intrinsics, allow_unverified=False):
    if not allow_unverified and not profile.get('mounting_verified'):
        raise ValueError('Floor geometry not independently verified')
    height = profile['camera_height_cm']
    pitch = profile['pitch_degrees']
    roll = profile['roll_degrees']
    if pitch is None or roll is None:
        raise ValueError('Mounting angles missing')
    if roll != 0:
        raise ValueError('This initial model supports zero roll only')
    if not all(math.isfinite(v) for v in (height,pitch)) or height <= 0:
        raise ValueError('Invalid camera geometry')
    ray = ray_from_pixel(pixel,intrinsics)
    angle = math.radians(pitch)
    down = ray[1]*math.cos(angle)+ray[2]*math.sin(angle)
    forward = ray[2]*math.cos(angle)-ray[1]*math.sin(angle)
    if down <= .05 or forward <= 0:
        raise ValueError('Point does not intersect usable floor ahead')
    scale = height/down
    return dict(lateral_cm=float(ray[0]*scale),forward_cm=float(forward*scale))
