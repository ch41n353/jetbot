"""Conservative illustration for an inspected turn; not an axle calibration."""
import os, math, json, base64
from functools import lru_cache
import cv2
import numpy as np
from point_controller import ROOT


@lru_cache(maxsize=24)
def turn_hull(degrees):
    if not math.isfinite(degrees) or not 0 < abs(degrees) <= 180:
        raise ValueError('Turn preview requires 0–180 degrees')
    # Same pivots, corners, 200 angles, 48 clearance samples and 7.1cm
    # allowance as the original loop. Vectorize only; do not shrink the sweep.
    pivots=np.array([[x,z] for x in (-6,6) for z in (-15,0)])
    delta=pivots[None,:,:]-pivots[:,None,:]
    angle=np.linspace(0,-math.copysign(math.radians(abs(degrees)+5),degrees),200)
    c,s=np.cos(angle),np.sin(angle)
    x=delta[:,:,0,None]*c-delta[:,:,1,None]*s
    z=delta[:,:,0,None]*s+delta[:,:,1,None]*c
    points=pivots[:,None,None,:]+np.stack((x,z),axis=-1)
    a=np.linspace(0,2*math.pi,48)
    offsets=7.1*np.column_stack((np.cos(a),np.sin(a)))
    expanded=(points[:,:,:,None,:]+offsets[None,None,None,:,:]).reshape(-1,2)
    hull=cv2.convexHull(np.float32(expanded)).reshape(-1,2)
    hull.setflags(write=False)
    return hull


def write_turn_preview(image_path, degrees, destination):
    hull=turn_hull(degrees)
    _,visible=cv2.intersectConvexConvex(hull,np.float32([[-100,0],[100,0],[100,100],[-100,100]]))
    profile=json.load(open(os.path.join(ROOT,'calibration/floor_geometry.json')))
    intrinsics=json.load(open(profile['intrinsics_path']))
    a=math.radians(profile['pitch_degrees']);h=profile['camera_height_cm']
    rays=np.array([[x,h*math.cos(a)-z*math.sin(a),h*math.sin(a)+z*math.cos(a)] for x,z in visible.reshape(-1,2)])
    uv,_=cv2.fisheye.projectPoints(rays.reshape(-1,1,3),np.zeros(3),np.zeros(3),np.array(intrinsics['K']),np.array(intrinsics['D']))
    with open(os.path.join(ROOT,'local_nav/assets/turn_sweep_preview.html')) as source:markup=source.read()
    with open(image_path,'rb') as source:encoded=base64.b64encode(source.read()).decode()
    values=dict(TURN_LABEL='%g° %s turn'%(abs(degrees),'right' if degrees>0 else 'left'),
                CAMERA_IMAGE='data:image/jpeg;base64,'+encoded,
                FLOOR_POLYGON=' '.join('%.1f,%.1f'%tuple(q) for q in uv.reshape(-1,2)),
                TOP_POLYGON=' '.join('%.1f,%.1f'%(160+3*x,120-3*z) for x,z in hull),
                DIRECTION_LINE='M160 100 Q210 100 210 145' if degrees>0 else 'M160 100 Q110 100 110 145')
    for key,value in values.items():markup=markup.replace(key,value)
    with open(destination,'w') as target:target.write(markup)
    return dict(preview_path=destination,bounds_cm=[*hull.min(0).tolist(),*hull.max(0).tolist()])
