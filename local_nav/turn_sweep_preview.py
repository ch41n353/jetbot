"""Conservative illustration for an inspected turn; not an axle calibration."""
import os, math, json, base64
import cv2
import numpy as np
from point_controller import ROOT


def write_turn_preview(image_path, degrees, destination):
    if not math.isfinite(degrees) or not 0 < abs(degrees) <= 180:
        raise ValueError('Turn preview requires 0–180 degrees')
    points=[]
    # Dense angular samples plus 0.1 cm numerical envelope allowance.
    for px in [-6,6]:
        for pz in [-15,0]:
            pivot=np.array([px,pz])
            for angle in np.linspace(0,-math.copysign(math.radians(abs(degrees)+5),degrees),200):
                r=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
                for x in [-6,6]:
                    for z in [-15,0]:
                        q=pivot+r.dot(np.array([x,z])-pivot)
                        for a in np.linspace(0,2*math.pi,48):
                            points.append(q+7.1*np.array([math.cos(a),math.sin(a)]))
    hull=cv2.convexHull(np.float32(points)).reshape(-1,2)
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
