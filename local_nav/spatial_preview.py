"""Project locally planned swept rectangles over an unchanged camera image."""
import base64
import json
import math
import os
import cv2
import numpy as np
from point_controller import ROOT
from spatial_planner import SpatialPlanner


def write_preview(plan,destination):
    goals=plan.get('goals_cm',[plan.get('goal_cm')])
    initial=tuple(plan.get('initial_pose_cm_degrees',[0.,0.,0.]))
    pose=initial
    route=dict(outcome='route_found',actions=[],elapsed_seconds=0.)
    for goal in goals:
        leg=SpatialPlanner(dict(plan,goal_cm=goal)).search(pose)
        if leg['outcome']!='route_found':
            raise ValueError('No complete local route to preview: '+leg['outcome'])
        route['actions'].extend(leg['actions'])
        route['elapsed_seconds']+=leg['elapsed_seconds']
        pose=tuple(leg['predicted_final_pose'])
    with open(os.path.join(ROOT,'calibration/floor_geometry.json')) as source:
        profile=json.load(source)
    with open(profile['intrinsics_path']) as source:
        intrinsics=json.load(source)
    pitch=math.radians(profile['pitch_degrees'])
    height=profile['camera_height_cm']
    polygons=[]
    for action in route['actions']:
        x0,z0,x1,z1=action['swept_bounds_cm']
        corners=[(x0,z0),(x0,z1),(x1,z1),(x1,z0)]
        angle=math.radians(initial[2])
        corners=[(math.cos(angle)*(x-initial[0])-math.sin(angle)*(z-initial[1]),
                  math.sin(angle)*(x-initial[0])+math.cos(angle)*(z-initial[1])) for x,z in corners]
        clipped=[]
        for a,b in zip(corners,corners[1:]+corners[:1]):
            if (a[1]>=0)!=(b[1]>=0):
                f=-a[1]/(b[1]-a[1])
                clipped.append((a[0]+f*(b[0]-a[0]),0.))
            if b[1]>=0:
                clipped.append(b)
        if len(clipped)<3:
            polygons.append([])
            continue
        points=[]
        for a,b in zip(clipped,clipped[1:]+clipped[:1]):
            points.extend(np.array(a)*(1-f)+np.array(b)*f for f in np.linspace(0,1,20))
        rays=np.array([[x,height*math.cos(pitch)-z*math.sin(pitch),
                        height*math.sin(pitch)+z*math.cos(pitch)] for x,z in points])
        uv,_=cv2.fisheye.projectPoints(rays.reshape(-1,1,3),np.zeros(3),np.zeros(3),
                                      np.array(intrinsics['K']),np.array(intrinsics['D']))
        polygons.append(uv.reshape(-1,2).round(2).tolist())
    with open(plan['image_path'],'rb') as source:
        image='data:image/jpeg;base64,'+base64.b64encode(source.read()).decode('ascii')
    data=dict(image=image,projected=polygons,actions=route['actions'],goal=goals[-1],goals=goals,initial_pose=initial,
              free=plan['inspected_free_rectangle_cm'],obstacles=plan['obstacle_rectangles_cm'])
    with open(os.path.join(ROOT,'local_nav/assets/spatial_preview.html')) as source:
        markup=source.read()
    with open(destination,'w') as output:
        output.write(markup.replace('SPATIAL_DATA',json.dumps(data).replace('</','<\\/')))
    return route
