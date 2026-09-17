"""Project locally planned swept rectangles over an unchanged camera image."""
import base64
import json
import math
import os
import cv2
import numpy as np
from point_controller import ROOT
from spatial_planner import SpatialPlanner
from spatial_executor import drive_group


def write_preview(plan,destination):
    goals=plan.get('goals_cm',[plan.get('goal_cm')])
    initial=tuple(plan.get('initial_pose_cm_degrees',[0.,0.,0.]))
    pose=initial
    route=dict(outcome='route_found',actions=[],elapsed_seconds=0.)
    for goal in goals:
        planner=SpatialPlanner(dict(plan,goal_cm=goal))
        if plan.get('direct_approach',False):
            from mapped_drive import nominal_bounds
            from spatial_planner import sweep,normalize
            heading=math.degrees(math.atan2(goal[0]-pose[0],goal[1]-pose[1]))
            distance=math.hypot(goal[0]-pose[0],goal[1]-pose[1])
            boxes=[nominal_bounds((pose[0],pose[1],heading),distance),
                   sweep(pose,('turn',normalize(heading-pose[2])),precise_turn=True)]
            envelope=[min(b[0] for b in boxes),min(b[1] for b in boxes),
                      max(b[2] for b in boxes),max(b[3] for b in boxes)]
            if not planner.clear(envelope):raise ValueError('Direct approach corridor is not clear')
            end=[goal[0],goal[1],heading]
            route['actions'].append(dict(kind='approach',value=distance,predicted_start_pose=pose,
                                         predicted_end_pose=end,swept_bounds_cm=envelope))
            pose=tuple(end)
            continue
        leg=planner.search(pose)
        if leg['outcome']!='route_found':
            raise ValueError('No complete local route to preview: '+leg['outcome'])
        pending=list(leg['actions'])
        while pending:
            action,count=drive_group(pending,plan.get('coalesce_drives',False))
            action['predicted_end_pose']=pending[count-1]['predicted_end_pose']
            action['swept_bounds_cm']=planner.envelope(action['predicted_start_pose'],
                                                       (action['kind'],action['value']))
            if count>1 and not planner.clear(action['swept_bounds_cm']):
                # Joining drives is a speed optimisation, not part of the route.
                # A longer run sweeps a wider envelope, so in a tight corridor
                # the join can fail where the planner's own primitives fit. Fall
                # back to the unjoined primitive rather than failing a route A*
                # already checked; the single drive is still verified below.
                action,count=drive_group(pending,False)
                action['predicted_end_pose']=pending[count-1]['predicted_end_pose']
                action['swept_bounds_cm']=planner.envelope(action['predicted_start_pose'],
                                                           (action['kind'],action['value']))
            if not planner.clear(action['swept_bounds_cm']):
                raise ValueError('Joined drive does not fit the inspected map')
            route['actions'].append(action)
            del pending[:count]
        route['elapsed_seconds']+=leg['elapsed_seconds']
        pose=tuple(leg['predicted_final_pose'])
    with open(os.path.join(ROOT,'calibration/floor_geometry.json')) as source:
        profile=json.load(source)
    with open(profile['intrinsics_path']) as source:
        intrinsics=json.load(source)
    pitch=math.radians(profile['pitch_degrees'])
    height=profile['camera_height_cm']
    down=np.array([0.,math.cos(pitch),math.sin(pitch)])
    if 'capture_attitude' in plan:
        down=np.asarray(plan['capture_attitude']['down_camera'],dtype=float)
        if down.shape!=(3,) or not np.isfinite(down).all() or np.linalg.norm(down)<.9:
            raise ValueError('Invalid preview capture attitude')
        down=down/np.linalg.norm(down)
    forward=np.array([0.,0.,1.])-down*down[2]
    forward=forward/np.linalg.norm(forward)
    right=np.cross(down,forward)
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
        rays=np.array([x*right+height*down+z*forward for x,z in points])
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
