"""Bind inspected map and local targets to the supervisor's exact fresh image."""
import math
import time
from spatial_planner import SpatialPlanner, predict


def prepare(capture, request):
    if request.get('image_path') != capture['image_path']:
        raise ValueError('Navigation request must name the latest captured image')
    age=time.monotonic()-capture['captured_monotonic']
    if not math.isfinite(age) or not 0<=age<=120:
        raise ValueError('Capture expired')
    plan={k:capture[k] for k in ('session_id','control_epoch','captured_monotonic','image_path')}
    plan.update(inspected_free_rectangle_cm=request['inspected_free_rectangle_cm'],
                obstacle_rectangles_cm=request['obstacle_rectangles_cm'],
                goals_cm=request['goals_cm'],goal_tolerance_cm=request.get('goal_tolerance_cm',3.))
    goals=plan['goals_cm']
    if not isinstance(goals,list) or not 1<=len(goals)<=4:
        raise ValueError('Expected one to four targets in the initial camera frame')
    pose=(0.,0.,0.)
    actions=[]
    travel=0.
    for goal in goals:
        route=SpatialPlanner(dict(plan,goal_cm=goal)).search(pose)
        if route['outcome']!='route_found':
            raise ValueError('Local planning: '+route['outcome'])
        for action in route['actions']:
            nxt=tuple(action['predicted_end_pose'])
            travel+=math.hypot(nxt[0]-pose[0],nxt[1]-pose[1])
            pose=nxt
            actions.append(action)
    if len(actions)>12 or travel>90:
        raise ValueError('Predicted mission exceeds 12 actions or 90 cm')
    return plan,actions
