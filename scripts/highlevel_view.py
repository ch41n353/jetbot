"""Display only the image and plan explicitly published after high-level review."""
import base64,json
from pathlib import Path
import cv2
from control_visualizer import interventions
from midlevel_view import render_pair,pixels_to_floor
from trajectory_executor import encode

def view(root):
    review_path=root/'highlevel-review.json'
    try: command=json.loads(review_path.read_text())
    except (OSError,ValueError):
        return dict(available=False,note='No image has been reviewed by the high-level planner in this run yet.')
    body=command.get('body',{})
    image_path=root/command.get('review_image','')
    image=cv2.imread(str(image_path))
    if image is None:return dict(available=False,prompt=command.get('prompt',''),note='Reviewed image unavailable.')
    supplied=command.get('high_level_visual') or {}
    pixels=supplied.get('route_pixels') or []
    route=body.get('trajectory',{}).get('waypoints_cm') or []
    goal=supplied.get('goal_pixel') or supplied.get('contact_pixel') or (supplied.get('target_was') or {}).get('contact_pixel')
    if route:
        image=render_pair(image,route,target=goal)['camera']
    else:
        # High-level routes originate in this exact RGB frame. Distant targets
        # may sit above the calibrated floor horizon and cannot be converted to
        # metric ground coordinates yet; that must not erase the RGB handoff.
        valid=[]
        for point in pixels:
            try:
                x,y=int(round(float(point['x']))),int(round(float(point['y'])))
                if 0<=x<image.shape[1] and 0<=y<image.shape[0]:valid.append((x,y))
            except (KeyError,TypeError,ValueError):pass
        if len(valid)>1:cv2.polylines(image,[__import__('numpy').array(valid,__import__('numpy').int32)],False,(220,100,255),4)
        for point in valid:cv2.circle(image,point,4,(220,100,255),-1)
        if isinstance(goal,dict):
            try:
                at=(int(round(float(goal['x']))),int(round(float(goal['y']))))
                if 0<=at[0]<image.shape[1] and 0<=at[1]<image.shape[0]:cv2.drawMarker(image,at,(60,220,220),cv2.MARKER_STAR,22,2)
            except (KeyError,TypeError,ValueError):pass
        image=encode(image)
    return dict(available=True,image=image,prompt=command.get('prompt',''),goal=goal,trajectory=route,
                trajectory_pixels=pixels,
                issued_at=command.get('reviewed_at'),command=command,
                note='Latest image explicitly reviewed by the high-level planner. Magenta: supplied reference. Yellow star: supplied goal.'+
                     (' No reference trajectory was supplied.' if not (route or pixels) else '')+
                     (' Goal was supplied in words only; no high-level pixel marker was recorded.' if not goal else ''))
