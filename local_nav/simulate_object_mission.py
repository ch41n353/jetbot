"""Production mission and tracker with rendered target, carpet, IMU and plant."""
import json
import math
import os
import tempfile
from collections import deque
from unittest.mock import patch
import cv2
import numpy as np
from simulate_route import Plant
from point_controller import ROOT,FloorTracker
from route_executor import load_json
from object_mission import execute


class ObjectPlant(Plant):
    def __init__(self,target=(0,80),occlude=None,**parameters):
        super().__init__(**parameters)
        self.target=target
        self.occlude=occlude
        rng=np.random.RandomState(145)
        self.label=cv2.resize(rng.randint(0,255,(16,8,3)).astype(np.uint8),(80,160),interpolation=cv2.INTER_NEAREST)
        self.intrinsics=load_json(load_json(os.path.join(ROOT,'calibration/floor_geometry.json'))['intrinsics_path'])

    def render(self):
        image=super().render()
        c,s=math.cos(self.yaw),math.sin(self.yaw)
        pitch=math.asin(self.down[2]);forward=np.array([0.,-math.sin(pitch),math.cos(pitch)])
        rays=[]
        for dx,h in [(-3,12),(3,12),(3,0),(-3,0)]:
            wx,wz=self.target[0]+dx-self.x,self.target[1]-self.z
            x,z=c*wx-s*wz,s*wx+c*wz
            rays.append(np.array([x,0,0])+self.down*(9.5-h)+forward*z)
        uv,_=cv2.fisheye.projectPoints(np.array(rays).reshape(-1,1,3),np.zeros(3),np.zeros(3),
                                      np.array(self.intrinsics['K']),np.array(self.intrinsics['D']))
        corners=uv.reshape(-1,2).astype(np.float32)
        self.target_box=np.r_[corners.min(0),corners.max(0)].tolist()
        elapsed=0 if self.powered_start is None else self.time-self.powered_start
        if self.occlude is not None and self.occlude[0]<=elapsed<=self.occlude[1]:return image
        matrix=cv2.getPerspectiveTransform(np.float32([[0,0],[79,0],[79,159],[0,159]]),corners)
        warped=cv2.warpPerspective(self.label,matrix,(640,480))
        mask=cv2.warpPerspective(np.full((160,80),255,np.uint8),matrix,(640,480))
        image[mask>0]=warped[mask>0]
        return image


def run_case(parameters=None,target=(0,80),occlude=None,perception_compute=.035,obstacles=None,free=None,
             mission_options=None,camera_pitch_bias_degrees=0.):
    plant=ObjectPlant(target=target,occlude=occlude,**(parameters or {}))
    profile=load_json(os.path.join(ROOT,'calibration/floor_geometry.json'))
    tracker=FloorTracker(profile,load_json(profile['intrinsics_path']),max_features=125)
    original=tracker.motion
    from target_tracker import TargetTracker
    original_update,original_reacquire=TargetTracker.update,TargetTracker.reacquire
    def timed_update(self,image):
        try:return original_update(self,image)
        finally:plant.advance(perception_compute)
    def timed_reacquire(self,image):
        try:return original_reacquire(self,image)
        finally:plant.advance(perception_compute)
    violations=[]
    imu_history=deque([dict(time=plant.time-.2+i*.02,gyro=np.zeros(3),
                           acceleration=np.array([0.,9.80665,0.]),yaw=0.) for i in range(11)],maxlen=600)
    imu_previous=[plant.time,plant.velocity]
    free=free or [-70,-40,70,140]
    obstacles=obstacles or []
    def audit(x,z,yaw):
        if plant.time-imu_previous[0]>=.019:
            acceleration=(plant.velocity-imu_previous[1])/(100.*(plant.time-imu_previous[0]))
            imu_history.append(dict(time=plant.time,gyro=np.array([0.,-plant.yaw_rate,0.]),
                                    acceleration=np.array([0.,9.80665,acceleration]),yaw=yaw))
            imu_previous[:]=[plant.time,plant.velocity]
        from spatial_planner import bounds,transform
        from route_geometry import contains,overlap
        body=bounds([transform((x,z,math.degrees(yaw)),u,v) for u in (-6,6) for v in (-15,0)],5)
        obstacle=[target[0]-3,target[1]-3,target[0]+3,target[1]+3]
        if not contains(free,body) or any(overlap(body,b) for b in obstacles+[obstacle]):violations.append([plant.time,x,z])
    plant.pose_audit=audit
    def motion(*args):
        try:return original(*args)
        finally:plant.advance(plant.compute)
    class Timeline:
        def __init__(self,mount):
            self.up=plant.initial_up.copy()
            self.states=imu_history
            # Plant frames already express camera coordinates. This matrix lets
            # the production mission apply its optional camera-frame alignment.
            self.matrix=np.eye(3)
    bias=cv2.Rodrigues(np.array([math.radians(camera_pitch_bias_degrees),0.,0.]))[0]
    def read_frame(timeline,settled=False):
        image,timestamp,attitude=plant.frame(timeline,settled=settled)
        attitude=dict(attitude,down_camera=timeline.matrix@bias@attitude['down_camera'])
        return image,timestamp,attitude
    with tempfile.TemporaryDirectory() as directory:
        path=os.path.join(directory,'anchor.jpg');cv2.imwrite(path,plant.render())
        plan=dict(image_path=path,captured_monotonic=plant.time,target_box=plant.target_box,
                  target_ground_cm=list(target),
                  target_label='simulated object',inspected_free_rectangle_cm=free,
                  obstacle_rectangles_cm=obstacles,standoff_cm=20,**plant.token)
        plan.update(mission_options or {})
        with patch('point_controller.call',side_effect=plant.call),patch('point_controller.frame',side_effect=read_frame), \
                patch('point_controller.FloorTracker',return_value=tracker),patch.object(tracker,'motion',side_effect=motion), \
                patch('state_estimator.AttitudeTimeline',Timeline), \
                patch.object(TargetTracker,'update',timed_update),patch.object(TargetTracker,'reacquire',timed_reacquire), \
                patch('object_mission.time.monotonic',side_effect=lambda:plant.time), \
                patch('object_mission.time.sleep',side_effect=plant.advance):
            result=execute(plan,os.path.join(directory,'mission.json'))
        plant.advance(1.)
    return dict(result=result,true_pose=[plant.x,plant.z,math.degrees(plant.yaw)],
                true_target_range_cm=math.hypot(target[0]-plant.x,target[1]-plant.z),
                clearance_violations=violations,commands=plant.commands,
                watchdog_stops=plant.watchdog_stops,final_motor_output=[plant.left,plant.right])


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    args=parser.parse_args();run=run_case(dict(speed=12,coast=.05,seed=3))
    with open(args.output,'w') as out:json.dump(run,out,indent=2)
    print(json.dumps({k:v for k,v in run.items() if k not in ('result','commands')}))
    print(json.dumps({k:v for k,v in run['result'].items() if k!='samples'}))
