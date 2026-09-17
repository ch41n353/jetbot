#!/usr/bin/env python3
"""Bounded, straight-ahead floor-point experiment on an operator-cleared path.

Tracks carpet features in calibrated floor coordinates. This is not obstacle
avoidance: use only an inspected, empty corridor. IMU gyro/gravity compensates camera tilt and supplies rotation prediction;
visual odometry supplies metric translation and corrects rotation. All motion uses expiring leases.
"""
import argparse
import base64
import json
import math
import os
import socket
import time
import cv2
import numpy as np
from floor_geometry import project_floor
from state_estimator import AttitudeTimeline, PlanarState, fuse_yaw, rotation2

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def call(action, **fields):
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as s:
        s.settimeout(.5)
        s.connect('/tmp/jetbot-local-nav/control.sock')
        s.sendall(json.dumps(dict(action=action,**fields)).encode()+b'\n')
        data=b''
        while b'\n' not in data:
            chunk=s.recv(65536)
            if not chunk:raise RuntimeError('Service disconnected')
            data+=chunk
        result=json.loads(data)
        if 'error' in result:raise RuntimeError(result['error'])
        return result


#: Where a tracked point is allowed to land, as (x0,y0,x1,y1). The default sits
#: barely 20px outside the source box, which suits forward driving but discards
#: nearly every point once the chassis rotates. Rotation-heavy callers widen it.
DRIVE_FLOW_WINDOW=(140,270,500,470)
TURN_FLOW_WINDOW=(30,250,610,478)

#: Accepted apparent-floor-scale range for one motion fit. Outside it the camera
#: height or tilt has genuinely changed; inside it is tracking noise.
SCALE_LIMITS=(.94,1.06)

#: Feature counts a floor fit needs. A similarity fit needs only a couple of
#: correspondences, so these are quality floors, not mathematical ones. They
#: were set for richly textured carpet; on plainer floor -- the doorway area the
#: robot scanned on 2026-09-16 -- they ended sweeps that were tracking fine.
#: Every fit is still checked for inlier fraction, residual and scale.
MIN_SOURCE_FEATURES=15
MIN_TRACKED_FEATURES=10
MIN_INLIERS=8
#: Fraction of tracked points the fit must agree with. On plain carpet with few
#: features a good fit routinely lands near half; 0.65 rejected a 12-of-27 fit
#: mid-approach. The absolute count above, the residual and the scale bound all
#: still apply, so a genuinely bad fit is still refused.
MIN_INLIER_FRACTION=.5


class FloorTracker:
    def __init__(self,profile,intrinsics,max_features=250,crop_flow=True,
                 flow_window=DRIVE_FLOW_WINDOW):
        if type(max_features) is not int or max_features not in (80, 125, 250):
            raise ValueError("Feature budget must be 80, 125 or 250")
        self.max_features=max_features
        if type(crop_flow) is not bool:
            raise ValueError('crop_flow must be boolean')
        self.crop_flow=crop_flow
        window=tuple(float(v) for v in flow_window)
        if len(window)!=4 or window[0]>=window[2] or window[1]>=window[3]:
            raise ValueError('flow_window must be (x0,y0,x1,y1) with x0<x1 and y0<y1')
        self.flow_window=window
        self.p,self.i=profile,intrinsics

    def ground(self,points,attitude=None):
        xy=cv2.fisheye.undistortPoints(np.asarray(points,dtype=float).reshape(-1,1,2),
            np.asarray(self.i['K'],dtype=float),np.asarray(self.i['D'],dtype=float)).reshape(-1,2)
        rays=np.column_stack((xy,np.ones(len(xy))))
        if attitude is None:
            a=math.radians(self.p['pitch_degrees'])
            down=np.array([0.,math.cos(a),math.sin(a)])
        else:
            down=np.asarray(attitude['down_camera'])
            down=down/np.linalg.norm(down)
        forward=np.array([0.,0.,1.])-down*down[2]
        forward=forward/np.linalg.norm(forward)
        right=np.cross(down,forward)
        denominator=rays.dot(down)
        if attitude is not None and np.any(denominator<.05):
            raise RuntimeError('Tilt moved tracked points off the usable floor')
        return np.column_stack((rays.dot(right),rays.dot(forward)))*self.p['camera_height_cm']/denominator[:,None]

    def motion(self,before,after,before_attitude=None,after_attitude=None):
        ox,oy=(40,168) if self.crop_flow else (0,0)
        region=(slice(168,480),slice(40,600)) if self.crop_flow else (slice(None),slice(None))
        offset=np.array([ox,oy],dtype=np.float32)
        gray0=cv2.cvtColor(before[region],cv2.COLOR_BGR2GRAY)
        gray1=cv2.cvtColor(after[region],cv2.COLOR_BGR2GRAY)
        # Remove global exposure offsets using the tracked floor region.
        # Geometric inlier and forward/backward checks still validate motion.
        floor=(slice(280-oy,460-oy),slice(160-ox,480-ox))
        gray0=np.clip(gray0.astype(np.float32)-gray0[floor].mean()+128,0,255).astype(np.uint8)
        gray1=np.clip(gray1.astype(np.float32)-gray1[floor].mean()+128,0,255).astype(np.uint8)
        mask=np.zeros_like(gray0)
        mask[floor]=255
        excluded=getattr(self,'excluded_box',None)
        if excluded is not None:
            x,y,r,b=np.round(excluded).astype(int)
            x,r=x-ox,r-ox
            y,b=y-oy,b-oy
            mask[max(0,y-5):max(0,min(mask.shape[0],b+5)),max(0,x-5):max(0,min(mask.shape[1],r+5))]=0
        p0=cv2.goodFeaturesToTrack(gray0,self.max_features,.005,7,mask=mask)
        if p0 is None or len(p0)<MIN_SOURCE_FEATURES:
            raise RuntimeError('Insufficient carpet texture (%d features)'
                               %(0 if p0 is None else len(p0)))
        p0=p0+offset
        guess=None
        flags=0
        if before_attitude is not None and after_attitude is not None:
            def basis(att):
                down=np.asarray(att['down_camera']);down=down/np.linalg.norm(down)
                forward=np.array([0.,0.,1.])-down*down[2];forward/=np.linalg.norm(forward)
                return np.column_stack((np.cross(down,forward),down,forward))
            # Predict tilt-induced pixel motion from IMU, before LK refinement.
            yaw=after_attitude['yaw']-before_attitude['yaw']
            c,s=math.cos(yaw),math.sin(yaw)
            yaw_rotation=np.array([[c,0.,-s],[0.,1.,0.],[s,0.,c]])
            cam_rotation=basis(after_attitude)@yaw_rotation@basis(before_attitude).T
            xy=cv2.fisheye.undistortPoints(p0.astype(float),np.asarray(self.i['K'],dtype=float),np.asarray(self.i['D'],dtype=float)).reshape(-1,2)
            rays=np.column_stack((xy,np.ones(len(xy))))@cam_rotation.T
            guess,_=cv2.fisheye.projectPoints(rays.reshape(-1,1,3),np.zeros(3),np.zeros(3),np.asarray(self.i['K'],dtype=float),np.asarray(self.i['D'],dtype=float))
            guess=guess.astype(np.float32)
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW
        # Keep >90px support around accepted floor endpoints for the 21px
        # window at pyramid level3. Align the crop origin to 2**3 so pyramid
        # samples match the full image. The bottom border is unchanged.
        local0=p0-offset
        local_guess=None if guess is None else guess-offset
        p1,ok1,_=cv2.calcOpticalFlowPyrLK(gray0,gray1,local0,local_guess,winSize=(21,21),maxLevel=3,flags=flags)
        if p1 is None:raise RuntimeError('Optical flow failed')
        back,ok2,_=cv2.calcOpticalFlowPyrLK(gray1,gray0,p1,local0.copy(),winSize=(21,21),maxLevel=3,flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        if back is None:raise RuntimeError('Reverse optical flow failed')
        p1=p1+offset
        back=back+offset
        good=(ok1.ravel()>0)&(ok2.ravel()>0)&(np.linalg.norm(back-p0,axis=2).ravel()<.8)
        q=p1.reshape(-1,2)
        wx0,wy0,wx1,wy1=self.flow_window
        good &= (q[:,0]>wx0)&(q[:,0]<wx1)&(q[:,1]>wy0)&(q[:,1]<wy1)
        a,b=self.ground(p0[good],before_attitude),self.ground(p1[good],after_attitude)
        if len(a)<MIN_TRACKED_FEATURES:
            raise RuntimeError('Lost carpet tracking (%d of %d survived)'%(len(a),len(p0)))
        transform,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=.35,maxIters=1000,confidence=.99)
        if transform is None or inliers is None:raise RuntimeError('Floor motion fit failed')
        select=inliers.ravel().astype(bool)
        if select.sum()<MIN_INLIERS or select.mean()<MIN_INLIER_FRACTION:
            raise RuntimeError('Floor motion is inconsistent (%d/%d inliers)'%(select.sum(),len(select)))
        scale=float(np.linalg.norm(transform[:,0]))
        # Catches the camera being knocked or the robot lifted, which change the
        # apparent floor scale a great deal. At +/-3% it also caught ordinary
        # carpet-tracking noise across a scan turn -- a 0.968 fit ended a sweep
        # 18 turns in -- so the band is the one that separates a moved camera
        # from a normal fit, not the tightest the tracker usually achieves.
        if not SCALE_LIMITS[0]<scale<SCALE_LIMITS[1]:
            raise RuntimeError('Camera tilt/height or tracking changed (scale %.3f)' % scale)
        rotation=transform[:,:2]/scale
        yaw_variance=math.radians(.8)**2
        if before_attitude is not None and after_attitude is not None:
            visual_yaw=math.atan2(rotation[1,0],rotation[0,0])
            inertial_yaw=after_attitude['yaw']-before_attitude['yaw']
            fused,yaw_variance=fuse_yaw(visual_yaw,inertial_yaw,after_attitude['time']-before_attitude['time'])
            rotation=rotation2(fused)
        translation=np.median(b[select]-a[select].dot(rotation.T),axis=0)
        residual=float(np.median(np.linalg.norm(a[select].dot(rotation.T)+translation-b[select],axis=1)))
        if residual>.3 or np.linalg.norm(translation)>5:raise RuntimeError('Implausible floor displacement')
        return rotation,translation,dict(matches=int(select.sum()),residual_cm=residual,scale=scale,yaw_variance=yaw_variance)


def record_observation(directory, snap, image):
    """Preserve initialization/history when multiple reads share one camera frame."""
    stem=os.path.join(directory,'%.6f' % snap['time'])
    record={k:v for k,v in snap.items() if k!='jpeg_base64'}
    if os.path.exists(stem+'.json'):
        with open(stem+'.json') as source:previous=json.load(source)
        samples={s['time']:s for s in previous['imu_samples']+record['imu_samples']}
        record['imu_samples']=[samples[t] for t in sorted(samples)]
        if 'attitude_initialization' in previous:
            record.setdefault('attitude_initialization',previous['attitude_initialization'])
    with open(stem+'.json','w') as target:json.dump(record,target)
    if not cv2.imwrite(stem+'.jpg',image):raise RuntimeError('Could not save observation image')


def frame(timeline=None, settled=True):
    snap=call('observation',since=timeline.last if timeline and timeline.last else time.monotonic()-.3) if timeline else call('snapshot')
    image=cv2.imdecode(np.frombuffer(base64.b64decode(snap['jpeg_base64']),dtype=np.uint8),cv2.IMREAD_COLOR) if timeline else cv2.imread(snap['path'])
    if image is None or image.shape[:2]!=(480,640):raise RuntimeError('Invalid camera frame')
    if timeline:
        if getattr(timeline,'keep_last_observation',False):
            timeline.last_sensor_observation={k:v for k,v in snap.items() if k!='jpeg_base64'}
            timeline.last_observation_image=image
        initialize = timeline.last is None
        if initialize:
            snap['attitude_initialization']='stationary_10deg'
        if hasattr(timeline, 'record_directory'):
            record_observation(timeline.record_directory,snap,image)
        if initialize:
            timeline.initialize_stationary(snap['imu_samples'],snap['time'])
        else:
            timeline.feed(snap['imu_samples'])
        return image,snap['time'],(timeline.settled_at(snap['time']) if settled else timeline.at(snap['time']))
    return image,snap['time']


def settled_frame(timeline):
    # Motors are already stopped by the caller. Never fit a floor-motion pair
    # using a quiet gravity estimate on one side and a drifting one on the other.
    deadline=time.monotonic()+.18
    while True:
        observation=frame(timeline)
        if observation[2]['settled_gravity_used']:
            return observation
        if time.monotonic()>=deadline:
            raise RuntimeError('Robot did not settle enough for floor tracking')
        time.sleep(.04)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pixel',nargs=2,type=float,default=[320,257])
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--settle-seconds',type=float,default=.25)
    parser.add_argument('--stop-distance-cm',type=float,default=10.)
    parser.add_argument('--log',default=os.path.join(ROOT,'local_nav','goals','point-run.json'))
    args=parser.parse_args()
    if not .25<=args.settle_seconds<=.5:
        parser.error('Settling interval must be between 0.25 and 0.5 seconds')
    if not 10<=args.stop_distance_cm<=20:
        parser.error('Stop distance must be between 10 and 20 cm')
    profile=json.load(open(os.path.join(ROOT,'calibration','floor_geometry.json')))
    intrinsics=json.load(open(profile['intrinsics_path']))
    point=project_floor(args.pixel,profile,intrinsics)
    target=np.array([point['lateral_cm'],point['forward_cm']])
    if not (30<=target[1]<=40 and abs(target[0])<5):
        raise RuntimeError('Initial test limited to a near-center goal 30–40 cm ahead')
    mount=json.load(open(os.path.join(ROOT,'calibration','imu_mount.json')))
    timeline=AttitudeTimeline(mount)
    timeline.record_directory=args.log+'.observations'
    os.makedirs(timeline.record_directory,exist_ok=True)
    state=PlanarState()
    tracker=FloorTracker(profile,intrinsics)
    result=dict(goal_pixel=args.pixel,stop_distance_cm=args.stop_distance_cm,initial_target_cm=target.tolist(),mode='execute' if args.execute else 'stationary_preflight',samples=[])
    start=time.monotonic()
    distance=0.
    angle=0.
    try:
        call('stop')
        status=call('status')
        if not status['healthy']:raise RuntimeError('Unhealthy sensors')
        if args.execute and not status['motion_enabled']:raise RuntimeError('Service is disarmed')
        old,t0,attitude0=settled_frame(timeline)
        target=tracker.ground([args.pixel],attitude0)[0]
        if not (30<=target[1]<=40 and abs(target[0])<5):
            raise RuntimeError('Measured attitude places goal outside initial test corridor')
        result['initial_target_cm']=target.tolist()
        cv2.imwrite(args.log+'.before.jpg',old)
        reference_gravity=np.array(status['imu']['acceleration'])
        reference_gravity/=np.linalg.norm(reference_gravity)
        while time.monotonic()-start<20:
            time.sleep(.10)
            current,t1,attitude1=settled_frame(timeline)
            if t1<=t0:raise RuntimeError('No new camera frame')
            dt=t1-t0
            if dt>.9:raise RuntimeError('Camera/control loop too slow')
            status=call('status')
            if not status['healthy']:raise RuntimeError('Sensor health lost')
            gyro=np.array(status['imu']['gyro'])
            if status['imu']['gyro_units']=='rad/s':gyro=np.degrees(gyro)
            if np.linalg.norm(gyro)>90:raise RuntimeError('Excessive IMU rotation')
            gravity=np.array(status['imu']['acceleration'])
            if not 7<np.linalg.norm(gravity)<13:raise RuntimeError('Robot bumped or lifted')
            # Filtered gravity alignment is only a gross tilt guard while moving.
            if np.dot(gravity/np.linalg.norm(gravity),reference_gravity)<.94:raise RuntimeError('Robot tilt changed')
            rotation,translation,quality=tracker.motion(old,current,attitude0,attitude1)
            yaw=math.degrees(math.atan2(rotation[1,0],rotation[0,0]))
            if abs(yaw)>7:raise RuntimeError('Unexpected visual rotation')
            angle+=yaw
            distance+=float(np.linalg.norm(translation))
            target=rotation.dot(target)+translation
            estimate=state.update(rotation,translation,dt,quality)
            result['samples'].append(dict(state=estimate,settled_gravity_used=attitude1['settled_gravity_used'],settled_correction_degrees=attitude1.get('settled_correction_degrees'),down_camera=attitude1['down_camera'].tolist(),imu_extrapolation_ms=attitude1['extrapolation_ms'],elapsed=time.monotonic()-start,target_cm=target.tolist(),travel_cm=distance,
                yaw_deg=angle,imu_rotation_rate=float(np.dot(gyro,reference_gravity)),**quality))
            if abs(angle)>20 or distance>40:raise RuntimeError('Travel/heading limit reached')
            if target[1]<0:raise RuntimeError('Overshot target')
            if np.linalg.norm(target)<=args.stop_distance_cm:
                result['outcome']='within_10cm_visual_estimate' if args.stop_distance_cm==10 else 'within_requested_distance_visual_estimate'
                break
            if not args.execute:
                if time.monotonic()-start>2:
                    if distance>1 or abs(angle)>2:raise RuntimeError('Stationary odometry unstable')
                    result['outcome']='stationary_tracking_passed'
                    break
            else:
                if time.monotonic()-start>3 and distance<1:raise RuntimeError('No measured progress')
                bearing=math.atan2(target[0],target[1])
                if abs(bearing)>.3:raise RuntimeError('Goal left the allowed straight corridor')
                steer=float(np.clip(.30*bearing,-.05,.05))
                result['samples'][-1]['command']=dict(left=.20+steer,right=.20-steer)
                # Never reverse or rotate in place in this first bounded test.
                call('motors',left=.20+steer,right=.20-steer)
                time.sleep(.10)
                call('stop')
                time.sleep(args.settle_seconds)
            old,t0,attitude0=current,t1,attitude1
        else:raise RuntimeError('Goal timeout')
    except Exception as exc:
        result['outcome']='stopped'
        result['reason']=str(exc)
    finally:
        try:call('stop')
        except Exception as exc:
            result['stop_error']=str(exc)
            result['outcome']='stopped'
        finally:
            result['elapsed']=time.monotonic()-start
            result['final_target_cm']=target.tolist()
            with open(args.log,'w') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2))

if __name__=='__main__':main()
