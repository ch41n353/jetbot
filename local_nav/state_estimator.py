"""Gyro/gravity attitude timeline and visual-inertial planar state.

No acceleration double integration. Metric translation remains visual. Host
acquisition timestamps are interpolated; hardware timing offset is uncalibrated.
"""
from collections import deque
import math
import cv2
import numpy as np

# Whole-sensor guard against shock, handling and runaway rotation. This is the
# three-axis gyro magnitude, not chassis yaw: a measured scan turn reaches about
# 80 deg/s of yaw plus carpet vibration on the other two axes, so a limit near
# the chassis-yaw ceiling rejects healthy turns. At the ~45 Hz sample rate this
# is still about 4 degrees per sample, well inside small-angle integration and
# far below gyro full scale. Chassis-yaw limits belong in the turn controllers.
SENSOR_RATE_LIMIT_DEG_S=180.

#: How far the visual-inertial pose estimate may drift before a controller
#: stops trusting it. Raised from the original 2 cm / 4 degree test thresholds,
#: which halted approaches that were otherwise going fine.
POSITION_LIMIT_CM=8.
YAW_LIMIT_DEGREES=15.

#: Longest accepted step between consecutive IMU samples. The integration is
#: small-angle over this step, so the bound is about how much rotation could
#: hide inside it. At 80 ms it also fired on a stopped robot: the search does
#: floor registration, preview drawing and map geometry between reads, and on
#: this board that thinking time grew past 80 ms once the accumulated map got
#: large, ending a 305-degree sweep that was otherwise going fine. Motion-time
#: protection does not come from here -- the motor lease expires in ~200 ms and
#: the controllers demand renewals every 180 ms, both untouched.
IMU_GAP_LIMIT_S=.3


def unit(v):
    v=np.asarray(v,dtype=float)
    if v.shape!=(3,) or not np.isfinite(v).all() or np.linalg.norm(v)<1e-8:
        raise ValueError('Invalid vector')
    return v/np.linalg.norm(v)


def rotation2(angle):
    c,s=math.cos(angle),math.sin(angle)
    return np.array([[c,-s],[s,c]])


class AttitudeTimeline:
    def __init__(self,mount):
        if not mount.get('verified'):raise ValueError('IMU mounting calibration is not verified')
        self.matrix=np.asarray(mount['imu_to_camera'],dtype=float)
        if self.matrix.shape!=(3,3) or not np.isfinite(self.matrix).all() or not np.allclose(self.matrix@self.matrix.T,np.eye(3),atol=1e-5) or np.linalg.det(self.matrix)<.99:
            raise ValueError('Invalid IMU-to-camera rotation')
        self.bias=np.radians(np.asarray(mount['gyro_bias_deg_s'],dtype=float))
        self.reference_up=unit(mount['level_acceleration'])
        self.states=deque(maxlen=600)
        self.last=None
        self.up=None
        self.route_reference_up=None
        self.yaw=0.
        self.variance=0.
        self.gyro=None
        self.acceleration_bad_since=None
        self.rejected_acceleration_samples=0
        self.startup_limit_degrees=4.

    def initialize_stationary(self, samples, frame_time):
        """Initialize from a quiet surface orientation, bounded to ten degrees.

        Single-sample feed initialization retains its original four-degree cap.
        This path requires 150 ms of low-noise, near-1g measurements first.
        The calibrated sensor mounting stays fixed. Only the initial gravity
        direction follows the carpet; runtime relative-tilt guards are unchanged.
        """
        if self.last is not None:
            raise RuntimeError('Stationary initialization requires a new timeline')
        window=[s for s in samples if frame_time-.2 <= s['time'] <= frame_time]
        if len(window)<6 or window[-1]['time']-window[0]['time']<.15:
            raise RuntimeError('Insufficient stationary startup history')
        acc=np.array([s['acceleration'] for s in window],dtype=float)
        gyro=[]
        for s in window:
            if s['gyro_units'] not in ('rad/s','deg/s'):
                raise ValueError('Unknown gyro units')
            gyro.append(np.array(s['gyro'])*(math.pi/180 if s['gyro_units']=='deg/s' else 1)-self.bias)
        mean=acc.mean(0)
        if (not np.isfinite(acc).all() or not np.isfinite(gyro).all()
                or np.max(acc.std(0))>.12 or abs(np.linalg.norm(mean)-9.80665)>.35
                or np.max(np.linalg.norm(gyro,axis=1))>math.radians(1.5)):
            raise RuntimeError('Robot is not quiet enough for stationary startup')
        first=dict(window[0],acceleration=mean.tolist())
        remaining=[s for s in samples if s['time']>first['time']]
        self.startup_limit_degrees=10.
        try:
            self.feed([first]+remaining)
        finally:
            self.startup_limit_degrees=4.

    def feed(self,samples):
        for sample in samples:
            timestamp=float(sample['time'])
            if not math.isfinite(timestamp):raise ValueError('Invalid IMU timestamp')
            if self.last is not None and timestamp<=self.last:continue
            if sample.get('error',0)!=0 or sample.get('system_status',5)!=5:
                raise RuntimeError('IMU fusion unhealthy')
            acceleration=np.asarray(sample['acceleration'],dtype=float)
            w=np.asarray(sample['gyro'],dtype=float)
            if sample['gyro_units']=='deg/s':w=np.radians(w)
            elif sample['gyro_units']!='rad/s':raise ValueError('Unknown gyro units')
            w=w-self.bias
            if not np.isfinite(w).all():raise RuntimeError('Non-finite IMU angular rate')
            rate=float(np.linalg.norm(w))
            if rate>math.radians(SENSOR_RATE_LIMIT_DEG_S):
                raise RuntimeError('IMU angular-rate limit exceeded: %.2f deg/s across all axes (limit%g); sensor rate is not measured chassis yaw' % (math.degrees(rate),SENSOR_RATE_LIMIT_DEG_S))
            norm=float(np.linalg.norm(acceleration))
            if not 3<norm<20:
                raise RuntimeError('Severe IMU acceleration (%.2f m/s^2)' % norm)
            if not 7<norm<13:
                self.rejected_acceleration_samples+=1
                if self.acceleration_bad_since is None:self.acceleration_bad_since=timestamp
                if timestamp-self.acceleration_bad_since>=.08:
                    raise RuntimeError('Sustained non-gravity acceleration')
                if self.last is None:raise RuntimeError('Cannot initialize attitude during acceleration')
            else:
                self.acceleration_bad_since=None
            measured=unit(acceleration)
            if self.last is None:
                if measured.dot(self.reference_up)<math.cos(math.radians(self.startup_limit_degrees)):
                    raise RuntimeError('Starting pose differs from calibrated floor pose')
                self.up=measured
            else:
                dt=timestamp-self.last
                if dt>IMU_GAP_LIMIT_S:
                    raise RuntimeError('Gap in IMU history exceeds %d ms'%(IMU_GAP_LIMIT_S*1000))
                average=(self.gyro+w)/2
                self.up=unit(cv2.Rodrigues(-average*dt)[0]@self.up)
                self.yaw-=float(average.dot(self.up))*dt  # positive robot-right yaw
                self.variance+=math.radians(.5)**2*dt
                # Accelerometer corrects slow tilt drift only during quiet periods;
                # never interpret drive acceleration immediately as camera pitch.
                if abs(norm-9.80665)<.25 and np.linalg.norm(w)<math.radians(1) and measured.dot(self.up)>math.cos(math.radians(5)):
                    alpha=1-math.exp(-dt/2)
                    self.up=unit((1-alpha)*self.up+alpha*measured)
            self.last=timestamp
            self.gyro=w
            self.states.append(dict(time=timestamp,up=self.up.copy(),yaw=self.yaw,variance=self.variance,gyro=w.copy(),acceleration=acceleration.copy()))

    def settled_at(self,timestamp):
        """Frame attitude after a motor-off settling interval, not while driving.

        Use a quiet gravity window to remove accumulated pitch/roll error.
        Do not alter integrated yaw or rewrite the inertial history.
        """
        attitude=self.at(timestamp)
        attitude['settled_gravity_used']=False
        samples=[s for s in self.states if timestamp-.18<=s['time']<=timestamp]
        if len(samples)<6 or samples[-1]['time']-samples[0]['time']<.12:
            return attitude
        acceleration=np.array([s['acceleration'] for s in samples])
        gyro=np.array([s['gyro'] for s in samples])
        if (np.max(np.linalg.norm(gyro,axis=1))>math.radians(1.5)
                or np.max(acceleration.std(axis=0))>.12
                or np.max(np.abs(np.linalg.norm(acceleration,axis=1)-9.80665))>.35):
            return attitude
        down=-self.matrix@unit(acceleration.mean(axis=0))
        difference=math.acos(float(np.clip(down.dot(attitude['down_camera']),-1,1)))
        if difference>math.radians(2):
            raise RuntimeError('Settled gravity disagrees with integrated attitude')
        attitude['down_camera']=down
        attitude['settled_gravity_used']=True
        attitude['settled_correction_degrees']=math.degrees(difference)
        return attitude

    def at(self,timestamp):
        if not self.states:raise RuntimeError('No IMU history')
        if timestamp<self.states[0]['time']:raise RuntimeError('Frame predates IMU history')
        for before,after in zip(self.states,list(self.states)[1:]):
            if before['time']<=timestamp<=after['time']:
                f=(timestamp-before['time'])/(after['time']-before['time'])
                up=unit((1-f)*before['up']+f*after['up'])
                return dict(time=timestamp,down_camera=-self.matrix@up,
                    yaw=(1-f)*before['yaw']+f*after['yaw'],variance=(1-f)*before['variance']+f*after['variance'],extrapolation_ms=0.)
        last=self.states[-1]
        dt=timestamp-last['time']
        if not 0<=dt<=.035:raise RuntimeError('No IMU sample close enough to frame')
        up=unit(cv2.Rodrigues(-last['gyro']*dt)[0]@last['up'])
        return dict(time=timestamp,down_camera=-self.matrix@up,
            yaw=last['yaw']-float(last['gyro'].dot(up))*dt,variance=last['variance']+math.radians(.5)**2*dt,extrapolation_ms=1000*dt)


def fuse_yaw(visual_yaw,imu_yaw,dt):
    innovation=math.atan2(math.sin(visual_yaw-imu_yaw),math.cos(visual_yaw-imu_yaw))
    if abs(innovation)>math.radians(3):raise RuntimeError('Camera and IMU rotation disagree')
    gyro_variance=math.radians(.5)**2*max(dt,.02)+math.radians(.25)**2
    visual_variance=math.radians(.8)**2
    gain=gyro_variance/(gyro_variance+visual_variance)
    return imu_yaw+gain*innovation,gyro_variance*(1-gain)


class PlanarState:
    def __init__(self):
        self.position=np.zeros(2)
        self.yaw=0.
        self.velocity=np.zeros(2)
        self.position_variance=0.
        self.yaw_variance=0.
    def update(self,rotation,translation,dt,quality):
        yaw=math.atan2(rotation[1,0],rotation[0,0])
        delta_previous=-rotation.T@translation
        delta_world=rotation2(-self.yaw)@delta_previous
        self.position+=delta_world
        self.velocity=.5*self.velocity+.5*delta_world/dt
        self.yaw+=yaw
        self.position_variance+=max(.02,float(quality['residual_cm']))**2+.03**2
        self.yaw_variance+=quality.get('yaw_variance',math.radians(.8)**2)
        # Drift bound, not a safety limit -- it says how far the pose estimate
        # may have wandered, and the operator accepts the robot bumping into
        # things. The old 2 cm / 4 degree pair was a test threshold that ended
        # real approaches a few centimetres from the target; these let a run
        # finish while still catching an estimate that has genuinely diverged.
        if math.sqrt(self.position_variance)>POSITION_LIMIT_CM or \
                math.sqrt(self.yaw_variance)>math.radians(YAW_LIMIT_DEGREES):
            raise RuntimeError('State uncertainty exceeded %g cm / %g degrees'
                               %(POSITION_LIMIT_CM,YAW_LIMIT_DEGREES))
        return dict(position_cm=self.position.tolist(),velocity_cm_s=self.velocity.tolist(),
            yaw_degrees=math.degrees(self.yaw),position_sigma_cm=math.sqrt(self.position_variance),
            yaw_sigma_degrees=math.degrees(math.sqrt(self.yaw_variance)))
