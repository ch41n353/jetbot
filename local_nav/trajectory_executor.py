"""Trajectory-only SE(2) execution. No GPT, target or obstacle interpretation.

Coordinates: cm right/forward in the camera frame at submission, clockwise yaw.
Vision estimates planar translation; IMU supplies yaw. Turns use measured IMU
rotation and calibrated camera pivot. This is planar visual/inertial odometry,
not a full 6-DoF VIO estimator and not an independent collision checker.
"""
import base64
import copy
import json
import math
import time
import threading
import uuid
from pathlib import Path
import cv2
import numpy as np
import fetch

# Hold the proven carpet duty instead of accelerating long legs.  The dynamic
# 20--32% profile produced about 10 cm/s with peaks around 16 cm/s and caused
# repeated blurred/lost floor tracks in the 2026-09-23 Advil run.
DRIVE_MIN_DUTY = .20
DRIVE_MAX_DUTY = .20
MAX_TRACKING_REJECTIONS = 6
MAX_TRACKING_GAP_S = .8

# The 20% figure was measured on a pack sitting around 12.2 V. Torque in a
# brushed motor goes with duty x supply, so the same command delivers less as
# the pack sags -- and over one session this pack went 12.35 V to 10.80 V.
DRIVE_NOMINAL_PACK_V = 12.2
# The service rejects any |speed| above 0.3; the steering trim adds up to 0.025
# on one wheel, so the compensated base must stay below that.
DRIVE_DUTY_CEILING = .27

# A turn is allowed to miss by this much before it counts as not having
# converged. Flat 8 degrees was 80% of a 10 degree request and 5% of a 160
# degree one: it waved through small turns and failed large ones that had
# effectively arrived. Measured: a -160.1 degree escape pivot off a wall was
# declared a failure and aborted the trajectory it had just made possible.
TURN_TOLERANCE_DEG = 8.
TURN_TOLERANCE_FRACTION = .08
# Residual sweeps allowed to close the gap. The pose is updated with the
# *measured* angle every time, so an under-turn is not a lie the executor then
# acts on -- it is simply unfinished, and finishing it is cheap.
TURN_ATTEMPTS = 3


def turn_tolerance(degrees):
    """How far a turn may miss before it has genuinely failed."""
    return max(TURN_TOLERANCE_DEG, abs(float(degrees)) * TURN_TOLERANCE_FRACTION)


def recoverable_tracking_error(exc):
    """Quality rejections that may clear on the next overlapping frame."""
    text = str(exc)
    return text.startswith(('Lost carpet tracking',
                            'Floor motion is inconsistent',
                            'Implausible floor displacement'))

def drive_duty(distance_cm, pack_v=None):
    """Carpet duty for a leg, compensated for how far the pack has sagged.

    `distance_cm` is ignored and kept only for the call site: the dynamic
    20-32% profile it used to scale was removed on 2026-09-23 after it measured
    16 cm/s peaks and lost floor tracking, and every branch of the old
    expression already clamped to the same 20%.

    Voltage is the part that was missing. 20% is the measured stiction wall on
    carpet at a full pack; at 10.9 V the same command is about a tenth less
    torque, and legs that drove fine at the start of a session stall at the end.
    Scaling by nominal/actual holds delivered torque roughly constant.

    Never below the measured floor -- 20% is a wall, not a target -- and never
    close enough to the service's +/-0.3 limit for the steering trim to breach
    it. First-order and uncalibrated: it wants a live check across the range.
    """
    duty = DRIVE_MIN_DUTY
    if pack_v and float(pack_v) > 0.:
        duty = DRIVE_MIN_DUTY * DRIVE_NOMINAL_PACK_V / float(pack_v)
    return min(DRIVE_DUTY_CEILING, max(DRIVE_MIN_DUTY, duty))


def compose(a, b):
    h = math.radians(a[2]); c, s = math.cos(h), math.sin(h)
    return [a[0]+c*b[0]+s*b[1], a[1]-s*b[0]+c*b[1], a[2]+b[2]]


def pivot_aim(relative):
    """Analytic heading accounting for the camera orbit around the wheel pivot."""
    shift=fetch.pivot_shift(180.)
    px,pz=shift[0]/2.,shift[1]/2.
    x,z=relative[0]-px,relative[1]-pz
    radius=math.hypot(x,z)
    if radius<=abs(px):raise fetch.Stop('waypoint inside turning circle')
    angle=math.degrees(math.atan2(x,z)+math.asin(px/radius))
    return (angle+180.)%360.-180.


def validate(body):
    if set(body)-{'trajectory', 'frame_token', 'visualization', 'stop_after_cm', 'stop_after_waypoints'}:
        raise ValueError('Only trajectory, frame_token, visualization, stop_after_cm and stop_after_waypoints are accepted')
    t=body.get('trajectory', {})
    if set(t)-{'waypoints_cm','initial_turn_deg'}:
        raise ValueError('Unknown trajectory fields')
    limit=body.get('stop_after_cm')
    if limit is not None and (isinstance(limit,bool) or not isinstance(limit,(int,float)) or not math.isfinite(limit) or not 0<limit<=400):raise ValueError('stop_after_cm must be in (0,400]')
    count=body.get('stop_after_waypoints')
    if count is not None and (type(count) is not int or not 1<=count<=64):raise ValueError('stop_after_waypoints must be an integer in [1,64]')
    points=t.get('waypoints_cm', [])
    angle=t.get('initial_turn_deg', 0)
    def number(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
    if not number(angle) or abs(angle)>180:raise ValueError('Initial turn must be within +/-180 degrees')
    if not isinstance(points,list) or len(points)>64:raise ValueError('At most 64 waypoints')
    length=0.;last=[0.,0.]
    for p in points:
        if not isinstance(p,list) or len(p)!=2 or not all(number(v) for v in p):raise ValueError('Waypoints are [right_cm, forward_cm]')
        if max(abs(v) for v in p)>200:raise ValueError('Waypoints must fit +/-2 m submission frame')
        length+=math.hypot(p[0]-last[0],p[1]-last[1]);last=p
    if length>400 or (not points and not angle):raise ValueError('Empty or longer than 4 m trajectory')
    return dict(waypoints_cm=points,initial_turn_deg=float(angle))


def encode(image):
    ok,data=cv2.imencode('.jpg',image)
    if not ok:raise ValueError('Image encoding failed')
    return 'data:image/jpeg;base64,'+base64.b64encode(data).decode('ascii')


class CameraFloorProjection:
    """Calibrated RGB sampling onto a camera-centered square of floor.

    `scale` is pixels per centimetre, so the default 1.6 gives the 4 m square
    the dashboard draws. A larger scale keeps the same 640x640 output and
    shrinks the ground it covers -- 6.4 px/cm is a 1 m square, which spends the
    whole image on the near floor where the projection is actually accurate
    instead of on a horizon that warps every upright object into a streak.
    """
    def __init__(self, robot, scale=1.6):
        yy,xx=np.indices((640,640),dtype=np.float32)
        right=(xx-320)/scale;forward=(320-yy)/scale
        down=np.array([0.,math.cos(robot.pitch),math.sin(robot.pitch)])
        ahead=np.array([0.,0.,1.])-down*down[2];ahead/=np.linalg.norm(ahead)
        side=np.cross(down,ahead)
        rays=right[...,None]*side+forward[...,None]*ahead+robot.height*down
        pixels,_=cv2.fisheye.projectPoints(rays.reshape(1,-1,3).astype(np.float64),
            np.zeros(3),np.zeros(3),robot.K,robot.D)
        pixels=pixels.reshape(640,640,2)
        self.x=pixels[:,:,0].astype(np.float32);self.y=pixels[:,:,1].astype(np.float32)
        self.visible=(forward>0)&(rays[:,:,2]>0)&(self.x>=0)&(self.x<=639)&(self.y>=0)&(self.y<=479)
    def apply(self,image):
        out=cv2.remap(image,self.x,self.y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        out[~self.visible]=(28,28,28)
        return out


class ZonedFloorProjection:
    """Ground plane on a radial map with two linear scales.

    A single scale cannot serve both jobs. Linear over 4 m and the near floor
    -- the only place a ground-plane warp is accurate, and the only distance
    that decides whether the robot may move -- is a few dozen pixels. Linear
    over 1 m and the world stops at the frame edge. A smoothly compressing
    scale fixes both and creates a third problem: nothing on the picture has a
    fixed size, so no distance can be judged without inverting the mapping.

    Two linear zones keep the compression but make it legible. Inside
    INNER_CM the scale is constant; outside it, constant at half. Past
    OUTER_CM everything is drawn on the rim, so a far target still has a
    bearing without claiming a range the projection cannot support.

    Measured 2026-09-25 on one frame with a block 15 cm ahead, 11 calls each
    at effort=none: this returned the correct "cannot drive, turn" 8 times,
    against 3 for a 1 m square at 6.4 px/cm and 1 for a smooth r = R*d/(d+k).
    """
    INNER_CM, OUTER_CM, EDGE_PX = 100., 200., 318.
    REAR_CM = 50.          # how far behind the robot is worth any pixels
    MARGIN_PX = 6
    INNER_PX_PER_CM = EDGE_PX / (INNER_CM + (OUTER_CM - INNER_CM) / 2.)
    OUTER_PX_PER_CM = INNER_PX_PER_CM / 2.
    INNER_EDGE_PX = INNER_CM * INNER_PX_PER_CM

    @classmethod
    def radius_px(cls, range_cm):
        if range_cm <= cls.INNER_CM:
            return range_cm * cls.INNER_PX_PER_CM
        if range_cm <= cls.OUTER_CM:
            return cls.INNER_EDGE_PX + (range_cm - cls.INNER_CM) * cls.OUTER_PX_PER_CM
        return cls.EDGE_PX

    def _sample(self, robot, px, py):
        """Camera pixel and validity for points on the polar canvas."""
        cls = type(self)
        dx, dy = px - 320., 320. - py
        rho = np.hypot(dx, dy)
        inside = rho < cls.EDGE_PX
        clipped = np.clip(rho, 0., cls.EDGE_PX)
        span = np.where(clipped <= cls.INNER_EDGE_PX,
                        clipped / cls.INNER_PX_PER_CM,
                        cls.INNER_CM + (clipped - cls.INNER_EDGE_PX) / cls.OUTER_PX_PER_CM)
        bearing = np.arctan2(dx, dy)
        right, forward = span * np.sin(bearing), span * np.cos(bearing)
        down = np.array([0., math.cos(robot.pitch), math.sin(robot.pitch)])
        ahead = np.array([0., 0., 1.]) - down * down[2]
        ahead /= np.linalg.norm(ahead)
        side = np.cross(down, ahead)
        rays = (right[..., None] * side + forward[..., None] * ahead
                + robot.height * down)
        pixels, _ = cv2.fisheye.projectPoints(
            rays.reshape(1, -1, 3).astype(np.float64),
            np.zeros(3), np.zeros(3), robot.K, robot.D)
        pixels = pixels.reshape(px.shape + (2,))
        x = pixels[..., 0].astype(np.float32)
        y = pixels[..., 1].astype(np.float32)
        good = (inside & (forward > 0) & (rays[..., 2] > 0)
                & (x >= 0) & (x <= 639) & (y >= 0) & (y <= 479))
        return x, y, good

    def __init__(self, robot):
        cls = type(self)
        # The camera sees a forward wedge, so a full circle of canvas is most
        # of the way empty -- measured at 27% of the frame carrying anything,
        # with everything below the centre row blank. Find what is actually
        # visible at low resolution, keep a little ground behind the robot so
        # the footprint has context, and spend the whole output on that.
        probe = np.linspace(0., 639., 160, dtype=np.float32)
        py, px = np.meshgrid(probe, probe, indexing='ij')
        _, _, good = self._sample(robot, px, py)
        cols = np.where(good.any(0))[0]
        rows = np.where(good.any(1))[0]
        if len(cols) and len(rows):
            x0, x1 = probe[cols.min()], probe[cols.max()]
            y0 = probe[rows.min()]
        else:                                   # nothing visible; keep it all
            x0, x1, y0 = 0., 639., 0.
        rear = cls.radius_px(cls.REAR_CM)
        x0 = max(0., x0 - cls.MARGIN_PX); x1 = min(639., x1 + cls.MARGIN_PX)
        y0 = max(0., y0 - cls.MARGIN_PX); y1 = min(639., 320. + rear)
        width, height = x1 - x0, y1 - y0
        self.zoom = min(640. / width, 640. / height)
        self.out_w = int(round(width * self.zoom))
        self.out_h = int(round(height * self.zoom))
        self.origin = (x0, y0)
        # One resample straight onto the output grid, so cropping costs nothing
        # in sharpness.
        v, u = np.indices((self.out_h, self.out_w), dtype=np.float32)
        px = x0 + u / self.zoom
        py = y0 + v / self.zoom
        self.x, self.y, self.visible = self._sample(robot, px, py)

    def place(self, right, forward):
        """Image pixel for a ground point, so overlays land where the warp put it."""
        r = type(self).radius_px(math.hypot(right, forward))
        bearing = math.atan2(right, forward)
        px = 320. + r * math.sin(bearing)
        py = 320. - r * math.cos(bearing)
        return (int(round((px - self.origin[0]) * self.zoom)),
                int(round((py - self.origin[1]) * self.zoom)))

    def apply(self, image):
        out = cv2.remap(image, self.x, self.y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT)
        out[~self.visible] = (28, 28, 28)
        return out


def camera_path(robot, camera, points):
    """Project sampled metric segments; label offscreen waypoints explicitly."""
    image=camera.copy();height,width=image.shape[:2];visible_segments=0
    color=(255,210,90)
    def project(p):
        q=robot.pixel(*p) if p[1]>0 else None
        return tuple(int(round(v)) for v in q) if q and all(math.isfinite(v) and abs(v)<1e6 for v in q) else None
    for a,b in zip(points,points[1:]):
        last=None
        steps=max(2,min(400,int(math.hypot(b[0]-a[0],b[1]-a[1])*2)))
        for t in np.linspace(0.,1.,steps+1):
            q=project([a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1])])
            if q and last:
                ok,start,end=cv2.clipLine((0,0,width,height),last,q)
                if ok:cv2.line(image,start,end,color,3);visible_segments+=1
            last=q
    offscreen=[]
    for i,p in enumerate(points[1:],1):
        q=project(p)
        if q and 0<=q[0]<width and 0<=q[1]<height:
            cv2.circle(image,q,5,color,-1);cv2.putText(image,str(i),(q[0]+7,q[1]-7),0,.5,color,1)
        else:offscreen.append((i,p,q))
    if offscreen:
        i,p,q=offscreen[0]
        # Border marker is a direction indicator, never an invented floor pixel.
        if q is None:edge=(width//2,height-12)
        else:
            dx,dy=q[0]-width/2,q[1]-height/2
            scale=min((width/2-12)/max(abs(dx),1),(height/2-12)/max(abs(dy),1))
            edge=(int(width/2+dx*scale),int(height/2+dy*scale))
        dx,dy=edge[0]-width/2,edge[1]-height/2;n=max(1,math.hypot(dx,dy))
        start=(int(edge[0]-28*dx/n),int(edge[1]-28*dy/n))
        cv2.arrowedLine(image,start,edge,color,3,tipLength=.4)
        text='Waypoint %d OFFSCREEN | %.1f cm away | see top-down'%(i,math.hypot(*p))
    elif len(points)<2:text='No remaining trajectory'
    else:text='Blue: remaining trajectory projected onto floor'
    cv2.rectangle(image,(0,0),(width,32),(12,14,17),-1)
    cv2.putText(image,text,(9,22),0,.46,color,1)
    return image,dict(visible_segments=visible_segments,offscreen_waypoints=len(offscreen))


def render(robot, camera, route, pose, trace, background=None, full_trace=None, completed=0):
    """4 m square, camera at center, current heading points up."""
    size=640;scale=size/400.;center=size/2
    grid=np.full((size,size,3),28,np.uint8)
    if background is not None:
        # Upstream top-down image contract is also centered 4 m square.
        h=math.radians(pose[2]);c,s=math.cos(h),math.sin(h)
        # Source submission pixel -> current egocentric pixel, including translation.
        origin=fetch.rebase([[0.,0.]],pose)[0]
        mat=np.array([[c,s,center-c*center-s*center+scale*origin[0]],
                      [-s,c,center+s*center-c*center-scale*origin[1]]],np.float32)
        grid=cv2.warpAffine(background,mat,(size,size),borderValue=(28,28,28))
    if all(hasattr(robot,k) for k in ('pitch','height','K','D')):
        projector=getattr(robot,'_local_floor_projection',None)
        if projector is None:
            projector=CameraFloorProjection(robot);robot._local_floor_projection=projector
        projected=projector.apply(camera)
        grid[projector.visible]=projected[projector.visible]
    for cm in range(-200,201,25):
        v=int(center+cm*scale);shade=(65,65,65) if cm%100 else (110,110,110)
        cv2.line(grid,(v,0),(v,size-1),shade,1);cv2.line(grid,(0,v),(size-1,v),shade,1)
    def gp(p):return (int(round(center+p[0]*scale)),int(round(center-p[1]*scale)))
    remaining=fetch.rebase(route[completed:],pose)
    current=([[0.,0.]]+remaining) if remaining else []
    past=fetch.rebase([p[:2] for p in trace],pose)
    full=fetch.rebase([p[:2] for p in (full_trace if full_trace is not None else trace)],pose)
    cam,_=camera_path(robot,camera,current)
    previous=None
    for i,p in enumerate(current):
        q=gp(p)
        if previous is not None:cv2.line(grid,previous,q,(255,210,90),6)
        cv2.circle(grid,q,4,(255,210,90),-1)
        if i:cv2.putText(grid,str(i),(q[0]+6,q[1]-6),0,.4,(255,210,90),1)
        previous=q
    # Thick full-run history remains visible around the thinner latest segment.
    if len(full)>1:
        cv2.polylines(grid,[np.array([gp(p) for p in full],np.int32)],False,(70,255,90),7)
        start=gp(full[0]);cv2.circle(grid,start,5,(70,255,90),1)
        cv2.putText(grid,'run start',(start[0]+9,start[1]+12),0,.4,(70,255,90),1)
    if len(past)>1:
        cv2.polylines(grid,[np.array([gp(p) for p in past],np.int32)],False,(0,160,255),2)
        for point in past:cv2.circle(grid,gp(point),2,(0,160,255),-1)
    if past:
        mark=gp(past[0]);cv2.drawMarker(grid,mark,(0,160,255),cv2.MARKER_DIAMOND,10,1)
        cv2.putText(grid,'instruction',(mark[0]+9,mark[1]-9),0,.4,(0,160,255),1)
    cv2.arrowedLine(grid,(320,320),(320,297),(0,220,255),3)
    cv2.putText(grid,'4 m x 4 m | 25 cm grid | camera at center',(10,20),0,.46,(230,230,230),1)
    cv2.putText(grid,'green: whole run | orange: since instruction',(10,604),0,.44,(230,230,230),1)
    cv2.putText(grid,'blue: remaining | yellow: current camera',(10,625),0,.44,(230,230,230),1)
    return dict(camera=encode(cam),floor=encode(grid))


def artifact_root():
    root=Path(json.loads(Path('/mnt/robotlogs/current-search.json').read_text())['root'])
    if not str(root.resolve()).startswith('/mnt/robotlogs/goals/'):
        raise ValueError('USB run root required')
    return root


class Pause(Exception):
    pass


class Execution:
    def __init__(self,bench,body,seconds=None):
        self.bench=bench;self.robot=bench.robot;self.command=validate(body)
        self.deadline=time.monotonic()+seconds if seconds is not None else None
        self.render_thread=None;self.render_error=None;self.control_intervals=[];self.last_control=None
        self.tracking_rejections=0;self.tracking_rejection_streak=0;self.tracking_gap_started=None
        self.drive_started=None;self.drive_start_distance=0.;self.pack_v=None
        self.stop_after=body.get('stop_after_cm');self.stop_waypoints=body.get('stop_after_waypoints');self.travelled=0.
        self.pose=[0.,0.,0.];self.trace=[list(self.pose)];self.phase='ready';self.error=None
        self.id=uuid.uuid4().hex;self.started=time.time();self.last_publish=0.;self.completed=0
        self.route=self.command['waypoints_cm'];self.background=None
        visual=body.get('visualization') or {}
        if set(visual)-{'rgb_jpeg_base64','topdown_jpeg_base64'}:raise ValueError('Unknown visualization fields')
        self.camera=bench.frame.copy()
        for key in visual:
            raw=base64.b64decode(visual[key],validate=True)
            im=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            if im is None:raise ValueError('Invalid visualization JPEG')
            if key=='topdown_jpeg_base64':
                if im.shape[:2]!=(640,640):raise ValueError('Top-down visualization must be 640x640 covering 4m square')
                self.background=im
            else:self.camera=im
        root=artifact_root()
        marker=(str(root),getattr(self.robot,'token',{}).get('session_id'))
        history=getattr(bench,'local_run_history',None)
        if history is None or history['marker']!=marker:
            history=dict(marker=marker,trace=[[0.,0.,0.]],pose=[0.,0.,0.],instructions=[])
            bench.local_run_history=history
        self.history=history
        self.origin=list(history['pose'])
        self.prefix=[list(p) for p in history['trace']]
        history['instructions'].append(dict(execution_id=self.id,origin_cm_deg=self.origin,
                                           trace_index=len(self.prefix)-1,issued_at=self.started))
        self.root=root/'local-executions'/self.id;self.root.mkdir(parents=True)
        (self.root/'command.json').write_text(json.dumps(body))
        cv2.imwrite(str(self.root/'initial.jpg'),self.camera)
        self.publish(force=True)

    def paths(self):
        whole=self.prefix+[compose(self.origin,p) for p in self.trace[1:]]
        world_pose=compose(self.origin,self.pose)
        # Renderer takes full history in the instruction frame; all three layers
        # then undergo the identical current-camera transform.
        local_xy=fetch.rebase([p[:2] for p in whole],self.origin)
        return whole,world_pose,local_xy

    def result(self):
        whole,world_pose,_=self.paths()
        return dict(control_intervals_s=list(self.control_intervals),travelled_cm=self.travelled,stop_after_cm=self.stop_after,
                    tracking_rejections=self.tracking_rejections,pack_voltage_v=self.pack_v,
                    drive_duty=drive_duty(0.,self.pack_v),
                    tracking_rejection_streak=self.tracking_rejection_streak,
                    run_pose_cm_deg=world_pose,run_estimated_trajectory=whole,
                    instruction_origin_cm_deg=self.origin,instruction_history=list(self.history['instructions']),
                    remaining_trajectory_cm=fetch.rebase(self.route[self.completed:],self.pose),
                    execution_id=self.id,phase=self.phase,error=self.error,pose_cm_deg=self.pose,
                    estimated_trajectory=self.trace,completed_waypoints=self.completed,
                    command=self.command,artifact_path=str(self.root),started_at=self.started,
                    updated_at=time.time(),odometry='planar vision translation + IMU yaw; measured pivot during turns')

    def publish(self,force=False):
        if not force and time.monotonic()-self.last_publish<.3:return
        if self.phase=='driving':
            # Snapshot a consistent observation; rendering and USB cannot stall control.
            if self.render_thread is not None and self.render_thread.is_alive():return
            self.last_publish=time.monotonic()
            whole,world_pose,_=self.paths()
            self.history['trace']=whole;self.history['pose']=world_pose
            frozen=copy.copy(self)
            for name in ('pose','trace','history','control_intervals'):
                setattr(frozen,name,copy.deepcopy(getattr(self,name)))
            frozen.camera=self.camera.copy()
            def render_snapshot():
                try:frozen._publish(True)
                except Exception as exc:self.render_error=str(exc)
            self.render_thread=threading.Thread(target=render_snapshot,daemon=True)
            self.render_thread.start()
        else:
            if self.render_thread is not None:self.render_thread.join()
            self._publish(force)

    def _publish(self,force=False):
        if not force and time.monotonic()-self.last_publish<.3:return
        self.last_publish=time.monotonic()
        whole,world_pose,full_local=self.paths()
        self.history['trace']=whole;self.history['pose']=world_pose
        b=self.bench;previous=getattr(b,'planner_snapshot',{})
        b.planner_snapshot=dict(id=previous.get('id',0)+1,phase=self.phase,image_time=time.time(),
            note='Local trajectory only | '+self.phase+' | '+str(self.error or ''),retained_plan=False,
            images=render(self.robot,self.camera,self.route,self.pose,self.trace,self.background,
                          full_trace=full_local,completed=self.completed),
            source_rgb=encode(self.camera),
            route_cm=fetch.rebase(self.route[self.completed:],self.pose),execution=self.result())
        snapshots=self.root/'snapshots';snapshots.mkdir(exist_ok=True)
        (snapshots/('%04d-source.jpg'%b.planner_snapshot['id'])).write_bytes(
            base64.b64decode(b.planner_snapshot['source_rgb'].split(',',1)[1]))
        for view,data in b.planner_snapshot['images'].items():
            (snapshots/('%04d-%s.jpg'%(b.planner_snapshot['id'],view))).write_bytes(base64.b64decode(data.split(',',1)[1]))
        with (self.root/'states.jsonl').open('a') as f:f.write(json.dumps(self.result())+'\n')

    def check(self):
        if self.bench.abort.is_set():raise fetch.Stop('cancelled')
        if time.time()-self.started>90:raise fetch.Stop('90 second execution limit')
        status=self.robot.check()
        # The health check already costs a status round trip; read the pack off
        # it rather than asking again, so drive duty tracks the sag in real time.
        self.pack_v=((status or {}).get('power') or {}).get('pack_voltage_v')
        if self.render_error:raise fetch.Stop('snapshot recording failed: '+self.render_error)
        if self.deadline is not None and time.monotonic()>=self.deadline:raise Pause('replanning interval reached')

    def turn(self,degrees):
        """Rotate by `degrees`, closing the residual over a few bounded sweeps.

        One sweep was the whole attempt, checked against a flat 8 degrees. That
        failed turns which had all but arrived -- and a failed turn aborts the
        trajectory, so a 160 degree pivot that came up 10 degrees short threw
        away the escape it had just performed.

        Retrying is safe because the pose is composed from the *measured* angle
        after every sweep: an under-turn leaves the executor's estimate correct
        and merely unfinished. The deadline, the 90 s limit and the sweep's own
        stall detector all still bound this.
        """
        tolerance=turn_tolerance(degrees)
        swept=0.
        for _ in range(TURN_ATTEMPTS):
            remaining=float(degrees)-swept
            self.check();self.phase='turning';self.publish(True)
            measured=getattr(self.robot,'turn_continuous',self.robot.turn)(remaining)
            swept+=measured
            self.pose=compose(self.pose,list(fetch.pivot_shift(measured))+[measured])
            self.trace.append(list(self.pose));self.camera,_=self.robot.frame();self.publish(True)
            if abs(float(degrees)-swept)<=tolerance:
                return swept
            if abs(measured)<1.:
                break            # a sweep that moved nothing will not move next time
        raise fetch.Stop('turn did not converge: %.1f of %.1f deg (%.1f short, '
                         'tolerance %.1f)'
                         %(swept,float(degrees),float(degrees)-swept,tolerance))

    def run(self):
        try:
            if self.command['initial_turn_deg']:self.turn(self.command['initial_turn_deg'])
            before,ob=self.robot.frame();h0=self.robot.heading(ob['imu_samples'][-1])
            for index,goal in enumerate(self.route):
                began=time.monotonic();best=1e9;progress=time.monotonic()
                while True:
                    self.check()
                    relative=fetch.rebase([goal],self.pose)[0];distance=math.hypot(*relative)
                    if distance<=3:break
                    if time.monotonic()-began>25:raise fetch.Stop('waypoint timeout')
                    if distance<best-.5:best=distance;progress=time.monotonic()
                    if time.monotonic()-progress>5:raise fetch.Stop('no progress toward next waypoint')
                    bearing=pivot_aim(relative)
                    if abs(bearing)>12:
                        self.robot.hold(0,0);self.last_control=None;self.turn(bearing)
                        # Aiming is progress of a different kind. A pivot barely
                        # changes range to the waypoint -- only by the 7.28 cm
                        # pivot offset -- so charging its seconds to the
                        # no-progress timer made the robot give up for having
                        # done exactly what the bearing asked. The 25 s waypoint
                        # timeout still bounds any turn/aim oscillation.
                        progress=time.monotonic()
                        before,ob=self.robot.frame();h0=self.robot.heading(ob['imu_samples'][-1]);continue
                    self.phase='driving'
                    duty=drive_duty(distance,self.pack_v)
                    trim=max(-.025,min(.025,bearing*.002))
                    now=time.monotonic()
                    if self.last_control is not None:self.control_intervals.append(now-self.last_control)
                    self.last_control=now
                    if self.drive_started is None:
                        self.drive_started=now;self.drive_start_distance=self.travelled
                    # Keep moving across camera measurements; renew only after
                    # the previous observation passed. Fault/lease stops remain.
                    self.robot.hold(min(DRIVE_DUTY_CEILING, duty+trim),
                                    min(DRIVE_DUTY_CEILING, duty-trim))
                    time.sleep(.06)
                    after,ob=self.robot.frame();h1=self.robot.heading(ob['imu_samples'][-1])
                    # Invert the observed static floor transform for camera translation.
                    try:
                        rotation,translation,quality=self.bench.odometer.tracker.motion(before,after)
                    except Exception as exc:
                        if not recoverable_tracking_error(exc):raise
                        # One weak optical-flow pair is not a controller
                        # failure.  Keep the motor lease alive, retain the IMU
                        # yaw increment, and anchor the next comparison on the
                        # newest frame.  Translation remains unknown for this
                        # one interval, so never fabricate distance travelled.
                        now=time.monotonic()
                        self.tracking_rejections+=1
                        self.tracking_rejection_streak+=1
                        if self.tracking_gap_started is None:self.tracking_gap_started=now
                        yaw=(h1-h0+180)%360-180
                        self.pose=compose(self.pose,[0.,0.,yaw])
                        self.trace.append(list(self.pose));self.camera=after
                        before=after;h0=h1;self.publish()
                        if (self.tracking_rejection_streak>=MAX_TRACKING_REJECTIONS
                                or now-self.tracking_gap_started>=MAX_TRACKING_GAP_S):
                            raise fetch.Stop('floor tracking unavailable after %d consecutive rejected frames: %s'
                                             %(self.tracking_rejection_streak,exc))
                        continue
                    self.tracking_rejection_streak=0;self.tracking_gap_started=None
                    delta=-np.dot(rotation.T,translation)
                    if not np.isfinite(delta).all() or np.linalg.norm(delta)>8:raise fetch.Stop('invalid visual displacement')
                    yaw=(h1-h0+180)%360-180
                    self.pose=compose(self.pose,[float(delta[0]),float(delta[1]),yaw])
                    self.travelled+=float(np.linalg.norm(delta))
                    self.trace.append(list(self.pose));self.camera=after
                    before=after;h0=h1;self.publish()
                    if self.stop_after is not None and self.travelled>=self.stop_after:
                        raise Pause('requested travel limit reached')
                    # Renew only after valid observations; service leases remain active.
                self.robot.hold(0,0);self.completed=index+1;self.publish(True)
                if self.stop_waypoints is not None and self.completed>=self.stop_waypoints:raise Pause('requested waypoint limit reached')
            self.robot.hold(0,0)
            self.phase='completed'
        except Pause:
            powered=(time.monotonic()-self.drive_started) if self.drive_started is not None else 0.
            progress=self.travelled-self.drive_start_distance
            if self.deadline is not None and powered>=2. and progress<.5:
                self.phase='failed'
                self.error='drive stalled: %.1f cm measured in %.1f s powered' % (progress,powered)
            else:
                self.phase='paused';self.error=None
        except Exception as exc:
            self.phase='cancelled' if self.bench.abort.is_set() else 'failed';self.error=str(exc)
        finally:
            try:self.robot.hold(0,0)
            except Exception as exc:
                # Never overwrite the reason the run actually ended. Once the
                # control generation is invalidated, every later command fails
                # the same way -- including this stop -- so an unconditional
                # overwrite replaced the first cause with its own consequence.
                # A 29.5 cm segment died of some transient the service rejected
                # and reported only "stop verification failed", which is the one
                # thing that was guaranteed to be true and says nothing.
                self.phase='failed'
                self.error=(self.error+' | stop verification also failed: '+str(exc)
                            if self.error else 'stop verification failed: '+str(exc))
            try:self.camera,_=self.robot.frame()
            except Exception:pass
            self.publish(True);(self.root/'result.json').write_text(json.dumps(self.result()))
            self.bench.say('local execution '+self.phase+': '+str(self.error or self.pose))


def launch(bench,body):
    command=validate(body)
    if bench.running():raise ValueError('Controller busy')
    if bench.frame is None or body.get('frame_token')!=bench.frame_token:raise ValueError('Fresh frame_token required')
    # A new instruction is not the continuation of a cancelled one. The robot is
    # idle here -- bench.running() was just checked -- so adopting the service's
    # current generation cannot let a cancelled controller resume, and it is the
    # only thing standing between one transient rejection and a planner that
    # refuses every subsequent execution until someone presses stop.
    bench.robot.resync()
    bench.abort.clear();job=Execution(bench,body);bench.local_execution=job
    bench.flight=threading.Thread(target=job.run,daemon=True);bench.flight.start()
    return job.result()
