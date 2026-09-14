"""Experimental drive adapter checking measured chassis against retained map.

Unlike StraightRoute's fixed wide rectangle, the nominal search tube uses the
same chassis/braking geometry checked at every camera update. Heading, powered
time, estimator, progress and motor-token guards remain in the executor.
This is a static-map, discretely observed guard, not obstacle sensing.
"""
import math
from route_geometry import finite, swept_pose_bounds


def nominal_bounds(pose,distance):
    direction=1 if distance>0 else -1
    boxes=[]
    # Translation is linear; union of endpoint extrema covers intermediate poses.
    for progress in (0.,abs(distance)):
        a=math.radians(pose[2])
        x=pose[0]+math.sin(a)*direction*progress
        z=pose[1]+math.cos(a)*direction*progress
        # swept_pose_bounds' z-dependent remaining coast is expressed in the
        # local route frame. Force its full four-cm braking allowance here.
        local=swept_pose_bounds(0.,0.,pose[2],direction,abs(distance))
        boxes.append([local[0]+x,local[1]+z,local[2]+x,local[3]+z])
    return [min(b[0] for b in boxes),min(b[1] for b in boxes),
            max(b[2] for b in boxes),max(b[3] for b in boxes)]


class MappedDriveRoute:
    def __init__(self,planner,pose,distance,coalesced=False):
        distance=finite(distance)
        if type(coalesced) is not bool:
            raise ValueError('coalesced must be boolean')
        limit=30 if coalesced else 15
        if not 1<=abs(distance)<=limit:
            raise ValueError('Mapped drive exceeds its distance limit')
        self.powered_limit_seconds=4. if coalesced and abs(distance)>15 else 2.
        self.waypoints=[abs(distance)]
        self.direction=1 if distance>0 else -1
        self.planner=planner
        self.pose=tuple(pose)
        if not planner.clear(nominal_bounds(pose,distance)):
            raise ValueError('Nominal chassis path conflicts with inspected map')

    def check_pose(self,x,z,yaw_degrees):
        from spatial_planner import transform
        x,z,yaw_degrees=map(finite,(x,z,yaw_degrees))
        progress=self.direction*z
        if abs(yaw_degrees)>5 or progress<-.5 or progress>self.waypoints[-1]+4:
            raise RuntimeError('Mapped drive exceeded heading or progress limits')
        gx,gz=transform(self.pose,x,z)
        # Full four-cm braking allowance, even close to the requested endpoint.
        local=swept_pose_bounds(0.,0.,self.pose[2]+yaw_degrees,self.direction,self.waypoints[-1])
        body=[local[0]+gx,local[1]+gz,local[2]+gx,local[3]+gz]
        if not self.planner.clear(body):
            raise RuntimeError('Measured chassis and braking space conflict with inspected map')
