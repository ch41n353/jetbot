"""Conservative static-map checks for a straight, bounded route (centimetres).

Map coordinates stay fixed at the initial lens position: x right, z forward.
Rectangles include offscreen obstacles; absence of a rectangle is NOT free space.
"""
import math


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Expected finite numeric centimetres')
    return value


def rectangle(value):
    if len(value) != 4:
        raise ValueError('Rectangle must be [left, back, right, front]')
    x0, z0, x1, z1 = [finite(v) for v in value]
    if x0 >= x1 or z0 >= z1:
        raise ValueError('Empty or reversed rectangle')
    return [x0, z0, x1, z1]


def overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def contains(outer, inner):
    return all((outer[0] <= inner[0], outer[1] <= inner[1],
                outer[2] >= inner[2], outer[3] >= inner[3]))


def swept_pose_bounds(x, z, yaw_degrees, direction, goal):
    """Actual chassis plus 5 cm clearance, 2 cm position uncertainty and braking.

    Include +/-2 degrees of heading uncertainty using analytic extrema, rather
    than spending the maximum heading allowance at every measured heading.
    The existing four-centimetre overrun cap bounds remaining braking travel.
    """
    x, z, yaw_degrees, goal = map(finite, (x, z, yaw_degrees, goal))
    if direction not in (-1, 1):
        raise ValueError('Invalid travel direction')
    coast = min(4., max(0., goal+4-direction*z))
    rear, front = (-15., coast) if direction==1 else (-15.-coast, 0.)
    lo, hi = math.radians(yaw_degrees-2), math.radians(yaw_degrees+2)
    points=[]
    for px in (-6.,6.):
        for pz in (rear,front):
            angles=[lo,hi]
            for root in (math.atan2(pz,px), math.atan2(-px,pz)):
                angles += [root+k*math.pi for k in (-1,0,1) if lo<=root+k*math.pi<=hi]
            for a in angles:
                points.append((x+px*math.cos(a)+pz*math.sin(a),
                               z-px*math.sin(a)+pz*math.cos(a)))
    return [min(p[0] for p in points)-7, min(p[1] for p in points)-7,
            max(p[0] for p in points)+7, max(p[1] for p in points)+7]


class StraightRoute:
    def __init__(self, plan):
        self.waypoints = [finite(v) for v in plan['waypoints_cm']]
        if not self.waypoints or len(self.waypoints) > 15:
            raise ValueError('Expected 1 to 15 forward waypoints')
        if any(b <= a for a, b in zip([0] + self.waypoints, self.waypoints)):
            raise ValueError('Waypoints must be increasing positive travel distances')
        if not 1 <= self.waypoints[-1] <= 15:
            raise ValueError('Total route distance must be 1 to 15 cm')
        self.free = rectangle(plan['inspected_free_rectangle_cm'])
        self.obstacles = [rectangle(r) for r in plan['obstacle_rectangles_cm']]
        direction = plan.get('travel_direction', 'forward')
        if direction not in ('forward', 'reverse'):
            raise ValueError('Unknown travel direction')
        self.direction = 1 if direction == 'forward' else -1
        # 12x15 body, 5 clearance, 2 pose uncertainty, 2 heading envelope,
        # 1 lateral tracking allowance; plus 4 forward braking allowance.
        # 2 cm covers rotation of every chassis corner through +/-5 degrees.
        self.corridor = ([-16, -24, 16, self.waypoints[-1] + 13] if self.direction == 1
                         else [-16, -self.waypoints[-1]-28, 16, 9])
        if not contains(self.free, self.corridor):
            raise ValueError('Swept chassis corridor enters uninspected space')
        if any(overlap(self.corridor, obstacle) for obstacle in self.obstacles):
            raise ValueError('Swept chassis corridor intersects a retained obstacle')

    def check_pose(self, x, z, yaw_degrees):
        x, z, yaw_degrees = map(finite, (x, z, yaw_degrees))
        progress = self.direction*z
        if abs(yaw_degrees) > 5 or progress < -.5 or progress > self.waypoints[-1] + 4:
            raise RuntimeError('Measured pose left the checked route envelope')
        if not contains(self.corridor, swept_pose_bounds(x,z,yaw_degrees,self.direction,self.waypoints[-1])):
            raise RuntimeError('Swept chassis left the checked route envelope')


class ProgressGuard:
    def __init__(self):
        self.samples = []

    def check(self, elapsed, forward):
        self.samples.append((elapsed, forward))
        if elapsed >= .4 and forward < .5:
            raise RuntimeError('No forward progress after 400 ms')
        old = [s for s in self.samples if elapsed - s[0] >= .4]
        if old and forward - old[-1][1] < .3:
            raise RuntimeError('Drive stalled; stopping')
        self.samples = self.samples[-100:]
