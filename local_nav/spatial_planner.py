"""Bounded SE(2) search in an explicitly inspected static map.

No sensors, motor access or model calls. Coordinates reference the camera lens.
The nominal turn pivot is ONLY a search prediction: execution must measure its
translation and replan after each primitive. Collision checks cover every fixed
pivot inside the chassis, including heading overrun and position uncertainty.
"""
import heapq
import math
import time

from route_geometry import rectangle, finite, contains, overlap


def transform(pose, x, z):
    a = math.radians(pose[2])
    c, s = math.cos(a), math.sin(a)
    return pose[0]+c*x+s*z, pose[1]-s*x+c*z


def bounds(points, margin=0.):
    return [min(p[0] for p in points)-margin,
            min(p[1] for p in points)-margin,
            max(p[0] for p in points)+margin,
            max(p[1] for p in points)+margin]


def normalize(angle):
    return (angle+180.) % 360.-180.


def sweep(pose, action, precise_turn=False):
    kind, value = action
    value = finite(value)
    if kind == 'drive':
        if not 1 <= abs(value) <= 15:
            raise ValueError('Drive primitive must be 1 to 15 cm')
        local = [-16, -24, 16, value+13] if value > 0 else [-16, value-28, 16, 9]
        return bounds([transform(pose, x, z)
                       for x in (local[0], local[2]) for z in (local[1], local[3])])
    if kind != 'turn' or not 0 < abs(value) <= 30:
        raise ValueError('Turn primitive must be within 30 degrees')
    # Extrema of pivot + R(angle)*(corner-pivot) occur at pivot/body vertices.
    # Analytic angle extrema avoid missing collisions between discrete samples.
    lo, hi = sorted((0., value+math.copysign(5., value)))
    lo, hi = lo-2., hi+2.
    lo, hi = math.radians(lo), math.radians(hi)
    heading = math.radians(pose[2]) if precise_turn else 0.
    points = []
    for px in (-6., 6.):
        for pz in (-15., 0.):
            for bx in (-6., 6.):
                for bz in (-15., 0.):
                    dx, dz = bx-px, bz-pz
                    angles = [lo, hi]
                    for root in (math.atan2(dz, dx)-heading,
                                 math.atan2(-dx, dz)-heading):
                        angles.extend(root+k*math.pi for k in range(-3, 4)
                                      if lo <= root+k*math.pi <= hi)
                    for a in angles:
                        x = px+math.cos(a)*dx+math.sin(a)*dz
                        z = pz-math.sin(a)*dx+math.cos(a)*dz
                        points.append(transform(pose,x,z) if precise_turn else (x,z))
    if precise_turn:
        return bounds(points,7.)
    local = bounds(points, 7.)  # 5 cm clearance + 2 cm pose allowance
    # The executor checks this local rectangle. Transform that same rectangle,
    # rather than a tighter envelope that would not cover every accepted pose.
    return bounds([transform(pose,x,z) for x in (local[0],local[2])
                   for z in (local[1],local[3])])


def predict(pose, action):
    kind, value = action
    if kind == 'drive':
        x, z = transform(pose, 0., value)
        return (x, z, pose[2])
    # Search seed only, never substitute this for measured turn odometry.
    a = math.radians(value)
    x, z = transform(pose, 7.5*math.sin(a), 7.5*(math.cos(a)-1.))
    return (x, z, normalize(pose[2]+value))


class SpatialPlanner:
    def __init__(self, plan):
        self.free = rectangle(plan['inspected_free_rectangle_cm'])
        self.obstacles = [rectangle(r) for r in plan['obstacle_rectangles_cm']]
        self.goal = tuple(finite(v) for v in plan['goal_cm'])
        if len(self.goal) != 2:
            raise ValueError('Goal must be an x,z floor point')
        self.tolerance = finite(plan.get('goal_tolerance_cm', 3.))
        self._sweeps = {}
        self.measured_map_drive=plan.get('measured_map_drive',False)
        if type(self.measured_map_drive) is not bool:
            raise ValueError('measured_map_drive must be boolean')
        self.precise_turn_sweep=plan.get('precise_turn_sweep',False)
        self.goal_heading_turns=plan.get('goal_heading_turns',False)
        if type(self.goal_heading_turns) is not bool:
            raise ValueError('goal_heading_turns must be boolean')
        if type(self.precise_turn_sweep) is not bool:
            raise ValueError('precise_turn_sweep must be boolean')
        if not 1 <= self.tolerance <= 5:
            raise ValueError('Goal tolerance must be 1 to 5 cm')

    def clear(self, envelope):
        return contains(self.free, envelope) and not any(overlap(envelope, r) for r in self.obstacles)

    def arrived(self, pose):
        return math.hypot(pose[0]-self.goal[0], pose[1]-self.goal[1]) <= self.tolerance

    def envelope(self, pose, action):
        if self.measured_map_drive and action[0]=='drive':
            from mapped_drive import nominal_bounds
            return nominal_bounds(pose,action[1])
        key = (round(normalize(pose[2]), 8), action)
        if key not in self._sweeps:
            self._sweeps[key] = sweep((0., 0., key[0]), action,self.precise_turn_sweep)
        box = self._sweeps[key]
        return [box[0]+pose[0], box[1]+pose[1], box[2]+pose[0], box[3]+pose[1]]

    def search(self, pose=(0., 0., 0.), max_nodes=10000, timeout_seconds=1.):
        pose = tuple(finite(v) for v in pose)
        if len(pose) != 3:
            raise ValueError('Pose must be x,z,yaw')
        started = time.perf_counter()
        stationary = bounds([transform(pose, x, z) for x in (-6., 6.) for z in (-15., 0.)], 7.)
        if not self.clear(stationary):
            return dict(outcome='blocked_start', actions=[], expanded=0)
        queue = [(0., 0, 0., pose, [])]
        visited = {}
        serial = expanded = 0
        while queue and expanded < max_nodes and time.perf_counter()-started < timeout_seconds:
            _, _, cost, current, path = heapq.heappop(queue)
            key = (round(current[0]/5.), round(current[1]/5.), round(normalize(current[2])/2.))
            if visited.get(key, float('inf')) <= cost:
                continue
            visited[key] = cost
            expanded += 1
            if self.arrived(current):
                return dict(outcome='route_found', actions=path, expanded=expanded,
                            elapsed_seconds=time.perf_counter()-started,
                            predicted_final_pose=list(current),
                            requires_measured_replanning=True, model_calls=0)
            actions = [('drive', n) for n in (15., 10., 5., -10., -5.)]+[('turn', -30.), ('turn', 30.)]
            if self.goal_heading_turns:
                bearing=math.degrees(math.atan2(self.goal[0]-current[0],self.goal[1]-current[1]))
                turn=normalize(bearing-current[2])
                if 1.5<abs(turn)<30:
                    actions.append(('turn',round(turn,1)))
            correction = normalize(round(current[2]/30.)*30.-current[2])
            if 1.5 < abs(correction) <= 5.:
                actions.append(('turn',correction))
            for action in actions:
                envelope = self.envelope(current, action)
                if not self.clear(envelope):
                    continue
                nxt = predict(current, action)
                step = abs(action[1])/15. if action[0]=='drive' else 2.
                if action[0]=='drive' and action[1]<0:
                    step += .5
                newcost = cost+step+.2
                heuristic = math.hypot(nxt[0]-self.goal[0], nxt[1]-self.goal[1])/15.
                serial += 1
                record = dict(kind=action[0], value=action[1], swept_bounds_cm=envelope,
                              predicted_start_pose=list(current), predicted_end_pose=list(nxt))
                # Weighted best-first search prioritizes quick feasible routes;
                # it does not claim the shortest or minimum-time route.
                heapq.heappush(queue, (newcost+2.5*heuristic, serial, newcost, nxt, path+[record]))
        return dict(outcome='search_budget_exhausted' if queue else 'no_checked_route',
                    actions=[], expanded=expanded, elapsed_seconds=time.perf_counter()-started,
                    model_calls=0)
