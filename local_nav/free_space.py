"""Accumulate per-observation free regions into map-frame free rectangles.

Each camera observation certifies a narrow forward fan at one heading, given
here as a camera-local rectangle (x right, z forward). Rotated into the map by
a non-zero yaw that fan is no longer axis aligned, and its axis-aligned
bounding box would claim corners that were never inspected, so these rectangles
are an INSCRIBED (inner) approximation instead.

Approach: the rotated footprint is a convex quadrilateral. Cut it into
grid-aligned z slabs; inside one slab keep the x span common to both slab
edges. Because the quadrilateral is convex its left boundary is a convex and
its right boundary a concave function of z, so that common span is exactly the
widest axis-aligned rectangle contained in the quadrilateral over that slab.
The union of the slab rectangles is therefore always a subset of the true
observed footprint.

Conservatism: what is dropped is the staircase wedge along each slanted edge,
bounded per slab by the slab height times the edge slope, plus the inward snap
of every coordinate to `grid` (which also makes rectangles from different
observations line up so they coalesce). `merge` trims to a cap by discarding
the smallest rectangles, which again only removes certified space. Every step
under-reports free space; the planner then refuses routes it cannot prove, and
never invents floor that was not inspected.
"""
import math

from route_geometry import finite, rectangle
from spatial_planner import transform


def _up(value, grid):
    return math.ceil(value/grid-1e-9)*grid


def _down(value, grid):
    return math.floor(value/grid+1e-9)*grid


def _span(quad, z):
    """x interval of the convex quad at height z, or None below/above it."""
    xs = []
    for i in range(len(quad)):
        (ax, az), (bx, bz) = quad[i], quad[(i+1) % len(quad)]
        if az == bz:
            if az == z:
                xs.extend((ax, bx))
        elif min(az, bz) <= z <= max(az, bz):
            xs.append(ax+(bx-ax)*(z-az)/(bz-az))
    return (min(xs), max(xs)) if xs else None


def local_to_map(pose, local, slabs=12, grid=.5):
    """Inscribed axis-aligned cover of one camera-local free rectangle."""
    pose = [finite(v) for v in pose]
    if len(pose) != 3:
        raise ValueError('Pose must be x,z,yaw')
    grid = finite(grid)
    if not 0 < grid <= 10:
        raise ValueError('Grid must be a positive size in centimetres')
    x0, z0, x1, z1 = rectangle(local)
    # Rounding to a picometre keeps an axis-aligned edge exactly horizontal, so
    # rotations by multiples of 90 degrees do not lose a whole end slab to the
    # rounding noise in cos/sin. The outward error is far below the map's units.
    quad = [tuple(round(v, 9) for v in transform(pose, x, z))
            for x, z in ((x0, z0), (x1, z0), (x1, z1), (x0, z1))]
    lo, hi = _up(min(p[1] for p in quad), grid), _down(max(p[1] for p in quad), grid)
    steps = int(round((hi-lo)/grid))
    if steps < 1:
        return []
    count = min(max(int(slabs), 1), steps)
    edges = [lo+grid*int(round(steps*i/float(count))) for i in range(count+1)]
    spans = [_span(quad, z) for z in edges]
    out = []
    for i in range(count):
        a, b = spans[i], spans[i+1]
        if a is None or b is None:
            continue
        left, right = _up(max(a[0], b[0]), grid), _down(min(a[1], b[1]), grid)
        if right > left+1e-9 and edges[i+1] > edges[i]+1e-9:
            out.append([left, edges[i], right, edges[i+1]])
    return out


def _coalesce(rects, axis):
    """Join rectangles with an identical span on the other axis that touch or
    overlap, so the union is preserved apart from seams below a picometre."""
    groups = {}
    for r in rects:
        groups.setdefault((r[axis], r[axis+2]), []).append(r)
    out = []
    other = 1-axis
    for key in groups:
        run = sorted(groups[key], key=lambda r: r[other])
        current = list(run[0])
        for r in run[1:]:
            if r[other] <= current[other+2]+1e-9:
                current[other+2] = max(current[other+2], r[other+2])
            else:
                out.append(current)
                current = list(r)
        out.append(current)
    return out


def _dedupe(rects):
    out, seen = [], set()
    for r in rects:
        key = tuple(r)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def merge(rects, cap=256):
    """Coalesce, drop contained rectangles, then cap the list by keeping the largest."""
    out = _dedupe([rectangle(r) for r in rects])
    for axis in (1, 0, 1, 0):
        out = _dedupe(_coalesce(out, axis))
    out = [r for r in out
           if not any(o is not r and o[0] <= r[0] and o[1] <= r[1] and o[2] >= r[2] and o[3] >= r[3]
                      for o in out)]
    if len(out) > cap:
        # Trimming discards inspected space rather than inventing any.
        out.sort(key=lambda r: (r[2]-r[0])*(r[3]-r[1]), reverse=True)
        out = out[:cap]
    out.sort()
    return out


def free_rectangles(observations, slabs=12, grid=.5, cap=256):
    """Map-frame rectangles for a plan's inspected_free_rectangles_cm."""
    out = []
    for pose, local in observations:
        out.extend(local_to_map(pose, local, slabs, grid))
    return merge(out, cap)


class FreeSpace:
    """Incremental accumulator; merges after every observation to stay bounded."""

    def __init__(self, slabs=12, grid=.5, cap=256):
        self.slabs, self.grid, self.cap, self.rects = slabs, grid, cap, []

    def add(self, pose, local):
        self.rects = merge(self.rects+local_to_map(pose, local, self.slabs, self.grid), self.cap)
        return self.rects

    def rectangles(self):
        return [list(r) for r in self.rects]
