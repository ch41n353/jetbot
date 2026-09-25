#!/usr/bin/env python3
"""Re-derive camera pitch from floor references at measured distances.

calibration/floor_geometry.json says "Revalidate after moving the camera
mount", and it means it: every metric range in this stack comes from
intersecting a pixel ray with the floor plane at the stored pitch. Move the
mount and leave the number alone, and the robot keeps reporting confident
distances that are all wrong in the same direction.

Give it one or more references -- an object whose base sits a measured distance
straight ahead, and the pixel where that base meets the floor -- and it solves
for the pitch that reproduces the measurement. Two references at different
distances are worth far more than one: a single point can be fitted by a wrong
pitch and a wrong height together, while two at different ranges cannot.

Nothing is written unless --apply is given, and the previous file is kept.
"""
import argparse
import json
import math
import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, 'local_nav'))

PROFILE = None


def load():
    global PROFILE
    import point_controller as pc
    path = os.path.join(pc.ROOT, 'calibration/floor_geometry.json')
    with open(path) as handle:
        PROFILE = json.load(handle)
    with open(PROFILE['intrinsics_path']) as handle:
        intrinsics = json.load(handle)
    return path, PROFILE, intrinsics


def ground(x, y, pitch_deg, height_cm, K, D):
    """Floor position of a pixel at a candidate pitch, as (right, forward)."""
    xy = cv2.fisheye.undistortPoints(np.array([[[float(x), float(y)]]]),
                                     K, D).reshape(2)
    ray = np.array([xy[0], xy[1], 1.])
    a = math.radians(pitch_deg)
    down = np.array([0., math.cos(a), math.sin(a)])
    forward = np.array([0., 0., 1.]) - down * down[2]
    forward /= np.linalg.norm(forward)
    denominator = ray.dot(down)
    if denominator < .05:
        return None
    scale = height_cm / denominator
    return float(ray.dot(np.cross(down, forward)) * scale), float(ray.dot(forward) * scale)


def solve(references, height_cm, K, D, lo=0., hi=75.):
    """Pitch minimising squared range error over the references."""
    def cost(pitch):
        total = 0.
        for x, y, want in references:
            spot = ground(x, y, pitch, height_cm, K, D)
            if spot is None:
                return float('inf')
            total += (math.hypot(*spot) - want) ** 2
        return total
    # Coarse sweep then golden-section: the cost is smooth and unimodal in
    # pitch over any range where every reference stays below the horizon.
    best, step = lo, .5
    grid = [lo + i * step for i in range(int((hi - lo) / step) + 1)]
    best = min(grid, key=cost)
    left, right = max(lo, best - step), min(hi, best + step)
    for _ in range(80):
        a = left + (right - left) * .382
        b = left + (right - left) * .618
        if cost(a) < cost(b):
            right = b
        else:
            left = a
    return (left + right) / 2.


def base_edge(image, band=(130, 158), columns=(250, 376), search=(200, 440)):
    """Locate a flat object's floor contact edge and its two ends.

    The contact shadow under an object standing on carpet is by far the darkest
    thing in a narrow band around it, and it exists only where the object is.
    So the same signal gives both the line (fit across every column that has
    one) and the extent (where it stops). Brightness cannot do this job here:
    a white box on light carpet separates by about ten grey levels, while the
    shadow separates by a hundred.
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    top, bottom = band
    xs, ys = [], []
    for x in range(*columns):
        xs.append(float(x))
        ys.append(float(top + int(np.argmin(grey[top:bottom, x]))))
    xs, ys = np.array(xs), np.array(ys)
    for _ in range(4):                       # robust fit: drop columns whose
        A = np.vstack([xs, np.ones(len(xs))]).T   # darkest row was not the shadow
        slope, intercept = np.linalg.lstsq(A, ys, rcond=-1)[0]
        residual = np.abs(ys - (slope * xs + intercept))
        keep = residual < max(1.0, 2.5 * np.median(residual))
        xs, ys = xs[keep], ys[keep]

    depth, cols = [], []
    for x in range(*search):
        y = int(round(slope * x + intercept))
        near = grey[y - 4:y + 5, x]
        away = np.concatenate([grey[y - 14:y - 8, x], grey[y + 8:y + 14, x]])
        cols.append(x)
        depth.append(float(np.median(away) - near.min()))
    cols = np.array(cols)
    depth = np.convolve(np.array(depth), np.ones(5) / 5, mode='same')
    lit = np.where(depth > depth.max() * .45)[0]
    best = (lit[0], lit[0]); start = prev = lit[0]
    for i in lit[1:]:
        if i != prev + 1:
            if prev - start > best[1] - best[0]:
                best = (start, prev)
            start = i
        prev = i
    if prev - start > best[1] - best[0]:
        best = (start, prev)
    left, right = float(cols[best[0]]), float(cols[best[1]])
    return ((left, slope * left + intercept), (right, slope * right + intercept))


def solve_span(left, right, length_cm, height_cm, K, D, stored):
    """Pitch that makes two floor points exactly `length_cm` apart.

    Span shrinks monotonically as the camera tilts further down, so this is a
    plain bisection -- but only over pitches where BOTH endpoints fall below the
    horizon. Outside that range the span is not merely wrong, it does not exist,
    and a solver that does not check first fails with a confusing TypeError
    rather than saying so.
    """
    def span(pitch):
        a = ground(left[0], left[1], pitch, height_cm, K, D)
        b = ground(right[0], right[1], pitch, height_cm, K, D)
        if a is None or b is None:
            return None
        return math.hypot(a[0] - b[0], a[1] - b[1])

    grid = [stored + i * .25 for i in range(int((85. - stored) / .25))]
    usable = [p for p in grid if span(p) is not None]
    if not usable:
        raise SystemExit('no pitch puts both endpoints below the horizon')
    lo, hi = usable[0], usable[-1]
    if not (span(hi) <= length_cm <= span(lo)):
        raise SystemExit('%.1f cm is outside the achievable span %.1f..%.1f cm'
                         % (length_cm, span(hi), span(lo)))
    for _ in range(200):
        mid = (lo + hi) / 2.
        if span(mid) > length_cm:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2., span


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', action='append', default=[], metavar='X,Y,CM',
                        help='contact pixel and its measured range, e.g. 320,430,30')
    parser.add_argument('--height-cm', type=float, default=None,
                        help='lens height above the floor; defaults to the stored value')
    parser.add_argument('--image', default=None,
                        help='frame containing a flat object of known length on the floor')
    parser.add_argument('--length-cm', type=float, default=None,
                        help='that object\'s length, measured')
    parser.add_argument('--apply', action='store_true',
                        help='write the solved pitch back to floor_geometry.json')
    args = parser.parse_args()

    path, profile, intrinsics = load()
    K = np.asarray(intrinsics['K'], float)
    D = np.asarray(intrinsics['D'], float)
    height = args.height_cm if args.height_cm is not None else float(profile['camera_height_cm'])

    stored = float(profile['pitch_degrees'])

    if args.image and args.length_cm:
        image = cv2.imread(args.image)
        if image is None:
            raise SystemExit('cannot read ' + args.image)
        left, right = base_edge(image)
        print('base edge  px(%.1f, %.2f) -> px(%.1f, %.2f)   %.0f px apart'
              % (left[0], left[1], right[0], right[1], right[0] - left[0]))
        pitch, span = solve_span(left, right, args.length_cm, height, K, D, 0.5)
        print()
        print('solved pitch  %.3f deg   (%+.2f deg from the stored %.3f)'
              % (pitch, pitch - stored, stored))
        print('  span at that pitch %.3f cm against %.1f known' % (span(pitch), args.length_cm))
        middle = ((left[0] + right[0]) / 2., (left[1] + right[1]) / 2.)
        spot = ground(middle[0], middle[1], pitch, height, K, D)
        print('  predicts the middle of that edge is %.1f cm ahead, %.1f cm right'
              % (spot[1], spot[0]))
        near = ground(320., 479., pitch, height, K, D)
        print('  nearest visible floor straight ahead: %.1f cm' % near[1])
        for slip in (1., 2.):
            shifted, _ = solve_span((left[0], left[1] + slip), (right[0], right[1] + slip),
                                    args.length_cm, height, K, D, 0.5)
            print('  a %.0f px error in the edge row moves pitch %.2f deg'
                  % (slip, abs(shifted - pitch)))
        if args.apply:
            write(path, profile, pitch, height, 'known %.1f cm object' % args.length_cm, 0.)
        else:
            print('\n(dry run: pass --apply to write it)')
        return

    references = []
    for spec in args.reference:
        x, y, cm = (float(v) for v in spec.split(','))
        references.append((x, y, cm))
    if not references:
        raise SystemExit('need at least one --reference X,Y,CM')

    print('stored pitch  %.4f deg   height %.2f cm' % (stored, height))
    print()
    print('%-22s %10s %10s %10s' % ('reference', 'measured', 'at stored', 'error'))
    for x, y, want in references:
        spot = ground(x, y, stored, height, K, D)
        got = math.hypot(*spot) if spot else float('nan')
        print('%-22s %9.1f %10.1f %9.1f' % ('px(%.0f,%.0f)' % (x, y), want, got, got - want))

    pitch = solve(references, height, K, D)
    print()
    print('solved pitch  %.4f deg   (%+.2f deg from stored)' % (pitch, pitch - stored))
    print()
    print('%-22s %10s %10s %10s' % ('reference', 'measured', 'at solved', 'error'))
    worst = 0.
    for x, y, want in references:
        spot = ground(x, y, pitch, height, K, D)
        got = math.hypot(*spot) if spot else float('nan')
        worst = max(worst, abs(got - want))
        print('%-22s %9.1f %10.1f %9.1f' % ('px(%.0f,%.0f)' % (x, y), want, got, got - want))
    print()
    if len(references) < 2:
        print('NOTE: one reference cannot separate pitch from height. Add a second at a '
              'clearly different distance before trusting this.')
    print('worst residual %.2f cm' % worst)
    # What the new geometry buys, in the terms that actually bit us.
    near = ground(320, 479, pitch, height, K, D)
    if near:
        print('nearest visible floor straight ahead: %.1f cm' % near[1])

    if args.apply:
        write(path, profile, pitch, height, '%d reference(s)' % len(references), worst)
    else:
        print('(dry run: pass --apply to write it)')


def write(path, profile, pitch, height, how, worst):
        shutil.copyfile(path, path + '.bak')
        profile['pitch_degrees'] = pitch
        profile['camera_height_cm'] = height
        profile['mounting_verified'] = True
        profile['notes'] = ('Pitch re-solved after the camera mount was moved, from %s; '
                            'worst residual %.2f cm. Previous file kept as '
                            'floor_geometry.json.bak.' % (how, worst))
        with open(path + '.tmp', 'w') as handle:
            json.dump(profile, handle, indent=4)
        os.replace(path + '.tmp', path)
        print('written to %s (previous kept as .bak)' % path)


if __name__ == '__main__':
    main()
