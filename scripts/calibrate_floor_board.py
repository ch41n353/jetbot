#!/usr/bin/env python3
"""Solve camera height, pitch AND roll from a checkerboard lying on the floor.

The previous method solved one unknown from one measurement, holding height at
a tape reading and roll at a value the profile itself described as "Assumed
zero; not independently measured". A single equation with a single unknown is
exact by construction: it cannot disagree with itself, so it cannot tell you it
is wrong.

A planar board fixes that. Two dozen corners at known spacing over a spread of
ranges give the full camera pose relative to the floor -- height, pitch and
roll together -- and leave enough redundancy that the reprojection residual
means something. A residual of a few tenths of a pixel says the model fits; a
residual of several pixels says it does not, which is the sentence the old
method had no way to say.

What this still cannot check is the fisheye intrinsics it stands on. An error
there is partly absorbed by the pose and can leave the residual looking
healthy. Only driving a measured distance and comparing tests that.

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


def grab(path):
    """Newest camera frame, straight from the sensor service."""
    import fetch
    answer = fetch.call('snapshot')
    image = cv2.imread(answer['path'])
    if image is None:
        raise SystemExit('could not read the camera frame')
    if path:
        cv2.imwrite(path, image)
    return image


def detect(image, corners_x, corners_y):
    """Board corners, sub-pixel, or None.

    findChessboardCornersSB is tried first. A board lying on the floor is seen
    at a shallow angle, so its squares are heavily sheared, and the classic
    detector gives up on exactly the views this calibration needs -- on
    2026-09-25 it found nothing at any pattern size from 3 to 9 in a frame
    where SB found all 24 corners immediately. SB also returns sub-pixel
    positions directly, so it does not need cornerSubPix refining it.
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2, 'findChessboardCornersSB'):
        try:
            found, corners = cv2.findChessboardCornersSB(
                grey, (corners_x, corners_y),
                cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
            if found:
                return corners
        except cv2.error:
            pass
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE |
             cv2.CALIB_CB_FAST_CHECK)
    found, corners = cv2.findChessboardCorners(grey, (corners_x, corners_y), flags)
    if not found:
        found, corners = cv2.findChessboardCorners(
            grey, (corners_x, corners_y),
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        return None
    return cv2.cornerSubPix(
        grey, corners, (7, 7), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, .001))


def pose(corners, corners_x, corners_y, square_mm, K, D):
    """Camera pose relative to the board plane, plus the reprojection residual."""
    board = np.zeros((corners_y * corners_x, 3), np.float64)
    board[:, :2] = np.mgrid[0:corners_x, 0:corners_y].T.reshape(-1, 2)
    board *= square_mm / 10.                      # centimetres, to match the profile

    # Undistort into normalised rays so the pose solve sees an ideal pinhole;
    # the fisheye model is applied once, here, rather than approximated inside
    # solvePnP, which has no fisheye variant.
    rays = cv2.fisheye.undistortPoints(corners.reshape(-1, 1, 2), K, D)
    identity = np.eye(3)
    zero = np.zeros(5)
    ok, rvec, tvec = cv2.solvePnP(board, rays.reshape(-1, 1, 2), identity, zero,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise SystemExit('pose solve failed')

    projected, _ = cv2.projectPoints(board, rvec, tvec, identity, zero)
    residual_rays = np.linalg.norm(projected.reshape(-1, 2) - rays.reshape(-1, 2), axis=1)
    # Convert the normalised residual back to pixels, so it is comparable with
    # the sub-pixel accuracy the corner detector claims.
    focal = float((K[0, 0] + K[1, 1]) / 2.)
    return rvec, tvec, residual_rays * focal


def geometry(rvec, tvec):
    """Height, pitch and roll in the profile's convention.

    The profile writes the floor normal in camera coordinates as
    down = [0, cos(pitch), sin(pitch)], so pitch is how far that down-vector
    leans forward and roll is how far it leans sideways. Recovering both from
    the board normal keeps this consistent with ground(), rather than inventing
    a second convention that happens to agree at zero roll.
    """
    R, _ = cv2.Rodrigues(rvec)
    normal = R.dot(np.array([0., 0., 1.]))
    origin = tvec.reshape(3)
    if normal.dot(origin) < 0:                    # point it away from the camera
        normal = -normal
    normal = normal / np.linalg.norm(normal)
    height = float(abs(normal.dot(origin)))
    pitch = math.degrees(math.atan2(normal[2], normal[1]))
    roll = math.degrees(math.atan2(normal[0], math.hypot(normal[1], normal[2])))
    return height, pitch, roll, normal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default=None, help='use a saved frame instead of the camera')
    parser.add_argument('--save', default=None, help='where to write the captured frame')
    parser.add_argument('--corners-x', type=int, default=6)
    parser.add_argument('--corners-y', type=int, default=4)
    parser.add_argument('--square-mm', type=float, default=25.)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()

    import point_controller as pc
    path = os.path.join(pc.ROOT, 'calibration/floor_geometry.json')
    with open(path) as handle:
        profile = json.load(handle)
    with open(profile['intrinsics_path']) as handle:
        intrinsics = json.load(handle)
    K = np.asarray(intrinsics['K'], float)
    D = np.asarray(intrinsics['D'], float)

    image = cv2.imread(args.image) if args.image else grab(args.save)
    if image is None:
        raise SystemExit('could not read ' + str(args.image))

    corners = detect(image, args.corners_x, args.corners_y)
    if corners is None:
        raise SystemExit('no %dx%d board found. Check it is fully in frame, flat, '
                         'unshadowed, and that the interior-corner counts match.'
                         % (args.corners_x, args.corners_y))
    print('found %d corners' % len(corners))

    rvec, tvec, residual = pose(corners, args.corners_x, args.corners_y,
                                args.square_mm, K, D)
    height, pitch, roll, normal = geometry(rvec, tvec)

    print()
    print('%-22s %10s %10s' % ('', 'stored', 'solved'))
    print('%-22s %10.3f %10.3f' % ('camera height cm', profile['camera_height_cm'], height))
    print('%-22s %10.3f %10.3f' % ('pitch degrees', profile['pitch_degrees'], pitch))
    print('%-22s %10.3f %10.3f' % ('roll degrees', profile.get('roll_degrees', 0.), roll))
    print()
    print('floor normal in camera coords  [%+.4f %+.4f %+.4f]' % tuple(normal))
    print('reprojection residual  median %.3f px   worst %.3f px   (%d corners)'
          % (np.median(residual), residual.max(), len(residual)))
    if np.median(residual) > 1.:
        print('  WARNING: a residual above a pixel means the model does not fit. '
              'Suspect the board scale, the interior-corner counts, or the intrinsics.')

    board_range = float(np.linalg.norm(tvec))
    print('board origin corner is %.1f cm from the lens' % board_range)

    if args.apply:
        shutil.copyfile(path, path + '.bak')
        profile['camera_height_cm'] = height
        profile['pitch_degrees'] = pitch
        profile['roll_degrees'] = roll
        profile['height_source'] = ('Solved with pitch and roll from a %d x %d checkerboard '
                                    'of %.1f mm squares' % (args.corners_x, args.corners_y,
                                                            args.square_mm))
        profile['roll_source'] = 'Measured from the same board, not assumed'
        profile['mounting_verified'] = True
        # Any recorded validation described the geometry being replaced, and a
        # stale pass is worse than none: this file claimed "Passed independent
        # near-center 50 cm check, predicted 48.6 cm" while the geometry then
        # in it computed 17.2 cm for that very pixel, because an earlier
        # --apply overwrote the pose and left the claim behind.
        stale = profile.pop('validation', None)
        profile.pop('verification_status', None)
        if stale:
            profile['validation_note'] = (
                'Cleared on re-solve: the previous check was made against the '
                'superseded pose. Re-validate against measured floor points '
                'before trusting this profile.')
        profile['notes'] = ('Height, pitch and roll solved together from a planar board; '
                            'median reprojection residual %.3f px over %d corners. Previous '
                            'file kept as floor_geometry.json.bak.'
                            % (np.median(residual), len(residual)))
        with open(path + '.tmp', 'w') as handle:
            json.dump(profile, handle, indent=4)
        os.replace(path + '.tmp', path)
        print('written to %s (previous kept as .bak)' % path)
    else:
        print('\n(dry run: pass --apply to write it)')


if __name__ == '__main__':
    main()
