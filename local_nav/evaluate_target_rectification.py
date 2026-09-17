"""Offline raw/rectified target replay; no service, camera or motor access.

Use the same decoded frames and recorded floor pose in both arms. Rectified
pixels are PINHOLE coordinates, not fisheye coordinates with zero distortion.
"""
import argparse
import glob
import json
import math
import os
import time
import cv2
import numpy as np
from target_tracker import TargetTracker, TargetLost

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(ROOT, 'reports', 'iteration-2026-09-14')
DEFAULT_DIRECTORY = os.path.join(ROOT, 'local_nav', 'goals',
    'session-20260914-203315-d1f494-route-02.json.action-02.json.observations')


def rectify_points(points, K, D):
    return cv2.fisheye.undistortPoints(np.asarray(points, dtype=float).reshape(-1, 1, 2),
                                      K, D, P=K).reshape(-1, 2)


def raw_points(points, K, D):
    xy = np.asarray(points, dtype=float).reshape(-1, 2)
    rays = np.column_stack((xy, np.ones(len(xy)))) @ np.linalg.inv(K).T
    rays = rays[:, :2] / rays[:, 2, None]
    return cv2.fisheye.distortPoints(rays.reshape(-1, 1, 2), K, D).reshape(-1, 2)


def ground(points, down, K, D, rectified, height):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if rectified:
        rays = np.column_stack((points, np.ones(len(points)))) @ np.linalg.inv(K).T
    else:
        xy = cv2.fisheye.undistortPoints(points.reshape(-1, 1, 2), K, D).reshape(-1, 2)
        rays = np.column_stack((xy, np.ones(len(xy))))
    down = np.asarray(down, dtype=float)
    down = down / np.linalg.norm(down)
    forward = np.array([0., 0., 1.]) - down * down[2]
    forward /= np.linalg.norm(forward)
    right = np.cross(down, forward)
    denominator = rays @ down
    if np.any(denominator < .05):
        raise ValueError('Floor ray outside usable geometry')
    return np.column_stack((rays @ right, rays @ forward)) * height / denominator[:, None]


def summary(values):
    return dict(median=float(np.median(values)), p95=float(np.percentile(values, 95)),
                maximum=float(max(values)))


def run(directory=DEFAULT_DIRECTORY, repeats=3):
    cv2.setNumThreads(1)
    profile = json.load(open(os.path.join(ROOT, 'calibration', 'floor_geometry.json')))
    intrinsics = json.load(open(profile['intrinsics_path']))
    K, D = np.asarray(intrinsics['K'], dtype=float), np.asarray(intrinsics['D'], dtype=float)
    # The earliest frame has insufficient startup history; use the same next
    # anchor as the prior validated 44-frame target/plane replay.
    paths = sorted(glob.glob(os.path.join(directory, '*.jpg')))[1:]
    images = [cv2.imread(path) for path in paths]
    if len(images) != 45 or any(image is None for image in images):
        raise ValueError('Expected the exact 45-image reference sequence')
    existing = json.load(open(os.path.join(REPORTS, 'contact-plane-replay.json')))
    reference = next(case for case in existing['replay'] if case['name'] == 'plane_only')
    poses = {round(sample['time'], 6): sample for sample in reference['samples']}
    seed_box = np.array([289., 166., 308., 199.])
    seed_contact = np.array([298.5, 199.])
    x, y, r, b = seed_box
    corners = rectify_points([[x,y],[r,y],[r,b],[x,b]], K, D)
    rect_box = np.r_[corners.min(axis=0), corners.max(axis=0)]
    rect_contact = rectify_points([seed_contact], K, D)[0]
    started = time.perf_counter()
    mapx, mapy = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K,
                            (images[0].shape[1], images[0].shape[0]), cv2.CV_32FC1)
    map_ms = (time.perf_counter()-started)*1000
    report = dict(repeats=repeats, frames_per_repeat=44, Knew=K.tolist(),
                  seed_raw_box=seed_box.tolist(), seed_raw_contact=seed_contact.tolist(),
                  seed_rectified_box=rect_box.tolist(), seed_rectified_contact=rect_contact.tolist(),
                  map_precompute_ms=map_ms, cases=[], manual_contact_comparisons=[])
    for rectified in (False, True):
        all_samples=[]
        for repeat in range(repeats):
            anchor = cv2.remap(images[0], mapx, mapy, cv2.INTER_LINEAR) if rectified else images[0]
            tracker = TargetTracker(anchor, rect_box if rectified else seed_box, projective_contact=True)
            # A transformed box's bottom-center is not the transformed contact.
            tracker.contact = (rect_contact if rectified else seed_contact).copy()
            u,v,rr,bb = np.round(tracker.box).astype(int)
            tracker.recent_contact = (tracker.contact - [u,v]) / np.array([rr-u,bb-v])
            samples=[]
            for path, image in zip(paths[1:], images[1:]):
                start=time.perf_counter()
                current=cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR) if rectified else image
                remap_ms=(time.perf_counter()-start)*1000
                try:
                    observation=tracker.update(current)
                except TargetLost as exc:
                    samples.append(dict(path=path,state='lost',reason=str(exc),
                        total_ms=(time.perf_counter()-start)*1000,remap_ms=remap_ms))
                    break
                total_ms=(time.perf_counter()-start)*1000
                stamp=round(float(os.path.basename(path)[:-4]),6)
                pose=poses[stamp]
                local=ground([observation['base_pixel']],pose['down_camera'],K,D,rectified,
                             profile['camera_height_cm'])[0]
                previous_local=np.asarray(pose['local_target_cm'])
                vector=np.asarray(pose['target_world_cm'])-pose['position_cm']
                yaw=math.atan2(vector[0],vector[1])-math.atan2(previous_local[0],previous_local[1])
                c,s=math.cos(yaw),math.sin(yaw)
                world=np.asarray(pose['position_cm'])+np.array([[c,s],[-s,c]])@local
                error=float(np.linalg.norm(world-reference['initial_target_world_cm']))
                equivalent_raw=raw_points([observation['base_pixel']],K,D)[0] if rectified else np.array(observation['base_pixel'])
                samples.append(dict(path=path,state='tracked',total_ms=total_ms,remap_ms=remap_ms,
                    tracker_ms=total_ms-remap_ms,box=observation['box'],contact=observation['base_pixel'],
                    equivalent_raw_contact=equivalent_raw.tolist(),confidence=observation['confidence'],
                    local_target_cm=local.tolist(),world_error_cm=error,
                    guard_limit_cm=max(5.,.1*float(np.linalg.norm(local)))))
            all_samples.append(samples)
        tracked=[s for repeat in all_samples for s in repeat if s['state']=='tracked']
        report['cases'].append(dict(name='rectified' if rectified else 'raw',
            tracked_per_repeat=[sum(s['state']=='tracked' for s in repeat) for repeat in all_samples],
            failures=[s for repeat in all_samples for s in repeat if s['state']=='lost'],
            total_ms=summary([s['total_ms'] for s in tracked]),
            tracker_ms=summary([s['tracker_ms'] for s in tracked]),
            remap_ms=summary([s['remap_ms'] for s in tracked]),
            maximum_world_error_cm=max(s['world_error_cm'] for s in tracked),
            guard_violations_per_repeat=[sum(s['state']=='tracked' and s['world_error_cm']>s['guard_limit_cm'] for s in repeat) for repeat in all_samples],
            samples=all_samples[0]))
    manual=[('21172.254796',[306.,207.5]),('21173.318939',[313.,215.5]),('21174.153190',[317.,209.5])]
    for stamp,point in manual:
        entry=dict(frame=stamp,manual_raw_contact=point,manual_rectified_contact=rectify_points([point],K,D)[0].tolist(),
                   uncertainty_raw_pixels=dict(x=2.,y=1.))
        for case in report['cases']:
            match=next((s for s in case['samples'] if os.path.basename(s['path'])[:-4]==stamp and s['state']=='tracked'),None)
            entry[case['name']]=None if match is None else dict(raw_contact=match['equivalent_raw_contact'],
                raw_y_error_pixels=match['equivalent_raw_contact'][1]-point[1])
        report['manual_contact_comparisons'].append(entry)
    # Mathematical check only, independent of any measured target outcome.
    down=next(iter(poses.values()))['down_camera']
    mapped=rectify_points([seed_contact],K,D)
    report['equivalent_seed_ground_error_cm']=float(np.linalg.norm(
        ground([seed_contact],down,K,D,False,profile['camera_height_cm'])-
        ground(mapped,down,K,D,True,profile['camera_height_cm'])))
    report['limits']='Offline recorded Advil sequence only. Shared recorded floor poses isolate target tracking; this does not measure a complete rectified navigation pipeline. Timings exclude JPEG decoding and report map setup separately. Manual pixel contacts are visual estimates, not metric ground truth.'
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',default=DEFAULT_DIRECTORY)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if not 1<=args.repeats<=5:parser.error('repeats must be1–5')
    result=run(args.directory,args.repeats)
    with open(args.output,'w') as out:json.dump(result,out,indent=2)
    print(json.dumps([{k:v for k,v in case.items() if k!='samples'} for case in result['cases']]))
