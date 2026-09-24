import sys,glob,json,math,time,contextlib
from unittest.mock import patch
sys.path.insert(0,'local_nav')
import cv2,numpy as np
from point_controller import FloorTracker
from state_estimator import AttitudeTimeline,PlanarState
cv2.setNumThreads(1)
profile=json.load(open('calibration/floor_geometry.json'));intr=json.load(open(profile['intrinsics_path']));K=np.array(intr['K']);D=np.array(intr['D']);size=(640,480)
mapx,mapy=cv2.fisheye.initUndistortRectifyMap(K,D,np.eye(3),K,size,cv2.CV_32FC1)
paths=sorted(glob.glob('local_nav/goals/session-20260914-203315-d1f494-route-02.json.action-02.json.observations/*.json'))
tl=AttitudeTimeline(json.load(open('calibration/imu_mount.json')));Q=cv2.Rodrigues(np.array([math.radians(-4),0.,0.]))[0];frames=[]
for index,path in enumerate(paths):
 if index==0:continue
 obs=json.load(open(path))
 if tl.last is None:
  merged={s['time']:s for f in paths[:2] for s in json.load(open(f))['imu_samples']}
  tl.initialize_stationary([merged[k] for k in sorted(merged)],obs['time'])
 else:tl.feed(obs['imu_samples'])
 a=tl.at(obs['time']);a=dict(a,down_camera=Q@a['down_camera']);frames.append((cv2.imread(path[:-5]+'.jpg'),obs['time'],a))
def unproject(points,unusedK,unusedD):return cv2.undistortPoints(points,K,np.zeros(5))
def project(points,rvec,tvec,unusedK,unusedD):return cv2.projectPoints(points,rvec,tvec,K,np.zeros(5))
results=[]
for mode in ['raw','rectified']:
 runs=[]
 for repeat in range(3):
  floor=FloorTracker(profile,intr,max_features=125);state=PlanarState();previous=None;times=[];remap_ms=[];samples=[];reason=None
  with contextlib.ExitStack() as stack:
   if mode=='rectified':
    stack.enter_context(patch('cv2.fisheye.undistortPoints',unproject));stack.enter_context(patch('cv2.fisheye.projectPoints',project))
   for raw,t,a in frames:
    began=time.monotonic();im=cv2.remap(raw,mapx,mapy,cv2.INTER_LINEAR) if mode=='rectified' else raw;remap_ms.append((time.monotonic()-began)*1000)
    if previous is not None:
     start=time.monotonic()
     try:
      r,translation,q=floor.motion(previous[0],im,previous[2],a);pose=state.update(r,translation,t-previous[1],q)
      samples.append(dict(time=t,pose=pose,quality=q))
     except RuntimeError as e:reason=str(e);break
     times.append((time.monotonic()-start)*1000+remap_ms[-1])
    previous=(im,t,a)
  runs.append(dict(samples=samples,reason=reason,processing_ms=times,remap_ms=remap_ms))
 elapsed=[v for r in runs for v in r['processing_ms']];remap=[v for r in runs for v in r['remap_ms']]
 results.append(dict(mode=mode,accepted_pairs=[len(r['samples']) for r in runs],reasons=[r['reason'] for r in runs],median_processing_ms=float(np.median(elapsed)),p95_processing_ms=float(np.percentile(elapsed,95)),median_remap_ms=float(np.median(remap)),last_pose=runs[-1]['samples'][-1]['pose'],samples=runs[-1]['samples']))
report=dict(recording=paths[0].rsplit('/',1)[0],floor_alignment_degrees=-4,feature_budget=125,rectified_model='pinhole with original K, same 640x480 raster; cached remap',results=results,limits='Recorded replay, no hardware movement. Same image raster ROI covers different rays after rectification. No independent metric ground truth; timing includes remap+floor tracking, excludes JPEG decode and capture.')
json.dump(report,open('reports/iteration-2026-09-14/rectification-floor-ab.json','w'),indent=2)
print(json.dumps([{k:v for k,v in r.items() if k!='samples'} for r in results]))
