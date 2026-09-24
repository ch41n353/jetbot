"""Motor-free tests through actual local execution HTTP entry point."""
import sys,os,json,threading,tempfile,time,unittest,urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','local_nav'))
import trajectory_executor as ex
import planner_demo

class Robot:
    def __init__(self):self.yaw=0;self.outputs=[];self.turns=[];self.on=False;self.resyncs=0
    def check(self):pass
    # launch() adopts the service's control generation before every new
    # instruction, so that one transient rejection cannot wedge the planner.
    def resync(self):self.resyncs+=1;return 1
    def hold(self,l,r):self.outputs.append([l,r]);self.on=bool(l or r)
    def frame(self):return np.zeros((480,640,3),np.uint8),{'imu_samples':[{'yaw':self.yaw}]}
    def heading(self,s):return s['yaw']
    def turn(self,a):self.turns.append(a);self.yaw+=a;return a
    def pixel(self,x,z):return (320+x,400-z)

class ExecutorTests(unittest.TestCase):
    def test_drive_uses_fixed_proven_carpet_duty(self):
        self.assertEqual(ex.drive_duty(50), .20)
        self.assertEqual(ex.drive_duty(30), .20)
        self.assertEqual(ex.drive_duty(15), .20)
        self.assertEqual(ex.drive_duty(1), .20)

    def test_near_waypoint_aim_accounts_for_camera_pivot(self):
        target=[2.4321646778,5.7008451510]
        angle=ex.pivot_aim(target)
        pose=list(ex.fetch.pivot_shift(angle))+[angle]
        remaining=ex.fetch.rebase([target],pose)[0]
        self.assertAlmostEqual(remaining[0],0.,places=6)
        self.assertGreater(remaining[1],0.)
        self.assertLess(abs(angle),12)
    def test_validation(self):
        for body in [{'trajectory':{'waypoints_cm':[[float('nan'),2]]}}, {'obstacles':[]}, {'trajectory':{'initial_turn_deg':200}}]:
            with self.assertRaises(ValueError):ex.validate(body)
    def test_frame_transform(self):
        p=ex.compose([2,3,90],[0,10,20]);self.assertAlmostEqual(p[0],12);self.assertAlmostEqual(p[1],3);self.assertEqual(p[2],110)
    def execute(self,fail=False,pause=False,behind=False):
        robot=Robot();b=planner_demo.Bench.__new__(planner_demo.Bench)
        b.robot=robot;b.frame=np.zeros((480,640,3),np.uint8);b.frame_token=7;b.abort=threading.Event();b.flight=None;b.lock=threading.Lock();b.lines=[]
        b.odometer=SimpleNamespace(tracker=SimpleNamespace(motion=lambda a,c:(np.eye(2),np.array([0.,-1.]),{})))
        b.state=lambda **kw:kw
        if fail:b.odometer.tracker.motion=lambda a,c:(_ for _ in ()).throw(RuntimeError('lost carpet'))
        server=planner_demo.Server(('127.0.0.1',0),planner_demo.Handler);server.bench=b
        threading.Thread(target=server.serve_forever,daemon=True).start()
        angle=20;shift=ex.fetch.pivot_shift(angle);end=ex.compose([shift[0],shift[1],angle],[0,8,0])
        body={'frame_token':7,'trajectory':{'initial_turn_deg':angle,'waypoints_cm':[end[:2]]}}
        if behind:body={'frame_token':7,'trajectory':{'waypoints_cm':[[0,-30]]},'stop_after_cm':2};pause=True
        if pause:body['stop_after_cm']=2
        try:
            with tempfile.TemporaryDirectory() as root,patch.object(ex,'artifact_root',return_value=Path(root)),patch.dict(os.environ,JETBOT_GPT_AUDIT=''):
                req=urllib.request.Request('http://127.0.0.1:%d/api/local-execution'%server.server_port,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
                answer=json.load(urllib.request.urlopen(req));b.flight.join(8)
                self.assertFalse(b.flight.is_alive());result=b.local_execution.result()
                self.assertEqual(result['phase'],'failed' if fail else ('paused' if pause else 'completed'))
                self.assertEqual(robot.outputs[-1],[0,0])
                if behind:
                    self.assertLess(sum(robot.turns),-160)
                    self.assertTrue(all(l>=0 and r>=0 for l,r in robot.outputs))
                else:self.assertEqual(robot.turns,[20])
                self.assertEqual(result['completed_waypoints'],0 if fail or pause else 1)
                self.assertGreater(b.planner_snapshot['id'],2)
                self.assertIn('floor',b.planner_snapshot['images'])
                self.assertTrue((Path(result['artifact_path'])/'result.json').exists())
        finally:server.shutdown();server.server_close()
    def test_rearward_waypoint_turns_before_forward_drive(self):self.execute(behind=True)
    def test_requested_pause_keeps_remaining_route(self):self.execute(pause=True)
    def test_http_turn_then_trajectory(self):self.execute()
    def test_lost_vision_stops_and_reports_failure(self):self.execute(True)

    def test_one_rejected_flow_pair_recovers_without_motor_stop(self):
        robot=Robot();b=planner_demo.Bench.__new__(planner_demo.Bench)
        b.robot=robot;b.frame=np.zeros((480,640,3),np.uint8);b.frame_token=7
        b.abort=threading.Event();b.flight=None;b.lock=threading.Lock();b.lines=[]
        calls=[0]
        def motion(a,c):
            calls[0]+=1
            if calls[0]==1:raise RuntimeError('Floor motion is inconsistent (4/19 inliers)')
            return np.eye(2),np.array([0.,-1.]),{}
        b.odometer=SimpleNamespace(tracker=SimpleNamespace(motion=motion));b.state=lambda **kw:kw
        with tempfile.TemporaryDirectory() as root,patch.object(ex,'artifact_root',return_value=Path(root)),patch.dict(os.environ,JETBOT_GPT_AUDIT=''):
            job=ex.Execution(b,{'trajectory':{'waypoints_cm':[[0,20]]},'stop_after_cm':3})
            job.run()
        self.assertEqual(job.phase,'paused')
        self.assertEqual(job.tracking_rejections,1)
        self.assertGreater(job.travelled,0)
        # The rejected frame did not insert a zero-power command; only final
        # cleanup stops the wheels.
        self.assertTrue(all(any(output) for output in robot.outputs[:-1]))
        self.assertEqual(robot.outputs[-1],[0,0])

class ProjectionTests(unittest.TestCase):
    def test_metric_projection_matches_camera_lens_and_masks_rear(self):
        from evaluate_gpt_routes import Lens
        lens=Lens();p=ex.CameraFloorProjection(lens)
        image=np.full((480,640,3),(20,90,170),np.uint8)
        view=p.apply(image)
        self.assertTrue(p.visible[:320].any())
        self.assertFalse(p.visible[320:].any())
        self.assertTrue(np.all(view[400,200]==28))
        # The metric location (0, 50 cm) is 80 pixels above the center.
        pixel=ex.fetch.Robot.pixel(lens,0.,50.)
        self.assertAlmostEqual(p.x[240,320],pixel[0],places=3)
        self.assertAlmostEqual(p.y[240,320],pixel[1],places=3)
        self.assertTrue(np.all(view[240,320]==[20,90,170]))
        changed=p.apply(np.full_like(image,50))
        self.assertTrue(np.all(changed[240,320]==50))

class TraceRenderingTests(unittest.TestCase):
    def test_traversed_history_remains_visible_when_plan_overlaps(self):
        import base64,cv2
        result=ex.render(Robot(),np.zeros((480,640,3),np.uint8),
                         [[60,60]],[0,0,0],[[0,0,0],[30,30,0],[60,60,0]])
        floor=cv2.imdecode(np.frombuffer(base64.b64decode(result['floor'].split(',',1)[1]),np.uint8),1)
        # Latest segment stays orange even where the remaining path overlaps.
        b,g,r=map(int,floor[272,368]);self.assertGreater(r,b+80);self.assertGreater(g,b+40)

class RunHistoryTests(unittest.TestCase):
    def test_second_instruction_keeps_run_history_and_resets_segment(self):
        b=SimpleNamespace(robot=Robot(),frame=np.zeros((480,640,3),np.uint8))
        with tempfile.TemporaryDirectory() as root,patch.object(ex,'artifact_root',return_value=Path(root)):
            a=ex.Execution(b,{'trajectory':{'waypoints_cm':[[0,20]]}})
            a.pose=[0,20,90];a.trace=[[0,0,0],[0,20,90]];a.completed=1;a.publish(True)
            c=ex.Execution(b,{'trajectory':{'waypoints_cm':[[0,10],[0,20]]}})
            self.assertEqual(c.origin,[0,20,90]);self.assertEqual(c.trace,[[0,0,0]])
            c.pose=[0,10,0];c.trace.append(c.pose);c.completed=1
            r=c.result()
            self.assertAlmostEqual(r['run_pose_cm_deg'][0],10)
            self.assertAlmostEqual(r['run_pose_cm_deg'][1],20)
            self.assertEqual(len(r['instruction_history']),2)
            self.assertEqual(r['remaining_trajectory_cm'],[(0.0,10.0)])
            self.assertEqual(len(r['run_estimated_trajectory']),3)
            c.completed=2;self.assertEqual(c.result()['remaining_trajectory_cm'],[])

class CameraPathTests(unittest.TestCase):
    def lens(self):
        import types
        from evaluate_gpt_routes import Lens
        lens=Lens();lens.pixel=types.MethodType(ex.fetch.Robot.pixel,lens);return lens
    def test_near_route_gets_explicit_offscreen_indicator(self):
        im,meta=ex.camera_path(self.lens(),np.zeros((480,640,3),np.uint8),[[0,0],[5.93,3.52]])
        self.assertEqual(meta['offscreen_waypoints'],1)
        self.assertTrue(im[32:].any())
    def test_segment_into_visible_floor_is_not_lost_at_near_endpoint(self):
        im,meta=ex.camera_path(self.lens(),np.zeros((480,640,3),np.uint8),[[0,0],[0,60]])
        self.assertGreater(meta['visible_segments'],0)
        self.assertEqual(meta['offscreen_waypoints'],0)
