"""Exercise the real executor with a deliberately blocked visualization worker."""
import sys,threading,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'local_nav'),str(Path(__file__).parent)]
import trajectory_executor as ex
from test_trajectory_executor import Robot

class AsyncPublishing(unittest.TestCase):
    def test_slow_render_does_not_block_stop_and_measure(self):
        robot=Robot();stopped=threading.Event();rendering=threading.Event();release=threading.Event()
        hold=robot.hold
        def observed_hold(l,r):
            was_on=robot.on;hold(l,r)
            if was_on and not robot.on:stopped.set()
        robot.hold=observed_hold
        b=SimpleNamespace(robot=robot,frame=np.zeros((480,640,3),np.uint8),abort=threading.Event(),say=lambda x:None,
                          odometer=SimpleNamespace(tracker=SimpleNamespace(motion=lambda a,c:(np.eye(2),np.array([0.,-1.]),{}))))
        original=ex.Execution._publish
        def slow(job,force=False):
            if job.phase=='driving':rendering.set();release.wait(3)
            return original(job,force)
        with tempfile.TemporaryDirectory() as root,patch.object(ex,'artifact_root',return_value=Path(root)),patch.object(ex.Execution,'_publish',slow):
            job=ex.Execution(b,dict(trajectory=dict(waypoints_cm=[[0,6],[0,10]])))
            t=threading.Thread(target=job.run);t.start()
            try:
                self.assertTrue(rendering.wait(1))
                self.assertTrue(stopped.wait(1),'control waited for rendering')
                nonzero=[i for i,v in enumerate(robot.outputs) if any(v)]
                self.assertTrue(nonzero)
                self.assertIn([0,0],robot.outputs[nonzero[0]+1:])
                self.assertTrue(t.is_alive(),'final render should still be flushing')
            finally:release.set();t.join(5)
            self.assertFalse(t.is_alive());self.assertEqual(job.phase,'completed')
            self.assertEqual(job.completed,2)
            nonzero=[i for i,v in enumerate(robot.outputs) if any(v)]
            self.assertEqual(robot.outputs[-1],[0,0])
            self.assertEqual(b.planner_snapshot['phase'],'completed')
