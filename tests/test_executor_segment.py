import sys,threading,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'local_nav'),str(Path(__file__).parent)]
import trajectory_executor as ex
from test_trajectory_executor import Robot

class SegmentTests(unittest.TestCase):
    def exercise(self,cancel=False):
        robot=Robot();abort=threading.Event();samples=[]
        def motion(a,b):
            samples.append(robot.on)
            if cancel:abort.set()
            return np.eye(2),np.array([0.,-1.]),{}
        bench=SimpleNamespace(robot=robot,frame=np.zeros((480,640,3),np.uint8),abort=abort,say=lambda x:None,odometer=SimpleNamespace(tracker=SimpleNamespace(motion=motion)))
        with tempfile.TemporaryDirectory() as root,patch.object(ex,'artifact_root',return_value=Path(root)):
            job=ex.Execution(bench,dict(trajectory=dict(waypoints_cm=[[0,20]]),stop_after_cm=10))
            job.run()
        self.assertTrue(samples[0], 'vision must sample while motors remain on')
        self.assertEqual(robot.outputs[-1],[0,0])
        if cancel:self.assertEqual(job.phase,'cancelled')
        else:
            self.assertEqual(job.phase,'paused')
            self.assertGreaterEqual(sum(samples),3)
            self.assertTrue(any(any(a) and any(b) for a,b in zip(robot.outputs,robot.outputs[1:])))
    def test_measure_while_moving(self):self.exercise()
    def test_cancel_stops(self):self.exercise(True)
