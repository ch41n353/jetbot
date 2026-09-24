"""Continuous turn command stream, no physical motors."""
import sys,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'local_nav'))
import fetch
class ContinuousTurn(unittest.TestCase):
    def test_no_zero_between_feedback_updates(self):
        r=fetch.Robot.__new__(fetch.Robot);r.dry_run=False;r.coast_s=.1;r.duty=.12;r.turn_gain=600
        values=iter([(0,0),(0,0),(10,60),(20,50),(27,25),(29,0),(30,0),(30,0)])
        r.check=lambda:{'imu':next(values,(30,0))};r.heading=lambda i:i[0];r.yaw_rate=lambda i:i[1]
        r.floor_duty=lambda x:max(.1,x);r.note_motion=lambda *a:None
        outputs=[];r.hold=lambda l,rr:outputs.append([l,rr])
        with patch.object(fetch.time,'sleep',lambda t:None):measured=r.turn_continuous(30)
        self.assertEqual(measured,30);self.assertEqual(outputs[-1],[0,0])
        self.assertGreater(len(outputs),2)
        self.assertTrue(all(l>0 and rr<0 for l,rr in outputs[:-1]))
    def test_fault_always_stops(self):
        r=fetch.Robot.__new__(fetch.Robot);r.dry_run=False
        values=iter([{'imu':0},fetch.Stop('sensor failure')]);outputs=[]
        def check():
            value=next(values)
            if isinstance(value,Exception):raise value
            return value
        r.check=check;r.heading=lambda i:i;r.hold=lambda l,rr:outputs.append([l,rr])
        with self.assertRaises(fetch.Stop):r.turn_continuous(30)
        self.assertEqual(outputs[-1],[0,0])
