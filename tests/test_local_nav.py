import os,sys,unittest
sys.path.insert(0,os.path.join(os.path.dirname(os.path.dirname(__file__)),'local_nav'))
from core import allowed

class LeaseTests(unittest.TestCase):
    def command(self):
        return dict(left=.2,right=.2,issued=10,expires=10.2,camera_time=9.9,imu_time=9.99)
    def test_valid(self): self.assertTrue(allowed(self.command(),10.05,True))
    def test_disabled(self): self.assertFalse(allowed(self.command(),10.05,False))
    def test_expiry(self): self.assertFalse(allowed(self.command(),10.21,True))
    def test_bad_fields(self):
        for field,val in [('left',float('nan')),('right',.4),('camera_time',9),('imu_time',9),('issued',11),('expires',20)]:
            c=self.command();c[field]=val
            self.assertFalse(allowed(c,10.05,True),field)
    def test_stop(self): self.assertFalse(allowed({},10,True))

if __name__=='__main__':unittest.main()

class WatchdogProcessTests(unittest.TestCase):
    def test_parent_silence_stops_output(self):
        import multiprocessing as mp
        import time
        import service
        ctx=mp.get_context('fork')
        events=ctx.Queue()
        class FakeRobot:
            def stop(self): events.put(('stop',time.monotonic()))
            def set_motors(self,l,r): events.put(('move',time.monotonic()))
        original=service.robot
        service.robot=lambda:FakeRobot()
        parent,child=ctx.Pipe()
        power_shared=ctx.Array('d',[time.monotonic(),5.04,1.])
        process=ctx.Process(target=service.motor_worker,args=(child,True,power_shared))
        try:
            process.start()
            child.close()
            self.assertTrue(parent.poll(2))
            self.assertTrue(parent.recv()['ready'])
            self.assertEqual(events.get(timeout=1)[0],'stop')
            now=time.monotonic()
            parent.send(dict(left=.2,right=.2,issued=now,expires=now+.15,camera_time=now,imu_time=now))
            self.assertEqual(events.get(timeout=1)[0],'move')
            # No further commands: watchdog must stop independently.
            event,stamp=events.get(timeout=1)
            self.assertEqual(event,'stop')
            self.assertLess(stamp-now,.35)
            parent.send({'shutdown':True})
            process.join(2)
            self.assertEqual(process.exitcode,0)
        finally:
            service.robot=original
            if process.is_alive():process.terminate();process.join()
            parent.close()
