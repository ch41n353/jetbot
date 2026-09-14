import os
import sys
import time
import unittest
import multiprocessing as mp
import queue
from unittest.mock import patch
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','local_nav'))
from battery import PowerGuard,power_permitted,read_shared


class BatteryTests(unittest.TestCase):
    def test_startup_and_low_voltage_latch(self):
        guard=PowerGuard()
        for index in range(3):
            status=guard.update(dict(time=index*.2,voltage_v=5.04),index*.2)
            self.assertEqual(status['motion_allowed'],index==2)
        low=guard.update(dict(time=1,voltage_v=4.72),1)
        self.assertFalse(low['motion_allowed'])
        self.assertEqual(low['stop_reason'],'input_undervoltage')
        self.assertFalse(guard.update(dict(time=2,voltage_v=5.04),2)['motion_allowed'])

    def test_warning_does_not_invent_charge_percentage(self):
        guard=PowerGuard()
        for index in range(3):
            status=guard.update(dict(time=index,voltage_v=4.88,battery_percent=None),index)
        self.assertTrue(status['warning'])
        self.assertTrue(status['motion_allowed'])
        self.assertIsNone(status['battery_percent'])

    def test_bad_or_stale_reading_disables_motion(self):
        for sample in ({'time':0,'voltage_v':5.0},{'time':1,'voltage_v':float('nan')},
                       {'time':1,'fault':'I2C unavailable'},{'time':1,'voltage_v':5.6}):
            self.assertFalse(PowerGuard().update(sample,1)['motion_allowed'])
        self.assertFalse(power_permitted([1,5.04,1],1.61))
        self.assertFalse(power_permitted([2,5.04,1],1))

    def test_locked_or_dead_sensor_does_not_block_motor_watchdog(self):
        class Lock:
            def acquire(self,blocking,timeout):return False
        class Shared:
            def get_lock(self):return Lock()
        old=[1.,5.04,1.]
        sample=read_shared(Shared(),old)
        self.assertFalse(power_permitted(sample,1.61))

    def test_independent_motor_watchdog_stops_and_does_not_restart(self):
        import service
        ctx=mp.get_context('fork')
        events=ctx.Queue()
        class Robot:
            def stop(self):events.put((0,0))
            def set_motors(self,l,r):events.put((l,r))
        shared=ctx.Array('d',[time.monotonic(),5.04,1.])
        parent,child=ctx.Pipe()
        with patch('service.robot',return_value=Robot()):
            process=ctx.Process(target=service.motor_worker,args=(child,True,shared))
            process.start()
        child.close()
        try:
            self.assertTrue(parent.poll(2));self.assertTrue(parent.recv()['ready'])
            self.assertEqual(events.get(timeout=1),(0,0))
            def command():
                now=time.monotonic()
                parent.send(dict(left=.16,right=.16,issued=now,expires=now+.2,camera_time=now,imu_time=now))
            command()
            self.assertEqual(events.get(timeout=1),(.16,.16))
            with shared.get_lock():shared[:]=[time.monotonic(),4.7,0.]
            self.assertEqual(events.get(timeout=1),(0,0))
            with shared.get_lock():shared[:]=[time.monotonic(),5.04,1.]
            command()
            with self.assertRaises(queue.Empty):events.get(timeout=.12)
        finally:
            parent.send({'shutdown':True})
            process.join(2)
            if process.is_alive():process.terminate();process.join()
            parent.close()


if __name__=='__main__':unittest.main()
