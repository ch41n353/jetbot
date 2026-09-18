import os
import sys
import time
import unittest
import multiprocessing as mp
import queue
from unittest.mock import patch
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','local_nav'))
from battery import PowerGuard,PowerSensors,power_permitted,read_shared
import battery
import shutil
import tempfile


class BatteryTests(unittest.TestCase):
    def test_combined_sample_preserves_oldest_sensor_age(self):
        sensors=PowerSensors.__new__(PowerSensors)
        from unittest.mock import Mock
        sensors.supply=Mock()
        sensors.pack=Mock()
        sensors.supply.sample.return_value=dict(time=1.,voltage_v=5.04)
        sensors.pack.sample.return_value=dict(pack_time=1.7,pack_voltage_v=12.2)
        sample=sensors.sample()
        self.assertEqual(sample['input_time'],1.)
        self.assertEqual(sample['pack_time'],1.7)
        self.assertEqual(sample['time'],1.)
        self.assertFalse(PowerGuard(require_pack=True).update(sample,1.7)['motion_allowed'])

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

    def test_low_pack_stops_even_while_regulated_supply_is_healthy(self):
        guard=PowerGuard(require_pack=True)
        for i in range(3):
            state=guard.update(dict(time=i,voltage_v=5.04,pack_voltage_v=12.2),i)
        self.assertTrue(state['motion_allowed'])
        low=guard.update(dict(time=3,voltage_v=5.04,pack_voltage_v=10.7),3)
        self.assertEqual(low['stop_reason'],'battery_pack_low')
        self.assertFalse(low['motion_allowed'])
        self.assertFalse(guard.update(dict(time=4,voltage_v=5.04,pack_voltage_v=12.2),4)['motion_allowed'])
        missing=PowerGuard(require_pack=True).update(dict(time=1,voltage_v=5.04),1)
        self.assertEqual(missing['stop_reason'],'pack_telemetry_unavailable')

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


class PowerLogRotationTests(unittest.TestCase):
    """The power log is written by the process that guards motion.

    It appends five records a second, every field of every sample, which came
    to 287 MB before anyone looked -- roughly 345 MB a day. Rotation bounds
    that. What matters as much as bounding it: rotation is housekeeping and the
    power monitor is not, so a failure to roll must never take the worker down.
    Losing that process loses the guard and latches motion off.
    """

    def setUp(self):
        self.room = tempfile.mkdtemp()
        self.path = os.path.join(self.room, 'power.jsonl')

    def tearDown(self):
        shutil.rmtree(self.room, ignore_errors=True)

    def test_rolling_keeps_a_bounded_window_newest_first(self):
        for cycle in range(6):
            with open(self.path, 'w') as handle:
                handle.write('cycle %d' % cycle)
            self.assertTrue(battery._roll(self.path))
        kept = sorted(os.listdir(self.room))
        self.assertEqual(kept, ['power.jsonl.%d' % n
                                for n in range(1, battery.LOG_KEEP + 1)])
        with open(self.path + '.1') as handle:
            self.assertEqual(handle.read(), 'cycle 5')
        with open(self.path + '.%d' % battery.LOG_KEEP) as handle:
            self.assertEqual(handle.read(), 'cycle %d' % (6 - battery.LOG_KEEP))

    def test_a_roll_that_fails_is_reported_not_raised(self):
        # The worker keeps writing to the handle it has; it must not die. A
        # missing path is not a failure -- there is simply nothing to move --
        # so the failure has to be provoked at the rename itself.
        with open(self.path, 'w') as handle:
            handle.write('live')
        def refuse(*args):
            raise OSError('read-only file system')
        with patch('battery.os.rename', refuse):
            self.assertIs(battery._roll(self.path), False)
        # and the log it was writing is still there to append to
        self.assertTrue(os.path.exists(self.path))

    def test_a_missing_log_is_nothing_to_roll(self):
        self.assertTrue(battery._roll(self.path))

    def test_the_window_is_bounded_and_small(self):
        # Four files of 16 MB is about 64 MB, against 345 MB a day unbounded.
        self.assertLessEqual(battery.LOG_MAX_BYTES * (battery.LOG_KEEP + 1),
                             128 * 1024 * 1024)


if __name__=='__main__':unittest.main()
