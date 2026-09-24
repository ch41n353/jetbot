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


class PowerLogPathTests(unittest.TestCase):
    """The power log is the one artifact here that grows with runtime.

    The battery worker appends a record five times a second for the life of the
    service. It used to default under --directory, which is /tmp -- on this
    robot /tmp is on the root filesystem, not a tmpfs, so that was a continuous
    write stream on the SD card. A card was already destroyed that way. The
    default belongs on the USB volume with every other run artifact, and a
    missing volume has to fail loudly: a silent fallback onto the card is how
    the last one filled up unnoticed.
    """

    def test_the_default_is_the_usb_run_root_not_the_runtime_directory(self):
        import service
        self.assertTrue(service.DEFAULT_POWER_LOG.endswith(
            os.path.join('local_nav','goals','power.jsonl')),service.DEFAULT_POWER_LOG)
        self.assertNotIn('/tmp/',service.DEFAULT_POWER_LOG)

    def test_an_unmounted_log_volume_is_refused_not_redirected(self):
        import service
        from unittest.mock import patch
        with patch('service.os.path.ismount',return_value=False):
            with self.assertRaises(RuntimeError) as caught:
                service.power_log_path(None)
        self.assertIn('not mounted',str(caught.exception))

    def test_a_path_on_the_root_filesystem_is_refused(self):
        import service
        from unittest.mock import patch
        # Mounted volume, but the log itself would land on the card anyway --
        # an unmounted goals symlink resolves back here, and so does a stray
        # --power-log. '/' is on the root device by definition.
        with patch('service.os.path.ismount',return_value=True):
            with self.assertRaises(RuntimeError) as caught:
                service.power_log_path('/power.jsonl')
        self.assertIn('root filesystem',str(caught.exception))

    def test_the_old_tmp_default_is_refused(self):
        import service
        from unittest.mock import patch
        old='/tmp/jetbot-local-nav/power.jsonl'
        if os.stat('/tmp').st_dev!=os.stat('/').st_dev:
            self.skipTest('/tmp is not on the root device on this host')
        with patch('service.os.path.ismount',return_value=True):
            with self.assertRaises(RuntimeError):
                service.power_log_path(old)

    def test_a_directory_off_the_root_device_is_accepted_and_created(self):
        import shutil
        import service
        from unittest.mock import patch
        if os.stat('/dev/shm').st_dev==os.stat('/').st_dev:
            self.skipTest('/dev/shm is on the root device on this host')
        room='/dev/shm/jetbot-power-log-test'
        shutil.rmtree(room,ignore_errors=True)
        try:
            with patch('service.os.path.ismount',return_value=True):
                resolved=service.power_log_path(os.path.join(room,'power.jsonl'))
            self.assertEqual(resolved,os.path.join(room,'power.jsonl'))
            self.assertTrue(os.path.isdir(room))
        finally:
            shutil.rmtree(room,ignore_errors=True)


class PowerLogBufferingTests(unittest.TestCase):
    """The power log must not cost a write syscall per sample.

    CLAUDE.md names this pattern directly: do not open streaming logs with
    buffering=1 at sensor rates, and battery.py samples at ~5 Hz. But a plain
    block buffer is not the answer either -- 64 KB holds roughly fifteen
    seconds of records, and the seconds immediately before a brownout or a hard
    power cut are the part of a power log anyone reads back. Hence a timed
    flush on top of the block buffer.
    """

    def test_the_stream_is_block_buffered_not_line_buffered(self):
        import battery
        self.assertGreater(battery.LOG_BUFFER_BYTES, 1,
                           'buffering=1 is a write syscall per sample')

    def test_the_flush_interval_bounds_what_a_power_cut_loses(self):
        import battery
        self.assertGreater(battery.LOG_FLUSH_SECONDS, 0.)
        self.assertLessEqual(battery.LOG_FLUSH_SECONDS, 5.,
                             'a power log that loses more than a few seconds '
                             'cannot explain the brownout that truncated it')

    def test_the_timer_flushes_sooner_than_the_block_would(self):
        # The point of the timer is to beat the buffer, so it has to.
        import battery
        seconds_per_sample = .2          # ~5 Hz
        bytes_per_sample = 850.          # measured record size
        seconds_to_fill = (battery.LOG_BUFFER_BYTES / bytes_per_sample
                           * seconds_per_sample)
        self.assertLess(battery.LOG_FLUSH_SECONDS, seconds_to_fill)
