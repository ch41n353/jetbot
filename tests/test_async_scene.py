import os
import sys
import threading
import time
import unittest
import numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','local_nav'))
from async_scene import AsyncScene,assess


class AsyncSceneTests(unittest.TestCase):
    def setUp(self):
        self.snapshot=dict(time=10.,pose=[0,0,0],attitude={'capture':True},generation=['s',1])
        self.response=dict(answer=dict(target_visible=True,
            target_box=dict(x0=300,y0=100,x1=340,y1=200),contact_pixel=dict(x=320,y=200)))

    def ground(self,pixels,attitude):
        self.assertTrue(attitude['capture'])
        return [[0,100]]

    def test_four_second_delay_uses_capture_pose_and_current_measured_motion(self):
        result=assess(self.snapshot,self.response,14.,[0,40,0],['s',1],self.ground)
        self.assertEqual(result['target_world_cm'],[0,100])
        self.assertEqual(result['target_current_cm'],[0,60])
        self.assertFalse(result['applied_to_control'])

    def test_rotation_rebase(self):
        self.snapshot['pose']=[20,30,90]
        result=assess(self.snapshot,self.response,14.,[40,30,90],['s',1],self.ground)
        np.testing.assert_allclose(result['target_current_cm'],[0,80],atol=1e-10)

    def test_rejects_age_generation_motion_and_invalid_contact(self):
        for now,pose,generation in [(19,[0,0,0],['s',1]),(14,[0,0,0],['s',2]),
                                    (14,[0,0,20],['s',1]),(14,[0,61,0],['s',1]),
                                    (14,[float('nan'),0,0],['s',1])]:
            self.assertEqual(assess(self.snapshot,self.response,now,pose,generation,self.ground)['status'],'rejected')
        self.response['answer']['contact_pixel']['y']=150
        self.assertEqual(assess(self.snapshot,self.response,14,[0,0,0],['s',1],self.ground)['status'],'rejected')

    def test_network_wait_does_not_block_ticks_or_queue_old_images(self):
        release=threading.Event()
        finished=threading.Event()
        seen=[]
        def request(image,target):
            release.wait(2.)
            seen.append(int(image[0,0,0]))
            finished.set()
            return self.response
        worker=AsyncScene(request=request,clock=lambda:10.)
        image=np.zeros((480,640,3),np.uint8)
        try:
            self.assertTrue(worker.submit(image,10.,[0,0,0],{},['s',1],'Advil'))
            image[:]=255
            for _ in range(100):
                self.assertIsNone(worker.poll())
                self.assertFalse(worker.submit(image,10.,[0,0,0],{},['s',1],'Advil'))
            release.set()
            self.assertTrue(finished.wait(2.))
            deadline=time.monotonic()+2.
            reply=None
            while reply is None and time.monotonic()<deadline:
                reply=worker.poll()
                time.sleep(.001)
            self.assertIsNotNone(reply)
            self.assertEqual(seen,[0])
            self.assertEqual(worker.sent,1)
        finally:
            release.set()
            worker.close()

    def test_close_discards_late_response(self):
        worker=AsyncScene(clock=lambda:10.)
        worker.close()
        worker.mailbox.put(dict(response=self.response))
        self.assertIsNone(worker.poll())
        self.assertFalse(worker.submit(np.zeros((480,640,3)),10.,[0,0,0],{},1,'Advil'))

    def test_configurable_cadence_and_context_are_owned_by_perception_worker(self):
        seen=[]
        def request(image,target,context):
            seen.append((target,context))
            return self.response
        worker=AsyncScene(request=request,clock=lambda:10.,min_interval_seconds=0.,max_requests=1)
        try:
            self.assertTrue(worker.submit(np.zeros((480,640,3),np.uint8),10.,[0,0,0],{},
                                          ['s',1],'Advil',{'policy':'host_owned'}))
            deadline=time.monotonic()+2.
            reply=None
            while reply is None and time.monotonic()<deadline:
                reply=worker.poll();time.sleep(.001)
            self.assertEqual(seen,[('Advil',{'policy':'host_owned'})])
            self.assertEqual(reply['response'],self.response)
            self.assertFalse(worker.submit(np.zeros((480,640,3),np.uint8),10.,[0,0,0],{},
                                           ['s',1],'Advil',{}))
        finally:worker.close()

    def test_invalid_async_limits_are_rejected(self):
        for interval,requests in [(-1,1),(0,0),(0,73)]:
            with self.assertRaises(ValueError):
                AsyncScene(min_interval_seconds=interval,max_requests=requests)


if __name__=='__main__':unittest.main()
