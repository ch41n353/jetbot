import os
import sys
import unittest
import cv2
import numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','local_nav'))
from target_tracker import TargetTracker,TargetLost


class TargetTrackerTests(unittest.TestCase):
    def scene(self):
        patch=cv2.resize(np.random.RandomState(13).randint(0,255,(24,16,3)).astype(np.uint8),
                         (64,96),interpolation=cv2.INTER_NEAREST)
        image=np.full((320,400,3),128,np.uint8)
        image[100:196,120:184]=patch
        return image,patch

    def test_original_registration_corrects_gradual_aspect_drift(self):
        image,patch=self.scene()
        tracker=TargetTracker(image,[120,100,184,196])
        original=tracker.template.copy()
        for step in range(1,21):
            width,height=64+step*2,96+step
            frame=np.full_like(image,128)
            frame[100:100+height,120:120+width]=cv2.resize(patch,(width,height))
            tracker.update(frame)
        np.testing.assert_array_equal(tracker.template,original)
        np.testing.assert_allclose(tracker.box,[120,100,224,216],atol=4)
        self.assertGreater(tracker.confidence,.7)

    def test_refinement_cannot_replace_identity(self):
        image,_=self.scene()
        tracker=TargetTracker(image,[120,100,184,196])
        image[100:196,120:184]=np.random.RandomState(9).randint(0,255,(96,64,3)).astype(np.uint8)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        box=tracker.box.copy()
        np.testing.assert_array_equal(tracker.refine(gray,box),box)
        with self.assertRaises(TargetLost):tracker.update(image)

    def test_roi_coordinates_remain_global_near_image_edges(self):
        _,patch=self.scene()
        for left,top in [(4,5),(200,175)]:
            image=np.full((320,400,3),128,np.uint8)
            image[top:top+96,left:left+64]=patch
            tracker=TargetTracker(image,[left,top,left+64,top+96])
            for step in range(1,6):
                frame=np.full_like(image,128)
                x,y=left+step*3,top+step*2
                frame[y:y+96,x:x+64]=patch
                tracker.update(frame)
            np.testing.assert_allclose(tracker.box,[x,y,x+64,y+96],atol=1)

    def test_target_leaving_bounded_flow_region_is_lost(self):
        image,patch=self.scene()
        tracker=TargetTracker(image,[120,100,184,196])
        frame=np.full_like(image,128)
        frame[100:196,310:374]=patch
        with self.assertRaises(TargetLost):tracker.update(frame)

    def test_projective_contact_follows_known_bottom_point(self):
        image,_=self.scene()
        tracker=TargetTracker(image,[120,100,184,196],projective_contact=True)
        original=tracker.template.copy()
        source=np.float32([[120,100],[184,100],[184,196],[120,196]])
        for step in range(1,11):
            destination=np.float32([[120-.2*step,100-.1*step],[184+.5*step,100],
                                    [184+.8*step,196+.8*step],[120-.5*step,196+.2*step]])
            matrix=cv2.getPerspectiveTransform(source,destination)
            frame=cv2.warpPerspective(image,matrix,(400,320),borderValue=(128,128,128))
            observation=tracker.update(frame)
            truth=matrix@np.array([152.,196.,1.]);truth=truth[:2]/truth[2]
            np.testing.assert_allclose(observation['base_pixel'],truth,atol=1.5)
        np.testing.assert_array_equal(tracker.template,original)

    def test_projective_contact_recovery_keeps_identity_and_contact(self):
        image,patch=self.scene()
        tracker=TargetTracker(image,[120,100,184,196],projective_contact=True)
        tracker.update(image)
        recovered=tracker.reacquire(image)
        np.testing.assert_allclose(recovered['base_pixel'],[152,196],atol=2)
        image[100:196,210:274]=patch
        with self.assertRaises(TargetLost):tracker.reacquire(image)

    def test_duplicate_identity_remains_ambiguous(self):
        image,patch=self.scene()
        tracker=TargetTracker(image,[120,100,184,196])
        image[100:196,210:274]=patch
        with self.assertRaises(TargetLost):tracker.reacquire(image)


if __name__=='__main__':
    cv2.setNumThreads(1)
    unittest.main()
