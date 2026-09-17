"""Local tracking of one explicitly identified object; no semantic recognition."""
import cv2
import numpy as np


class TargetLost(RuntimeError):
    pass


MINIMUM_BOX_PX=16


def _widen(low,high,limit):
    """Grow a span to the minimum trackable size without leaving the frame."""
    if limit<MINIMUM_BOX_PX:
        return 0.,float(limit)
    short=MINIMUM_BOX_PX-(high-low)
    if short>0:
        low-=short/2.;high+=short/2.
        if low<0: high-=low;low=0.
        if high>limit: low-=high-limit;high=float(limit)
    return max(0.,low),min(float(limit),high)


class TargetTracker:
    def __init__(self,image,box,projective_contact=False):
        if not isinstance(projective_contact,bool):raise ValueError("projective_contact must be boolean")
        self.projective_contact=projective_contact
        values=np.asarray(box,dtype=float)
        if values.shape!=(4,) or not np.isfinite(values).all():
            raise ValueError('Target box must be finite [left,top,right,bottom]')
        # A box that is small or clipped by the frame edge means the target is
        # far away or off to one side, not that the mission should end. Clamp it
        # into the frame and grow it to a trackable patch so the robot can drive
        # closer and get a better look, which is the whole point of approaching.
        height,width=image.shape[0],image.shape[1]
        x,y,r,b=(min(max(v,0.),limit) for v,limit in
                 zip(values,(width,height,width,height)))
        if r<x: x,r=r,x
        if b<y: y,b=b,y
        x,r=_widen(x,r,width)
        y,b=_widen(y,b,height)
        values=np.array([x,y,r,b],dtype=float)
        self.box=values
        self.contact=np.array([(x+r)/2,b],dtype=float)
        self.recent_contact=np.array([.5,1.])
        self.template=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)[int(y):int(b),int(x):int(r)].copy()
        if self.template.std()<10:
            raise ValueError('Target lacks distinctive texture')
        self.previous=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        factor=min(1.,128./max(self.template.shape))
        self.registration_template=cv2.resize(self.template,
            (max(12,int(self.template.shape[1]*factor)),max(12,int(self.template.shape[0]*factor)))).astype(np.float32)/255.
        self.recent_template=self.template.copy()
        self.confidence=1.

    def appearance(self,gray,box):
        x,y,r,b=np.round(box).astype(int)
        if x<0 or y<0 or r>gray.shape[1] or b>gray.shape[0] or min(r-x,b-y)<12:
            return -1.
        crop=cv2.resize(gray[y:b,x:r],(self.template.shape[1],self.template.shape[0]))
        return float(cv2.matchTemplate(cv2.GaussianBlur(crop,(5,5),1.),
                                      cv2.GaussianBlur(self.template,(5,5),1.),cv2.TM_CCOEFF_NORMED)[0,0])

    def refine(self,gray,box):
        """Remove accumulated flow drift against the immutable original texture.

        ECC is a bounded local registration, never a global identity search.
        Accept only when the resulting axis-aligned crop independently matches
        the original template; ECC's optimized score alone is insufficient.
        """
        x,y,r,b=np.round(box).astype(int)
        if x<0 or y<0 or r>gray.shape[1] or b>gray.shape[0] or min(r-x,b-y)<12:return box
        height,width=self.registration_template.shape
        crop=cv2.resize(gray[y:b,x:r],(width,height))
        try:
            score,matrix=cv2.findTransformECC(
                self.registration_template,crop.astype(np.float32)/255.,
                np.eye(2,3,dtype=np.float32),cv2.MOTION_AFFINE,
                (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,30,1e-4),None,5)
        except cv2.error:return box
        if not np.isfinite(matrix).all() or not np.isfinite(score):return box
        scales=np.linalg.svd(matrix[:,:2],compute_uv=False)
        if (score<.8 or np.linalg.det(matrix[:,:2])<=0
                or scales.min()<.85 or scales.max()>1.18
                or abs(matrix[0,2])>width*.15 or abs(matrix[1,2])>height*.15):return box
        center=matrix@np.array([width/2.,height/2.,1.])
        center=np.array([x,y])+center*np.array([(r-x)/width,(b-y)/height])
        size=np.array([r-x,b-y])*np.linalg.norm(matrix[:,:2],axis=0)
        refined=np.r_[center-size/2,center+size/2]
        if self.appearance(gray,refined)>.7:return refined
        return box

    def project_contact(self,points,next_points,box):
        """Track the contact separately from the appearance bounding rectangle."""
        before=points.reshape(-1,2)
        if np.any(np.ptp(before,axis=0)<(self.box[2:]-self.box[:2])*.25):
            raise TargetLost('Contact feature support is too narrow')
        matrix,inliers=cv2.findHomography(points,next_points,cv2.RANSAC,2.)
        if matrix is None or inliers is None or inliers.mean()<.7 or not np.isfinite(matrix).all():
            raise TargetLost('Contact projective geometry inconsistent')
        point=np.r_[self.contact,1.]
        denominator=float(matrix[2]@point)
        if abs(denominator)<1e-6:raise TargetLost('Contact projection singular')
        contact=(matrix@point)[:2]/denominator
        jacobian=(matrix[:2,:2]-np.outer(contact,matrix[2,:2]))/denominator
        scales=np.linalg.svd(jacobian,compute_uv=False)
        if np.linalg.det(jacobian)<=0 or not (.85<scales.min() and scales.max()<1.18):
            raise TargetLost('Contact scale changed too quickly')
        size=box[2:]-box[:2]
        if (not np.isfinite(contact).all() or abs(contact[0]-(box[0]+box[2])/2)>.35*size[0]
                or abs(contact[1]-box[3])>.2*size[1]):
            raise TargetLost('Contact left verified target boundary')
        return contact

    def update(self,image):
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        x,y,r,b=np.round(self.box).astype(int)
        # Both frames use the same padded ROI and coordinate origin. Padding
        # retains three pyramid levels for ordinary boxes and room for motion;
        # leaving this bounded region fails closed into stationary recovery.
        left,top=max(0,x-96),max(0,y-96)
        right,bottom=min(gray.shape[1],r+96),min(gray.shape[0],b+96)
        previous=self.previous[top:bottom,left:right]
        current=gray[top:bottom,left:right]
        mask=np.zeros_like(previous)
        inset=max(1,int(min(r-x,b-y)*.1))
        mask[max(0,y+inset-top):b-inset-top,max(0,x+inset-left):r-inset-left]=255
        points=cv2.goodFeaturesToTrack(previous,60,.01,3,mask=mask)
        if points is None or len(points)<8:raise TargetLost('Target features unavailable')
        next_points,ok,_=cv2.calcOpticalFlowPyrLK(previous,current,points,None,winSize=(21,21),maxLevel=3)
        if next_points is None:raise TargetLost('Target flow unavailable')
        back,back_ok,_=cv2.calcOpticalFlowPyrLK(current,previous,next_points,points.copy(),winSize=(21,21),maxLevel=3,
                                              flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        if back is None:raise TargetLost('Target reverse flow unavailable')
        coordinates=next_points.reshape(-1,2)
        inside=(coordinates[:,0]>=0)&(coordinates[:,0]<current.shape[1])&\
               (coordinates[:,1]>=0)&(coordinates[:,1]<current.shape[0])
        good=(ok.ravel()>0)&(back_ok.ravel()>0)&inside&(np.linalg.norm(points-back,axis=2).ravel()<1.)
        if good.sum()<8:raise TargetLost('Target flow inconsistent')
        origin=np.array([left,top],dtype=np.float32)
        points=points+origin
        next_points=next_points+origin
        matrix,inliers=cv2.estimateAffinePartial2D(points[good],next_points[good],method=cv2.RANSAC,ransacReprojThreshold=2.)
        if matrix is None or inliers.mean()<.7:raise TargetLost('Target geometry inconsistent')
        scale=float(np.linalg.norm(matrix[:,0]))
        if not .85<scale<1.18:raise TargetLost('Target scale changed too quickly')
        # Preserve subpixel dimensions. Repeatedly rounding/re-bounding rotated
        # rectangles enlarges the box and drifts onto background every frame.
        center=np.array([(self.box[0]+self.box[2])/2,(self.box[1]+self.box[3])/2,1])@matrix.T
        size=(self.box[2:]-self.box[:2])*scale
        box=np.r_[center-size/2,center+size/2]
        confidence=self.appearance(gray,box)
        if confidence<.75:
            refined=self.refine(gray,box)
            refined_confidence=self.appearance(gray,refined)
            if refined_confidence>confidence:box,confidence=refined,refined_confidence
        if confidence<.55:raise TargetLost('Target appearance no longer matches')
        contact=self.project_contact(points[good],next_points[good],box) if self.projective_contact else np.array([(box[0]+box[2])/2,box[3]])
        self.contact=contact
        self.box,self.previous,self.confidence=box,gray,confidence
        if confidence>.7:
            u,v,rr,bb=np.round(box).astype(int)
            self.recent_template=gray[v:bb,u:rr].copy()
            self.recent_contact=(self.contact-np.array([u,v]))/np.array([rr-u,bb-v])
        return self.observation()

    def reacquire(self,image):
        """Bounded stationary search; reject ambiguous matches, never change identity."""
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        x,y,r,b=self.box
        left,top=max(0,int(x)-100),max(0,int(y)-70)
        right,bottom=min(gray.shape[1],int(r)+100),min(gray.shape[0],int(b)+70)
        search=gray[top:bottom,left:right]
        candidates=[]
        for w in sorted(set(int(round((r-x)*scale)) for scale in np.linspace(.85,1.2,8))):
            h=int(round((b-y)*w/(r-x)))
            if min(w,h)<12 or w>search.shape[1] or h>search.shape[0]:continue
            template=cv2.resize(self.recent_template,(w,h))
            scores=cv2.matchTemplate(cv2.GaussianBlur(search,(5,5),1.),
                                     cv2.GaussianBlur(template,(5,5),1.),cv2.TM_CCOEFF_NORMED)
            _,score,_,location=cv2.minMaxLoc(scores)
            px,py=location
            suppressed=scores.copy()
            suppressed[max(0,py-h//2):py+h//2+1,max(0,px-w//2):px+w//2+1]=-1
            second=float(suppressed.max())
            candidates.append((score,second,[left+px,top+py,left+px+w,top+py+h]))
        if not candidates:raise TargetLost('Target not reacquired')
        score,second,box=max(candidates,key=lambda c:c[0])
        if score<.8 or score-second<.08 or self.appearance(gray,box)<.55:
            raise TargetLost('Target reacquisition uncertain or ambiguous')
        self.box=np.array(box,dtype=float)
        self.contact=self.box[:2]+(self.box[2:]-self.box[:2])*self.recent_contact
        self.previous=gray
        self.confidence=score
        return self.observation()

    def observation(self):
        x,y,r,b=self.box
        return dict(box=self.box.tolist(),base_pixel=self.contact.tolist() if self.projective_contact else [float((x+r)/2),float(b)],confidence=self.confidence)
