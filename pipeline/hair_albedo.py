"""Safe redrawn hair RGB transfer with albedo correction for hidden backing."""
import cv2
import numpy as np

def smoothstep(lo,hi,x):
 t=np.clip((x-lo)/(hi-lo),0,1)
 return t*t*(3-2*t)

def distance(a):return cv2.distanceTransform((a[:,:,3]>127).astype(np.uint8),cv2.DIST_L2,5)

def interior_tone(a):
 light=a[:,:,:3].mean(2);d=distance(a)
 sample=(d>15)&(light>=30)&(light<=88)
 lab=cv2.cvtColor(a[:,:,:3].copy(),cv2.COLOR_RGB2LAB).astype(np.float32)
 if not sample.any():raise ValueError('No opaque dark interior hair sample')
 return lab[sample].mean(0),lab[sample].std(0)

def neutralize_outline(a):
 a=a.copy();tone,_=interior_tone(a)
 rgb=cv2.cvtColor(np.uint8(np.clip(tone,0,255))[None,None],cv2.COLOR_LAB2RGB)[0,0]
 edge=(a[:,:,3]>0)&(distance(a)<=3)&(a[:,:,:3].mean(2)<26)
 a[edge,:3]=rgb
 return a

def delight(a,canvas_y,back=False):
 out=a.copy();tone,_=interior_tone(a)
 lab=cv2.cvtColor(a[:,:,:3].copy(),cv2.COLOR_RGB2LAB).astype(np.float32)
 low=cv2.GaussianBlur(lab[:,:,0],(0,0),2.5);high=lab[:,:,0]-low;excess=low-tone[0]
 yy=np.arange(a.shape[0])[:,None]+canvas_y
 if back:band=np.broadcast_to(1-smoothstep(300,340,yy),low.shape)
 else:
  band=np.maximum(1-smoothstep(15,42,distance(a)),1-smoothstep(230,260,yy))
  if canvas_y>0:band*=1-smoothstep(285,310,yy)
 rgb=a[:,:,:3].astype(float)
 weight=band*smoothstep(5,20,excess)*((excess>5)|(rgb[:,:,0]>145))
 warm=(rgb[:,:,0]-rgb[:,:,2]>35)&(rgb[:,:,0]>135)
 weight=np.maximum(weight,.75*band*warm)*(a[:,:,3]>0)
 target=np.broadcast_to(tone,lab.shape).copy()
 target[:,:,0]=tone[0]+.35*np.minimum(excess,12)+.8*high
 corrected=lab+(target-lab)*weight[:,:,None]
 changed=weight>0
 out[changed,:3]=cv2.cvtColor(np.uint8(np.clip(corrected,0,255)),cv2.COLOR_LAB2RGB)[changed]
 return out

def transfer_redrawn_hair(a,original,still,redrawn,canvas_x,canvas_y):
 """Use repainted RGB only where the part is visible in the original still.

 All inputs are the same canvas crop. Matching the immutable part excludes
 hidden face infill. A dark brown target gate rejects skin, blue accessories
 and white gaps. Eroding inward by two pixels softens only the safe side of
 the mask, so excluded colors never leak across the blend. Alpha is untouched.
 """
 rgb=redrawn.astype(np.float32);light=rgb.mean(2)
 match=np.abs(original[:,:,:3].astype(np.int16)-still.astype(np.int16)).max(2)<=3
 hair=(light>=18)&(light<125)&(rgb[:,:,0]>=rgb[:,:,2])&(rgb[:,:,0]-rgb[:,:,1]<45)&(rgb[:,:,1]-rgb[:,:,2]<35)
 yy,xx=np.indices(light.shape);xx+=canvas_x;yy+=canvas_y
 pin=(xx>=646)&(xx<703)&(yy>=241)&(yy<311)
 orig_light=original[:,:,:3].mean(2)
 not_darkened=~((orig_light<95)&(light<orig_light-8))
 safe=match&hair&not_darkened&~pin&(original[:,:,3]>200)
 if canvas_x>0 or canvas_y>0:
  safe=cv2.erode(safe.astype(np.uint8),np.ones((3,3),np.uint8),iterations=1)>0
 weight=smoothstep(0,2,cv2.distanceTransform(safe.astype(np.uint8),cv2.DIST_L2,5))
 if canvas_x>0 or canvas_y>0:
  rim_zone=np.maximum(1-smoothstep(280,305,yy),smoothstep(645,670,xx))
  weight*=rim_zone
 out=a.copy()
 out[:,:,:3]=np.rint(a[:,:,:3]+(rgb-a[:,:,:3])*weight[:,:,None]).astype(np.uint8)
 # The backing preparation may already have normalized black outline texels.
 # Restore the protected strand RGB from its immutable input as well, avoiding
 # both that earlier change and any zero-weight LAB round-trip quantization.
 if canvas_x>0 or canvas_y>0:
  protected=(yy>=310)&(xx<640)&(original[:,:,3]>200)
  out[protected,:3]=original[protected,:3]
 return out,weight
