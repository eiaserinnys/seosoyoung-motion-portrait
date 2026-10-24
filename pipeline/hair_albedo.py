"""Deterministic hair albedo correction and foreground-mask cleanup."""
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
 else:band=np.maximum(1-smoothstep(15,42,distance(a)),1-smoothstep(230,260,yy))
 rgb=a[:,:,:3].astype(float)
 weight=band*smoothstep(5,20,excess)*((excess>5)|(rgb[:,:,0]>145))
 warm=(rgb[:,:,0]-rgb[:,:,2]>35)&(rgb[:,:,0]>135)
 weight=np.maximum(weight,.75*band*warm)*(a[:,:,3]>0)
 target=np.broadcast_to(tone,lab.shape).copy()
 target[:,:,0]=tone[0]+.35*np.minimum(excess,12)+.8*high
 corrected=lab+(target-lab)*weight[:,:,None]
 out[:,:,:3]=cv2.cvtColor(np.uint8(np.clip(corrected,0,255)),cv2.COLOR_LAB2RGB)
 return out

def remove_brow_ghost(a,brow_mask,canvas_y):
 out=a.copy();yy=np.arange(a.shape[0])+canvas_y
 upper=a[(yy>=322)&(yy<=331),:,3];lower=a[(yy>=363)&(yy<=372),:,3]
 through=(upper.max(0)>80)&(lower.max(0)>80)
 brow=cv2.dilate((brow_mask>16).astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))>0
 erase=brow&((yy>=333)&(yy<=362))[:,None]&~through[None,:]
 fade=np.maximum(erase.astype(np.float32),cv2.GaussianBlur(erase.astype(np.float32),(3,3),1))
 out[:,:,3]=np.rint(a[:,:,3]*(1-fade)).astype(np.uint8)
 return out
