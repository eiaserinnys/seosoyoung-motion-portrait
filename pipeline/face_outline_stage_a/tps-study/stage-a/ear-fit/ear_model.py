"""7.6.3 ear similarity; full support alpha IoU, no ear-box crop."""
import cv2,numpy as np
from scipy.ndimage import distance_transform_edt
CENTERS={'left':np.array([333.,465.]),'right':np.array([671.,464.5])}
def affine(theta,side):
 tx,ty,sx,s,angle=theta;a=np.deg2rad(angle);r=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]);M=r@np.diag([s*sx,s]);center=CENTERS[side]
 return M,center+np.array([tx,ty])-M@center

def warp(source,target,theta,side):
 yy,xx=np.where(source>0);sx0,sx1=int(xx.min()),int(xx.max()+1);sy0,sy1=int(yy.min()),int(yy.max()+1)
 M,b=affine(theta,side);corners=np.array([[sx0,sy0],[sx1,sy0],[sx0,sy1],[sx1,sy1]])@M.T+b
 ty,tx=np.where(target);lo=np.maximum(0,np.floor(np.minimum(corners.min(0),[tx.min(),ty.min()])-2).astype(int));hi=np.minimum(1024,np.ceil(np.maximum(corners.max(0),[tx.max()+1,ty.max()+1])+2).astype(int))
 alpha=cv2.warpAffine(source[sy0:sy1,sx0:sx1],np.column_stack((M,M@[sx0,sy0]+b-lo)),tuple((hi-lo).tolist()),flags=cv2.INTER_LINEAR)
 truth=target[lo[1]:hi[1],lo[0]:hi[0]];assert truth.sum()==target.sum()
 return alpha>127,truth,lo.tolist()

def outline(mask):
 c,_=cv2.findContours(np.uint8(mask),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);out=np.zeros(mask.shape,bool)
 for q in c:out[q[:,0,1],q[:,0,0]]=True
 return out

def measure(pred,target,origin,box,side,distances=False):
 union=int((pred|target).sum());n=int(pred.sum());yy,xx=np.indices(pred.shape);x=xx+origin[0];y=yy+origin[1];x0,y0,x1,y1=box
 inside=(x>=x0)&(x<x1)&(y>=y0)&(y<y1)
 result={'wholeScreenIoU':float((pred&target).sum()/union),'earAlphaPixels':n,'outsideBoxPixels':int((pred&~inside).sum()),'outsideBoxFraction':float((pred&~inside).sum()/n) if n else 1.}
 if distances:
  aa=outline(pred);bb=outline(target);clip=x0 if side=='right' else x1-1;bb&=x!=clip
  if aa.any() and bb.any():
   symmetric_a=aa&(x!=clip)
   d=np.r_[distance_transform_edt(~bb)[symmetric_a],distance_transform_edt(~symmetric_a)[bb]];one=distance_transform_edt(~aa)[bb]
   result.update(helixSymmetricRMS=float(np.sqrt(np.mean(d*d))),helixSymmetricMax=float(d.max()),helixOneWayRMS=float(np.sqrt(np.mean(one*one))),helixOneWayMax=float(one.max()))
  else:result.update(helixSymmetricRMS=999.,helixSymmetricMax=999.,helixOneWayRMS=999.,helixOneWayMax=999.)
 return result
