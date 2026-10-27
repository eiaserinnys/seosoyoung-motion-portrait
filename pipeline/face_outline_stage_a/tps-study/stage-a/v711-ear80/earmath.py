"""Residual similarity in the posed B frame, plus source-rest composition."""
import importlib.util
import numpy as np,cv2
from config import A
spec=importlib.util.spec_from_file_location('old_ear_metrics',A/'ear-fit/ear_model.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
measure=module.measure

def residual_affine(theta,center):
 tx,ty,sx,s,angle=theta;a=np.deg2rad(angle);M=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])@np.diag([s*sx,s]);c=np.asarray(center)
 return M,c+[tx,ty]-M@c

def total_affine(base,theta):
 M,_=residual_affine(theta,base['Bcenter']);K=np.array(base['K']);F=M@K;b=np.array(base['Bcenter'])+theta[:2]-F@base['restCenter'];return F,b

def warp(source,target,theta,center):
 yy,xx=np.where(source>0);x0,x1=int(xx.min()),int(xx.max()+1);y0,y1=int(yy.min()),int(yy.max()+1);M,b=residual_affine(theta,center)
 corners=np.array([[x0,y0],[x1,y0],[x0,y1],[x1,y1]])@M.T+b;ty,tx=np.where(target)
 lo=np.maximum(0,np.floor(np.minimum(corners.min(0),[tx.min(),ty.min()])-2).astype(int));hi=np.minimum(1024,np.ceil(np.maximum(corners.max(0),[tx.max()+1,ty.max()+1])+2).astype(int))
 out=cv2.warpAffine(source[y0:y1,x0:x1],np.column_stack((M,M@[x0,y0]+b-lo)),tuple((hi-lo).tolist()),flags=cv2.INTER_LINEAR)
 truth=target[lo[1]:hi[1],lo[0]:hi[0]];assert truth.sum()==target.sum()
 return out>127,truth,lo.tolist()

def feasible_translation(theta):return float(np.linalg.norm(theta[:2]))<=40.

def inner_line(alpha,base,theta,side):
 rows=np.flatnonzero((alpha>127).any(1));q=np.array([[np.flatnonzero(alpha[y]>127)[0 if side=='right' else -1],y] for y in rows],float)
 M,b=total_affine(base,theta);line=q@M.T+b;slopes=np.diff(q[:,0])/np.diff(q[:,1]);first=int(np.flatnonzero(abs(slopes)<1)[0]);return line,float(line[first,1]),{'sourceXY':q[first].tolist(),'worldXY':line[first].tolist()}

def edge_at(line,ys,side):
 out=np.full(len(ys),np.nan)
 for i,y in enumerate(ys):
  hits=[]
  for a,b in zip(line,line[1:]):
   if min(a[1],b[1])<=y<=max(a[1],b[1]):
    if a[1]==b[1]:hits.extend([a[0],b[0]])
    else:hits.append(a[0]+(y-a[1])*(b[0]-a[0])/(b[1]-a[1]))
  if hits:out[i]=min(hits) if side=='right' else max(hits)
 return out
