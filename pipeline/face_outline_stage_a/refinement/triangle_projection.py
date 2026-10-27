"""The runtime sideTriangleCorrection X-only orthogonal projection, offline."""
import numpy as np
class ProjectionConflict(RuntimeError):pass

def project(rest,world,triangles,fixed):
 points=np.array(world,float).copy();before=points.copy();fixed=set(fixed);rest=np.array(rest,float)
 for iteration in range(16):
  changed=False
  for ti,tri in enumerate(triangles):
   a,b,c=points[tri,:2];u,v,w=rest[tri,:2]
   raw=(v[0]-u[0])*(w[1]-u[1])-(v[1]-u[1])*(w[0]-u[0]);sign=np.sign(raw)
   area=sign*((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))
   if area>0:continue
   gradient=np.array([b[1]-c[1],c[1]-a[1],a[1]-b[1]])*sign
   for k,i in enumerate(tri):
    if i in fixed:gradient[k]=0
   norm=gradient@gradient
   if norm==0:raise ProjectionConflict(f'triangle {ti} conflicts with fixed observations {tri}')
   step=(abs(raw)*1e-6-area)/norm
   for k,i in enumerate(tri):points[i,0]+=step*gradient[k]
   changed=True
  if not changed:return points,dict(iterations=iteration,movedIds=np.flatnonzero(points[:,0]!=before[:,0]).tolist(),maxMove=float(abs(points[:,0]-before[:,0]).max()))
 raise ProjectionConflict('Side triangle projection did not converge after the runtime 16 passes')
