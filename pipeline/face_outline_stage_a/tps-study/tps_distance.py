"""8x8 thin-plate displacement, signed-distance data and grid second differences."""
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import distance_transform_edt,map_coordinates
from scipy.optimize import least_squares

def sdf(mask):
 return distance_transform_edt(~mask)-distance_transform_edt(mask)

class TPS:
 def __init__(self,controls,coefficients=None):
  self.controls=np.array(controls,float);self.origin=self.controls.min(0)
  self.scale=np.ptp(self.controls,axis=0)
  self.rbf=RBFInterpolator((self.controls-self.origin)/self.scale,np.eye(64),kernel='thin_plate_spline',degree=1,smoothing=0)
  self.coefficients=np.zeros((64,2)) if coefficients is None else np.array(coefficients,float)
 def basis(self,points):return self.rbf((np.array(points)-self.origin)/self.scale)
 def __call__(self,points):
  points=np.array(points,float);return points+self.basis(points)@self.coefficients

def second_differences():
 rows=[]
 for y in range(8):
  for x in range(6):
   row=np.zeros((8,8));row[y,x:x+3]=[1,-2,1];rows.append(row.ravel())
 for x in range(8):
  for y in range(6):
   row=np.zeros((8,8));row[y:y+3,x]=[1,-2,1];rows.append(row.ravel())
 return np.array(rows)

def fit(source,target,brow,bottom,initial_linear,initial_shift,ear_region):
 sy,sx=np.where(source);xmin,xmax=sx.min(),sx.max();ymin=max(brow,int(sy.min()));ymax=min(bottom,source.shape[0]-1)
 controls=np.array([[x,y] for y in np.linspace(ymin,ymax,8) for x in np.linspace(xmin,xmax,8)])
 model=TPS(controls);s=sdf(source);t=sdf(target);ty,tx=np.gradient(t)
 yy,xx=np.indices(source.shape);sample=(yy>=ymin)&(yy<=ymax)&(np.abs(s)<48)&(xx%8==0)&(yy%8==0)
 points=np.column_stack((xx[sample],yy[sample])).astype(float);values=s[sample]
 B=model.basis(points);D=second_differences();reg=.25**.5
 initial=controls@initial_linear.T+initial_shift-controls
 q0=points+B@initial;ex,ey=ear_region
 weights=np.sqrt(np.where((q0[:,0]>ex)&(q0[:,1]>ey-30)&(q0[:,1]<ey+110),.5,1.))
 def at(image,q):return map_coordinates(image,[q[:,1],q[:,0]],order=1,mode='nearest')
 def residual(c):
  c=c.reshape(64,2);q=points+B@c
  return np.r_[weights*(at(t,q)-values),(reg*D@c).ravel()]
 def jacobian(c):
  q=points+B@c.reshape(64,2);g=np.column_stack((at(tx,q),at(ty,q)))
  data=(weights[:,None,None]*B[:,:,None]*g[:,None,:]).reshape(len(points),128)
  return np.vstack((data,reg*np.kron(D,np.eye(2))))
 result=least_squares(residual,initial.ravel(),jac=jacobian,max_nfev=120,ftol=1e-6,xtol=1e-6,gtol=1e-6)
 model.coefficients=result.x.reshape(64,2)
 return model,{'success':bool(result.success),'message':result.message,'nfev':int(result.nfev),'sourceSamples':len(points),
  'earHalfWeightSamples':int((weights<1).sum()),'secondDifferenceLambda':.25,
  'initialCost':float(np.sum(residual(initial.ravel())**2)),'finalCost':float(np.sum(result.fun**2)),
  'maxControlDisplacement':float(np.linalg.norm(model.coefficients,axis=1).max()),'controlGrid':[8,8]}

def smoke():
 yy,xx=np.indices((80,80));a=(xx-35)**2+(yy-40)**2<20**2;b=(xx-38)**2+(yy-40)**2<20**2
 model,info=fit(a,b,16,67,np.eye(2),np.zeros(2),(100,40))
 shift=model([[35,40]])[0]-[35,40]
 assert abs(shift[0]-3)<.5 and abs(shift[1])<.5,(shift,info)
 d=second_differences();assert np.max(abs(d@np.array([[x,y] for y in range(8) for x in range(8)])))==0
 print('TPS SMOKE PASS 8x8 SDF known translation',shift.tolist(),info['nfev'],flush=True)

if __name__=='__main__':smoke()
