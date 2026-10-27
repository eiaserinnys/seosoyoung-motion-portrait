"""Same 8x8 distance TPS; corrected source composition and observation weights."""
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from tps_distance import TPS,second_differences,sdf

def observation_weights(q0,hair_band,ear_regions,ear_boxes):
    ear=np.zeros(len(q0),bool);ear_box=np.zeros(len(q0),bool)
    for region in ear_regions:
        side,ex,ey=region
        outside=q0[:,0]<ex if side=='left' else q0[:,0]>ex
        ear|=outside&(q0[:,1]>ey-30)&(q0[:,1]<ey+110)
    for x0,y0,x1,y1 in ear_boxes:
        ear_box|=(q0[:,0]>=x0)&(q0[:,0]<x1)&(q0[:,1]>=y0)&(q0[:,1]<y1)
    covered=map_coordinates(hair_band.astype(float),[q0[:,1],q0[:,0]],order=0,mode='constant',cval=0)>0
    weights=np.where(ear|ear_box,.5,1.);weights[covered&~ear_box]=0
    return np.sqrt(weights)

def fit(source,target,brow,bottom,initial_linear,initial_shift,ear_regions,ear_boxes,hair_band,max_nfev=120):
    sy,sx=np.where(source);xmin,xmax=sx.min(),sx.max()
    ymin=max(brow,int(sy.min()));ymax=min(bottom,source.shape[0]-1)
    controls=np.array([[x,y] for y in np.linspace(ymin,ymax,8) for x in np.linspace(xmin,xmax,8)])
    model=TPS(controls);s=sdf(source);t=sdf(target);ty,tx=np.gradient(t)
    yy,xx=np.indices(source.shape)
    sample=(yy>=ymin)&(yy<=ymax)&(np.abs(s)<48)&(xx%8==0)&(yy%8==0)
    points=np.column_stack((xx[sample],yy[sample])).astype(float);values=s[sample]
    B=model.basis(points);D=second_differences();reg=.25**.5
    initial=controls@initial_linear.T+initial_shift-controls
    q0=points+B@initial;weights=observation_weights(q0,hair_band,ear_regions,ear_boxes)
    def at(image,q):return map_coordinates(image,[q[:,1],q[:,0]],order=1,mode='nearest')
    def residual(c):
        c=c.reshape(64,2);q=points+B@c
        return np.r_[weights*(at(t,q)-values),(reg*D@c).ravel()]
    def jacobian(c):
        q=points+B@c.reshape(64,2);g=np.column_stack((at(tx,q),at(ty,q)))
        data=(weights[:,None,None]*B[:,:,None]*g[:,None,:]).reshape(len(points),128)
        return np.vstack((data,reg*np.kron(D,np.eye(2))))
    result=least_squares(residual,initial.ravel(),jac=jacobian,max_nfev=max_nfev,ftol=1e-6,xtol=1e-6,gtol=1e-6)
    model.coefficients=result.x.reshape(64,2)
    return model,{'success':bool(result.success),'message':result.message,'nfev':int(result.nfev),'maxNfev':max_nfev,
        'sourceSamples':len(points),'zeroWeightSamples':int((weights==0).sum()),
        'earHalfWeightSamples':int(np.isclose(weights,np.sqrt(.5)).sum()),
        'secondDifferenceLambda':.25,'initialCost':float(np.sum(residual(initial.ravel())**2)),
        'finalCost':float(np.sum(result.fun**2)),
        'maxControlDisplacement':float(np.linalg.norm(model.coefficients,axis=1).max()),
        'controlGrid':[8,8],'initialLinear':initial_linear.tolist(),'initialShift':initial_shift.tolist(),
        'earRegions':ear_regions,'earBoxes':ear_boxes}
