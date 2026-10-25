"""Sparse trajectory fitting with velocity/acceleration priors and explicit CV masks."""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

TEMPORAL={'velocity':.5,'acceleration':16.,'componentScale':[1.,1.,1.,45.,.2,.2]}


def project_frames(model,poses):
    y,p,r=np.radians(poses[:,:3]).T
    cy,sy,cp,sp=np.cos(y),np.sin(y),np.cos(p),np.sin(p)
    rotation=np.zeros((len(poses),2,3))
    rotation[:,0,0]=cy;rotation[:,0,1]=sy*sp;rotation[:,0,2]=sy*cp
    rotation[:,1,1]=cp;rotation[:,1,2]=-sp
    X=np.array(model['points3D']);C=np.array(model['residual'])
    yy=poses[:,0]/45;pp=poses[:,1]/30
    B=np.column_stack((np.ones(len(poses)),yy,pp,yy*yy,yy*pp,pp*pp,yy**3,yy*yy*pp,yy*pp*pp,pp**3))
    q=np.einsum('tji,ni->tnj',rotation,X)+np.einsum('tb,bnd->tnd',B,C)
    c,s=np.cos(r)[:,None],np.sin(r)[:,None]
    result=np.stack((q[:,:,0]*c-q[:,:,1]*s,q[:,:,0]*s+q[:,:,1]*c),axis=2)
    return result*np.exp(poses[:,3,None,None])+poses[:,None,4:6]


def fit_trajectory(model,observations,initial,measurement_weights,free=None,max_nfev=18):
    """Training: held-out image weights are zero. Test: only held-out cameras free."""
    n=len(initial);free=np.ones(n,bool) if free is None else np.array(free,bool)
    frame_ids=np.flatnonzero(free);lookup={i:j for j,i in enumerate(frame_ids)}
    dimensions=observations.shape[1]*2;rows=n*dimensions+(n-1)*6+(n-2)*6
    sparsity=lil_matrix((rows,len(frame_ids)*6),dtype=np.int8)
    for i,j in lookup.items():sparsity[i*dimensions:(i+1)*dimensions,j*6:(j+1)*6]=1
    offset=n*dimensions
    for order,count in [(1,n-1),(2,n-2)]:
        for i in range(count):
            for k in range(i,i+order+1):
                if k in lookup:
                    j=lookup[k]
                    for d in range(6):sparsity[offset+i*6+d,j*6+d]=1
        offset+=count*6
    component=np.array(TEMPORAL['componentScale'])
    def unpack(values):
        poses=initial.copy();poses[free]=values.reshape(-1,6);return poses
    def residual(values):
        poses=unpack(values)
        data=(project_frames(model,poses)-observations)*np.sqrt(measurement_weights)[:,None,None]
        velocity=np.diff(poses,axis=0)*component*np.sqrt(TEMPORAL['velocity'])
        acceleration=np.diff(poses,n=2,axis=0)*component*np.sqrt(TEMPORAL['acceleration'])
        return np.r_[data.ravel(),velocity.ravel(),acceleration.ravel()]
    lo=np.tile([-65,-45,-30,-.6,0,0],len(frame_ids));hi=np.tile([65,45,30,.6,720,720],len(frame_ids))
    result=least_squares(residual,np.clip(initial[free].ravel(),lo+1e-6,hi-1e-6),bounds=(lo,hi),
                         jac_sparsity=sparsity.tocsr(),x_scale=np.tile([30,30,10,.1,100,100],len(frame_ids)),
                         max_nfev=max_nfev,ftol=1e-4,xtol=1e-4,gtol=1e-4)
    return unpack(result.x)


def normalized_key_pose(raw,affine):
    a=np.array(affine);scale=np.linalg.norm(a[:,0]);angle=np.degrees(np.arctan2(a[1,0],a[0,0]))
    return np.r_[raw[:2],raw[2]+angle,raw[3]+np.log(scale),a[:,:2]@raw[4:6]+a[:,2]]
