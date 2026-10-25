"""Convex polar surface fit with direct contour RMS and material-area constraints."""
import clarabel
import numpy as np
from scipy import sparse


def fit_radii(h,g,B,target,poses,grid_basis,grid_rigid,front,keys,center_error):
    n=len(front);count=len(keys);variables=10*n
    ag=grid_basis.copy();ag[:,0]-=.505
    # r(grid)>=.505*r(neutral) ensures every polar material triangle has
    # positive area >=.505². These are linear inequalities in cubic coefficients.
    A=[-sparse.kron(ag,sparse.eye(n))]
    b=[(grid_rigid-.505*front).ravel()]
    cones=[clarabel.NonnegativeConeT(len(grid_basis)*n)]
    theta=np.linspace(-np.pi,np.pi,n,endpoint=False)
    rays=np.column_stack((np.cos(theta),np.sin(theta)))
    radial_to_xy=sparse.csc_matrix((rays.ravel(),(np.arange(2*n),np.repeat(np.arange(n),2))),shape=(2*n,n))
    for i,key in enumerate(keys):
        scale=np.exp(poses[i,3]);angle=np.radians(poses[i,2])
        if abs(angle)>1e-7:raise ValueError('Representative camera must be unrolled before contour fitting')
        projection=scale*radial_to_xy@sparse.kron(B[i:i+1],sparse.eye(n))
        # b-Ax = [limit, observed-predicted] uses the actual 2D center error.
        offset=(scale*rays*target[i,:,None]-center_error[i]).ravel()
        A.append(sparse.vstack((sparse.csc_matrix((1,variables)),-projection)))
        b.append(np.r_[np.sqrt(n)*5.95,-offset])
        cones.append(clarabel.SecondOrderConeT(2*n+1))
    scale=np.trace(h)/10
    P=sparse.triu(sparse.kron(h/scale,sparse.eye(n))).tocsc()
    settings=clarabel.DefaultSettings();settings.verbose=False
    result=clarabel.DefaultSolver(P,-g.ravel()/scale,sparse.vstack(A).tocsc(),np.concatenate(b),cones,settings).solve()
    if str(result.status)!='Solved':raise ValueError('Contour RMS/Jacobian cones: '+str(result.status))
    return np.array(result.x).reshape(10,n)
