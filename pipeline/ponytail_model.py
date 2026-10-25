"""Weak-perspective alternating 3D/pose solve and weighted cubic shape residual.

The neutral XY gauge is the observed front. Roll, scale and translation remain
per-frame camera terms; shape residuals are fitted only on training frames.
"""
import numpy as np
from scipy.optimize import least_squares


def constrained_quadratic(h,g,matrices,lower,upper):
    """Solve the small linear-inequality silhouette QP in one convex path."""
    import clarabel
    from scipy import sparse
    a=np.vstack(matrices);lo=np.concatenate(lower);hi=np.concatenate(upper)
    finite_lo=np.isfinite(lo);finite_hi=np.isfinite(hi)
    A=sparse.csc_matrix(np.vstack((-a[finite_lo],a[finite_hi])))
    b=np.r_[-lo[finite_lo],hi[finite_hi]]
    scale=np.trace(h)/len(g)
    settings=clarabel.DefaultSettings();settings.verbose=False
    result=clarabel.DefaultSolver(sparse.triu(sparse.csc_matrix(h/scale)).tocsc(),-g/scale,A,b,
                                 [clarabel.NonnegativeConeT(len(b))],settings).solve()
    if str(result.status)!='Solved':raise ValueError('Silhouette QP: '+str(result.status))
    return np.array(result.x)


def basis(y,p):
    y=np.asarray(y);p=np.asarray(p)
    return np.stack(np.broadcast_arrays(np.ones_like(y),y,p,y*y,y*p,p*p,y**3,y*y*p,y*p*p,p**3),axis=-1)


def block_holdout(n):
    mask=np.zeros(n,bool)
    for start in [60,165,270]:mask[start:start+18]=True
    return mask


def rotation(yaw,pitch):
    a,b=np.radians([yaw,pitch]);ca,sa,cb,sb=np.cos(a),np.sin(a),np.cos(b),np.sin(b)
    # Local pitch followed by yaw; camera roll is the outer image rotation.
    return np.array([[ca,sa*sb,sa*cb],[0,cb,-sb],[-sa,ca*sb,ca*cb]])


def roll_matrix(degrees):
    a=np.radians(degrees);return np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])


def predict(model,pose):
    yaw,pitch,roll,logscale,tx,ty=pose
    X=np.asarray(model['points3D']);C=np.asarray(model['residual'])
    q=(X@rotation(yaw,pitch).T)[:,:2]+np.einsum('b,bnd->nd',basis(yaw/45,pitch/30),C)
    return np.exp(logscale)*(q@roll_matrix(roll).T)+[tx,ty]


def camera_points(points,pose):
    return (np.asarray(points)-pose[4:6])@roll_matrix(pose[2])/np.exp(pose[3])


def initial_poses(rows,keys,X):
    def descriptor(points):
        p=np.array(points);eye=p[:2].mean(0);distance=np.linalg.norm(p[1]-p[0])
        return ((p[:6]-eye)/distance).ravel()
    a=np.array([np.r_[1,descriptor(k['landmarks'])] for k in keys])
    inverse=np.linalg.lstsq(a,np.array([[k['yaw'],k['pitch']] for k in keys]),rcond=None)[0]
    result=[]
    for row in rows:
        obs=np.array(row['landmarks']);angles=np.r_[1,descriptor(obs)]@inverse
        if 'initialAngles' in row:angles=np.array(row['initialAngles'])
        if 'yaw' in row:angles=np.array([row['yaw'],row['pitch']])
        angles=np.clip(angles,[-55,-40],[55,40])
        angle=np.degrees(np.arctan2(*(obs[1]-obs[0])[::-1]))
        q=(X@rotation(*angles).T)[:,:2]@roll_matrix(angle).T
        scale=np.linalg.norm(obs[1]-obs[0])/np.linalg.norm(q[1]-q[0])
        translation=obs[:6].mean(0)-scale*q[:6].mean(0)
        result.append([*angles,angle,np.log(scale),*translation])
    return np.array(result)


def shape_system(poses,X,observations):
    rotations=np.array([rotation(*p[:2]) for p in poses])
    B=basis(poses[:,0]/45,poses[:,1]/30)[:,1:]
    a=np.zeros((len(poses),2,19))
    a[:,:,0]=rotations[:,:2,2]
    a[:,0,1:10]=B;a[:,1,10:19]=B
    canonical=np.array([camera_points(obs,p) for obs,p in zip(observations,poses)])
    plane=np.einsum('ni,tji->tnj',X[:,:2],rotations[:,:2,:2])
    return a.reshape(-1,19),(canonical-plane).transpose(0,2,1).reshape(-1,len(X))


def fit_continuous_shapes(keys,observations,X,poses,train):
    """Convex cubic residual fit on the final rigid camera trajectory.

    Each key's image-space RMS bound is a second-order cone. Keeping the
    Layer 1 camera/depth fixed prevents the residual from creating pose branches.
    The constant coefficient is zero, preserving the exact canonical front.
    """
    import clarabel
    from scipy import sparse
    count=len(keys);n=len(X);dimensions=n*2
    weights=np.r_[np.full(count,10.),train.astype(float)]
    B=basis(poses[:,0]/45,poses[:,1]/30)[:,1:]
    canonical=np.array([camera_points(obs,p) for obs,p in zip(observations,poses)])
    rigid=np.array([(X@rotation(*p[:2]).T)[:,:2] for p in poses])
    target=(canonical-rigid).reshape(len(poses),dimensions)
    # The prescribed weights multiply image-space squared residuals.
    w=weights*np.exp(2*poses[:,3])
    h=B.T@(B*w[:,None])+.1*np.eye(9)
    g=B.T@(target*w[:,None]);scale=np.trace(h)/9
    P=sparse.triu(sparse.kron(h/scale,sparse.eye(dimensions))).tocsc()
    q=-g.ravel()/scale
    blocks=[];rhs=[];cones=[]
    for i,key in enumerate(keys[1:],1):
        limit=.49 if key['name']=='front' else 3.45
        camera_scale=np.exp(poses[i,3])
        projection=sparse.kron(B[i:i+1],sparse.eye(dimensions))
        blocks.append(sparse.vstack((sparse.csc_matrix((1,9*dimensions)),-camera_scale*projection)))
        rhs.append(np.r_[np.sqrt(n)*limit,-camera_scale*target[i]])
        cones.append(clarabel.SecondOrderConeT(dimensions+1))
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=100
    solution=clarabel.DefaultSolver(P,q,sparse.vstack(blocks).tocsc(),np.concatenate(rhs),cones,settings).solve()
    if str(solution.status)!='Solved':raise ValueError('Cubic RMS cone solve: '+str(solution.status))
    C=np.zeros((10,n,2));C[1:]=np.array(solution.x).reshape(9,n,2)
    model=dict(points3D=X.tolist(),residual=C.tolist(),residualSolver='Clarabel SOCP; fixed temporal rigid cameras')
    return model,poses


def fit_layers(keys,video):
    from seedance_temporal import fit_trajectory,normalized_key_pose
    holdout=block_holdout(len(video));train=~holdout
    observations=np.array([r['landmarks'] for r in keys+video])
    front=observations[0];origin=front[:2].mean(0)
    X=np.column_stack((front-origin,[0,0,-15,-15,65,35,0,-50,-50]))
    prior=X[:,2].copy()
    poses=initial_poses(keys+video,keys,X)
    weights=train.astype(float)
    model=dict(points3D=X.tolist(),residual=np.zeros((10,len(X),2)).tolist())
    # Estimate cameras from Layer 1 only: inverting the cubic residual here
    # creates extra branches that a physical rigid camera does not have.
    poses[0]=[0,0,0,0,*origin]
    for iteration in range(5):
        poses[len(keys):]=fit_trajectory(model,observations[len(keys):],poses[len(keys):],weights,max_nfev=25)
        temporal=poses[len(keys):]
        canonical=np.array([camera_points(obs,p) for obs,p in zip(observations[len(keys):],temporal)])
        rotations=np.array([rotation(*p[:2]) for p in temporal])
        # Shared 3D depth is the linear ALS block; neutral XY is canonical.
        planes=np.einsum('ni,tji->tnj',X[:,:2],rotations[:,:2,:2])
        coefficients=rotations[:,:2,2]
        rhs=canonical-planes
        numerator=np.einsum('t,td,tnd->n',weights,coefficients,rhs)+20*prior
        denominator=np.einsum('t,td,td->',weights,coefficients,coefficients)+20
        X[:,2]=numerator/denominator
        model=dict(points3D=X.tolist(),residual=np.zeros((10,len(X),2)).tolist())
        print('rigid temporal ALS',iteration+1,'max step',float(np.max(np.abs(np.diff(temporal[:,:3],axis=0)))),flush=True)
    # Resolve remaining illustrated/nonrigid image motion without freezing a
    # poor rigid-only camera gauge. Every camera step keeps the same temporal
    # priors; held-out image coordinates never enter this training block.
    temporal=poses[len(keys):];image_points=observations[len(keys):]
    for iteration in range(8):
        B=basis(temporal[:,0]/45,temporal[:,1]/30)[:,1:]
        target=np.array([camera_points(o,p)-(X@rotation(*p[:2]).T)[:,:2] for o,p in zip(image_points,temporal)])
        C=np.zeros((10,len(X),2))
        C[1:]=np.linalg.solve(B.T@(B*weights[:,None])+np.eye(9),
                             B.T@(target.reshape(len(video),-1)*weights[:,None])).reshape(9,len(X),2)
        model=dict(points3D=X.tolist(),residual=C.tolist())
        temporal=fit_trajectory(model,image_points,temporal,weights,max_nfev=25)
    poses[len(keys):]=temporal
    for j,key in enumerate(keys[1:],1):poses[j]=normalized_key_pose(poses[len(keys)+key['sourceFrame']-1],key['normalization'])
    return model,poses,holdout


def infer_holdout(model,observations,poses,count,holdout):
    # Adjacent training poses initialize a held-out block; its observations
    # estimate camera only and never enter either shape coefficient system.
    from seedance_temporal import fit_trajectory
    poses[count:]=fit_trajectory(model,observations[count:],poses[count:],holdout.astype(float),free=holdout,max_nfev=30)
    return poses


def fit_silhouette(keys,video,poses,holdout):
    """Polar radii in the unrolled camera gauge, conditioned on learned pose."""
    count=len(keys)
    rows=keys+video;weights=np.r_[np.full(count,10.),(~holdout).astype(float)]
    B=basis(poses[:,0]/45,poses[:,1]/30)
    theta=np.linspace(-np.pi,np.pi,len(keys[0]['radii']),endpoint=False)
    def unroll(r,p):
        angle=(theta+np.radians(p[2])+np.pi)%(2*np.pi)-np.pi
        return np.interp(angle,np.r_[theta,np.pi],np.r_[r,r[0]])/np.exp(p[3])
    radii=np.array([unroll(np.array(r['radii']),p) for r,p in zip(rows,poses)])
    front=radii[0]
    theta=np.linspace(-np.pi,np.pi,len(front),endpoint=False)
    rotations=np.array([rotation(*p[:2]) for p in poses])
    a=B*np.sqrt(weights[:,None])
    h=a.T@a+.2*np.eye(10)
    inverse=np.linalg.solve(h,a.T)
    depth=[]
    def shell(r,t,z,rot):
        ring=np.array([r*np.cos(t),r*np.sin(t),z])
        return np.linalg.norm(rot[:,:2,:]@ring,axis=1)
    for j,t in enumerate(theta):
        def error(z):
            target=(radii[:,j]-shell(front[j],t,z[0],rotations))*np.sqrt(weights)
            return np.r_[target-a@(inverse@target),.005*z]
        fitted=least_squares(error,[0.],bounds=(-250,250),max_nfev=60)
        depth.append(float(fitted.x[0]))
    rigid=np.column_stack([shell(r,t,z,rotations) for r,t,z in zip(front,theta,depth)])
    target=radii-rigid
    right=a.T@(target*np.sqrt(weights[:,None]))
    grid=np.array([[y,p,0,0,0,0] for y in np.linspace(-45,45,19) for p in np.linspace(-30,30,13)])
    bg=basis(grid[:,0]/45,grid[:,1]/30);rg=np.array([rotation(*p[:2]) for p in grid])
    # The silhouette centroid is a separate material control. It is not the
    # nose/eye midpoint, which shifts toward the near side under yaw.
    centers=np.array([r['center'] for r in rows])
    xy=camera_points([centers[0]],poses[0])[0]
    canonical=np.array([camera_points([q],p)[0] for q,p in zip(centers,poses)])
    plane=np.einsum('i,tji->tj',xy,rotations[:,:2,:2])
    ac=np.zeros((len(rows),2,21));ac[:,:,0]=rotations[:,:2,2];ac[:,0,1:11]=B;ac[:,1,11:21]=B
    ac=ac.reshape(-1,21);bc=(canonical-plane).ravel()
    w=np.repeat(weights,2)
    hc=ac.T@(ac*w[:,None])+.1*np.eye(21);gc=ac.T@(bc*w)
    tolerance=np.repeat(1/np.exp(poses[:count,3]),2)
    cc=constrained_quadratic(hc,gc,[ac[:count*2]],[bc[:count*2]-tolerance],[bc[:count*2]+tolerance])
    center=dict(point3D=[*xy,float(cc[0])],residual=np.column_stack((cc[1:11],cc[11:21])).tolist())
    from seedance_silhouette import fit_radii
    center_error=np.array([predict_center({'center':center},p)-q for p,q in zip(poses[:count],centers[:count])])
    grid_rigid=np.column_stack([shell(r,t,z,rg) for r,t,z in zip(front,theta,depth)])
    coefficient=fit_radii(h,right,B,target,poses,bg,grid_rigid,front,keys,center_error)
    return dict(front=front.tolist(),depth=depth,residual=coefficient.tolist(),center=center)


def predict_radii(model,pose):
    r=np.array(model['front']);theta=np.linspace(-np.pi,np.pi,len(r),endpoint=False)
    points=np.column_stack((r*np.cos(theta),r*np.sin(theta),model['depth']))
    shell=np.linalg.norm((points@rotation(*pose[:2]).T)[:,:2],axis=1)
    radius=shell+basis(pose[0]/45,pose[1]/30)@np.array(model['residual'])
    angle=(theta-np.radians(pose[2])+np.pi)%(2*np.pi)-np.pi
    return np.exp(pose[3])*np.interp(angle,np.r_[theta,np.pi],np.r_[radius,radius[0]])


def predict_center(model,pose):
    control=model['center']
    point=(np.array(control['point3D'])@rotation(*pose[:2]).T)[:2]
    point+=basis(pose[0]/45,pose[1]/30)@np.array(control['residual'])
    return np.exp(pose[3])*(point@roll_matrix(pose[2]).T)+pose[4:6]


def model_field(model,silhouette):
    """Polar skin field on a regular neutral material grid, camera similarity removed.

    Facial landmarks are independent controls (including occluding ears). The
    continuous skin domain is the measured star-shaped facial silhouette.
    """
    front=predict_radii(silhouette,[0,0,0,0,0,0]);theta=np.linspace(-np.pi,np.pi,len(front),endpoint=False)
    rays=np.column_stack((np.cos(theta),np.sin(theta)));n=len(front)
    rest=np.vstack(([[0,0]],*[rays*(front*f)[:,None] for f in np.linspace(1/12,1,12)]))
    triangles=[[0,1+i,1+(i+1)%n] for i in range(n)]
    for ring in range(11):
        for i in range(n):
            j=(i+1)%n;a=1+ring*n+i;b=1+ring*n+j;c=a+n;d=b+n
            triangles.extend([[a,c,d],[a,d,b]])
    tris=np.array(triangles)
    def area(p):
        a=p[tris[:,1]]-p[tris[:,0]];b=p[tris[:,2]]-p[tris[:,0]]
        return a[:,0]*b[:,1]-a[:,1]*b[:,0]
    reference=area(rest);samples=[]
    for y in np.linspace(-45,45,19):
        for p in np.linspace(-30,30,13):
            observed=predict_radii(silhouette,[y,p,0,0,0,0])
            ratio=observed/front
            xy=rest*np.r_[1,np.tile(ratio,12)][:,None];d=area(xy)/reference
            samples.append([float(y),float(p),float(np.min(d))])
    return dict(domain='polar skin; 12x96 material grid; 19x13 yaw/pitch sweep',points=rest.tolist(),triangles=tris.tolist(),grid=samples,minDetJ=min(s[2] for s in samples),reversed=sum(s[2]<=0 for s in samples))


def measure(data):
    key_errors={}
    for row in data['keys']:
        predicted=predict(data['model'],row['pose']);e=predicted-row['landmarks']
        r=predict_radii(data['silhouette'],row['pose'])-row['radii']
        theta=np.array(data['theta']);rays=np.column_stack((np.cos(theta),np.sin(theta)))
        center=predict_center(data['silhouette'],row['pose'])
        contour_error=center-np.array(row['center'])+rays*r[:,None]
        key_errors[row['name']]=dict(featureRms=float(np.sqrt(np.mean(np.sum(e*e,axis=1)))),contourRms=float(np.sqrt(np.mean(np.sum(contour_error**2,axis=1)))))
    held=data['holdout']['indices']
    errors=[predict(data['model'],data['video'][i]['pose'])-data['video'][i]['landmarks'] for i in held]
    training=[predict(data['model'],data['video'][i]['pose'])-data['video'][i]['landmarks'] for i in data['holdout']['trainingIndices']]
    rest=data['frontConstraint'];front=predict(data['model'],[0,0,0,0,*np.array(rest['landmarks'])[:2].mean(0)])
    trajectory=np.array([r['pose'] for r in data['video']])
    return dict(keyframes=key_errors,frontCanonicalRms=float(np.sqrt(np.mean(np.sum((front-rest['landmarks'])**2,axis=1)))),holdoutFeatureRms=float(np.sqrt(np.mean(np.sum(np.array(errors)**2,axis=2)))),trainingFeatureRms=float(np.sqrt(np.mean(np.sum(np.array(training)**2,axis=2)))),maxAngularStep=float(np.max(np.abs(np.diff(trajectory[:,:3],axis=0)))),field=model_field(data['model'],data['silhouette']))


def assert_registration(data):
    result=measure(data)
    from seedance_references import direction_cue
    from ponytail_observations import NAMES,ANGLES
    assert data['temporal']['velocity']>0 and data['temporal']['acceleration']>0
    assert result['maxAngularStep']<20,result['maxAngularStep']
    directions=dict(zip(NAMES,ANGLES))
    for row in data['keys']:
        assert row['requestedAngles']==list(directions[row['name']])
        cue=direction_cue(row['landmarks'],data['frontConstraint']['landmarks'])
        for target,value,angle in zip(row['requestedAngles'],cue,row['pose'][:2]):
            assert not target or (target*value>0 and target*angle>0),(row['name'],cue,row['pose'][:2])
    assert result['frontCanonicalRms']<=.5,result['frontCanonicalRms']
    for name,row in result['keyframes'].items():
        assert row['featureRms'] <= (.5 if name=='front' else 3.5),(name,row)
        assert row['contourRms']<=6.,(name,row)
    assert result['holdoutFeatureRms']<=11.1,result['holdoutFeatureRms']
    assert result['field']['minDetJ']>=.25,result['field']['minDetJ']
    assert data['field']['minDetJ']>=.25
    assert np.allclose(data['field']['grid'],result['field']['grid'],atol=1e-7)
    assert set(data['holdout']['indices']).isdisjoint(data['holdout']['trainingIndices'])
    return result
