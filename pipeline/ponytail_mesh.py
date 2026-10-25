"""Constrained upper-skin integration; lower measured rim and facial anchors stay fixed."""
import json
import numpy as np
from ponytail_observations import ROOT, digest, similarity
from ponytail_model import assert_registration


def registration_matrix(row,feature):
    source=np.array(row['landmarks'])[:2]
    target=np.array([feature['centers'][k] for k in ['left','right']])
    linear,shift=similarity(source,target)
    return np.column_stack((linear,shift))


def registered(row,feature):
    a=registration_matrix(row,feature)
    return np.array(row['contour'])@a[:,:2].T+a[:,2]


def intersect(contour,y):
    a=contour;b=np.roll(contour,-1,axis=0)
    valid=(np.minimum(a[:,1],b[:,1])<=y)&(np.maximum(a[:,1],b[:,1])>=y)&(np.abs(a[:,1]-b[:,1])>1e-7)
    xs=a[valid,0]+(y-a[valid,1])*(b[valid,0]-a[valid,0])/(b[valid,1]-a[valid,1])
    if len(xs)<2:return None
    return [float(xs.min()),float(xs.max())]


def integrate(result,inputs,lap):
    from optimize_3d_2d import projection, determinants
    path=ROOT/'source/viewer/ponytail-seedance-registration.json'
    registration=json.loads(path.read_text())
    assert_registration(registration)
    features=json.loads((ROOT/'source/viewer/face-feature-fit.json').read_text())
    points=np.array(result['points']);tris=result['triangles'];anchors=list(result['anchors'].values())
    # Only the previously unobserved upper skin is free. Lower artwork, feature
    # controls, neutral yaw and all hair attachment evaluators are unchanged.
    fixed=points[:,1]>=410
    fixed[anchors]=True
    boundary=set(i for s in result['samples'] if s['y']<410 for i in [s['left'],s['right']])
    # Include the upper material hull, not just the extra query rim nodes.
    counts={}
    for tri in tris:
        for a,b in zip(tri,np.roll(tri,-1)):
            edge=tuple(sorted((int(a),int(b))));counts[edge]=counts.get(edge,0)+1
    boundary.update(i for edge,n in counts.items() if n==1 for i in edge if points[i,1]<410)
    boundary=np.array(sorted(boundary),int)
    hold=fixed.copy();hold[boundary]=True;free=~hold
    matrix=lap+.01*np.eye(len(points))
    targets=[];metadata=[]
    for pose in result['poses']:
        a,b=projection(points,pose['yaw'],pose['pitch'])
        old=a+np.array(result['Z'])[:,None]*b+pose['t']+pose['du']
        delta=np.zeros_like(old)
        key=next(k for k in registration['keys'] if k['name']==pose['name'])
        feature=next(f for f in features['poses'] if f['name']==pose['name'])
        contour=registered(key,feature)
        if pose['pitch']:
            for i in boundary:
                row=intersect(contour,old[i,1])
                if row is None:continue
                y=points[i,1]
                near=old[np.abs(points[:,1]-y)<3,0]
                lo,hi=float(near.min()),float(near.max())
                u=np.clip((old[i,0]-lo)/max(1,hi-lo),0,1)
                # Fade into the measured old cheek rim; the material boundary
                # owns the top/temple while the old measured jaw owns y>=410.
                fade=np.clip((410-y)/40,0,1)
                delta[i,0]=(row[0]+u*(row[1]-row[0])-old[i,0])*fade
            if np.any(free):
                delta[free]=np.linalg.solve(matrix[np.ix_(free,free)],-matrix[np.ix_(free,hold)]@delta[hold])
            # A convex step preserves the actual 673-triangle orientation,
            # rather than projecting measured features or changing topology.
            step=1.
            while determinants(points,old+step*delta,tris).min()<.0034 and step>1/1024:
                step*=.5
            delta*=step
        else:step=0.
        actual=old+delta
        pose['du']=(np.array(pose['du'])+delta).tolist()
        displacement=np.array(pose['du'])
        pose['stats']['duRms']=float(np.sqrt(np.mean(np.sum(displacement**2,axis=1))))
        pose['stats']['duMax']=float(np.linalg.norm(displacement,axis=1).max())
        det=determinants(points,actual,tris)
        pose['stats']['minDetJ']=float(det.min());pose['stats']['reversedTriangles']=int(np.count_nonzero(det<=0))
        targets.append(dict(name=pose['name'],upper={str(i):actual[i].tolist() for i in range(len(points)) if points[i,1]<410 and i not in anchors}))
        metadata.append(dict(name=pose['name'],step=step,maxDisplacement=float(np.linalg.norm(delta,axis=1).max()),minDetJ=float(det.min())))
    traces=json.loads((ROOT/'source/viewer/outline-traces.json').read_text())
    traces['ponytail']=dict(source='ponytail-seedance-registration.json',keys=[dict(name=k['name'],contour=k['contour'],earContours=k['earContours']) for k in registration['keys']],meshTargets=targets)
    (ROOT/'source/viewer/outline-traces.json').write_text(json.dumps(traces,ensure_ascii=False,indent=2)+'\n')
    result['sources']['outlineTraces']=digest(ROOT/'source/viewer/outline-traces.json')
    shell=registration['silhouette'];indices=[0,24,48]
    result['ponytail']=dict(registrationSha256=digest(path),poses=metadata,
        continuous=dict(front=[shell['front'][i] for i in indices],depth=[shell['depth'][i] for i in indices],
                        residual=np.array(shell['residual'])[:,indices].tolist()))
    return result
