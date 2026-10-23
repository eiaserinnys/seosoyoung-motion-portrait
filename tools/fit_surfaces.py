"""Fit candidate depth surfaces on cardinal poses; score unseen diagonals/intermediates.
No scipy required. Deterministic bounded coordinate search, NOT a global optimum proof.
All distances are normalized by neutral face width. Inputs and output remain local.
"""
import gzip,json,sys
from pathlib import Path
import numpy as np

TRAIN=[(-30,0),(30,0),(0,-30),(0,30)]
TEST=[(x,y) for x in [-30,-15,0,15,30] for y in [-30,-15,0,15,30] if (x,y) not in TRAIN+[(0,0)]]
SHAPES=['plane','sphere','cylinder','ellipsoid','superellipse4','ellipsoid_relief']


def depth(points, labels, shape, p):
    x,y=(points-p[:2]).T
    if shape=='plane':return np.full(len(points),p[4])
    if shape=='sphere':return np.sqrt(np.maximum(0,p[4]**2-x*x-y*y))
    rx,rz=p[4:6]
    if shape=='cylinder':return rz*np.sqrt(np.maximum(0,1-(x/rx)**2))
    ry=p[6]; power=4 if shape=='superellipse4' else 2
    z=rz*np.sqrt(np.maximum(0,1-np.abs(x/rx)**power-np.abs(y/ry)**power))
    if shape=='ellipsoid_relief':
        for k,label in enumerate(['nose','mouth','eyeballs_left','eyeballs_right']):z=z+(labels==label)*p[7+(min(k,2))]
    return z


def predict(points,labels,shape,p,angles):
    local=points-p[:2];x,y=local.T;z=depth(points,labels,shape,p)
    a,b=np.radians(np.asarray(angles,float)*p[2:4]).T
    X=x[None,:]*np.cos(a[:,None])+z[None,:]*np.sin(a[:,None])
    Z=-x[None,:]*np.sin(a[:,None])+z[None,:]*np.cos(a[:,None])
    Y=y[None,:]*np.cos(b[:,None])-Z*np.sin(b[:,None])
    return np.stack([X,Y],axis=-1)+p[:2]


def setup(shape,height):
    # Common center and parameter-to-physical-angle calibration for every family.
    initial=[0,0,.65,.65]; lower=[-.15,-.25,.2,.2];upper=[.15,.25,1.5,1.5]
    if shape=='plane': initial+=[.4];lower+=[.01];upper+=[1.5]
    elif shape=='sphere':initial+=[max(.55,height*.55)];lower+=[.3];upper+=[1.5]
    elif shape=='cylinder':initial+=[.6,.4];lower+=[.3,.01];upper+=[1.5,1.5]
    else:
        initial+=[.6,.4,height*.6];lower+=[.3,.01,.3];upper+=[1.5,1.5,2.0]
        if shape=='ellipsoid_relief':initial +=[.05,.02,.02];lower +=[-.2,-.2,-.2];upper +=[.4,.4,.4]
    return np.array(initial),np.array(lower),np.array(upper)


def fit(points,labels,observations,weights,shape,height,calibrate=True):
    initial,lower,upper=setup(shape,height)
    if not calibrate:initial[2:4]=lower[2:4]=upper[2:4]=1
    def loss(p):
        delta=predict(points,labels,shape,p,TRAIN)-observations
        return float(np.mean(np.sum(delta*delta,axis=-1)@weights))
    candidates=[]
    for scale in [.4,.75,1.1]:
        p=initial.copy();p[2:4]=scale if calibrate else 1;score=loss(p);steps=(upper-lower)*.12;evaluations=1
        for iteration in range(180):
            improved=False
            for k in range(len(p)):
                best=p;value=score
                for sign in [-1,1]:
                    trial=p.copy();trial[k]=np.clip(trial[k]+sign*steps[k],lower[k],upper[k]);val=loss(trial);evaluations+=1
                    if val<value:best,value=trial,val
                if value<score-1e-14:p,score=best,value;improved=True
            if not improved:steps*=.5
            if steps.max()<2e-5:break
        candidates.append((score,p,evaluations))
    score,p,_=min(candidates,key=lambda v:v[0]);return p,score,[v[0]**.5 for v in candidates]


def run(samples,config,calibrate=True):
    ids=samples['meshIds'];frames=samples['poses']
    def pts(p,i):return np.array(frames[p]['vertices'][i]).reshape(-1,2)*[1,-1]
    face=pts('neutral',ids.index(config['face']));lo=face.min(axis=0);hi=face.max(axis=0);width=hi[0]-lo[0];center=(lo+hi)/2;height=(hi[1]-lo[1])/width
    scopes={'skin_only':['skin'], 'face_and_features':['skin','nose','mouth','eyes_left','eyes_right','eyeballs_left','eyeballs_right','ears']}
    report={'method':'three-start bounded coordinate search; cardinal training only; equal weight per semantic region', 'train':TRAIN,'test':TEST,'angle_calibration':calibrate,'normalization':'neutral face width','scopes':{}}
    for scope,regions in scopes.items():
        points=[];labels=[];targets={pose:[] for pose in TRAIN+TEST}
        for region in regions:
            idx=[ids.index(id) for id in config['regions'][region]]
            a=np.concatenate([pts('neutral',i) for i in idx]);points.extend((a-center)/width);labels.extend([region]*len(a))
            for pose in targets:targets[pose].extend((np.concatenate([pts(f'xy_{pose[0]}_{pose[1]}',i) for i in idx])-center)/width)
        points=np.array(points);labels=np.array(labels);weights=np.zeros(len(points))
        # Fit a deterministic spatial/mesh-order sample; all vertices are scored afterwards.
        chosen=[]
        for region in regions:
            ix=np.flatnonzero(labels==region);weights[ix]=1/(len(regions)*len(ix));chosen.extend(ix[np.linspace(0,len(ix)-1,min(180,len(ix))).astype(int)])
        chosen=np.array(chosen);fitweights=np.zeros(len(chosen))
        for region in regions:
            mask=labels[chosen]==region;fitweights[mask]=1/(len(regions)*mask.sum())
        train=np.array([targets[p] for p in TRAIN]);test=np.array([targets[p] for p in TEST]);scores={}
        for shape in SHAPES:
            if scope=='skin_only' and shape=='ellipsoid_relief':continue
            p,_,starts=fit(points[chosen],labels[chosen],train[:,chosen],fitweights,shape,height,calibrate)
            prediction=predict(points,labels,shape,p,TEST);delta=prediction-test
            train_delta=predict(points,labels,shape,p,TRAIN)-train
            per_pose=np.sqrt(np.sum(delta*delta,axis=-1)@weights)
            by_region={r:float(np.sqrt(np.mean(np.sum(delta[:,labels==r]**2,axis=-1)))) for r in regions}
            scores[shape]={'parameters':p.tolist(),'parameter_count':len(p),'train_rms':float(np.sqrt(np.mean(np.sum(train_delta*train_delta,axis=-1)@weights))), 'heldout_rms':float(np.sqrt(np.mean(per_pose**2))), 'worst_pose_rms':float(per_pose.max()),'per_region_rms':by_region,'start_train_rms':starts, 'heldout_pose_rms':per_pose.tolist()}
            print(scope,shape,'train',round(scores[shape]['train_rms'],5),'heldout',round(scores[shape]['heldout_rms'],5),flush=True)
        report['scopes'][scope]=scores
    return report

if __name__=='__main__':
    sf,cf,out=sys.argv[1:4];s=json.load(gzip.open(sf,'rt'));config=json.loads(Path(cf).read_text());r=run(s,config,'--fixed-angles' not in sys.argv);Path(out).write_text(json.dumps(r,ensure_ascii=False,indent=2))
