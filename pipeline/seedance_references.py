"""Select sharp training-only representatives; normalize camera similarity, never facial shape."""
import copy
from pathlib import Path
import cv2
import numpy as np
from ponytail_observations import ROOT,NAMES,ANGLES,digest,similarity
from ponytail_model import block_holdout,initial_poses,roll_matrix


def direction_cue(points,front):
    def values(p):
        p=np.array(p);eye=p[:2].mean(0);u=p[1]-p[0];distance=np.linalg.norm(u);u=u/distance;v=np.array([-u[1],u[0]])
        # The nose-to-mouth ratio changes under yaw in this illustrated face.
        # Eye height relative to the two ear/side controls resolves nod direction.
        return np.dot(p[4]-eye,u)/distance,np.dot(p[7:9].mean(0)-eye,v)/distance
    x,y=values(points);fx,fy=values(front)
    return np.array([x-fx,y-fy])


def representatives(inputs,canonical,calibration,video,trajectory=None):
    front=np.array(canonical['landmarks']);origin=front[:2].mean(0)
    X=np.column_stack((front-origin,[0,0,-15,-15,65,35,0,-50,-50]))
    poses=initial_poses(video,calibration,X) if trajectory is None else np.array(trajectory)
    holdout=block_holdout(len(video));chosen=set();keys=[]
    target=ROOT/'source/references/seedance_9dir';target.mkdir(exist_ok=True)
    for name,(yaw,pitch) in zip(NAMES,ANGLES):
        distance=((poses[:,0]-yaw)/45)**2+((poses[:,1]-pitch)/30)**2
        if name=='front' and trajectory is not None:
            # A frontal return can be sharp while its small mouth patch has
            # drifted. Prefer actual neutral feature consistency within the
            # final near-frontal trajectory, not an almost-zero camera alone.
            for i,row in enumerate(video):
                a,b=similarity(np.array(row['landmarks'])[:2],front[:2])
                residual=np.array(row['landmarks'])[:6]@a.T+b-front[:6]
                distance[i]+=np.mean(np.sum(residual**2,axis=1))/25
        cues=np.array([direction_cue(row['landmarks'],front) for row in video])
        candidates=[int(i) for i in np.argsort(distance) if not holdout[i] and i not in chosen and
                    (trajectory is None or ((not yaw or (cues[i,0]*yaw>0 and poses[i,0]*yaw>0)) and
                                            (not pitch or (cues[i,1]*pitch>0 and poses[i,1]*pitch>0))))][:8]
        if not candidates:raise ValueError('No observed frame supports direction '+name)
        records=[]
        for i in candidates:
            path=inputs/'frames'/('frame_'+str(i+1).zfill(4)+'.png')
            rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
            rgb=cv2.resize(rgb,(720,720),interpolation=cv2.INTER_CUBIC)
            sharp=float(cv2.Laplacian(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),cv2.CV_64F).var())
            records.append((distance[i]+.005/(1+sharp/100),i,rgb,sharp,path))
        _,i,rgb,sharp,path=min(records,key=lambda r:r[0]);i=int(i);chosen.add(i)
        p=poses[i];linear=roll_matrix(p[2]).T/np.exp(p[3]);shift=origin-linear@p[4:6]
        affine=np.column_stack((linear,shift))
        normalized=cv2.warpAffine(rgb,affine,(720,720),flags=cv2.INTER_CUBIC,borderValue=(255,255,255))
        filename=name.replace('-','_')+'.png'
        cv2.imwrite(str(target/filename),cv2.cvtColor(normalized,cv2.COLOR_RGB2BGR))
        row=copy.deepcopy(video[i])
        for key in ['landmarks','contour']:
            row[key]=(np.array(row[key])@linear.T+shift).tolist()
        row['center']=(np.array(row['center'])@linear.T+shift).tolist()
        row['earContours']=[(np.array(arc)@linear.T+shift).tolist() if arc else [] for arc in row['earContours']]
        theta=np.linspace(-np.pi,np.pi,len(row['radii']),endpoint=False)
        angle=(theta+np.radians(p[2])+np.pi)%(2*np.pi)-np.pi
        row['radii']=(np.interp(angle,np.r_[theta,np.pi],np.r_[row['radii'],row['radii'][0]])/np.exp(p[3])).tolist()
        row.update(name=name,yaw=float(p[0]),pitch=float(p[1]),requestedAngles=[yaw,pitch],sourceFrame=i+1,
                   sourceSha256=digest(path),image='source/references/seedance_9dir/'+filename,
                   normalization=affine.tolist(),sharpness=sharp,source='seedance')
        row['directionCue']=cues[i].tolist()
        row['normalizedSha256']=digest(target/filename)
        keys.append(row)
    for row,p in zip(video,poses):row['initialAngles']=p[:2].tolist()
    # The input front is a separate exact rest constraint, not a substituted
    # video observation. The nine saved representatives keep their measured pose.
    rest=copy.deepcopy(canonical);rest['name']='canonical-rest';rest['source']='canonical front input'
    rest['image']='source/references/ponytail/ponytail_front.png'
    rest['yaw']=0.;rest['pitch']=0.
    return rest,keys
