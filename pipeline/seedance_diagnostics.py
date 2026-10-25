"""Report frozen-fit diagnostics without estimating any held-out camera."""
import numpy as np
from ponytail_model import predict


def interpolated_holdout(poses,indices):
    result=np.array(poses,float).copy()
    indices=np.array(sorted(indices),int)
    blocks=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
    for block in blocks:
        before,after=int(block[0]-1),int(block[-1]+1)
        for i in block:
            t=(i-before)/(after-before)
            result[i]=(1-t)*result[before]+t*result[after]
    return result


def diagnostics(data,applied_poses):
    poses=np.array([r['pose'] for r in data['video']])
    held=data['holdout']['indices']
    interpolated=interpolated_holdout(poses,held)
    squared=[np.sum((predict(data['model'],interpolated[i])-data['video'][i]['landmarks'])**2,axis=1) for i in held]
    step=np.r_[0,np.max(np.abs(np.diff(poses[:,:3],axis=0)),axis=1)]
    peak=int(np.argmax(step))
    applied={(p['yaw'],p['pitch']):p for p in applied_poses}
    return dict(holdoutInterpolatedPoseFeatureRms=float(np.sqrt(np.mean(squared))),
                holdoutConditionalPoseFeatureRms=float(data['metrics']['holdoutFeatureRms']),
                maxAngularStepWindow=[dict(frameIndex=i+1,yaw=float(poses[i,0]),pitch=float(poses[i,1]),roll=float(poses[i,2]),stepDeg=float(step[i])) for i in range(max(0,peak-5),min(len(poses),peak+6))],
                achievedAngleRange=[dict(name=k['name'],requestedAngles=k['requestedAngles'],
                    sourceYaw=k['pose'][0],sourcePitch=k['pose'][1],
                    appliedYaw=applied[tuple(k['requestedAngles'])]['sourceYaw'],
                    appliedPitch=applied[tuple(k['requestedAngles'])]['sourcePitch']) for k in data['keys']])
