"""Frozen-model diagnostics with explicit camera information splits."""
import numpy as np
from scipy.optimize import least_squares
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


def point_split_cameras(model,observations,initial):
    """Fit cameras only from the eyes, nose and chin; all shape is frozen."""
    fit=[0,1,4,6]
    result=[]
    for observation,start in zip(observations,initial):
        solved=least_squares(lambda pose:(predict(model,pose)[fit]-observation[fit]).ravel(),
            start,bounds=([-65,-45,-30,-.6,0,0],[65,45,30,.6,720,720]),
            x_scale=[30,30,10,.1,100,100],max_nfev=200)
        result.append(solved.x)
    return np.array(result)


def diagnostics(data,applied_poses):
    poses=np.array([r['pose'] for r in data['video']])
    held=data['holdout']['indices']
    interpolated=interpolated_holdout(poses,held)
    squared=[np.sum((predict(data['model'],interpolated[i])-data['video'][i]['landmarks'])**2,axis=1) for i in held]
    observations=np.array([data['video'][i]['landmarks'] for i in held])
    # Initialization uses adjacent training frames, never the old nine-point
    # held-out camera estimate or any of the five evaluation landmarks.
    cameras=point_split_cameras(data['model'],observations,interpolated[held])
    point_split=np.array([predict(data['model'],p) for p in cameras])[:,[2,3,5,7,8]]-observations[:,[2,3,5,7,8]]
    step=np.r_[0,np.max(np.abs(np.diff(poses[:,:3],axis=0)),axis=1)]
    peak=int(np.argmax(step))
    applied={(p['yaw'],p['pitch']):p for p in applied_poses}
    return dict(holdoutInterpolatedPoseFeatureRms=float(np.sqrt(np.mean(squared))),
                holdoutPointSplitFeatureRms=float(np.sqrt(np.mean(np.sum(point_split**2,axis=2)))),
                holdoutPointSplitProtocol=dict(cameraLandmarks=[0,1,4,6],evaluationLandmarks=[2,3,5,7,8],frames=len(held),
                    initialization='Adjacent training camera interpolation; independent four-point camera least squares; frozen shape'),
                holdoutConditionalPoseFeatureRms=float(data['metrics']['holdoutFeatureRms']),
                maxAngularStepWindow=[dict(frameIndex=i+1,yaw=float(poses[i,0]),pitch=float(poses[i,1]),roll=float(poses[i,2]),stepDeg=float(step[i])) for i in range(max(0,peak-5),min(len(poses),peak+6))],
                achievedAngleRange=[dict(name=k['name'],requestedAngles=k['requestedAngles'],
                    sourceYaw=k['pose'][0],sourcePitch=k['pose'][1],
                    appliedYaw=applied[tuple(k['requestedAngles'])]['sourceYaw'],
                    appliedPitch=applied[tuple(k['requestedAngles'])]['sourcePitch']) for k in data['keys']])
