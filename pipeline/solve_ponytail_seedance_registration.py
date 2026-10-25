"""Reproducible ponytail references + 361-frame Seedance registration."""
import argparse
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from ponytail_observations import ROOT, FEATURES, THETA, seedance_bank, track_video, digest
from ponytail_model import fit_layers, fit_silhouette, predict, model_field, measure,fit_continuous_shapes,infer_holdout,assert_registration
from seedance_temporal import TEMPORAL,normalized_key_pose
from seedance_references import representatives


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--observations',type=Path)
    args=parser.parse_args()
    cv2.setNumThreads(1);cv2.setRNGSeed(42)
    target=ROOT/'source/references/ponytail';target.mkdir(exist_ok=True)
    shutil.copyfile(args.inputs/'front_ponytail_canonical.png',target/'ponytail_front.png')
    cache=ROOT/'debug/ponytail-observations.json'
    canonical,calibration,bank=seedance_bank(args.inputs)
    if args.observations:
        observations=json.loads(args.observations.read_text());video=observations['video']
    else:
        video=track_video(args.inputs,bank)
        cache.write_text(json.dumps(dict(keys=calibration,canonical=canonical,video=video),ensure_ascii=False)+'\n')
    rest,keys=representatives(args.inputs,canonical,calibration,video)
    fitting_keys=[rest,*keys]
    model,poses,holdout=fit_layers(fitting_keys,video)
    trajectory=poses[len(fitting_keys):].copy()
    (ROOT/'debug/seedance-temporal-initial.json').write_text(json.dumps(dict(model=model,trajectory=trajectory.tolist(),holdout=holdout.tolist()),ensure_ascii=False)+'\n')
    rest,keys=representatives(args.inputs,canonical,calibration,video,trajectory=trajectory)
    fitting_keys=[rest,*keys]
    origin=np.array(rest['landmarks'])[:2].mean(0)
    key_poses=[[0,0,0,0,*origin]]+[normalized_key_pose(trajectory[k['sourceFrame']-1],k['normalization']) for k in keys]
    poses=np.vstack((key_poses,trajectory))
    observations=np.array([r['landmarks'] for r in fitting_keys+video])
    model,poses=fit_continuous_shapes(fitting_keys,observations,np.array(model['points3D']),poses,~holdout)
    poses=infer_holdout(model,observations,poses,len(fitting_keys),holdout)
    (ROOT/'debug/ponytail-pose-fit.json').write_text(json.dumps(dict(model=model,poses=poses.tolist(),holdout=holdout.tolist()),ensure_ascii=False)+'\n')
    write_registration(args.inputs,rest,keys,video,model,poses,holdout)


def write_registration(inputs,rest,keys,video,model,poses,holdout):
    fitting_keys=[rest,*keys]
    silhouette=fit_silhouette(fitting_keys,video,poses,holdout)
    for i,row in enumerate(fitting_keys+video):
        row.pop('confidence',None)
        row['pose']=poses[i].tolist();row['prediction']=predict(model,poses[i]).tolist()
    data=dict(schema=1,landmarkNames=FEATURES,theta=THETA.tolist(),weights=dict(keyframe=10.,video=1.),
              frontConstraint=rest,keys=keys,video=video,model=model,silhouette=silhouette,field=model_field(model,silhouette),temporal=TEMPORAL,
              holdout=dict(indices=np.flatnonzero(holdout).tolist(),trainingIndices=np.flatnonzero(~holdout).tolist(),
                           poseInference='Camera inferred from all nine tracked points; all held-out frames excluded from shape/contour fitting; conditional shape CV, not future-frame forecasting'),
              sources=dict(video=digest(inputs/'seedance_ref_9dir_sweep_15s_480p.mp4'),
                           seedManifest=digest(ROOT/'source/viewer/seedance-reference-frames.json'),
                           featureFit=digest(ROOT/'source/viewer/face-feature-fit.json')))
    data['metrics']=measure(data)
    print(json.dumps(dict(keyframes=data['metrics']['keyframes'],holdoutFeatureRms=data['metrics']['holdoutFeatureRms'],minDetJ=data['field']['minDetJ']),ensure_ascii=False,indent=2))
    try:assert_registration(data)
    except AssertionError:
        (ROOT/'debug/seedance-rejected-registration.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'))+'\n')
        raise
    (ROOT/'source/viewer/ponytail-seedance-registration.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'))+'\n')

if __name__=='__main__':main()
