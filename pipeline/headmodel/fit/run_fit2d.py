#!/usr/bin/env python3
"""Bounded 2d experiment; stop at camera, global-shape and B3 review gates."""
import argparse
import time
import cv2
import numpy as np
import torch
from fit_core import load_targets, write_json, read_json
from fit2d_core import RESULTS, pose_affine, frame_sets, template, global_fit, fit_cameras
from fit2d_sheets import init_sheet, shape_sheet, center_checks
from run_falsify import initial_poses


def init():
    started=time.perf_counter()
    a33,allframes,six=frame_sets()
    v,tri,_,_,faces=template()
    # Shared camera gauge is anchored on front key 3 for every frame set.
    selected=[3]+six
    targets=load_targets(selected,pose_affine)
    poses,gauge=initial_poses(v,tri,faces,targets)
    checks=init_sheet(v,tri,faces,poses[1:],targets[1:])
    passed=all(c['center_error_px']<=30 for c in checks)
    write_json(RESULTS/'init.json',{'frames_a33':a33,'frames_all':allframes,
               'sample_checks':checks,'camera_gauge':gauge,'passed':passed,
               'elapsed_seconds':time.perf_counter()-started})
    print('INIT',checks,'passed',passed,flush=True)
    if not passed: raise SystemExit('STOP: initial head center exceeds 30px')


def fit_a():
    started=time.perf_counter()
    a33,allframes,six=frame_sets()
    v,tri,modes,masks,faces=template()
    targets=load_targets(a33,pose_affine)
    poses,gauge=initial_poses(v,tri,faces,targets)
    vertices,poses,shared=global_fit(v,tri,modes,faces,targets,poses)
    # Save the global solution before the fixed-shape camera extension.
    np.savez_compressed(RESULTS/'a-state.npz',vertices=vertices,triangles=tri)
    shared['camera_gauge']=gauge
    shared['initial_camera_review']={'commit':'0be6d4fa506820e4714e50b1bb6c1c43bd72b682',
                                    'assignee_verdict':'passed; proceed to A'}
    write_json(RESULTS/'a-shared.json',shared)
    write_json(RESULTS/'a33-poses.json',{'frames':{str(f):p.tolist() for f,p in zip(a33,poses)}})
    del targets
    targets=load_targets([3]+[f for f in allframes if f!=3],pose_affine)
    all_initial,_=initial_poses(v,tri,faces,targets)
    fitted={f:p for f,p in zip(a33,poses)}
    for i,target in enumerate(targets):
        if target['frame'] in fitted: all_initial[i]=fitted[target['frame']]
    final,trace=fit_cameras(vertices,tri,faces,targets,all_initial,shared['thickness_px'])
    cameras={t['frame']:p for t,p in zip(targets,final)}
    write_json(RESULTS/'a-poses.json',{'frames':{str(f):cameras[f].tolist() for f in allframes},
                                    'trace':trace})
    six_targets=[next(t for t in targets if t['frame']==f) for f in six]
    shared['sample_center_checks']=center_checks(vertices,tri,faces,[cameras[f] for f in six],six_targets)
    shared['camera_extension']=trace
    shared['total_elapsed_seconds']=time.perf_counter()-started
    axes=np.asarray(shared['scale_xyz'])
    shared['passed']=bool(((axes>=.9)&(axes<=1.25)).all() and shared['beta_at_limit']<=3 and
                          all(c['center_error_px']<=30 for c in shared['sample_center_checks']))
    write_json(RESULTS/'a-shared.json',shared)
    shape_sheet([('EGG V5',v),('A GLOBAL',vertices)],tri,faces['head'],'fit2d-shape-a.png')
    print('A GATE',shared,flush=True)
    if not shared['passed']: raise SystemExit('STOP: A shape or camera-center gate failed')


def fit_b3():
    from fit2d_core import prepare_b_targets, solve_icp, measure_b
    from fit2d_sheets import b3_sheets
    started=time.perf_counter()
    a=read_json(RESULTS/'a-shared.json')
    if not a['passed']: raise SystemExit('STOP: A gate failed; B3 is prohibited')
    v,tri,_,masks,faces=template()
    baseline=np.load(RESULTS/'a-state.npz')['vertices']
    cameras=read_json(RESULTS/'a-poses.json')['frames']
    sref=float(np.exp(np.median([p[3] for p in cameras.values()])))
    frames=(3,42,264)
    targets=prepare_b_targets(load_targets(frames,pose_affine),a['thickness_px'])
    poses=np.array([cameras[str(f)] for f in frames])
    result,trace=solve_icp(baseline,tri,masks,faces,poses,targets,sref,a['thickness_px'])
    displacement=(result-baseline)*sref; magnitude=np.linalg.norm(displacement,axis=1)
    np.savez_compressed(RESULTS/'b3-state.npz',vertices=result,triangles=tri,
                        displacement_px=displacement,s_ref=sref)
    keys=[r['frame'] for r in read_json(RESULTS.parents[2]/'data/frames.json')['frames'] if r['role']=='key']
    evaluation=prepare_b_targets(load_targets(keys,pose_affine),a['thickness_px'])
    rows=measure_b(result,tri,masks,faces,np.array([cameras[str(f)] for f in keys]),evaluation)
    shared={key:a[key] for key in ('beta','scale_xyz','thickness_px','beta_at_limit')}
    shared.update(lambda_value=300.,mu=.01,s_ref=sref,icp=trace,
        displacement_stats_px={'max':float(magnitude.max()),'mean':float(magnitude.mean()),
                               'p95':float(np.percentile(magnitude,95))},
        neck_displacement_finite=bool(np.isfinite(displacement[~masks['head']]).all()),
        elapsed_seconds=time.perf_counter()-started,
        target_spacing_px=1,fit_frames=list(frames),evaluation_frames=keys,
        adoption_reason='Assignee final directive: lambda 300 only; no lambda correction/search')
    write_json(RESULTS/'b3-shared.json',shared)
    worst=sorted(rows,key=lambda r:r['face']['max_px'],reverse=True)[:int(np.ceil(len(rows)*.1))]
    write_json(RESULTS/'b3-residuals.json',{'frames':rows,
        'top_10_percent_frames_by_face_max':[r['frame'] for r in worst],
        'head_iou_min':min(r['head_iou'] for r in rows),
        'head_iou_median':float(np.median([r['head_iou'] for r in rows])),
        'notes':['Nine-key evaluation; shared displacement fitted only to 3/42/264.',
                 'Face targets use classified pixels only; ears excluded at the contour pixel and outward probe.',
                 'Mask jitter is a spatial staircase estimate; temporal jitter was not measured.']})
    shape_sheet([('EGG V5',v),('A GLOBAL',baseline),('B3 LAMBDA 300',result)],tri,faces['head'],'fit2d-shape-b3.png')
    b3_sheets(result,tri,masks,faces,poses,targets)
    print('B3 FINAL',shared,rows,flush=True)


def diagnose_b3():
    """Split saved-state contour residual directions; no fitting or geometry edit."""
    from scipy.spatial import cKDTree
    from fit_core import boundary
    from fit2d_core import prepare_b_targets, model_contours
    _,tri,_,masks,faces=template()
    vertices=np.load(RESULTS/'b3-state.npz')['vertices']
    a=read_json(RESULTS/'a-shared.json')
    cameras=read_json(RESULTS/'a-poses.json')['frames']
    keys=read_json(RESULTS/'b3-shared.json')['evaluation_frames']
    targets=prepare_b_targets(load_targets(keys,pose_affine),a['thickness_px'])
    def directed(points,wanted,used=None):
        distance=cKDTree(wanted).query(points)[0]
        worst=np.argsort(distance)[-3:][::-1]
        return {'median_px':float(np.median(distance)),'max_px':float(distance.max()),
                'worst':[{'xy':points[i].tolist(),'distance_px':float(distance[i]),
                          **({'nearest_full_boundary_is_used':bool(used[i])} if used is not None else {})}
                         for i in worst]}
    rows=[]
    for target in targets:
        contours=model_contours(vertices,tri,masks,faces,cameras[str(target['frame'])])
        row={'frame':target['frame'],'contours':{}}
        pairs=[('face',target['face_points']),('head',target['head_boundary'])]
        pairs += [(key,boundary(mask)) for key,mask in target['ears'].items()]
        for key,wanted in pairs:
            points=contours[key]['xy']; used=None
            if key=='face':
                nearest=cKDTree(target['face_all_points']).query(points)[1]
                used=target['face_keep'][nearest]
            row['contours'][key]={'target_to_model':directed(wanted,points),
                                  'model_to_target':directed(points,wanted,used)}
        rows.append(row)
    write_json(RESULTS/'b3-diagnostics.json',{'frames':rows,
        'note':'Separate directions explain raw symmetric maxima. Original fit, metrics and geometry are unchanged.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['init','a','b3','b3-diagnostics'])
    args=parser.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1); cv2.setNumThreads(2)
    {'init':init,'a':fit_a,'b3':fit_b3,'b3-diagnostics':diagnose_b3}[args.phase]()
