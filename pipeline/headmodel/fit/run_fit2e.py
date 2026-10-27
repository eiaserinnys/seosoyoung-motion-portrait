"""Bounded 2e v2 stages. K9 is the first mandatory reporting stop."""
import argparse
import hashlib
import time
import unittest
import numpy as np
import cv2
from scipy.spatial import cKDTree
from fit_core import FIT, ROOT, WORKSPACE, read_json, write_json
from fit2f_core import (RESULTS as FRESULTS, load_targets_f, surface_contour, boundary_edges)
import fit2e_core as e

RESULTS=FIT/'results/fit2e'


def context():
    with np.load(FRESULTS/'fit-state-n3.npz') as source:
        model={k:source[k].copy() for k in source.files}
    with np.load(FRESULTS/'surface-a.npz') as source: ear=source['ear_excluded'].copy()
    shared=read_json(FRESULTS/'shared-n3.json')
    poses=read_json(FRESULTS/'poses-n3.json')['frames']
    frames=read_json(ROOT/'pipeline/headmodel/data/frames.json')['frames']
    return model,ear,shared,poses,frames


def prepare():
    started=time.perf_counter(); RESULTS.mkdir(parents=True,exist_ok=True)
    if (RESULTS/'prepare.json').exists(): raise FileExistsError('Preparation already executed')
    required=['fit-state-n3.npz','poses-n3.json','surface-a.npz','shared-n3.json','b3-step2-residuals.json']
    for name in required:
        if not (FRESULTS/name).is_file(): raise FileNotFoundError(FRESULTS/name)
    model,ear,shared,poses,rows=context()
    for key in ['vertices','triangles','base_vertices','displacement_px','excluded','s_ref','open_edges']:
        if key not in model: raise KeyError(key)
    if not np.isfinite(model['vertices']).all(): raise ValueError('Nonfinite V_B')
    np.testing.assert_array_equal(model['open_edges'],boundary_edges(model['triangles']))
    if ear.shape!=(len(model['vertices']),): raise ValueError('Ear exclusion dimensions')
    if len(rows)!=130 or len(poses)!=130: raise ValueError('Expected 130 frames')
    if {r['frame'] for r in rows}!={int(f) for f in poses}: raise ValueError('Frame IDs differ')
    for row in rows:
        p=np.asarray(poses[str(row['frame'])]); assert p.shape==(6,) and np.isfinite(p).all()
        for name in ['head.png','raw/face skin.png','raw/neck.png','raw/ear.png']:
            path=ROOT/f'pipeline/headmodel/data/frame_{row["frame"]:04d}'/name
            image=cv2.imread(str(path),0)
            if image is None or image.shape!=(720,720): raise ValueError(path)
    split=e.split_frames(rows); assert len(split['train'])==99 and len(split['heldout'])==31
    p_ref=read_json(FRESULTS/'b3-step2-residuals.json')['icp'][0]['pairs']
    excluded=e.excluded_vertices_3ring(model['triangles'],len(ear),ear)
    np.savez_compressed(RESULTS/'prepared.npz',excluded=excluded)
    from test_fit2e_core import Contracts
    from test_fit2f_core import Contracts as FContracts
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    for name in ['test_taper','test_skin_conflict_removed','test_open_boundary_not_paired','test_ear_neighborhood_not_paired']:
        suite.addTest(FContracts(name))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'P_ref':int(p_ref),'vertices':len(ear),'faces':len(model['triangles']),
            'open_edges':len(model['open_edges']),'s_ref':float(model['s_ref']),
            'thickness_px':shared['thickness_px'],'excluded_vertices_3ring':int(excluded.sum()),
            'split':split,'tests_run':result.testsRun,'tests_passed':result.wasSuccessful(),
            'input_sha256':{name:hashlib.sha256((FRESULTS/name).read_bytes()).hexdigest() for name in required},
            'elapsed_seconds':time.perf_counter()-started}
    write_json(RESULTS/'prepare.json',report); print('PREPARE',report,flush=True)
    if not result.wasSuccessful(): raise RuntimeError('Preparation contracts failed')


def measure(vertices,tri,pose,target):
    model=surface_contour(vertices,tri,pose); tree=cKDTree(model['xy'])
    row={'frame':target['frame']}
    for label,name in [(0,'skin'),(1,'hair')]:
        points=target['points'][target['labels']==label]
        if len(target['ear_points']): points=points[cKDTree(target['ear_points']).query(points)[0]>10.]
        distances=tree.query(points)[0]; worst=int(distances.argmax())
        row[name]={'median_px':float(np.median(distances)),'max_px':float(distances.max()),
                   'worst_point':points[worst].tolist(),'count':len(points)}
    wanted=np.zeros((720,720),np.uint8)
    cv2.fillPoly(wanted,[np.rint(target['head_points_all']).astype(np.int32)],1)
    row['head_iou']=float((model['head_mask']&(wanted>0)).sum()/(model['head_mask']|(wanted>0)).sum())
    return row


def pose_stage():
    started=time.perf_counter()
    if (RESULTS/'poses-refined.json').exists(): raise FileExistsError('Pose fit already executed')
    prep=read_json(RESULTS/'prepare.json')
    if not prep['tests_passed']: raise ValueError('Preparation failed')
    model,ear,shared,cameras,rows=context()
    frames=[r['frame'] for r in rows]; initial=np.array([cameras[str(f)] for f in frames])
    targets=load_targets_f(frames,shared['thickness_px'])
    before=[measure(model['vertices'],model['triangles'],pose,t) for pose,t in zip(initial,targets)]
    refined,trace=e.refine_poses(model['vertices'],model['triangles'],targets,initial)
    after=[measure(model['vertices'],model['triangles'],pose,t) for pose,t in zip(refined,targets)]
    delta=refined[:,:3]-initial[:,:3]; excluded=[]; entries=[]
    for i,frame in enumerate(frames):
        reasons=[]
        boundary=bool((np.abs(delta[i])>=12.-1e-8).any())
        if after[i]['hair']['median_px']>8.: reasons.append('hair_median_above_8px')
        if boundary: reasons.append('rotation_clamp_boundary')
        if reasons: excluded.append({'frame':frame,'reasons':reasons,'hair_median_px':after[i]['hair']['median_px'],
                                     'rotation_delta_deg':delta[i].tolist()})
        entries.append({'frame':frame,'registered_pose':initial[i].tolist(),'refined_pose':refined[i].tolist(),
                        'delta_deg':delta[i].tolist(),'before':before[i],'after':after[i],
                        'at_clamp_boundary':boundary})
    summary={'refine_rotation':True,'frames':entries,'optimizer':trace,
             'delta_abs_median_deg':float(np.median(np.abs(delta))),
             'delta_abs_max_deg':float(np.abs(delta).max()),
             'axis_abs_median_deg':np.median(np.abs(delta),axis=0).tolist(),
             'boundary_frames':[entry['frame'] for entry in entries if entry['at_clamp_boundary']],
             'pre_hair_worst_18':[entry['frame'] for entry in sorted(entries,key=lambda r:r['before']['hair']['median_px'],reverse=True)[:18]]}
    write_json(RESULTS/'poses-refined.json',summary)
    write_json(RESULTS/'excluded-frames.json',{'frames':excluded,'count':len(excluded),'stop_above_25':len(excluded)>25})
    from fit2e_sheets import pose_sheet
    manifest=pose_sheet(model,targets,entries)
    summary['sheet']=manifest; summary['elapsed_seconds']=time.perf_counter()-started
    write_json(RESULTS/'poses-refined.json',summary)
    print('POSE COMPLETE',{'excluded':len(excluded),'boundary':summary['boundary_frames'],
          'median_abs_delta':summary['delta_abs_median_deg'],'maximum_abs_delta':summary['delta_abs_max_deg'],
          'seconds':summary['elapsed_seconds']},flush=True)
    if len(excluded)>25: raise RuntimeError('More than 25 excluded frames: reporting stop')


def refined_context():
    model,ear,shared,_,rows=context()
    refined=read_json(RESULTS/'poses-refined.json')
    exclusions=read_json(RESULTS/'excluded-frames.json')
    if exclusions['count']>25: raise RuntimeError('Exclusion threshold exceeded')
    excluded={entry['frame'] for entry in exclusions['frames']}
    poses={r['frame']:np.array(r['refined_pose']) for r in refined['frames']}
    split=e.split_frames(rows,excluded)
    return model,ear,shared,poses,split


def noise_stage():
    started=time.perf_counter()
    if (RESULTS/'noise-floor.json').exists(): raise FileExistsError('Noise already measured')
    _,_,shared,poses,split=refined_context(); frames=split['noise']
    targets=load_targets_f(frames,shared['thickness_px']); pairs=[]
    for i,f in enumerate(frames):
        for j in range(i+1,len(frames)):
            g=frames[j]; distance=np.linalg.norm((poses[f][:2]-poses[g][:2])*[1.,1.4])
            if distance>1.: continue
            record={'frames':[f,g],'angle_distance_deg':float(distance)}
            for label,name in [(0,'skin'),(1,'hair')]:
                record[name]=e.symmetric_distance(targets[i]['points'][targets[i]['labels']==label],
                                                  targets[j]['points'][targets[j]['labels']==label])
            pairs.append(record)
    aggregate={name:{key:float(np.median([p[name][key] for p in pairs])) if pairs else None
                    for key in ['median_px','max_px']} for name in ['skin','hair']}
    report={'pairs':pairs,'pair_count':len(pairs),'aggregate':aggregate,
            'frames':frames,'excluded':split['excluded'],'elapsed_seconds':time.perf_counter()-started}
    write_json(RESULTS/'noise-floor.json',report); print('NOISE',report,flush=True)


def aggregate(rows):
    result={}
    for group in ['train','heldout']:
        selected=[r for r in rows if r['set']==group]; result[group]={}
        for name in ['skin','hair']:
            values=[r[name]['median_px'] for r in selected]; maxima=[r[name]['max_px'] for r in selected]
            worst=max(selected,key=lambda r:r[name]['max_px'])
            result[group][name]={'median_px':float(np.median(values)),
                'frame_max_p95_px':float(np.percentile(maxima,95)),
                'top_5pct_frames':sorted([{'frame':r['frame'],'max_px':r[name]['max_px']} for r in selected],
                                         key=lambda r:r['max_px'],reverse=True)[:max(1,int(np.ceil(len(selected)*.05)))],
                'worst_point':{'frame':worst['frame'],'xy':worst[name]['worst_point'],'max_px':worst[name]['max_px']}}
    return result


def k9_stage():
    started=time.perf_counter()
    if (RESULTS/'k9-state.npz').exists(): raise FileExistsError('K9 already executed')
    model,ear,shared,poses,split=refined_context()
    prep=read_json(RESULTS/'prepare.json'); noise=read_json(RESULTS/'noise-floor.json')
    nodes=np.array([poses[f][:2] for f in split['keys']]); self_weights=np.diag(e.phi(nodes,nodes))
    basis_check={'key_frames':split['keys'],'nodes_deg':nodes.tolist(),
                 'self_weights':self_weights.tolist(),'passed':bool((self_weights>=.6).all())}
    write_json(RESULTS/'basis-check.json',basis_check)
    if not basis_check['passed']: raise RuntimeError('Actual corrected key Gaussian self weights below .6: design decision required')
    frames=split['train']; targets=load_targets_f(frames,shared['thickness_px'])
    def failure(history,attempts):
        write_json(RESULTS/'k9-cg-failure.json',{'icp':history,'failed_attempts':attempts})
    d,loads,trace,elapsed=e.solve_field(model['vertices'],model['triangles'],[poses[f] for f in frames],
                        targets,float(model['s_ref']),nodes,prep['P_ref'],failure_callback=failure)
    np.savez_compressed(RESULTS/'k9-state.npz',D_k=d,theta_k=nodes,sigma=18.,s_k=loads['s'],
                        vertices=model['vertices'],triangles=model['triangles'],s_ref=model['s_ref'],
                        excluded=e.excluded_vertices_3ring(model['triangles'],len(ear),ear))
    sref=float(model['s_ref']); base=model['vertices']*sref
    from fit2f_core import geometric_operator
    lap,mass=geometric_operator(base,model['triangles'])
    excluded=e.excluded_vertices_3ring(model['triangles'],len(ear),ear)
    node_stats=[]
    for k,field in enumerate(d):
        stats,_=e.field_stats(lap,mass,field,excluded); magnitude=np.linalg.norm(field,axis=1)
        node_stats.append({'frame':split['keys'][k],'theta_deg':nodes[k].tolist(),'s_k':float(loads['s'][k]),
            'lambda_k':float(loads['lambda'][k]),'mu_k':float(loads['mu'][k]),
            'd_px':{'mean':float(magnitude.mean()),'p95':float(np.percentile(magnitude,95)),'max':float(magnitude.max())},
            'mean_vector_px':field.mean(0).tolist(),'scale_minus_one':e.similarity_scale(base,field),'curvature':stats})
    all_frames=[r['frame'] for r in context()[-1]]
    all_targets=load_targets_f(all_frames,shared['thickness_px']); rows=[]
    registered={r['frame']:r['registered_pose'] for r in read_json(RESULTS/'poses-refined.json')['frames']}
    for frame,target in zip(all_frames,all_targets):
        field=np.einsum('k,kic->ic',e.phi(poses[frame][None,:2],nodes)[0],d)
        row=measure((base+field)/sref,model['triangles'],poses[frame],target)
        row.update(set='train' if frame in split['train'] else 'heldout' if frame in split['heldout'] else 'excluded',
                   registered_rotation_deg=registered[frame][:3],refined_rotation_deg=poses[frame][:3].tolist())
        rows.append(row)
    aggregate_values=aggregate(rows)
    skin_noise=noise['aggregate']['skin']['median_px']; hair_noise=noise['aggregate']['hair']['median_px']
    targets_met={group:bool(aggregate_values[group]['skin']['median_px']<=.5 and aggregate_values[group]['skin']['frame_max_p95_px']<=1.5
                   and aggregate_values[group]['hair']['median_px']<=1. and aggregate_values[group]['hair']['frame_max_p95_px']<=3.)
                 for group in ['train','heldout']}
    report={'K':9,'sigma_deg':18.,'pitch_weight':1.4,'rho':1e4,'P_ref':prep['P_ref'],'nu':loads['nu'].tolist(),
            'frames':rows,'aggregate':aggregate_values,'noise_floor':noise['aggregate'],'noise_pairs':noise['pair_count'],
            'excluded_frames':read_json(RESULTS/'excluded-frames.json')['frames'],'nodes':node_stats,'icp':trace,
            'train_frames':frames,'heldout_frames':split['heldout'],'refine_rotation':True,
            'targets_met':targets_met,'heldout_within_noise_plus_point3':{
                'skin':None if skin_noise is None else aggregate_values['heldout']['skin']['median_px']<=skin_noise+.3,
                'hair':None if hair_noise is None else aggregate_values['heldout']['hair']['median_px']<=hair_noise+.3},
            'fit_seconds':elapsed,'elapsed_seconds':time.perf_counter()-started,
            'insertion_history':[],'sigma_correction':False,'reporting_stop':'K9: no insertion authorized yet',
            'maximum_definition':'p95 of frame pixel maxima; top 5% frame list and global worst point also recorded'}
    write_json(RESULTS/'k9-residuals.json',report)
    print('K9 COMPLETE',{'aggregate':aggregate_values,'seconds':report['elapsed_seconds']},flush=True)


def sheet_stage():
    from fit2e_sheets import k9_sheets
    k9_sheets()


if __name__=='__main__':
    phases={'prepare':prepare,'pose':pose_stage,'noise':noise_stage,'k9':k9_stage,'sheets':sheet_stage}
    parser=argparse.ArgumentParser(); parser.add_argument('phase',choices=phases)
    args=parser.parse_args(); phases[args.phase]()
