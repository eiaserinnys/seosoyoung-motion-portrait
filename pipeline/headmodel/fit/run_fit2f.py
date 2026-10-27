"""Six-runner 2f/Tripo experiment entry points, without changing 2d artifacts."""
import argparse
import sys
import subprocess
import time
import unittest
import cv2
import numpy as np
import torch
from fit_core import FIT, ROOT, read_json, write_json
from fit2d_core import template, frame_sets
from fit2f_core import (RESULTS, compact_surface, boundary_edges, load_surface,
                        load_targets_f, remesh_surface, surface_distances,
                        solve_surface, curvature_stats, measure_surface)


def context():
    poses=read_json(FIT/'results/fit2d/a-poses.json')['frames']
    shared=read_json(FIT/'results/fit2d/a-shared.json')
    sref=float(np.exp(np.median([p[3] for p in poses.values()])))
    return poses,shared,sref


def save_surface(name,v,tri,**extra):
    RESULTS.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(RESULTS/(name+'.npz'),vertices=v,triangles=tri,
                        open_edges=boundary_edges(tri),**extra)


def prepare():
    started=time.perf_counter(); RESULTS.mkdir(parents=True,exist_ok=True)
    # Installation is inside the preparation resource runner, never prod venv.
    status={}
    for package in ['trimesh','rtree']:
        subprocess.run([sys.executable,'-m','pip','install',package],check=True)
    result=subprocess.run([sys.executable,'-m','pip','install','pymeshlab==2025.7'])
    status['pymeshlab_install_returncode']=result.returncode
    if result.returncode:
        fallback=subprocess.run([sys.executable,'-m','pip','install','pyacvd'])
        status['pyacvd_install_returncode']=fallback.returncode
    cameras,shared,sref=context()
    old=np.load(FIT/'results/fit2d/a-state.npz'); v,tri=old['vertices'],old['triangles']
    _,_,_,masks,_=template()
    tri=tri[masks['head'][tri].all(1)]
    original,original_tri,used=compact_surface(v,tri)
    original_ear=(masks['ear-left']|masks['ear-right'])[used]
    save_surface('surface-original-a',original,original_tri,ear_excluded=original_ear)
    try:
        new,new_tri,tool=remesh_surface(original,original_tri,sref)
    except ImportError:
        if status.get('pyacvd_install_returncode',0)==0: raise
        new,new_tri,tool=original.copy(),original_tri.copy(),'no remesh: both dependency installs failed'
    from scipy.spatial import cKDTree
    new_ear=original_ear[cKDTree(original).query(new)[1]]
    save_surface('surface-a',new,new_tri,ear_excluded=new_ear)
    from fit2f_core import edges
    unique=np.unique(edges(new_tri),axis=0)
    length=np.linalg.norm(new[unique[:,0]]-new[unique[:,1]],axis=1)*sref
    info={'tool':tool,'installation':status,'s_ref':sref,'target_edge_length_px':8.,
          'iterations':10,'original_vertices':len(original),'original_faces':len(original_tri),
          'vertices':len(new),'faces':len(new_tri),'open_edges':len(boundary_edges(new_tri)),
          'edge_length_px':{'median':float(np.median(length)),'p10':float(np.percentile(length,10)),
                            'p90':float(np.percentile(length,90))},
          'distance_px':surface_distances(original,original_tri,new,new_tri,sref)}
    targets=load_targets_f([3,42,264],shared['thickness_px'])
    from scipy.spatial import cKDTree
    from fit_core import boundary, eroded_target
    info['target_collision']=[]
    for target in targets:
        skin=target['points'][target['labels']==0]
        old_distance=cKDTree(boundary(eroded_target(target['head'],shared['thickness_px']))).query(skin)[0]
        new_distance=cKDTree(target['head_points_all']).query(skin)[0]
        info['target_collision'].append({'frame':target['frame'],'before_median_px':float(np.median(old_distance)),
                                         'after_median_px':float(np.median(new_distance)),
                                         'skin_points':len(skin),'ear_excluded_points':len(target['ear_points'])})
    info['elapsed_seconds']=time.perf_counter()-started
    write_json(RESULTS/'remesh.json',info); print('PREPARE',info,flush=True)
    suite=unittest.defaultTestLoader.discover(str(FIT),pattern='test_fit2f_core.py')
    tick=time.perf_counter(); result=unittest.TextTestRunner(verbosity=2).run(suite)
    write_json(RESULTS/'tests.json',{'run':result.testsRun,'failures':[(str(t),e) for t,e in result.failures],
        'errors':[(str(t),e) for t,e in result.errors],'passed':result.wasSuccessful(),
        'elapsed_seconds':time.perf_counter()-tick})
    if not result.wasSuccessful(): print('CONTRACT FAILURES recorded; no parameter changes',flush=True)


def step(number):
    cameras,shared,sref=context(); frames=[3,42,264]
    surface=load_surface('surface-original-a' if number==1 else 'surface-a')
    v,tri=surface['vertices'],surface['triangles']
    targets=load_targets_f(frames,shared['thickness_px'])
    vertices,d,trace,elapsed=solve_surface(v,tri,[cameras[str(f)] for f in frames],targets,sref,number==2)
    stats,curvature,excluded=curvature_stats(v*sref,tri,d,surface['ear_excluded'])
    name=f'b3-step{number}'
    save_surface(name+'-state',vertices,tri,base_vertices=v,displacement_px=d,
                 curvature=curvature,excluded=excluded,s_ref=sref)
    keys=frame_sets()[0][:9]
    rows=measure_surface(vertices,tri,[cameras[str(f)] for f in keys],load_targets_f(keys,shared['thickness_px']))
    magnitude=np.linalg.norm(d,axis=1)
    summary={'name':name,'lambda':300. if number==1 else 1000.,'mu':.01,
             's_ref':sref,'curvature':stats,'icp':trace,'elapsed_seconds':elapsed,'frames':rows,
             'd_px':{'max':float(magnitude.max()),'mean':float(magnitude.mean()),'p95':float(np.percentile(magnitude,95))},
             'note':'재메시 전이라 원본 얼굴 영역 추정 잡음 있음' if number==1 else 'Remeshed cotangent stiffness with barycentric mass'}
    write_json(RESULTS/(name+'-residuals.json'),summary); print('STEP',summary,flush=True)




def refresh_step1_curvature():
    """Only remeasure the existing uniform-L solution; no ICP or rendering."""
    _,_,sref=context(); model=load_surface('b3-step1-state')
    original=load_surface('surface-original-a')
    stats,curvature,excluded=curvature_stats(model['base_vertices']*sref,model['triangles'],
                     model['displacement_px'],original['ear_excluded'])
    model.update(curvature=curvature,excluded=excluded)
    np.savez_compressed(RESULTS/'b3-step1-state.npz',**model)
    report=read_json(RESULTS/'b3-step1-residuals.json'); report['curvature']=stats
    report['operator_correction']='Uniform-L deformation is unchanged; curvature remeasured after edge-weight clamp correction.'
    write_json(RESULTS/'b3-step1-residuals.json',report)
    write_json(RESULTS/'pre-edge-clamp/reason.json',{
        'incorrect':'sum(max(cot contribution,0))/2',
        'specified':'max(sum(cot contribution)/2,0)',
        'observed_contract':'edge opposite cots -1.75 and 2.4: old weight1.2, corrected .325',
        'step1':'displacement unchanged, curvature remeasured',
        'step2':'old deformation and curvature invalid for the specified operator; corrective solve pending',
        'test':'RED weight1.2 != .325; GREEN 1 test .009s, Gaussian bump .02212374877928758 unchanged'})
    print('STEP1 CORRECTED CURVATURE',stats,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('phase',choices=['prepare','step1','step2','refresh-step1-curvature'])
    phase=parser.parse_args().phase
    torch.set_num_threads(2); torch.set_num_interop_threads(1); cv2.setNumThreads(2)
    {'prepare':prepare,'step1':lambda:step(1),'step2':lambda:step(2),
     'refresh-step1-curvature':refresh_step1_curvature}[phase]()
