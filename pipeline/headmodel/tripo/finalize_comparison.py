"""Approved final runner: six-model comparison, evidence and measured status."""
import hashlib
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree

HERE=Path(__file__).resolve().parent
FIT=HERE.parent/'fit'
sys.path.insert(0,str(FIT))
from fit_core import ROOT, read_json, write_json, project
from fit2d_core import frame_sets
from fit2f_core import load_surface, boundary_edges, excluded_vertices, load_targets_f, surface_contour
from fit2f_sheets import (all_sheets, tripo_models, common_view, save, draw_points,
                         padded_stack, image_for)
from fit_sheets import shade, label
from run_fit2f import context

OUT=HERE/'results/tripo'
FOUT=FIT/'results/fit2f'


def remesh_maximum_location():
    """Locate the already measured maximum; do not rerun the rejected test."""
    import trimesh
    _,_,sref=context()
    original=load_surface('surface-original-a'); remesh=load_surface('surface-a')
    mesh=trimesh.Trimesh(remesh['vertices'],remesh['triangles'],process=False)
    closest,distance,_=trimesh.proximity.closest_point(mesh,original['vertices'])
    index=int(distance.argmax()); point=original['vertices'][index]
    open_vertices=np.unique(boundary_edges(original['triangles']))
    open_distance=float(cKDTree(original['vertices'][open_vertices]).query(point)[0]*sref)
    ear=bool(original['ear_excluded'][index])
    two_ring=bool(excluded_vertices(original['triangles'],len(original['vertices']))[index])
    record={'original_vertex':index,'point_ict':point.tolist(),
            'closest_remesh_point_ict':closest[index].tolist(),
            'distance_px':float(distance[index]*sref),'ear_excluded':ear,
            'open_boundary_two_rings':two_ring,'nearest_open_vertex_px':open_distance,
            'classification':'ear' if ear else 'neck cut boundary vicinity' if two_ring else 'other; inspect marked views',
            'test6_rerun':False,'action':'location only, geometry and remeshing parameters unchanged'}
    write_json(FOUT/'remesh-maximum-location.json',record)
    center,scale=common_view(); panels=[]
    for yaw in (0,-90,90):
        pose=[yaw,0,0,np.log(scale),200,200]
        image,_,_=shade(original['vertices']-center,original['triangles'],pose,400,cull_backfaces=True)
        q=project(torch.tensor((point-center)[None]),torch.tensor(pose,dtype=torch.float64)).numpy()[0,:2]
        cv2.circle(image,tuple(np.rint(q).astype(int)),6,(0,0,255),2,cv2.LINE_AA)
        label(image,f'max {record["distance_px"]:.3f}px / yaw {yaw}')
        panels.append(image)
    save(FIT/'sheets/fit2f-remesh-maximum.png',np.hstack(panels))
    print('REMESH MAXIMUM',record,flush=True)


def eyelashes():
    """Inspect unmodified dense T1-A and the approved remesh in the nine poses."""
    with np.load(OUT/'T1-a-state.npz') as source:
        v,tri=source['vertices'],source['triangles']
    model=tripo_models()['T1-A']; keys=frame_sets()[0][:9]
    targets=load_targets_f(keys,model['t']); panels=[]; rows=[]
    for target in targets:
        frame=target['frame']; pose=model['poses'][str(frame)]
        dense=surface_contour(v,tri,pose)
        coarse=surface_contour(model['vertices'],model['triangles'],pose)
        image=image_for(frame)
        draw_points(image,dense['xy'],(0,145,245))
        draw_points(image,coarse['xy'],(25,25,235))
        skin=target['points'][target['labels']==0]
        lo=np.maximum(np.floor(skin.min(0)).astype(int)-35,0)
        hi=np.minimum(np.ceil(skin.max(0)).astype(int)+36,720)
        hi[1]=min(hi[1],int(lo[1]+.65*(hi[1]-lo[1])))
        crop=cv2.resize(image[lo[1]:hi[1],lo[0]:hi[0]],None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
        label(crop,f'f{frame} yaw {pose[0]:.1f}: dense orange / remesh red')
        panels.append(crop)
        rows.append({'frame':frame,'pose':pose,'crop_720':[lo.tolist(),hi.tolist()],
                     'dense_contour_points':len(dense['xy']),'remesh_contour_points':len(coarse['xy'])})
    evidence=save(HERE/'sheets/tripo-eyelashes.png',padded_stack(panels))
    write_json(OUT/'eyelash-inspection.json',{'frames':rows,'sheet':evidence,
               'definition':'Visual comparison only; no eyelash labels, edits or losses.'})


def manifest(directory,pattern):
    return [{'path':str(p.relative_to(ROOT)),
             'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
             'validity':'superseded operator archive' if 'pre-edge-clamp' in p.name else 'current'}
            for p in sorted(directory.glob(pattern))]


def main():
    started=time.perf_counter()
    previous=read_json(FOUT/'validation.json')
    status=read_json(OUT/'status.json')
    status.update(status='final_comparison_in_progress',additional_runners_pending=0,
                  additional_runners_approved=3,T2='excluded by caller; no topology repair or fit')
    write_json(OUT/'status.json',status)
    all_sheets(tripo_only=True)
    remesh_maximum_location()
    eyelashes()
    elapsed=time.perf_counter()-started
    write_json(FOUT/'sheet-sha256.json',{'sheets':manifest(FIT/'sheets','fit2f-*.png')})
    write_json(OUT/'sheet-sha256.json',{'sheets':manifest(HERE/'sheets','*.png'),'elapsed_seconds':elapsed})
    compare=read_json(OUT/'compare.json')
    status.update(status='six_model_comparison_complete',T1='T1-A and T1-B3 measured and rendered')
    write_json(OUT/'status.json',status)
    compare.update(status='complete within approved six-model scope',tripo_status=status)
    write_json(OUT/'compare.json',compare)
    previous.update(status='six-model comparison complete; shape verdict recorded separately',
        runner_calls=12,additional_runners_approved=3,additional_runners_requested=0,pending_runs=[],
        pending_design_decision=None,step2_status='Recomputed with edge-wise cotangent clamp, lambda1000, mu.01.',
        T2={'status':'excluded by caller','repair_applied':False,'alignment_executed':False},
        T1_A_evaluation='8px remesh; original global alignment and neck cut reused without refitting',
        remesh_test6_disposition='Caller accepts p99 .716px; no test rerun, maximum location recorded.',
        final_runner_seconds=elapsed,
        not_executed=['T2 repair/alignment','B130','2e','A refit','parameter search','PR','merge','deployment'])
    previous['runner_history']+=['approved corrected ICT step2','approved remesh T1-B3','approved final comparison and sheets']
    previous['models']={name:{'curvature':value.get('curvature'),'frames':len(value['frames'])}
                        for name,value in compare['models'].items()}
    write_json(FOUT/'validation.json',previous)
    print('FINAL RUNNER SECONDS',elapsed,flush=True)


if __name__=='__main__':
    torch.set_num_threads(2); torch.set_num_interop_threads(1); cv2.setNumThreads(2)
    main()
