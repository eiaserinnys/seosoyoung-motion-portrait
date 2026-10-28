#!/usr/bin/env python3
"""Run exactly the bounded three-frame ICT experiment: prepare, fit, sheets."""
from __future__ import annotations
import argparse
import hashlib
import time
import unittest

import cv2
import numpy as np
import torch

from fit_core import (FIT, ROOT, WORKSPACE, FRAMES, read_json, write_json, load_model,
                      define_regions, load_targets, project, silhouettes, loss_frame,
                      rasterize, lower_boundary, attach_contour)
from fit_sheets import regions_sheet,overlay_sheet,shape_sheet,measure

RESULTS=FIT/'results/falsify'
STAGES=(('initialize_similarity',300),('pose_only',500),('joint',800))


def input_hashes():
    paths=[ROOT/'source/viewer/ponytail-seedance-registration.json']
    for frame in FRAMES:
        directory=ROOT/f'pipeline/headmodel/data/frame_{frame:04d}'
        paths += [directory/x for x in ('head.png','raw/face skin.png','raw/neck.png',
                                      'ear-left.png','ear-right.png','manifest.json')]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def initial_poses(v,tri,faces,targets):
    poses=np.stack([t['normalized_pose'] for t in targets])
    front=targets[0]
    q=project(torch.tensor(v),torch.tensor(poses[0],dtype=torch.float32)).numpy()
    keep=np.unique(tri[faces['head']]); xy=q[keep,:2]
    ys,xs=np.where(front['head'])
    model_size=np.ptp(xy,axis=0)
    desired=np.array([np.ptp(xs)-14,np.ptp(ys)-14])
    scale=float(np.sqrt(np.prod(desired/model_size)))
    center_model=(xy.min(0)+xy.max(0))/2-poses[0,4:6]
    center_target=np.array([(xs.min()+xs.max())/2,(ys.min()+ys.max())/2])
    shift=center_target-poses[0,4:6]-scale*center_model
    # Registration's translation is located on the facial plane, whereas ICT's
    # origin is inside the head. A fixed neutral facial-plane datum is a camera
    # gauge conversion, not a measured image landmark or an extra shape term.
    pivot=torch.tensor([[0.,3.,9.]])
    pivot_images=np.stack([project(pivot,torch.tensor(p,dtype=torch.float32)).numpy()[0,:2]-p[4:6] for p in poses])
    gauge_correction=-scale*(pivot_images-pivot_images[0])
    poses[:,3]+=np.log(scale); poses[:,4:6]+=shift+gauge_correction
    return poses.astype(np.float32), {'base_scale':scale,'gauge_shift_px':shift.tolist(),
                                    'neutral_facial_plane_datum':[0.,3.,9.],
                                    'per_frame_gauge_correction_px':gauge_correction.tolist(),
                                    'normalized_poses':[t['normalized_pose'].tolist() for t in targets]}


def prepare():
    suite=unittest.defaultTestLoader.discover(str(FIT),pattern='test_fit_core.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    v,tri,modes=load_model(); regions,faces=define_regions(v,tri); targets=load_targets()
    poses,gauge=initial_poses(v,tri,faces,targets)
    definitions={'neck_plane':'neutral y + 0.4*z < -5.0 excluded, including every incident triangle',
                 'ears':'abs(x)>7.6, -4<y<4.8, z<3.5; side by sign(x); majority triangle labels',
                 'face_lower':'Face vertices 0..9408, neck and ears removed, y<=1.5 (blue); only triangles fully in this lower region enter loss and metrics',
                 'vertices':{k:len(ids) for k,ids in regions.items()},
                 'triangles':{k:len(ids) for k,ids in faces.items()},
                 'vertex_indices':{k:ids.tolist() for k,ids in regions.items()},
                 'pose_gauge':gauge,'initial_poses':poses.tolist(),
                 'input_hashes':input_hashes()}
    write_json(RESULTS/'preparation.json',definitions)
    regions_sheet(v,tri,regions)
    overlay_sheet(v,tri,poses,targets,7.,'initial-overlay.png')
    started=time.perf_counter()
    for target,pose in zip(targets,poses):
        q=project(torch.tensor(v),torch.tensor(pose)).numpy()
        contours=silhouettes(q,tri,faces)
        cost,values=loss_frame(torch.tensor(q,requires_grad=True),contours,target,torch.tensor(7.))
        cost.backward()
        print('initial',target['frame'],{k:round(float(x.detach()),3) for k,x in values.items()},flush=True)
    print('preparation contour audit seconds',time.perf_counter()-started,flush=True)
    print('regions',definitions['vertices'],flush=True)


def fit():
    preparation=read_json(RESULTS/'preparation.json')
    if input_hashes()!=preparation['input_hashes']:
        raise ValueError('Prepared input changed; stop rather than mixing data revisions')
    v,tri,modes=load_model(); regions,faces=define_regions(v,tri); targets=load_targets()
    # The initial preparation sheet exposed the camera-origin offset. Rebuild
    # just that initialization here, within the same bounded fit runner.
    converted,gauge=initial_poses(v,tri,faces,targets)
    preparation['initial_poses']=converted.tolist(); preparation['pose_gauge']=gauge
    preparation['triangles']={k:len(ids) for k,ids in faces.items()}
    preparation['face_lower']='Face without neck/ears, neutral y<=1.5; all three triangle vertices must be in this region for L_skin and metrics'
    write_json(RESULTS/'preparation.json',preparation)
    overlay_sheet(v,tri,converted,targets,7.,'initial-overlay.png')
    neutral=torch.tensor(v); basis=torch.tensor(modes.reshape(100,-1))
    beta=torch.nn.Parameter(torch.zeros(100))
    logaxes=torch.nn.Parameter(torch.zeros(3))
    thickness=torch.nn.Parameter(torch.tensor(7.))
    # Scaled parameters keep degrees, logscale, and pixel translations distinct.
    pose_base=torch.tensor(preparation['initial_poses'])
    delta=torch.nn.Parameter(torch.zeros_like(pose_base))
    history=[]; times=[]; status='finished'; correspondences=None
    run_start=time.perf_counter()
    for stage,steps in STAGES:
        if stage=='initialize_similarity':
            pose_rate=torch.tensor([0.,0.,0.,.004,.35,.35])
        elif stage=='pose_only':
            pose_rate=torch.tensor([.10,.10,.05,.0015,.20,.20])
        else:
            pose_rate=torch.tensor([.06,.06,.03,.001,.15,.15])
        # Rebase the camera so an Adam restart does not change the current pose.
        if history:
            pose_base=poses.detach().clone()
        with torch.no_grad(): delta.zero_()
        groups=[{'params':[delta],'lr':1.}]
        if stage=='joint':
            groups += [{'params':[beta],'lr':.012},{'params':[logaxes],'lr':.002},
                       {'params':[thickness],'lr':.025}]
        optimizer=torch.optim.Adam(groups)
        stage_start=time.perf_counter()
        for iteration in range(steps):
            tick=time.perf_counter(); optimizer.zero_grad(set_to_none=True)
            vertices=(neutral+(beta@basis).reshape(-1,3))*logaxes.exp()
            poses=pose_base+delta*pose_rate
            projected=[project(vertices,p) for p in poses]
            if iteration%5==0:
                correspondences=[silhouettes(q.detach().numpy(),tri,faces) for q in projected]
            terms=[loss_frame(q,c,t,thickness) for q,c,t in zip(projected,correspondences,targets)]
            data=torch.stack([x[0] for x in terms]).mean()
            # Prescribed L2 sums. t's stated prior 7 is the starting value;
            # no unrequested thickness regularizer is added.
            prior=.01*(beta**2).sum()+.1*(logaxes**2).sum()
            loss=data+prior
            if not torch.isfinite(loss):
                status='diverged'; break
            loss.backward(); optimizer.step()
            with torch.no_grad(): thickness.clamp_(3.,14.)
            times.append(time.perf_counter()-tick)
            if iteration%50==0 or iteration==steps-1:
                row={'stage':stage,'iteration':iteration+1,'loss':float(loss.detach()),
                     'head':[float(x[1]['head'].detach()) for x in terms],
                     'skin':[float(x[1]['skin'].detach()) for x in terms],
                     'ear':[float(x[1]['ear'].detach()) for x in terms],
                     'beta_abs_max':float(beta.detach().abs().max()),
                     'axes':logaxes.detach().exp().tolist(),'t':float(thickness.detach()),
                     'seconds':time.perf_counter()-run_start}
                history.append(row); print(row,flush=True)
        print('stage seconds',stage,time.perf_counter()-stage_start,flush=True)
        if status=='diverged': break
    with torch.no_grad():
        vertices=(neutral+(beta@basis).reshape(-1,3))*logaxes.exp()
        poses=pose_base+delta*pose_rate
    write_json(RESULTS/'shared.json',{'beta':beta.detach().tolist(),'scale_xyz':logaxes.detach().exp().tolist(),
                                    'thickness_px':float(thickness.detach()),'status':status,
                                    'weights':{'head':1.,'skin':.5,'ear':.5,'beta_L2_sum':.01,'logscale_L2_sum':.1},
                                    'iterations':dict(STAGES),'silhouette_refresh':5})
    write_json(RESULTS/'poses.json',{'coordinate_system':'normalized 720; ICT Y flipped before Rx(pitch), Ry(yaw), screen roll',
                                   'frames':{str(f):p.tolist() for f,p in zip(FRAMES,poses)}})
    write_json(RESULTS/'trace.json',{'history':history,'iteration_mean_seconds':float(np.mean(times)),
                                   'iteration_median_seconds':float(np.median(times)),
                                   'iterations_executed':len(times),'elapsed_seconds':time.perf_counter()-run_start,
                                   'input_hashes':input_hashes()})
    with (RESULTS/'fitted_neutral.obj').open('w') as out:
        out.write('# ICT fitted neutral, original axes and units, full neck retained\n')
        for x,y,z in vertices.numpy():out.write(f'v {x:.7f} {y:.7f} {z:.7f}\n')
        for a,b,c in tri+1:out.write(f'f {a} {b} {c}\n')
    rows=measure(vertices.numpy(),tri,faces,poses.detach().numpy(),targets,float(thickness.detach()))
    write_json(RESULTS/'fit-end-metrics.json',rows)
    print('END metrics',rows,flush=True)
    if any(x['head_iou']<.80 for x in rows):
        print('STOP: stage 3 head IoU below .80; no weight or step tuning',flush=True)


def sheets():
    shared=read_json(RESULTS/'shared.json'); trace=read_json(RESULTS/'trace.json')
    if input_hashes()!=trace['input_hashes']:
        raise ValueError('Fitted input changed')
    neutral,tri,modes=load_model(); _,faces=define_regions(neutral,tri); targets=load_targets()
    vertices=(neutral+np.einsum('i,ivd->vd',np.array(shared['beta'],np.float32),modes))*np.array(shared['scale_xyz'],np.float32)
    poses=np.array(list(read_json(RESULTS/'poses.json')['frames'].values()),np.float32)
    rows=read_json(RESULTS/'fit-end-metrics.json')
    for row in rows:
        row.update(beta_abs_max=max(abs(x) for x in shared['beta']),scale_xyz=shared['scale_xyz'],
                   thickness_px=shared['thickness_px'],iteration_mean_seconds=trace['iteration_mean_seconds'])
    ear264=next(x for x in rows if x['frame']==264)['ears']
    criteria={'1_all_three_head_iou_below_090':all(x['head_iou']<.9 for x in rows),
              '2_any_skin_over_10px':any(x['skin_over_10px'] for x in rows),
              '2_any_displacement_over_26px_failure_candidate':any(x['failure_candidate_over_26px'] for x in rows),
              '3_down_right_visible_ear_iou_below_030':any(x['visible_iou']<.3 for x in ear264.values())}
    joint=[x for x in trace['history'] if x['stage']=='joint']
    joint_mean=(joint[-1]['seconds']-joint[0]['seconds'])/(joint[-1]['iteration']-joint[0]['iteration'])
    write_json(RESULTS/'residuals.json',{'frames':rows,'criteria':criteria,
                                      'joint_iteration_mean_seconds':joint_mean,
                                      'stage3_130frames_seconds_linear_estimate':joint_mean*800*130/3,
                                      'notes':['Distances are symmetric dense contour DT measurements at 720.',
                                               '26px estimate is projected contour displacement, not a recovered vertex delta.',
                                               'No weight search or alternative base was run.']})
    overlay_sheet(vertices,tri,poses,targets,shared['thickness_px'],'falsify-overlay.png')
    shape_sheet(vertices,neutral,tri)
    print('RESIDUALS',rows,flush=True); print('CRITERIA',criteria,flush=True)


def diagnose_skin_surface():
    """Preserve the observed upper-Face contamination before the single refit."""
    if not (RESULTS/'fit-end-metrics.json').exists():
        return
    neutral,tri,modes=load_model(); regions,faces=define_regions(neutral,tri)
    shared=read_json(RESULTS/'shared.json')
    vertices=(neutral+np.einsum('i,ivd->vd',np.array(shared['beta'],np.float32),modes))*np.array(shared['scale_xyz'],np.float32)
    poses=read_json(RESULTS/'poses.json')
    full_face=np.zeros(len(neutral),bool); full_face[regions['face']]=True
    oldids=np.flatnonzero(full_face[tri].all(1))
    evidence=[]
    for target in load_targets():
        q=project(torch.tensor(vertices),torch.tensor(poses['frames'][str(target['frame'])],dtype=torch.float32)).numpy()
        _,owner=rasterize(q,tri[oldids]); valid=owner>=0
        global_owner=np.full_like(owner,-1); global_owner[valid]=oldids[owner[valid]]
        points=lower_boundary(valid)
        field=np.ones((720,720),np.uint8)
        sp=target['lower'].astype(int); field[sp[:,1],sp[:,0]]=0
        dt=cv2.distanceTransform(field,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        samples=dt[points[:,1].astype(int),points[:,0].astype(int)]
        worst=int(np.argmax(samples))
        indices,weights=attach_contour(points,q,tri,global_owner)
        neutral_y=float((neutral[indices[worst].numpy(),1]*weights[worst].numpy()).sum())
        row={'frame':target['frame'],'worst_model_point_px':points[worst].tolist(),
             'distance_to_target_lower_px':float(samples[worst]),'neutral_y':neutral_y,
             'outside_lower_face_region':neutral_y>1.5}
        evidence.append(row); print('SKIN DIAGNOSIS',row,flush=True)
    write_json(RESULTS/'first-pass-diagnostics.json',{'invalid_lower_surface_metrics':read_json(RESULTS/'fit-end-metrics.json'),
                                                   'evidence':evidence,'shared':shared,'poses':poses})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','fit','sheets'])
    args=parser.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1); cv2.setNumThreads(2)
    RESULTS.mkdir(parents=True,exist_ok=True)
    if args.phase=='fit':
        diagnose_skin_surface()
        fit()
        sheets()
    else:
        {'prepare':prepare,'sheets':sheets}[args.phase]()


if __name__=='__main__':main()
