#!/usr/bin/env python3
"""Bounded 2b experiment: fixed registered rotations, beta limits, head-only sheet."""
import argparse
import time
import cv2
import numpy as np
import torch

from fit_core import (FIT, FRAMES, load_model, define_regions, load_targets, write_json,
                      read_json, project, silhouettes, loss_frame)
from fit_sheets import shade, source_image, label, save_image, overlay_sheet, measure
from run_falsify import initial_poses, input_hashes, STAGES

RESULTS = FIT/'results/falsify2'


def init_checkpoint():
    neutral,tri,_ = load_model()
    _,faces = define_regions(neutral,tri)
    targets = load_targets()
    poses,gauge = initial_poses(neutral,tri,faces,targets)
    panels=[]
    for frame,target,pose in zip(FRAMES,targets,poses):
        # Keep the verified key-normalized yaw/pitch/roll exactly. The previous
        # origin/units conversion changes only translation and logscale.
        np.testing.assert_allclose(pose[:3],target['normalized_pose'][:3],atol=2e-6)
        original = source_image(target)
        gray,mask,_ = shade(neutral,tri[faces['head']],pose)
        image = original.copy()
        image[mask] = (.48*original[mask]+.52*gray[mask]).astype(np.uint8)
        label(image,f'frame {frame}: ICT NEUTRAL / NO OPTIMIZATION')
        label(image,f'yaw {pose[0]:.2f}  pitch {pose[1]:.2f}  roll {pose[2]:.2f}',(12,50))
        panels.append(image)
    save_image('init-overlay.png',np.hstack(panels))
    write_json(RESULTS/'init.json',{
        'status':'awaiting_assignee_camera_check',
        'frames':{str(frame):pose.tolist() for frame,pose in zip(FRAMES,poses)},
        'camera_gauge':gauge,'input_hashes':input_hashes(),
        'shape':'ICT neutral; beta=0, scale_xyz=[1,1,1]',
        'optimization_iterations':0,
        'planned_fit':{'rotation':'fixed to these registered yaw/pitch/roll values',
                       'free_camera':['logscale','tx','ty'],
                       'beta_abs_limit':3.,'beta_L2_sum_weight':.1,
                       'shape_sheet':'same ten views; neck triangles excluded in both rows'}
    })
    print('No optimization run; awaiting assignee camera check.',flush=True)


def fixed_cameras(base,delta,rate):
    """Only logscale/tx/ty have optimizer parameters; rotations are immutable."""
    return torch.cat([base[:,:3],base[:,3:]+delta*rate],dim=1)


def constrain_parameters(beta,thickness):
    with torch.no_grad():
        beta.clamp_(-3.,3.)
        thickness.clamp_(3.,14.)


def shape_prior(beta,logaxes):
    return .1*(beta**2).sum()+.1*(logaxes**2).sum()


def head_shape_sheet(vertices,neutral,tri,head_faces):
    views=[(0,0),(-45,0),(45,0),(-67.5,0),(67.5,0),(-78.75,0),(78.75,0),
           (0,30),(0,-30),(45,-30)]
    visible=np.unique(tri[head_faces])
    # One center, scale and light for both rows. Excluded neck does not set
    # either the crop or scale, and no triangle incident to it is rendered.
    center=(neutral[visible].min(0)+neutral[visible].max(0))/2
    radius=np.linalg.norm(np.r_[vertices[visible],neutral[visible]]-center,axis=1).max()
    cell=400; scale=(cell-50)/(2*radius)
    rows=[]
    for name,v in [('FITTED 2b',vertices),('ICT NEUTRAL',neutral)]:
        panels=[]
        for yaw,pitch in views:
            image,_,_=shade(v-center,tri[head_faces],
                            [yaw,pitch,0,np.log(scale),cell/2,cell/2],cell)
            label(image,f'{name} y {yaw:g} p {pitch:g}')
            panels.append(image)
        rows.append(np.hstack(panels))
    save_image('falsify2-shape.png',np.vstack(rows))


def fit_and_sheets():
    checkpoint=read_json(RESULTS/'init.json')
    if input_hashes()!=checkpoint['input_hashes']:
        raise ValueError('Approved checkpoint inputs changed')
    v,tri,modes=load_model(); regions,faces=define_regions(v,tri); targets=load_targets()
    neutral=torch.tensor(v); basis=torch.tensor(modes.reshape(100,-1))
    initial=torch.tensor([checkpoint['frames'][str(frame)] for frame in FRAMES])
    base=initial.clone()
    beta=torch.nn.Parameter(torch.zeros(100))
    logaxes=torch.nn.Parameter(torch.zeros(3))
    thickness=torch.nn.Parameter(torch.tensor(7.))
    history=[]; times=[]; started=time.perf_counter()
    for stage,steps in STAGES:
        rate=torch.tensor({'initialize_similarity':[.004,.35,.35],
                           'pose_only':[.0015,.20,.20],
                           'joint':[.001,.15,.15]}[stage])
        delta=torch.nn.Parameter(torch.zeros(len(FRAMES),3))
        joint=stage=='joint'
        for parameter in (beta,logaxes,thickness):parameter.requires_grad_(joint)
        groups=[{'params':[delta],'lr':1.}]
        if joint:
            groups += [{'params':[beta],'lr':.012},{'params':[logaxes],'lr':.002},
                       {'params':[thickness],'lr':.025}]
        optimizer=torch.optim.Adam(groups)
        for iteration in range(steps):
            tick=time.perf_counter(); optimizer.zero_grad(set_to_none=True)
            vertices=(neutral+(beta@basis).reshape(-1,3))*logaxes.exp()
            poses=fixed_cameras(base,delta,rate)
            projected=[project(vertices,p) for p in poses]
            if iteration%5==0:
                contours=[silhouettes(q.detach().numpy(),tri,faces) for q in projected]
            terms=[loss_frame(q,c,t,thickness) for q,c,t in zip(projected,contours,targets)]
            loss=torch.stack([x[0] for x in terms]).mean()+shape_prior(beta,logaxes)
            if not torch.isfinite(loss):raise FloatingPointError('2b fit diverged')
            loss.backward(); optimizer.step(); constrain_parameters(beta,thickness)
            times.append(time.perf_counter()-tick)
            if iteration%50==0 or iteration==steps-1:
                row={'stage':stage,'iteration':iteration+1,'loss':float(loss.detach()),
                     'head':[float(x[1]['head'].detach()) for x in terms],
                     'skin':[float(x[1]['skin'].detach()) for x in terms],
                     'ear':[float(x[1]['ear'].detach()) for x in terms],
                     'beta_abs_max':float(beta.detach().abs().max()),
                     'axes':logaxes.detach().exp().tolist(),'t':float(thickness.detach()),
                     'seconds':time.perf_counter()-started}
                history.append(row); print(row,flush=True)
        # Carry the final updated similarity into the next stage. Rotations
        # remain exactly the approved values throughout all three stages.
        base=fixed_cameras(base,delta,rate).detach()
    with torch.no_grad():
        vertices=(neutral+(beta@basis).reshape(-1,3))*logaxes.exp()
    torch.testing.assert_close(base[:,:3],initial[:,:3],rtol=0,atol=0)
    if float(beta.detach().abs().max())>3.:raise AssertionError('beta clamp violated')
    shared={'beta':beta.detach().tolist(),'scale_xyz':logaxes.detach().exp().tolist(),
            'thickness_px':float(thickness.detach()),'status':'finished',
            'weights':{'head':1.,'skin':.5,'ear':.5,'beta_L2_sum':.1,'logscale_L2_sum':.1},
            'beta_abs_limit':3.,'iterations':dict(STAGES),'silhouette_refresh':5,
            'rotation_max_abs_delta_deg':float((base[:,:3]-initial[:,:3]).abs().max()),
            'approved_checkpoint_commit':'f62b4449fcca26c634f92880e795db6011c75dd3'}
    write_json(RESULTS/'shared.json',shared)
    write_json(RESULTS/'poses.json',{'coordinate_system':'unchanged from approved init.json',
                                   'frames':{str(f):p.tolist() for f,p in zip(FRAMES,base)}})
    write_json(RESULTS/'trace.json',{'history':history,'iteration_mean_seconds':float(np.mean(times)),
                                   'iterations_executed':len(times),'elapsed_seconds':time.perf_counter()-started,
                                   'input_hashes':input_hashes()})
    with (RESULTS/'fitted_neutral.obj').open('w') as out:
        out.write('# ICT 2b fitted neutral, original axes and units; full neck retained in OBJ\n')
        for x,y,z in vertices.numpy():out.write(f'v {x:.7f} {y:.7f} {z:.7f}\n')
        for a,b,c in tri+1:out.write(f'f {a} {b} {c}\n')
    rows=measure(vertices.numpy(),tri,faces,base.numpy(),targets,shared['thickness_px'])
    for row in rows:
        row.update(beta_abs_max=float(beta.detach().abs().max()),scale_xyz=shared['scale_xyz'],
                   thickness_px=shared['thickness_px'],iteration_mean_seconds=float(np.mean(times)))
    ear264=next(row for row in rows if row['frame']==264)['ears']
    criteria={'1_all_three_head_iou_below_090':all(row['head_iou']<.9 for row in rows),
              '3_down_right_visible_ear_iou_below_030':any(x['visible_iou']<.3 for x in ear264.values())}
    write_json(RESULTS/'residuals.json',{'frames':rows,'criteria':criteria,
                                      'rotation_max_abs_delta_deg':shared['rotation_max_abs_delta_deg'],
                                      'notes':['Same masks, regions, contour losses and distance metrics as falsify.',
                                               'Only rotation freedom, beta bound/prior and head-only shape sheet changed.',
                                               'No landmarks, vertex delta, alternative base or weight search.']})
    print('RESIDUALS',rows,flush=True); print('CRITERIA',criteria,flush=True)
    overlay_sheet(vertices.numpy(),tri,base.numpy(),targets,shared['thickness_px'],'falsify2-overlay.png')
    head_shape_sheet(vertices.numpy(),v,tri,faces['head'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['init','fit'])
    args=parser.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1); cv2.setNumThreads(2)
    {'init':init_checkpoint,'fit':fit_and_sheets}[args.phase]()
