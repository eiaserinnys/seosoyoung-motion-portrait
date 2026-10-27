"""Exactly one bounded four-stage fit of shared egg PCA to frames 3,42,264."""
from __future__ import annotations
import time
import numpy as np
import torch
import cv2
from fit_core import FRAMES,write_json,read_json,project,silhouettes
from fit2c_core import (RESULTS,load_template,ear_transform,ear_prior,normal_deform,
                        delta_prior,constrain,frame_loss)
from egg_pca import vertex_normals
from run_falsify import input_hashes
from run_falsify2 import fixed_cameras,shape_prior
from fit_core import load_targets,FIT

STAGES=(('similarity',300),('shape',500),('ears',500),('normal_delta_and_nose',800))


def run():
    checkpoint=read_json(FIT/'results/falsify2/init.json')
    if input_hashes()!=checkpoint['input_hashes']:raise ValueError('Approved registration or target masks changed')
    v,tri,modes,masks,faces,groups,weights,edge_array,observations,nose=load_template()
    neutral=torch.tensor(v); basis=torch.tensor(modes.reshape(100,-1))
    head=torch.tensor(masks['head']); face=torch.tensor(masks['face']); edges=torch.tensor(edge_array)
    beta=torch.nn.Parameter(torch.zeros(100)); logaxes=torch.nn.Parameter(torch.zeros(3))
    thickness=torch.nn.Parameter(torch.tensor(7.)); ears=torch.nn.Parameter(torch.zeros(2,7))
    delta=torch.nn.Parameter(torch.zeros(len(v)))
    initial=torch.tensor([checkpoint['frames'][str(f)] for f in FRAMES]); base=initial.clone()
    targets=load_targets(); times=[]; trace=[]; started=time.perf_counter()
    normals=torch.tensor(vertex_normals(v,tri)); final_vertices=None
    for stage_index,(stage,steps) in enumerate(STAGES):
        rate=torch.tensor([.004,.35,.35] if stage_index==0 else [.001,.15,.15])
        camera=torch.nn.Parameter(torch.zeros(3,3))
        for param,enabled in [(beta,stage_index>=1),(logaxes,stage_index>=1),
                              (thickness,stage_index>=1),(ears,stage_index>=2),(delta,stage_index>=3)]:
            param.requires_grad_(enabled)
        optim_groups=[{'params':[camera],'lr':1.}]
        if stage_index>=1:
            optim_groups.extend([{'params':[beta],'lr':.012},{'params':[logaxes],'lr':.002},
                                 {'params':[thickness],'lr':.025}])
        if stage_index>=2:optim_groups.append({'params':[ears],'lr':.006})
        if stage_index>=3:optim_groups.append({'params':[delta],'lr':.08})
        optimizer=torch.optim.Adam(optim_groups)
        for iteration in range(steps):
            tick=time.perf_counter(); optimizer.zero_grad(set_to_none=True)
            shaped=(neutral+(beta@basis).reshape(-1,3))*logaxes.exp()
            shaped=ear_transform(shaped,ears,weights,groups)
            poses=fixed_cameras(base,camera,rate)
            if stage_index>=3 and iteration%5==0:
                normals=torch.tensor(vertex_normals(shaped.detach().numpy(),tri))
            max_scale=poses[:,3].exp().max().detach()
            vertices=normal_deform(shaped,normals,delta,head,max_scale)
            projected=[project(vertices,p) for p in poses]
            if iteration%5==0:contours=[silhouettes(q.detach().numpy(),tri,faces) for q in projected]
            terms=[frame_loss(q,c,target,thickness,nose,observations[target['frame']],stage_index>=3)
                   for q,c,target in zip(projected,contours,targets)]
            prior=shape_prior(beta,logaxes)+ear_prior(ears)
            if stage_index>=3:prior=prior+delta_prior(delta,edges,head,face)
            loss=torch.stack([item[0] for item in terms]).mean()+prior
            if not torch.isfinite(loss):raise FloatingPointError('2c fit diverged')
            loss.backward(); optimizer.step(); constrain(beta,thickness,delta,head)
            times.append(time.perf_counter()-tick)
            if iteration%50==0 or iteration==steps-1:
                row={'stage':stage,'iteration':iteration+1,'loss':float(loss.detach()),
                     'head':[float(item[1]['head'].detach()) for item in terms],
                     'skin':[float(item[1]['skin'].detach()) for item in terms],
                     'ear':[float(item[1]['ear'].detach()) for item in terms],
                     'nose':[float(item[1]['nose'].detach()) for item in terms],
                     'beta_abs_max':float(beta.detach().abs().max()),'axes':logaxes.detach().exp().tolist(),
                     't':float(thickness.detach()),'delta_abs_max_px':float(delta.detach().abs().max()),
                     'seconds':time.perf_counter()-started}
                trace.append(row); print(row,flush=True)
        base=fixed_cameras(base,camera,rate).detach()
    with torch.no_grad():
        shaped=(neutral+(beta@basis).reshape(-1,3))*logaxes.exp()
        shaped=ear_transform(shaped,ears,weights,groups)
        # Keep the last frozen normal frame used by the optimization; save it so
        # sheet generation reproduces the final surface exactly.
        max_scale=base[:,3].exp().max()
        final_vertices=normal_deform(shaped,normals,delta,head,max_scale).numpy()
    torch.testing.assert_close(base[:,:3],initial[:,:3],rtol=0,atol=0)
    delta_np=delta.detach().numpy()
    saturated=np.flatnonzero((np.abs(delta_np)>=25.999)&masks['head'])
    shared={'status':'finished','template':'egg-template-v5.npz','beta':beta.detach().tolist(),
            'beta_abs_max':float(beta.detach().abs().max()),'scale_xyz':logaxes.detach().exp().tolist(),
            'thickness_px':float(thickness.detach()),'ear_parameters':ears.detach().tolist(),
            'ear_parameter_order':['rx_rad','ry_rad','rz_rad','tx_ICT','ty_ICT','tz_ICT','logscale'],
            'delta_px':delta_np.tolist(),'delta_world_scale':float(1/max_scale),
            'delta_stats':{'abs_max_px':float(np.abs(delta_np).max()),'abs_mean_px':float(np.abs(delta_np[masks['head']]).mean()),
                           'saturated_vertex_count':len(saturated),'saturated_vertex_ids':saturated.tolist(),
                           'limit_px':26.,'neck_nonzero_count':int(np.count_nonzero(delta_np[~masks['head']]))},
            'nose_vertex':nose,'nose_observations':observations,
            'weights':{'head':1.,'skin':1.,'ear':.5,'nose_Huber':1.,'nose_Huber_transition_px':1.,
                       'beta_L2_sum':.1,'logaxes_L2_sum':.1,'ear_rotation_L2_sum':.1,
                       'ear_translation_ICT_L2_sum':.1,'ear_logscale_L2_sum':.1,
                       'delta_laplacian_mean_square':1.,'delta_face_laplacian_multiplier':.3,'delta_L2_mean_square':.1},
            'rotation_max_abs_delta_deg':float((base[:,:3]-initial[:,:3]).abs().max()),
            'iterations':dict(STAGES),'silhouette_refresh':5,'normal_refresh':5,
            'iteration_mean_seconds':float(np.mean(times)),
            'iteration_mean_by_stage':{stage:float(np.mean(times[sum(n for _,n in STAGES[:i]):sum(n for _,n in STAGES[:i+1])]))
                                       for i,(stage,_) in enumerate(STAGES)},
            'elapsed_seconds':time.perf_counter()-started,'input_hashes':input_hashes(),
            'notes':['Ear loss uses the established isolated projection; reported visible ear IoU includes head occlusion.',
                     'Normal delta is applied after ear and anisotropic transforms; its 26px bound holds in all three cameras.',
                     'Normal frames are frozen for five ICP steps. Regularizers use mean squares for mesh-sized delta fields.']}
    write_json(RESULTS/'shared.json',shared)
    write_json(RESULTS/'poses.json',{'coordinate_system':'same approved key-normalized orthographic registration as 2b',
                                   'frames':{str(f):p.tolist() for f,p in zip(FRAMES,base)}})
    write_json(RESULTS/'trace.json',{'history':trace,'iterations_executed':len(times),'iteration_mean_seconds':float(np.mean(times))})
    np.savez_compressed(RESULTS/'fit-state.npz',vertices=final_vertices,normals=normals.numpy(),triangles=tri)
    with (RESULTS/'fitted_neutral.obj').open('w',encoding='utf-8') as out:
        out.write('# Fitted shared egg v5 shape in neutral ICT axes; full neck retained\n')
        for x,y,z in final_vertices:out.write(f'v {x:.7f} {y:.7f} {z:.7f}\n')
        for a,b,c in tri+1:out.write(f'f {a} {b} {c}\n')
    print('FIT FINISHED',shared['delta_stats'],'MEAN SECONDS',shared['iteration_mean_seconds'],flush=True)


if __name__=='__main__':
    torch.set_num_threads(2); torch.set_num_interop_threads(1); cv2.setNumThreads(2)
    run()
