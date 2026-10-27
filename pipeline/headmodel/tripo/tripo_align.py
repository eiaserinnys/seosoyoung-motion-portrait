"""Shared affine shape and camera alignment using 2f's label-free targets."""
import time
import numpy as np
import torch
from fit_core import project
from fit2f_core import surface_contour, pair_rows, load_targets_f, boundary_edges


def rotation(offset):
    x,y,z=offset.unbind()
    zero=x*0
    skew=torch.stack([zero,-z,y,z,zero,-x,-y,x,zero]).reshape(3,3)
    return torch.matrix_exp(skew)


def align(v,tri,frames,initial_poses,thickness,hair=False,initial_parameters=None,initial_cameras=None):
    """Ten frozen-correspondence ICP updates, 40 Adam steps per update.

    Only a shared rotation, diagonal scales, translation, thickness and each
    camera's scale/translation change. Registration rotations stay fixed.
    Redundant global/camera similarity is reported in this initialization gauge.
    """
    started=time.perf_counter(); dtype=torch.float64
    base=torch.tensor(initial_poses,dtype=dtype)
    vertices=torch.tensor(v,dtype=dtype)
    angles=torch.nn.Parameter(torch.zeros(3,dtype=dtype))
    axes=torch.nn.Parameter(torch.zeros(3,dtype=dtype))
    translation=torch.nn.Parameter(torch.zeros(3,dtype=dtype))
    cameras=torch.nn.Parameter(torch.zeros(len(frames),3,dtype=dtype))
    t=torch.nn.Parameter(torch.tensor(0. if hair else thickness,dtype=dtype),requires_grad=not hair)
    if initial_parameters is not None:
        with torch.no_grad():
            angles.copy_(torch.tensor(np.radians(initial_parameters['rotation_offset_deg']),dtype=dtype))
            axes.copy_(torch.tensor(np.log(initial_parameters['scale_xyz']),dtype=dtype))
            translation.copy_(torch.tensor(initial_parameters['translation_ict'],dtype=dtype))
    if initial_cameras is not None:
        with torch.no_grad(): cameras.copy_((torch.tensor(initial_cameras,dtype=dtype)[:,3:]-base[:,3:])/torch.tensor([.0015,.20,.20],dtype=dtype))
    groups=[{'params':[angles],'lr':.002},{'params':[axes],'lr':.002},
            {'params':[translation],'lr':.02},{'params':[cameras],'lr':1.}]
    if not hair: groups.append({'params':[t],'lr':.025})
    optimizer=torch.optim.Adam(groups); rate=torch.tensor([.0015,.20,.20],dtype=dtype)
    trace=[]; cut_edges=boundary_edges(tri)
    for outer in range(10):
        tick=time.perf_counter()
        with torch.no_grad():
            transformed=(vertices*axes.exp())@rotation(angles).T+translation
            poses=torch.cat([base[:,:3],base[:,3:]+cameras*rate],1)
        targets=load_targets_f(frames,float(t.detach()))
        rows=[]; counts=[]
        gate=15.-9.*min(outer,7)/7
        for index,(pose,target) in enumerate(zip(poses.numpy(),targets)):
            model=surface_contour(transformed.numpy(),tri,pose,boundary_cache=cut_edges)
            pairs=pair_rows(model,target,gate)
            mi,ti=pairs['model_indices'],pairs['target_indices']
            ids=model['ids'][mi]; weights=model['weights'][mi]
            wanted=target['points'][ti]
            # Hair locations depend linearly on t; refresh classification only
            # between ICP iterations, as the mask labels themselves are fixed.
            normals=target['normals'][ti]
            dt=np.zeros(len(ti))
            hp=target['head_boundary']; hn=__import__('fit2d_core').contour_normals(hp)
            from fit2f_core import taper_amount
            hair_ids=np.flatnonzero(target['head_labels']==1)
            all_derivative=taper_amount(target['skin']|target['ear'],hp,1.)
            skin_count=int((target['labels']==0).sum())
            hair_row=target['labels'][ti]==1
            dt[hair_row]=all_derivative[hair_ids[ti[hair_row]-skin_count]]
            fixed_wanted=wanted+float(t.detach())*dt[:,None]*normals
            rows.append((index,torch.tensor(ids),torch.tensor(weights,dtype=dtype),
                         torch.tensor(fixed_wanted,dtype=dtype),torch.tensor(normals,dtype=dtype),
                         torch.tensor(dt,dtype=dtype),torch.tensor(np.where(target['labels'][ti]==0,10.,1.)/3.,dtype=dtype)))
            counts.append(len(mi))
        if sum(counts)==0: raise ValueError('No Tripo-A correspondences within gate')
        for _ in range(40):
            optimizer.zero_grad(set_to_none=True)
            rot=rotation(angles); loss=vertices.sum()*0
            for index,ids,weights,wanted,normals,derivative,w in rows:
                original=(vertices[ids]*weights[:,:,None]).sum(1)
                fitted=(original*axes.exp())@rot.T+translation
                pose=torch.cat([base[index,:3],base[index,3:]+cameras[index]*rate])
                xy=project(fitted,pose)[:,:2]
                residual=((xy-wanted)*normals).sum(1)+t*derivative
                loss=loss+(w*residual.square()).sum()/sum(counts)
            if not torch.isfinite(loss): raise FloatingPointError('Tripo-A diverged')
            loss.backward(); optimizer.step()
            if not hair:
                with torch.no_grad(): t.clamp_(3.,14.)
        record={'iteration':outer+1,'gate_px':gate,'pairs':sum(counts),'loss':float(loss.detach()),
                'rotation_offset_deg':np.degrees(angles.detach().numpy()).tolist(),
                'scales_xyz':axes.detach().exp().tolist(),'t':float(t.detach()),'seconds':time.perf_counter()-tick}
        trace.append(record); print('TRIPO ALIGN',record,flush=True)
    with torch.no_grad():
        output=((vertices*axes.exp())@rotation(angles).T+translation).numpy()
        poses=torch.cat([base[:,:3],base[:,3:]+cameras*rate],1).numpy()
    return output,poses,{'rotation_offset_deg':np.degrees(angles.detach().numpy()).tolist(),
        'scale_xyz':axes.detach().exp().tolist(),'translation_ict':translation.detach().tolist(),
        'thickness_px':float(t.detach()),'icp':trace,'elapsed_seconds':time.perf_counter()-started,
        'camera_rotations_unchanged':bool(np.array_equal(poses[:,:3],np.asarray(initial_poses)[:,:3])),
        'gauge':'global/camera similarity retains initialization gauge; no geometric vertex displacement'}
