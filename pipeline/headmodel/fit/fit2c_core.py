"""Shared-shape 2c: fixed camera rotations, pinna similarities and normal deltas."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from fit_core import FIT,ROOT,read_json,loss_frame
from egg_pca import mesh_edges,nose_observations

RESULTS=FIT/'results/fit2c'


def load_template():
    bundle=np.load(RESULTS/'egg-template-v5.npz')
    v=bundle['neutral_vertices']; tri=bundle['triangles']
    head=bundle['head_mask']; face=bundle['face_mask']
    masks={'head':head,'face':face,'ear-left':bundle['ear_left_mask'],'ear-right':bundle['ear_right_mask']}
    faces={'head':np.flatnonzero(head[tri].all(1)),
           'face':np.flatnonzero((face&(v[:,1]<=1.5))[tri].all(1))}
    groups=[]; feathers=[]; edges=mesh_edges(tri)
    lengths=np.linalg.norm(v[edges[:,0]]-v[edges[:,1]],axis=1)
    graph=sparse.csr_matrix((np.r_[lengths,lengths],
              (np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(v),len(v)))
    for key in ('ear-left','ear-right'):
        ids=np.flatnonzero(masks[key]); groups.append(torch.tensor(ids))
        faces[key]=np.flatnonzero((masks[key][tri].sum(1)>=2)&head[tri].all(1))
        distance=dijkstra(graph,directed=False,indices=ids,min_only=True)
        width=np.ptp(v[ids,1])*.25
        u=np.clip(1-distance/width,0,1)
        feathers.append(u*u*(3-2*u)*head)
    observations=nose_observations(read_json(ROOT/'pipeline/headmodel/data/landmarks.json'))
    nose=read_json(RESULTS/'egg-template-v5.json')['nose_vertex']
    return v,tri,bundle['identity_modes'],masks,faces,groups,torch.tensor(np.array(feathers),dtype=torch.float32),edges,observations,nose


def ear_transform(vertices,params,weights,groups):
    result=vertices
    for side,ids in enumerate(groups):
        angle=params[side,:3]; zero=angle[0]*0
        x,y,z=angle
        skew=torch.stack([zero,-z,y,z,zero,-x,-y,x,zero]).reshape(3,3)
        rotation=torch.matrix_exp(skew)
        center=vertices[ids].mean(0)
        moved=(vertices-center)@rotation.T*params[side,6].exp()+center+params[side,3:6]
        result=result+(moved-vertices)*weights[side,:,None]
    return result


def ear_prior(params):
    # Rotation vector in radians, translation in ICT units, scale in log units.
    return .1*(params[:,:3].square().sum()+params[:,3:6].square().sum()+params[:,6].square().sum())


def normal_deform(vertices,normals,delta_px,head,max_camera_scale):
    # A normal displacement of this length projects to <= |delta_px| in every
    # camera, since every orthographic camera scale is <= max_camera_scale.
    return vertices+normals*(delta_px*head/max_camera_scale)[:,None]


def delta_prior(delta,edges,head,face):
    values=delta*head
    degree=torch.zeros_like(values); neighbors=torch.zeros_like(values)
    for a,b in ((edges[:,0],edges[:,1]),(edges[:,1],edges[:,0])):
        neighbors.index_add_(0,a,values[b]); degree.index_add_(0,a,torch.ones_like(values[b]))
    lap=values-neighbors/degree.clamp_min(1)
    weights=torch.where(face,.3,1.)
    return (weights[head]*lap[head].square()).mean()+.1*values[head].square().mean()


def constrain(beta,thickness,delta,head):
    with torch.no_grad():
        beta.clamp_(-3.,3.); thickness.clamp_(3.,14.)
        delta.clamp_(-26.,26.); delta[~head]=0.


def frame_loss(q,contours,target,thickness,nose_id,observation,use_nose):
    old,terms=loss_frame(q,contours,target,thickness)
    # The base experiment uses skin .5; 2c requires exactly 1.
    total=old+.5*terms['skin']
    nose=q.sum()*0
    if use_nose and observation['used']:
        wanted=torch.tensor(observation['xy'],dtype=q.dtype)
        error=(q[nose_id,:2]-wanted).norm()
        nose=F.smooth_l1_loss(error,torch.zeros_like(error),beta=1.,reduction='sum')
    terms['nose']=nose
    return total+nose,terms
