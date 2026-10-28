"""Dirichlet implicit uniform-Laplacian smoothing for the neutral and PCA fields."""
from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from fit_core import FIT, ROOT, FRAMES, load_model, define_regions, read_json, write_json

RESULTS=FIT/'results/fit2c'
STRENGTHS=(100.,1000.,10000.)
NOSE_VERTEX=4857  # Original ICT Multi-PIE 68 list[30], nose tip.
NOSE_INDEX=23    # Anime-face-detector 28-point nose.


def mesh_edges(tri):
    return np.unique(np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1),axis=0)


def adjacency(edges,count,weights=None):
    weights=np.ones(len(edges)) if weights is None else weights
    return sparse.csr_matrix((np.r_[weights,weights],
                              (np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(count,count))


def boundary_vertices(face_mask,edges):
    cross=face_mask[edges[:,0]]!=face_mask[edges[:,1]]
    endpoints=edges[cross].ravel()
    return np.unique(endpoints[face_mask[endpoints]])


def implicit_smooth(fields,edges,interior,lam):
    """Apply (I+lambda L)^-1 on free vertices with exact original boundary.

    One factorization solves all XYZ right-hand sides, including displacement
    fields. Positive uniform weights avoid cotangent sign/degeneracy issues.
    """
    count=len(fields); adj=adjacency(edges,count)
    degree=np.asarray(adj.sum(1)).ravel()
    fixed=np.setdiff1d(np.arange(count),interior)
    matrix=sparse.eye(len(interior),format='csc')+lam*(sparse.diags(degree[interior])-adj[interior][:,interior])
    rhs=fields[interior]+lam*(adj[interior][:,fixed]@fields[fixed])
    output=np.array(fields,dtype=np.float64,copy=True)
    output[interior]=splu(matrix.tocsc()).solve(np.asarray(rhs,dtype=np.float64))
    return output


def nose_observations(document):
    result={}
    for row in document['frames']:
        frame=int(row['frame'])
        if frame not in FRAMES:continue
        x,y,score=row['keypoints'][NOSE_INDEX]
        result[frame]={'xy':[float(x),float(y)],'score':float(score),'used':float(score)>=.5}
    return result


def vertex_normals(v,tri):
    normal=np.zeros_like(v)
    face=np.cross(v[tri[:,1]]-v[tri[:,0]],v[tri[:,2]]-v[tri[:,0]])
    for column in range(3):np.add.at(normal,tri[:,column],face)
    return normal/np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-12)


def prepare_strength():
    from fit_sheets import shade,label,save_image
    neutral,tri,_=load_model(); regions,faces=define_regions(neutral,tri)
    edges=mesh_edges(tri)
    face_mask=np.zeros(len(neutral),bool); face_mask[regions['face']]=True
    boundary=boundary_vertices(face_mask,edges)
    interior=np.setdiff1d(regions['face'],boundary)
    candidates=[neutral]+[implicit_smooth(neutral,edges,interior,lam).astype(np.float32) for lam in STRENGTHS]
    visible=np.unique(tri[faces['head']]); center=(neutral[visible].min(0)+neutral[visible].max(0))/2
    cell=600; radius=max(np.linalg.norm(v[visible]-center,axis=1).max() for v in candidates)
    scale=(cell-60)/(2*radius)
    panels=[]
    for index,v in enumerate(candidates):
        row=[]
        for title,yaw in [('FRONT',0),('YAW 45',45),('SIDE 90',90)]:
            image,_,_=shade(v-center,tri[faces['head']],[yaw,0,0,np.log(scale),cell/2,cell/2],cell)
            name='ICT original' if not index else f'lambda {STRENGTHS[index-1]:g}'
            label(image,f'{name} / {title}')
            row.append(image)
        panels.append(np.hstack(row))
    save_image('egg-strength.png',np.vstack(panels))
    observations=nose_observations(read_json(ROOT/'pipeline/headmodel/data/landmarks.json'))
    RESULTS.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(RESULTS/'egg-candidates.npz',neutral=np.stack(candidates),edges=edges,
                        boundary=boundary,interior=interior)
    write_json(RESULTS/'egg-preparation.json',{
        'operator':'uniform combinatorial L=D-A; (I+lambda L)x=x0, Dirichlet boundary',
        'strengths':list(STRENGTHS),'face_vertices':len(regions['face']),
        'fixed_boundary_vertices':len(boundary),'interior_vertices':len(interior),
        'boundary_definition':'Face excluding ears/neck, adjacent by a triangle edge to a vertex outside that region',
        'nose_vertex':NOSE_VERTEX,'nose_original_xyz':neutral[NOSE_VERTEX].tolist(),
        'nose_detector_index':NOSE_INDEX,'nose_observations':observations,
        'sources':{'scipy':'https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html',
                   'anime_landmarks':'https://github.com/hysts/anime-face-detector/blob/main/assets/landmarks.jpg'}})
    print('EGG REGION',len(interior),'interior',len(boundary),'fixed boundary',flush=True)
    print('NOSE OBSERVATIONS',observations,flush=True)


if __name__=='__main__':
    import torch,cv2
    torch.set_num_threads(2); torch.set_num_interop_threads(1); cv2.setNumThreads(2)
    prepare_strength()
