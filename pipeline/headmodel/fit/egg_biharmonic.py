"""Close interior Face holes and fair all PCA fields with two fixed outer rings."""
from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from fit_core import FIT,load_model,define_regions,write_json,boundary,rasterize
from egg_pca import mesh_edges,adjacency,boundary_vertices,NOSE_VERTEX
from fit_sheets import shade,label,save_image

RESULTS=FIT/'results/fit2c'


def boundary_loops(tri):
    directed=np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]])
    _,inverse,counts=np.unique(np.sort(directed,axis=1),axis=0,return_inverse=True,return_counts=True)
    border=directed[counts[inverse]==1]
    successor={int(a):int(b) for a,b in border}
    if len(successor)!=len(border) or len(set(successor.values()))!=len(border):
        raise ValueError('Face patch boundary is not a disjoint set of oriented rings')
    unseen=set(successor); loops=[]
    while unseen:
        first=min(unseen); ring=[first]; cursor=successor[first]
        while cursor!=first:
            if cursor not in unseen:raise ValueError('Boundary ring does not close')
            ring.append(cursor); cursor=successor[cursor]
        unseen.difference_update(ring); loops.append(np.array(ring,dtype=np.int64))
    return loops


def cap_loops(v,tri,modes,loops):
    vertices=[v]; fields=[modes]; caps=[]
    for index,ring in enumerate(loops):
        center=len(v)+index
        vertices.append(v[ring].mean(0,keepdims=True))
        fields.append(modes[:,ring].mean(1,keepdims=True))
        # Reverse the oriented open edge, so the new fan has consistent normals.
        caps.append(np.column_stack([np.roll(ring,-1),ring,np.full(len(ring),center)]))
    return np.concatenate(vertices),np.concatenate([tri]+caps),np.concatenate(fields,axis=1)


def two_ring_boundary(face_mask,edges,outer):
    selected=np.zeros(len(face_mask),bool); selected[outer]=True
    touched=edges[np.any(selected[edges],axis=1)].ravel()
    selected[touched]=True
    return np.flatnonzero(selected & face_mask)


def biharmonic_fields(fields,edges,interior):
    count=len(fields); adj=adjacency(edges,count)
    lap=sparse.diags(np.asarray(adj.sum(1)).ravel())-adj
    energy=(lap@lap).tocsr()
    fixed=np.setdiff1d(np.arange(count),interior)
    rhs=-(energy[interior][:,fixed]@fields[fixed])
    out=np.array(fields,dtype=np.float64,copy=True)
    out[interior]=splu(energy[interior][:,interior].tocsc()).solve(np.asarray(rhs,dtype=np.float64))
    return out


def front_metrics(original,egg,tri,face_mask):
    face_tri=tri[face_mask[tri].all(1)]
    original_face=face_tri[face_tri.max(1)<len(original)]
    center=(original.min(0)+original.max(0))/2
    scale=20.; pose=[0.,0.,0.,np.log(scale),360.,360.]
    _,ma,qa=shade(original-center,original_face,pose)
    _,mb,qb=shade(egg-center,face_tri,pose)
    points=boundary(ma).astype(int)
    outside=__import__('cv2').distanceTransform((~mb).astype(np.uint8),__import__('cv2').DIST_L2,__import__('cv2').DIST_MASK_PRECISE)
    inward=outside[points[:,1],points[:,0]]
    width=float(np.ptp(points[:,0]))
    return {'reference_face_width_px':width,'max_inward_px':float(inward.max()),
            'max_inward_fraction_of_width':float(inward.max()/width),
            'within_five_percent':bool(inward.max()/width<=.05),
            'definition':'maximum original outer Face contour distance outside the new Face silhouette, divided by original Face width'}


def prepare_template():
    neutral,tri,modes=load_model(); regions,faces=define_regions(neutral,tri)
    original_edges=mesh_edges(tri)
    face_mask=np.zeros(len(neutral),bool); face_mask[regions['face']]=True
    outer=boundary_vertices(face_mask,original_edges)
    patch=tri[face_mask[tri].all(1)]
    loops=boundary_loops(patch)
    near_outer=two_ring_boundary(face_mask,original_edges,outer)
    internal=[loop for loop in loops if not np.intersect1d(loop,near_outer).size]
    if not internal:raise ValueError('No internal Face holes found')
    augmented,new_tri,new_modes=cap_loops(neutral,tri,modes,internal)
    extended_face=np.r_[face_mask,np.ones(len(internal),bool)]
    edges=mesh_edges(new_tri)
    fixed_rings=two_ring_boundary(extended_face,edges,outer)
    free=np.setdiff1d(np.flatnonzero(extended_face),fixed_rings)
    # All 303 RHS columns use exactly the same factorization and boundary map.
    fields=np.column_stack([augmented,new_modes.transpose(1,0,2).reshape(len(augmented),-1)])
    solved=biharmonic_fields(fields,edges,free)
    egg=solved[:,:3].astype(np.float32)
    egg_modes=solved[:,3:].reshape(len(augmented),100,3).transpose(1,0,2).astype(np.float32)
    np.testing.assert_array_equal(solved[fixed_rings],fields[fixed_rings])
    remainder=boundary_loops(new_tri[extended_face[new_tri].all(1)])
    open_internal=[loop for loop in remainder if not np.intersect1d(loop,near_outer).size]
    head_mask=np.r_[np.isin(np.arange(len(neutral)),regions['head']),np.ones(len(internal),bool)]
    head_faces=np.flatnonzero(head_mask[new_tri].all(1))
    visible=np.unique(new_tri[head_faces]); center=(neutral[regions['head']].min(0)+neutral[regions['head']].max(0))/2
    radius=max(np.linalg.norm(v[visible]-center,axis=1).max() for v in [augmented,egg])
    cell=600; scale=(cell-60)/(2*radius)
    panels=[]
    for name,v,draw_tri in [('ICT ORIGINAL',neutral,tri[faces['head']]),('EGG BIHARMONIC',egg,new_tri[head_faces])]:
        row=[]
        for title,yaw,pitch in [('FRONT',0,0),('YAW 45',45,0),('SIDE 90',90,0),('FROM BELOW 30',0,30)]:
            image,_,_=shade(v-center,draw_tri,[yaw,pitch,0,np.log(scale),cell/2,cell/2],cell)
            label(image,name+' / '+title); row.append(image)
        panels.append(np.hstack(row))
    save_image('egg-template.png',np.vstack(panels))
    nose=int(np.argmin(np.linalg.norm(egg-neutral[NOSE_VERTEX],axis=1)))
    metrics=front_metrics(neutral,egg,new_tri,extended_face)
    RESULTS.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(RESULTS/'egg-template.npz',neutral_vertices=egg,identity_modes=egg_modes,
                        triangles=new_tri,original_augmented=augmented,face_mask=extended_face,
                        head_mask=head_mask,fixed_rings=fixed_rings,edges=edges)
    write_json(RESULTS/'egg-template.json',{
        'operator':'uniform combinatorial L squared, interior solution Q_II x_I=-Q_IF x_F, Q=L@L',
        'capped_holes':[{'vertices':loop.tolist(),'count':len(loop),'original_center':neutral[loop].mean(0).tolist()} for loop in internal],
        'remaining_internal_holes':len(open_internal),'outer_ring_count':len(outer),
        'fixed_two_ring_count':len(fixed_rings),'free_count':len(free),
        'mode_count':100,'same_factorization_for_neutral_and_modes':True,
        'front_silhouette':metrics,'nose_vertex':nose,'original_nose_vertex':NOSE_VERTEX,
        'original_nose_xyz':neutral[NOSE_VERTEX].tolist(),'selected_egg_nose_xyz':egg[nose].tolist(),
        'visual_conditions':'pending direct sheet inspection',
        'source':'https://libigl.github.io/tutorial/#biharmonic-surfaces'})
    print('HOLES',[(len(loop),neutral[loop].mean(0).tolist()) for loop in internal],flush=True)
    print('REMAINING INTERNAL HOLES',len(open_internal),'FRONT',metrics,'NOSE',nose,flush=True)


if __name__=='__main__':
    import torch,cv2
    torch.set_num_threads(2); torch.set_num_interop_threads(1); cv2.setNumThreads(2)
    prepare_template()
