"""Replace folded feature topology by a single UV patch with a fixed linear TPS map.

Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html
UV locations are fixed in neutral XY. Thus the very same interpolator acts on
neutral XYZ and every displacement field, rather than re-solving on each shape.
"""
from __future__ import annotations
import numpy as np
import cv2
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay
from fit_core import FIT,load_model,define_regions,write_json,boundary,rasterize
from fit_sheets import shade,label,save_image
from egg_pca import mesh_edges,NOSE_VERTEX
from egg_biharmonic import boundary_loops

RESULTS=FIT/'results/fit2c'


def area2(a,b,c):
    return (b[...,0]-a[...,0])*(c[...,1]-a[...,1])-(b[...,1]-a[...,1])*(c[...,0]-a[...,0])


def triangulate_polygon(xy):
    """Ear clipping retains every original boundary edge, including concavities."""
    remaining=list(range(len(xy))); triangles=[]
    while len(remaining)>3:
        for j,i in enumerate(remaining):
            prev=remaining[j-1]; nxt=remaining[(j+1)%len(remaining)]
            a,b,c=xy[[prev,i,nxt]]
            if area2(a,b,c)<=1e-10:continue
            other=xy[[k for k in remaining if k not in (prev,i,nxt)]]
            inside=(area2(a,b,other)>=-1e-10)&(area2(b,c,other)>=-1e-10)&(area2(c,a,other)>=-1e-10)
            if inside.any():continue
            triangles.append([prev,i,nxt]); remaining.pop(j); break
        else:raise ValueError('Feature rim is not a simple triangulatable CCW polygon')
    triangles.append(remaining)
    return np.asarray(triangles,dtype=np.int64)


def refine_centroids(xy,tri,levels):
    """Refine interiors only: avoid T junctions with retained outside triangles."""
    for _ in range(levels):
        centers=np.arange(len(xy),len(xy)+len(tri))
        xy=np.r_[xy,xy[tri].mean(1)]
        tri=np.concatenate([np.c_[tri[:,0],tri[:,1],centers],
                            np.c_[tri[:,1],tri[:,2],centers],
                            np.c_[tri[:,2],tri[:,0],centers]])
    return xy,tri


def interpolate_fields(anchors,queries,fields):
    return RBFInterpolator(anchors,fields,kernel='thin_plate_spline',degree=1,smoothing=0.)(queries)


def crossing(a,b,c,d):
    return area2(a,b,c)*area2(a,b,d)<-1e-12 and area2(c,d,a)*area2(c,d,b)<-1e-12


def grid_patch(polygon,spacing):
    """Recover the exact rim in a planar Delaunay mesh of a staggered grid."""
    contour=polygon.astype(np.float32)
    rows=[]
    for row,y in enumerate(np.arange(polygon[:,1].min()+spacing/2,polygon[:,1].max(),spacing)):
        for x in np.arange(polygon[:,0].min()+spacing/2+(row%2)*spacing/2,polygon[:,0].max(),spacing):
            if cv2.pointPolygonTest(contour,(float(x),float(y)),True)>spacing*.15:rows.append([x,y])
    xy=np.r_[polygon,np.asarray(rows)]; tri=Delaunay(xy).simplices.copy()
    wrong=area2(xy[tri[:,0]],xy[tri[:,1]],xy[tri[:,2]])<0
    tri[wrong]=tri[wrong][:,[0,2,1]]
    constraints={tuple(sorted((i,(i+1)%len(polygon)))) for i in range(len(polygon))}
    for start,end in sorted(constraints):
        for _ in range(len(tri)*2):
            owners={}
            for f,triangle in enumerate(tri):
                for a,b in zip(triangle,np.roll(triangle,-1)):
                    owners.setdefault(tuple(sorted((int(a),int(b)))),[]).append(f)
            if (start,end) in owners:break
            candidates=[]
            for (a,b),adjacent in owners.items():
                if len(adjacent)!=2 or (a,b) in constraints:continue
                if not crossing(xy[start],xy[end],xy[a],xy[b]):continue
                c=next(int(k) for k in tri[adjacent[0]] if k not in (a,b))
                d=next(int(k) for k in tri[adjacent[1]] if k not in (a,b))
                if not crossing(xy[a],xy[b],xy[c],xy[d]):continue
                candidates.append((crossing(xy[start],xy[end],xy[c],xy[d]),adjacent,c,d,a,b))
            if not candidates:raise ValueError('Cannot recover feature rim edge')
            _,adjacent,c,d,a,b=min(candidates,key=lambda item:item[0])
            replacement=np.array([[c,d,a],[d,c,b]])
            wrong=area2(xy[replacement[:,0]],xy[replacement[:,1]],xy[replacement[:,2]])<0
            replacement[wrong]=replacement[wrong][:,[0,2,1]]
            tri[adjacent]=replacement
        else:raise ValueError('Rim recovery did not converge')
    inside=[cv2.pointPolygonTest(contour,tuple(map(float,point)),False)>=0 for point in xy[tri].mean(1)]
    tri=tri[inside]
    edges=np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1)
    unique,counts=np.unique(edges,axis=0,return_counts=True)
    if set(map(tuple,unique[counts==1]))!=constraints or (counts>2).any():
        raise ValueError('Patch does not preserve the exact manifold rim')
    return xy,tri


def contour_change(reference,new):
    def directed(a,b):
        points=boundary(a).astype(int)
        dt=cv2.distanceTransform((~b).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        return float(dt[points[:,1],points[:,0]].max())
    width=float(np.ptp(boundary(reference)[:,0]))
    inward=directed(reference,new); outward=directed(new,reference)
    return {'reference_face_width_px':width,'max_inward_px':inward,'max_outward_px':outward,
            'max_change_fraction_of_reference_width':max(inward,outward)/width,
            'within_five_percent':bool(max(inward,outward)/width<=.05)}


def prepare_template():
    original,tri,modes=load_model(); old_regions,old_faces=define_regions(original,tri)
    previous=np.load(RESULTS/'egg-template-v4.npz')
    selected=previous['feature_mask'][:len(original)]
    keep_faces=~selected[tri].any(1); retained=tri[keep_faces]
    loops=boundary_loops(retained)
    rim=max(loops,key=lambda loop:float(original[loop,1].mean()))
    uv=original[rim,:2].astype(np.float64)
    if area2(np.zeros(2),uv,np.roll(uv,-1,axis=0)).sum()<0:
        rim=rim[::-1]; uv=uv[::-1]
    patch_uv,patch_tri=grid_patch(uv,.23)
    # Two unmodified exterior rings constrain both height and its local slope.
    edges=mesh_edges(tri); collar=np.zeros(len(original),bool); collar[rim]=True
    for _ in range(2):
        collar[edges[collar[edges].any(1)].ravel()]=True
        collar &= ~selected
    collar &= np.isin(np.arange(len(original)),old_regions['face'])
    anchors=np.flatnonzero(collar)
    _,unique=np.unique(np.round(original[anchors,:2],7),axis=0,return_index=True)
    anchors=anchors[np.sort(unique)]
    fields=np.column_stack([original,modes.transpose(1,0,2).reshape(len(original),-1)])
    interpolated=interpolate_fields(original[anchors,:2].astype(float),patch_uv[len(rim):],fields[anchors])
    kept=np.flatnonzero(~selected); old_to_new=np.full(len(original),-1,int); old_to_new[kept]=np.arange(len(kept))
    patch_ids=np.r_[old_to_new[rim],np.arange(len(kept),len(kept)+len(interpolated))]
    out_fields=np.r_[fields[kept],interpolated]
    vertices=out_fields[:,:3].astype(np.float32)
    output_modes=out_fields[:,3:].reshape(len(vertices),100,3).transpose(1,0,2).astype(np.float32)
    output_tri=np.r_[old_to_new[retained],patch_ids[patch_tri]]
    np.testing.assert_array_equal(vertices[:len(kept)],original[kept])
    np.testing.assert_array_equal(output_modes[:,:len(kept)],modes[:,kept])
    np.testing.assert_allclose(vertices[len(kept):,:2],patch_uv[len(rim):],atol=2e-6)
    def transferred(ids):return np.r_[np.isin(kept,ids),np.ones(len(interpolated),bool)]
    head=transferred(old_regions['head']); face=transferred(old_regions['face'])
    ear_left=np.r_[np.isin(kept,old_regions['ear-left']),np.zeros(len(interpolated),bool)]
    ear_right=np.r_[np.isin(kept,old_regions['ear-right']),np.zeros(len(interpolated),bool)]
    head_tri=output_tri[head[output_tri].all(1)]
    center=(original[old_regions['head']].min(0)+original[old_regions['head']].max(0))/2
    radius=max(np.linalg.norm(original[np.unique(tri[old_faces['head']])]-center,axis=1).max(),
               np.linalg.norm(vertices[np.unique(head_tri)]-center,axis=1).max())
    cell=600; scale=(cell-60)/(2*radius); colors=np.full_like(original,205.)
    colors[selected]=(70.,170.,250.)
    panels=[]
    for name,v,t,color in [('ICT ORIGINAL',original,tri[old_faces['head']],None),
                            ('REPLACED FEATURE PATCH',original,tri[old_faces['head']],colors),
                            ('EGG V5 / SINGLE UV PATCH',vertices,head_tri,None)]:
        row=[]
        for title,yaw,pitch in [('FRONT',0,0),('YAW 45',45,0),('SIDE 90',90,0),('FROM BELOW 30',0,30)]:
            image,_,_=shade(v-center,t,[yaw,pitch,0,np.log(scale),cell/2,cell/2],cell,color)
            label(image,name+' / '+title); row.append(image)
        panels.append(np.hstack(row))
    save_image('egg-template-v5.png',np.vstack(panels))
    pose=[0.,0.,0.,np.log(20.),360.,360.]
    original_face=np.isin(np.arange(len(original)),old_regions['face'])
    _,reference,_=shade(original-center,tri[original_face[tri].all(1)],pose)
    _,result,_=shade(vertices-center,output_tri[face[output_tri].all(1)],pose)
    silhouette=contour_change(reference,result)
    patch_area=area2(vertices[patch_ids[patch_tri[:,0]],:2],vertices[patch_ids[patch_tri[:,1]],:2],vertices[patch_ids[patch_tri[:,2]],:2])
    all_edges=np.sort(np.concatenate([output_tri[:,[0,1]],output_tri[:,[1,2]],output_tri[:,[2,0]]]),axis=1)
    _,counts=np.unique(all_edges,axis=0,return_counts=True)
    nose=int(np.argmin(np.linalg.norm(vertices-original[NOSE_VERTEX],axis=1)))
    np.savez_compressed(RESULTS/'egg-template-v5.npz',neutral_vertices=vertices,identity_modes=output_modes,
                        triangles=output_tri,face_mask=face,head_mask=head,ear_left_mask=ear_left,ear_right_mask=ear_right,
                        original_ids=np.r_[kept,np.full(len(interpolated),-1)],patch_vertex_ids=patch_ids,
                        patch_faces=np.arange(len(retained),len(output_tri)),removed_vertex_ids=np.flatnonzero(selected))
    report={'method':'single neutral-XY staggered-grid constrained Delaunay patch, exact original rim, two-ring collar TPS, frozen UV linear map on all 303 RHS',
            'deleted_original_vertices':int(selected.sum()),'deleted_original_faces':int((~keep_faces).sum()),
            'new_patch_interior_vertices':len(interpolated),'new_patch_faces':len(patch_tri),'boundary_vertices':len(rim),
            'interpolation_anchors':len(anchors),'vertex_count':len(vertices),'face_count':len(output_tri),
            'outside_neutral_and_100_modes_exact':True,'patch_negative_xy_triangles':int((patch_area<=0).sum()),
            'patch_min_xy_area2':float(patch_area.min()),'nonmanifold_edges':int((counts>2).sum()),
            'remaining_boundary_loop_sizes':[len(loop) for loop in boundary_loops(output_tri)],
            'front_silhouette':silhouette,'nose_vertex':nose,'original_nose_vertex':NOSE_VERTEX,
            'nose_xyz':vertices[nose].tolist(),'visual_conditions':'pending direct sheet inspection',
            'source':'https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html'}
    write_json(RESULTS/'egg-template-v5.json',report); print(report,flush=True)


if __name__=='__main__':
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(1); cv2.setNumThreads(2)
    prepare_template()
