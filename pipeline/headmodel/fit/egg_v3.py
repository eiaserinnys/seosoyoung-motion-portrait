"""Fair only the forward-facing feature hull; keep the outer head untouched."""
from __future__ import annotations
import numpy as np
from scipy.spatial import ConvexHull

from fit_core import FIT,ROOT,load_model,define_regions,read_json,write_json
from egg_pca import mesh_edges,vertex_normals,NOSE_VERTEX
from egg_biharmonic import boundary_loops,cap_loops,biharmonic_fields,front_metrics
from fit_sheets import shade,label,save_image

RESULTS=FIT/'results/fit2c'


def expanded_hull(points,margin):
    """Offset every convex polygon support line by the same world-space margin."""
    hull=points[ConvexHull(points).vertices]
    direction=np.roll(hull,-1,axis=0)-hull
    outward=np.column_stack([direction[:,1],-direction[:,0]])
    outward/=np.linalg.norm(outward,axis=1)[:,None]
    constant=(outward*hull).sum(1)+margin
    return np.array([np.linalg.solve(np.array([outward[i-1],outward[i]]),
                                    [constant[i-1],constant[i]]) for i in range(len(hull))])


def inside_polygon(points,polygon):
    edges=np.roll(polygon,-1,axis=0)-polygon
    relative=points[:,None,:]-polygon[None,:,:]
    cross=edges[None,:,0]*relative[:,:,1]-edges[None,:,1]*relative[:,:,0]
    return (cross>=-1e-9).all(1)


def feature_mask(vertices,normals,face_mask,polygon):
    return face_mask & inside_polygon(vertices[:,:2],polygon) & (normals[:,2]>0)


def extended_selection(original,vertices,tri,face,polygon,selector=feature_mask):
    # Original vertices keep the neutral mask; new cap centers use their own
    # normals rather than inheriting back-facing aperture wall membership.
    added=selector(vertices,vertex_normals(vertices,tri),face,polygon)[len(original):]
    return np.r_[original,added]


def prepare_template(version='v3',selector=feature_mask,
                     selection_text='Face excluding ears and neck, expanded feature polygon in neutral XY, original vertex normal z>0'):
    basename=f'egg-template-{version}'
    neutral,tri,modes=load_model(); regions,faces=define_regions(neutral,tri)
    landmarks=read_json(ROOT/'pipeline/headmodel/model/ict_landmarks.json')
    if landmarks['index_base']!=0 or len(landmarks['indices'])!=68:
        raise ValueError('Expected zero-based Multi-PIE 68 landmarks')
    ids=np.array(landmarks['indices']); landmark_xyz=neutral[ids]
    eye_centers=np.array([landmark_xyz[36:42,:2].mean(0),landmark_xyz[42:48,:2].mean(0)])
    eye_distance=float(np.linalg.norm(eye_centers[0]-eye_centers[1]))
    polygon=expanded_hull(landmark_xyz[17:68,:2],eye_distance*.15)
    face=np.isin(np.arange(len(neutral)),regions['face'])
    normals=vertex_normals(neutral,tri)
    selected=selector(neutral,normals,face,polygon)
    # Inspect the actual open mesh, avoiding artificial borders from the mask.
    loops=boundary_loops(tri)
    internal=[loop for loop in loops if face[loop].all() and
              inside_polygon(neutral[loop,:2],polygon).all()]
    augmented,new_tri,new_modes=cap_loops(neutral,tri,modes,internal)
    extended_face=np.r_[face,np.ones(len(internal),bool)]
    selected=extended_selection(selected,augmented,new_tri,extended_face,polygon,selector)
    head=np.r_[np.isin(np.arange(len(neutral)),regions['head']),np.ones(len(internal),bool)]
    free=np.flatnonzero(selected); fixed=np.flatnonzero(~selected)
    edges=mesh_edges(new_tri)
    fields=np.column_stack([augmented,new_modes.transpose(1,0,2).reshape(len(augmented),-1)])
    solved=biharmonic_fields(fields,edges,free)
    np.testing.assert_array_equal(solved[fixed],fields[fixed])
    egg=solved[:,:3].astype(np.float32)
    egg_modes=solved[:,3:].reshape(len(augmented),100,3).transpose(1,0,2).astype(np.float32)
    remaining=boundary_loops(new_tri)
    remaining_features=[loop for loop in remaining if extended_face[loop].all() and
                        inside_polygon(egg[loop,:2],polygon).all()]
    head_tri=new_tri[head[new_tri].all(1)]
    center=(neutral[regions['head']].min(0)+neutral[regions['head']].max(0))/2
    visible=np.unique(head_tri); cell=600
    radius=max(np.linalg.norm(v[visible]-center,axis=1).max() for v in [augmented,egg])
    scale=(cell-60)/(2*radius)
    color=np.full_like(neutral,205.); color[selected[:len(neutral)]]=(70.,170.,250.)
    panels=[]
    for name,v,draw_tri,colors in [('ICT ORIGINAL',neutral,tri[faces['head']],None),
                                 ('FEATURE MASK',neutral,tri[faces['head']],color),
                                 ('EGG '+version.upper(),egg,head_tri,None)]:
        row=[]
        for title,yaw,pitch in [('FRONT',0,0),('YAW 45',45,0),('SIDE 90',90,0),('FROM BELOW 30',0,30)]:
            image,_,_=shade(v-center,draw_tri,[yaw,pitch,0,np.log(scale),cell/2,cell/2],cell,colors)
            label(image,name+' / '+title); row.append(image)
        panels.append(np.hstack(row))
    save_image(basename+'.png',np.vstack(panels))
    nose=int(np.argmin(np.linalg.norm(egg-neutral[NOSE_VERTEX],axis=1)))
    metrics=front_metrics(neutral,egg,new_tri,extended_face)
    # Also measure outward contour changes; the older helper checks inward only.
    reverse=front_metrics(egg,augmented,new_tri,extended_face)
    metrics['max_outward_px']=reverse['max_inward_px']
    metrics['max_change_fraction_of_reference_width']=max(metrics['max_inward_px'],reverse['max_inward_px'])/metrics['reference_face_width_px']
    metrics['within_five_percent']=bool(metrics['max_change_fraction_of_reference_width']<=.05)
    jaw_untouched=bool((~selected[ids[:17]]).all())
    RESULTS.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(RESULTS/(basename+'.npz'),neutral_vertices=egg,identity_modes=egg_modes,
                        triangles=new_tri,original_augmented=augmented,face_mask=extended_face,
                        head_mask=head,feature_mask=selected,fixed_vertices=fixed,edges=edges)
    report={
        'landmark_convention':landmarks['convention'],'index_base':0,'feature_landmark_range':[17,67],
        'eye_distance_xy':eye_distance,'outward_margin':eye_distance*.15,'polygon_xy':polygon.tolist(),
        'selection':selection_text,
        'fixed':'all vertices outside selection, including all exterior boundary rings',
        'original_selected_count':int(selected[:len(neutral)].sum()),'free_count':len(free),'fixed_count':len(fixed),
        'cap_selected_count':int(selected[len(neutral):].sum()),
        'jaw_landmarks_all_fixed':jaw_untouched,
        'capped_holes':[{'vertices':loop.tolist(),'count':len(loop),'center_xyz':neutral[loop].mean(0).tolist(),
                         'selected_ring_count':int(selected[loop].sum())} for loop in internal],
        'remaining_feature_holes':len(remaining_features),
        'operator':'uniform L=D-A, Q=L@L, Q_II x_I=-Q_IF x_F; same 303-column solve for neutral and 100 modes',
        'fixed_fields_max_difference':float(np.abs(solved[fixed]-fields[fixed]).max()),
        'front_silhouette':metrics,'nose_vertex':nose,'original_nose_vertex':NOSE_VERTEX,
        'original_nose_xyz':neutral[NOSE_VERTEX].tolist(),'selected_egg_nose_xyz':egg[nose].tolist(),
        'visual_conditions':'pending direct sheet inspection'}
    write_json(RESULTS/(basename+'.json'),report)
    print('MASK',report['original_selected_count'],'FREE',len(free),'FIXED',len(fixed),'JAW FIXED',jaw_untouched,flush=True)
    print('HOLES',[(len(loop),int(selected[loop].sum())) for loop in internal],'REMAINING',len(remaining_features),flush=True)
    print('FRONT',metrics,'NOSE',nose,flush=True)


if __name__=='__main__':
    import torch,cv2
    torch.set_num_threads(2); torch.set_num_interop_threads(1); cv2.setNumThreads(2)
    prepare_template()
