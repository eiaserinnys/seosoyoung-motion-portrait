"""Exact 720px residuals and fixed-light sheets for the saved 2c surface."""
from __future__ import annotations
import numpy as np
import cv2
import torch
from fit_core import FIT,FRAMES,read_json,write_json,project,rasterize as raw_rasterize,boundary,lower_boundary,eroded_target,load_targets,load_model,define_regions
from fit2c_core import RESULTS,load_template
from fit_sheets import shade as raw_shade,label,save_image,source_image,measure
from egg_v5 import area2,contour_change


def shade(vertices,tri,pose,size=720,colors=None):
    return raw_shade(vertices,tri,pose,size,colors,cull_backfaces=True)


def rasterize(q,tri,size=720):
    return raw_rasterize(q,tri,size,cull_backfaces=True)


def template_sheet(template,tri,masks):
    original,original_tri,_=load_model(); regions,faces=define_regions(original,original_tri)
    bundle=np.load(RESULTS/'egg-template-v5.npz')
    head_tri=tri[masks['head'][tri].all(1)]
    center=(original[regions['head']].min(0)+original[regions['head']].max(0))/2
    radius=max(np.linalg.norm(original[np.unique(original_tri[faces['head']])]-center,axis=1).max(),
               np.linalg.norm(template[np.unique(head_tri)]-center,axis=1).max())
    cell=600; scale=(cell-60)/(2*radius); colors=np.full_like(original,205.)
    colors[bundle['removed_vertex_ids']]=(70.,170.,250.); panels=[]
    for name,v,t,color in [('ICT ORIGINAL',original,original_tri[faces['head']],None),
                          ('REPLACED FEATURE PATCH',original,original_tri[faces['head']],colors),
                          ('EGG V5 / DEPTH + CULL',template,head_tri,None)]:
        row=[]
        for title,yaw,pitch in [('FRONT',0,0),('YAW 45',45,0),('SIDE 90',90,0),('FROM BELOW 30',0,30)]:
            image,_,_=shade(v-center,t,[yaw,pitch,0,np.log(scale),cell/2,cell/2],cell,color)
            label(image,name+' / '+title); row.append(image)
        panels.append(np.hstack(row))
    save_image('egg-template-v5.png',np.vstack(panels))
    pose=[0,0,0,np.log(20.),360,360]
    original_face=np.isin(np.arange(len(original)),regions['face'])
    _,reference,_=shade(original-center,original_tri[original_face[original_tri].all(1)],pose)
    _,result,_=shade(template-center,tri[masks['face'][tri].all(1)],pose)
    return contour_change(reference,result)


def v4_culling_diagnostic():
    bundle=np.load(RESULTS/'egg-template-v4.npz')
    v=bundle['neutral_vertices']; tri=bundle['triangles']; head=bundle['head_mask']
    tri=tri[head[tri].all(1)]; visible=np.unique(tri)
    center=(v[visible].min(0)+v[visible].max(0))/2
    cell=600; scale=(cell-60)/(2*np.linalg.norm(v[visible]-center,axis=1).max())
    panels=[]; counts=[]
    for cull in (False,True):
        row=[]
        for title,yaw,pitch in [('FRONT',0,0),('YAW 45',45,0),('SIDE 90',90,0),('FROM BELOW 30',0,30)]:
            pose=[yaw,pitch,0,np.log(scale),cell/2,cell/2]
            image,_,q=raw_shade(v-center,tri,pose,cell,cull_backfaces=cull)
            _,owner=raw_rasterize(q,tri,cell,cull_backfaces=cull)
            projected_area=area2(q[tri[:,0],:2],q[tri[:,1],:2],q[tri[:,2],:2])
            back=np.flatnonzero(projected_area>=0)
            pixels=int(np.isin(owner,back).sum())
            counts.append({'view':title,'backface_culling':cull,'visible_backface_pixels':pixels})
            label(image,('DEPTH + CULL' if cull else 'DEPTH ONLY')+' / '+title); row.append(image)
        panels.append(np.hstack(row))
    save_image('v4-culling-diagnostic.png',np.vstack(panels))
    write_json(RESULTS/'v4-culling-diagnostic.json',{'views':counts,
               'purpose':'Existing v4 geometry only; no new template, smoothing, optimization or geometry change.'})


def directed_distances(points,reference):
    field=np.ones((720,720),np.uint8)
    other=np.rint(reference).astype(int); field[other[:,1],other[:,0]]=0
    dt=cv2.distanceTransform(field,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    ids=np.rint(points).astype(int)
    return dt[ids[:,1],ids[:,0]]


def jaw_details(model,wanted):
    a=directed_distances(model,wanted); b=directed_distances(wanted,model)
    distances=np.r_[a,b]; points=np.r_[model,wanted]
    worst=int(distances.argmax()); low,high=wanted[:,0].min(),wanted[:,0].max()
    thirds=np.linspace(low,high,4); regions={}
    for index,name in enumerate(('screen-left','center-chin','screen-right')):
        keep=(points[:,0]>=thirds[index])&(points[:,0]<=thirds[index+1])
        regions[name]={'mean_px':float(distances[keep].mean()),'max_px':float(distances[keep].max())} if keep.any() else None
    return {'median_px':float(np.median(distances)),'mean_px':float((a.mean()+b.mean())/2),
            'max_px':float(distances.max()),'worst_point_xy':points[worst].tolist(),
            'worst_direction':'model_to_target' if worst<len(model) else 'target_to_model',
            'regions':regions,'target_median_le_2':bool(np.median(distances)<=2.),'target_max_le_5':bool(distances.max()<=5.)}


def shape_sheet(v,template,tri,head_faces):
    views=[(0,0),(-45,0),(45,0),(-67.5,0),(67.5,0),(-78.75,0),(78.75,0),(0,30),(0,-30),(45,-30)]
    visible=np.unique(tri[head_faces]); center=(template[visible].min(0)+template[visible].max(0))/2
    radius=np.linalg.norm(np.r_[v[visible],template[visible]]-center,axis=1).max()
    cell=400; scale=(cell-50)/(2*radius); rows=[]
    for name,vertices in [('FITTED 2c',v),('EGG V5 TEMPLATE',template)]:
        panels=[]
        for yaw,pitch in views:
            image,_,_=shade(vertices-center,tri[head_faces],[yaw,pitch,0,np.log(scale),cell/2,cell/2],cell)
            label(image,f'{name} y {yaw:g} p {pitch:g}'); panels.append(image)
        rows.append(np.hstack(panels))
    save_image('fit2c-shape.png',np.vstack(rows))


def run():
    state=np.load(RESULTS/'fit-state.npz'); vertices=state['vertices']; tri=state['triangles']
    template,template_tri,_,masks,faces,_,_,_,observations,nose=load_template()
    np.testing.assert_array_equal(tri,template_tri)
    shared=read_json(RESULTS/'shared.json'); poses_doc=read_json(RESULTS/'poses.json')
    poses=np.array([poses_doc['frames'][str(f)] for f in FRAMES],np.float32)
    targets=load_targets(); thickness=shared['thickness_px']; rows=measure(vertices,tri,faces,poses,targets,thickness,cull_backfaces=True)
    template_metrics=template_sheet(template,tri,masks)
    v4_culling_diagnostic()
    overlays=[]; crops=[]
    patch_faces=np.load(RESULTS/'egg-template-v5.npz')['patch_faces']
    patch_tri=tri[patch_faces]
    areas=area2(vertices[patch_tri[:,0],:2],vertices[patch_tri[:,1],:2],vertices[patch_tri[:,2],:2])
    reversed_faces=patch_faces[areas<=0]
    for row,pose,target in zip(rows,poses,targets):
        frame=target['frame']; image=source_image(target)
        gray,head_mask,q=shade(vertices,tri[faces['head']],pose)
        _,head_owner=rasterize(q,tri[faces['head']])
        owned=faces['head'][head_owner[head_owner>=0]]
        row['visible_reversed_patch_faces']=int(np.isin(np.unique(owned),reversed_faces).sum())
        overlay=image.copy(); overlay[head_mask]=(.48*image[head_mask]+.52*gray[head_mask]).astype(np.uint8)
        outline=image.copy(); head_target=eroded_target(target['head'],thickness)
        cv2.polylines(outline,[boundary(head_target).astype(np.int32)],True,(0,210,0),2,cv2.LINE_AA)
        cv2.polylines(outline,[target['lower'].astype(np.int32)],False,(255,70,0),2,cv2.LINE_AA)
        for ear in target['ears'].values():cv2.polylines(outline,[boundary(ear).astype(np.int32)],True,(0,230,255),2,cv2.LINE_AA)
        cv2.polylines(outline,[boundary(head_mask).astype(np.int32)],True,(0,0,240),2,cv2.LINE_AA)
        observation=observations[frame]
        if observation['used']:
            point=tuple(np.rint(observation['xy']).astype(int))
            for panel in (overlay,outline):cv2.drawMarker(panel,point,(255,255,255),cv2.MARKER_TILTED_CROSS,14,2,cv2.LINE_AA)
        predicted=tuple(np.rint(q[nose,:2]).astype(int))
        for panel in (overlay,outline):cv2.circle(panel,predicted,4,(0,0,255),-1,cv2.LINE_AA)
        label(overlay,f'frame {frame}: fitted shared mesh / alpha .52')
        label(outline,'H-t green / lower S blue / E yellow / model red')
        overlays.append(np.hstack([overlay,outline]))
        _,owner=rasterize(q,tri[faces['face']]); lower=lower_boundary(owner>=0)
        details=jaw_details(lower,target['lower']); row['skin_lower']=details
        row['nose']={'score':observation['score'],'used':observation['used'],'detected_xy':observation['xy'],
                     'projected_xy':q[nose,:2].tolist(),
                     'error_px':float(np.linalg.norm(q[nose,:2]-observation['xy'])) if observation['used'] else None}
        row['targets']={'head_iou_ge_096':bool(row['head_iou']>=.96),
                        'jaw_median_le_2':details['target_median_le_2'],'jaw_max_le_5':details['target_max_le_5'],
                        'nose_error_le_3':bool(row['nose']['error_px']<=3.) if observation['used'] else None,
                        'visible_ear_iou_ge_030':{key:bool(value['visible_iou']>=.3) for key,value in row['ears'].items()}}
        row.update(beta_abs_max=shared['beta_abs_max'],scale_xyz=shared['scale_xyz'],thickness_px=thickness,
                   iteration_mean_seconds=shared['iteration_mean_seconds'])
        jaw=image.copy()
        cv2.polylines(jaw,[target['lower'].astype(np.int32)],False,(255,70,0),2,cv2.LINE_AA)
        cv2.polylines(jaw,[lower.astype(np.int32)],False,(0,0,240),2,cv2.LINE_AA)
        joint=np.r_[target['lower'],lower]
        x0,y0=np.maximum(np.floor(joint.min(0)).astype(int)-18,0)
        x1,y1=np.minimum(np.ceil(joint.max(0)).astype(int)+18,719)
        crop=cv2.resize(jaw[y0:y1+1,x0:x1+1],None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
        label(crop,f'frame {frame} / jaw 2x / target blue / model red'); crops.append(crop)
    save_image('fit2c-overlay.png',np.vstack(overlays))
    width=max(crop.shape[1] for crop in crops)
    padded=[cv2.copyMakeBorder(crop,0,0,0,width-crop.shape[1],cv2.BORDER_CONSTANT,value=(248,248,248)) for crop in crops]
    save_image('fit2c-jaw.png',np.vstack(padded)); shape_sheet(vertices,template,tri,faces['head'])
    report={'frames':rows,'delta_stats':shared['delta_stats'],'beta_abs_max':shared['beta_abs_max'],
            'mesh_quality':{'patch_nonpositive_front_xy_triangles':len(reversed_faces),
                            'note':'Diagnostic only; this is not a failure gate after the director removed the overlap/reversal constraint.'},
            'rendering':{'depth_test':True,'backface_culling':True,'template_front_silhouette':template_metrics,
                         'optimization':'In-flight bounded fit retained its depth-only silhouette objective; final assessment and sheets use depth plus culling.'},
            'scale_xyz':shared['scale_xyz'],'thickness_px':thickness,
            'iteration_mean_seconds':shared['iteration_mean_seconds'],
            'stage_mean_seconds':shared['iteration_mean_by_stage'],
            'estimated_130_frames_800_iterations_seconds':shared['iteration_mean_by_stage']['normal_delta_and_nose']*130/3*800,
            'rotation_max_abs_delta_deg':shared['rotation_max_abs_delta_deg'],
            'criteria':{'1_all_three_head_iou_below_090':all(row['head_iou']<.9 for row in rows),
                        '3_down_right_visible_ear_iou_below_030':any(value['visible_iou']<.3 for row in rows if row['frame']==264 for value in row['ears'].values())},
            'notes':['Exact DT residuals in 720px coordinates; symmetric jaw median pools both directed contour samples.',
                     'Visible ear IoU includes head occlusion; isolated ear IoU is reported separately.',
                     'No weight search, new template attempt, full-video fit or rig connection after this bounded fit.']}
    write_json(RESULTS/'residuals.json',report)
    print('RESIDUALS',report,flush=True)


if __name__=='__main__':
    torch.set_num_threads(2); torch.set_num_interop_threads(1); cv2.setNumThreads(2)
    run()
