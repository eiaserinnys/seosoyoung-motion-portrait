"""Measured 2e geometry evidence with fixed-light depth-tested renders."""
import hashlib
import shutil
import cv2
import numpy as np
from scipy.spatial import cKDTree
from fit_core import FIT, WORKSPACE, read_json, write_json, boundary
from fit2d_sheets import image_for
from fit2f_core import surface_contour
from fit2f_sheets import draw_points, padded_stack, common_view, heat, panel
from fit_sheets import label, shade
import fit2e_core as e


def save(name,image):
    path=FIT/'sheets'/name; path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),image): raise RuntimeError(path)
    copy=WORKSPACE/'.local/artifacts/fit2e'/name; copy.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(path,copy)
    return {'path':str(path),'artifact_path':str(copy),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def pose_sheet(model,targets,entries):
    lookup={t['frame']:t for t in targets}; images=[]
    worst=sorted(entries,key=lambda row:row['before']['hair']['median_px'],reverse=True)[:12]
    for entry in worst:
        target=lookup[entry['frame']]; panels=[]
        for side,key in [('BEFORE','registered_pose'),('AFTER','refined_pose')]:
            image=image_for(entry['frame'])
            contour=surface_contour(model['vertices'],model['triangles'],entry[key])
            draw_points(image,target['points'][target['labels']==0],(240,80,0))
            draw_points(image,target['points'][target['labels']==1],(0,190,25))
            draw_points(image,contour['xy'],(20,25,240))
            metrics=entry['before' if side=='BEFORE' else 'after']
            label(image,f'{entry["frame"]} {side} hair {metrics["hair"]["median_px"]:.2f} skin {metrics["skin"]["median_px"]:.2f}')
            panels.append(image)
        images.append(np.hstack(panels))
    return save('fit2e-pose-refine.png',padded_stack(images))


def grid(images,columns):
    height=max(im.shape[0] for im in images); width=max(im.shape[1] for im in images)
    canvas=np.full((height*int(np.ceil(len(images)/columns)),width*columns,3),248,np.uint8)
    for i,im in enumerate(images):
        y,x=divmod(i,columns); canvas[y*height:y*height+im.shape[0],x*width:x*width+im.shape[1]]=im
    return canvas


def k9_sheets():
    import time
    from fit2f_core import geometric_operator, load_targets_f
    from run_fit2e import RESULTS, refined_context
    started=time.perf_counter()
    if (RESULTS/'k9-sheet-sha256.json').exists(): raise FileExistsError('K9 sheets already rendered')
    model,ear,shared,poses,split=refined_context(); source=np.load(RESULTS/'k9-state.npz')
    d,nodes=source['D_k'],source['theta_k']; base=model['vertices']; tri=model['triangles']; sref=float(model['s_ref'])
    excluded=source['excluded']; lap,mass=geometric_operator(base*sref,tri)
    center,scale=common_view(); scale*=300/400; synthetic_stats=[]; view_cache={}
    def field(yaw,pitch):
        return np.einsum('k,kic->ic',e.phi(np.array([[yaw,pitch]]),nodes)[0],d)
    def render(yaw,pitch,color=False):
        key=(yaw,pitch); displacement=field(yaw,pitch)
        stats,curvature=e.field_stats(lap,mass,displacement,excluded)
        if key not in view_cache:
            synthetic_stats.append({'yaw':yaw,'pitch':pitch,'curvature':stats,'extrapolated':abs(yaw)>45})
        colors=heat(curvature,.05,excluded) if color else None
        image,mask,_=shade(base+displacement/sref-center,tri,[yaw,pitch,0,np.log(scale),150,150],300,
                           colors=colors,cull_backfaces=True)
        if key not in view_cache: view_cache[key]=(mask,displacement)
        label(image,f'y{yaw:g} p{pitch:g}',(6,18))
        label(image,f'p99 {stats["p99"]:.3f} max {stats["max"]:.3f}',(6,290))
        return image
    yaws=list(range(-45,46,5)); rotation_rows=[]
    for pitch in [-25,0,25]: rotation_rows.append(np.hstack([render(yaw,pitch) for yaw in yaws]))
    extras=[-60,-55,-50,50,55,60]; extrap=[render(yaw,0) for yaw in extras]
    for image in extrap: label(image,'extrapolated',(6,42))
    extra_row=np.full_like(rotation_rows[0],248)
    extra_row[:,:300*len(extrap)]=np.hstack(extrap); rotation_rows.append(extra_row)
    rotation_rows.append(np.hstack([render(yaw,0,True) for yaw in yaws]))
    movement=[]; intervals=[]; stacks=[]
    palette=[(230,80,30),(20,160,60),(30,50,230),(180,30,180),(30,180,180),(180,100,30)]
    for pitch in [-25,0,25]:
        image=np.full((300,300,3),248,np.uint8); previous=None
        for i,yaw in enumerate(yaws):
            mask,displacement=view_cache[(yaw,pitch)]; points=boundary(mask)
            draw_points(image,points,palette[i%len(palette)])
            if previous is not None:
                other_points,other_d=previous
                movement.append(float(np.linalg.norm(displacement-other_d,axis=1).max()))
                gaps=np.r_[cKDTree(points).query(other_points)[0],cKDTree(other_points).query(points)[0]]
                intervals.append({'pitch':pitch,'yaw_from':yaws[i-1],'yaw_to':yaw,
                                  'min_render_px':float(gaps.min()),'max_render_px':float(gaps.max())})
            previous=(points,displacement)
        label(image,f'pitch {pitch:g} silhouette stack',(5,20)); stacks.append(image)
    stack_row=np.full_like(rotation_rows[0],248); stack_row[:,:900]=np.hstack(stacks); rotation_rows.append(stack_row)
    manifest=[save('fit2e-rotation-k9.png',np.vstack(rotation_rows))]
    targets=load_targets_f(split['keys'],shared['thickness_px']); rows=[[] for _ in range(5)]
    for target in targets:
        f=target['frame']; pose=poses[f]; displacement=field(*pose[:2]); v=base+displacement/sref
        stats,curvature=e.field_stats(lap,mass,displacement,excluded)
        rendered,mask,_=shade(v,tri,pose,720,cull_backfaces=True)
        original=image_for(f); blended=original.copy()
        blended[mask]=np.rint(.48*original[mask]+.52*rendered[mask]).astype(np.uint8)
        status=' EXCLUDED' if f in split['excluded'] else ''
        first=cv2.resize(blended,(300,300)); label(first,f'key {f}{status}',(6,20)); rows[0].append(first)
        for index,yaw,colors in [(1,0,None),(2,90,None),(3,0,heat(np.linalg.norm(displacement,axis=1),6.)),
                                  (4,0,heat(curvature,.05,excluded))]:
            image,_,_=shade(v-center,tri,[yaw,0,0,np.log(scale),150,150],300,colors=colors,cull_backfaces=True)
            title=['','','','|D| 0-6px','|L D/M| 0-.05'][index]
            label(image,f'{f}{status} {title}',(6,20))
            if index==4: label(image,f'p99 {stats["p99"]:.3f} max {stats["max"]:.3f}',(6,290))
            rows[index].append(image)
    manifest.append(save('fit2e-keys-k9.png',np.vstack([np.hstack(row) for row in rows])))
    def crop_sheet(frames,name,title,full_head=False):
        targets=load_targets_f(frames,shared['thickness_px']); images=[]
        for target in targets:
            f=target['frame']; pose=poses[f]; v=base+field(*pose[:2])/sref
            if not full_head:
                status=' EXCLUDED' if f in split['excluded'] else ''
                image=panel({'vertices':v,'triangles':tri},target,pose,f'{title} {f}{status}')
            else:
                image=image_for(f); contours=surface_contour(v,tri,pose)
                draw_points(image,contours['xy'],(20,25,240))
                draw_points(image,target['points'][target['labels']==0],(240,80,0))
                draw_points(image,target['points'][target['labels']==1],(0,190,25))
                points=np.vstack([target['head_boundary'],contours['xy']])
                lo=np.maximum(np.floor(points.min(0)).astype(int)-25,0)
                hi=np.minimum(np.ceil(points.max(0)).astype(int)+26,720)
                image=cv2.resize(image[lo[1]:hi[1],lo[0]:hi[0]],None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
                status=' EXCLUDED' if f in split['excluded'] else ''
                label(image,f'{title} {f}{status}')
            images.append(image)
        manifest.append(save(name,grid(images,3)))
    crop_sheet(split['keys'],'fit2e-jaw-keys-k9.png','KEY')
    heldout=sorted(split['heldout'],key=lambda f:abs(poses[f][0])+abs(poses[f][1]),reverse=True)[:12]
    crop_sheet(heldout,'fit2e-jaw-heldout-k9.png','HELD-OUT')
    profile=[172,184,192,240,219]
    right=max(poses,key=lambda f:poses[f][0])
    if right not in profile: profile.append(right)
    crop_sheet(profile,'fit2e-profile-k9.png','PROFILE',full_head=True)
    metadata={'rotation_poses':synthetic_stats,'pose_maximum_curvature':max(r['curvature']['max'] for r in synthetic_stats),
              'all_rotation_poses_passed':all(r['curvature']['passed'] for r in synthetic_stats),
              'maximum_vertex_movement_per_5deg_px':max(movement),'silhouette_adjacent_intervals':intervals,
              'silhouette_interval_min_render_px':min(r['min_render_px'] for r in intervals),
              'silhouette_interval_max_render_px':max(r['max_render_px'] for r in intervals),
              'profile_frames':profile,'heldout_crop_frames':heldout,
              'elapsed_seconds':time.perf_counter()-started,'sheets':manifest}
    write_json(RESULTS/'k9-sheet-sha256.json',metadata)
    report=read_json(RESULTS/'k9-residuals.json'); report['angle_axis']=metadata
    write_json(RESULTS/'k9-residuals.json',report)
    print('SHEETS',metadata,flush=True)
