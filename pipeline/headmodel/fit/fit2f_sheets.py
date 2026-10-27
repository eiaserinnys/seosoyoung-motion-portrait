"""Common-light 2f evidence and mask-defined comparison, no generated visuals."""
import hashlib
import csv
from pathlib import Path
import time
import cv2
import numpy as np
from scipy.spatial import cKDTree
from fit_core import FIT, ROOT, read_json, write_json
from fit2d_core import template, frame_sets
from fit2d_sheets import image_for
from fit_sheets import shade, label
from fit2f_core import (RESULTS, load_surface, load_targets_f, surface_contour,
                        measure_surface, curvature_stats, compact_surface)
from run_fit2f import context

TRIPO=ROOT/'pipeline/headmodel/tripo'
TOUT=TRIPO/'results/tripo'
VIEWS=[(0,0),(-45,0),(45,0),(-67.5,0),(67.5,0),(-78.75,0),(78.75,0),(0,30),(0,-30),(45,-30)]


def save(path,image):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),image): raise RuntimeError(path)
    return {'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def heat(values,maximum,excluded=None):
    t=np.clip(np.asarray(values)/maximum,0,1)
    green=np.array([45,190,35]); yellow=np.array([0,225,235]); red=np.array([30,35,240])
    colors=np.where((t<.5)[:,None],green+(yellow-green)*np.minimum(t*2,1)[:,None],
                    yellow+(red-yellow)*np.maximum(t*2-1,0)[:,None])
    if excluded is not None: colors[excluded]=[145,145,145]
    return colors


def dihedral_change(base,v,tri):
    def normals(vertices):
        n=np.cross(vertices[tri[:,1]]-vertices[tri[:,0]],vertices[tri[:,2]]-vertices[tri[:,0]])
        return n/np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-15)
    before,after=normals(base),normals(v)
    edge=np.sort(np.vstack([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1)
    owners=np.tile(np.arange(len(tri)),3)
    order=np.lexsort((edge[:,1],edge[:,0])); edge,owners=edge[order],owners[order]
    same=np.flatnonzero((edge[:-1]==edge[1:]).all(1))
    a,b=owners[same],owners[same+1]
    original=np.arccos(np.clip((before[a]*before[b]).sum(1),-1,1))
    changed=np.arccos(np.clip((after[a]*after[b]).sum(1),-1,1))
    delta=np.degrees(np.abs(changed-original))
    result=np.zeros(len(v))
    np.maximum.at(result,edge[same,0],delta); np.maximum.at(result,edge[same,1],delta)
    return result


def common_view():
    v,tri,_,masks,faces=template()
    a=np.load(FIT/'results/fit2d/a-state.npz')['vertices']
    b=np.load(FIT/'results/fit2d/b3-state-w3.npz')['vertices']
    visible=np.unique(tri[faces['head']])
    center=(v[visible].min(0)+v[visible].max(0))/2
    radius=max(np.linalg.norm(w[visible]-center,axis=1).max() for w in (v,a,b))
    return center,350/(2*radius)


def shape_rows(rows):
    center,scale=common_view(); images=[]
    for title,v,tri,colors in rows:
        panels=[]
        for yaw,pitch in VIEWS:
            image,_,_=shade(v-center,tri,[yaw,pitch,0,np.log(scale),200,200],400,
                            colors=colors,cull_backfaces=True)
            label(image,f'{title} y{yaw:g} p{pitch:g}'); panels.append(image)
        images.append(np.hstack(panels))
    return np.vstack(images)


def draw_points(image,points,color):
    for x,y in np.rint(points).astype(int):
        if 0<=x<image.shape[1] and 0<=y<image.shape[0]:
            cv2.circle(image,(x,y),1,color,-1,cv2.LINE_AA)


def crop_box(target):
    skin=target['points'][target['labels']==0]
    lo=np.maximum(np.floor(skin.min(0)).astype(int)-25,0)
    hi=np.minimum(np.ceil(skin.max(0)).astype(int)+26,720)
    return int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])


def panel(model,target,pose,title,error=False):
    image=image_for(target['frame']); contours=surface_contour(model['vertices'],model['triangles'],pose)
    if error:
        allowed=target['labels']!=2
        wanted=target['points'][allowed]
        if len(target['ear_points']): wanted=wanted[cKDTree(target['ear_points']).query(wanted)[0]>10]
        distance=cKDTree(wanted).query(contours['xy'])[0]
        colors=heat(distance,3.)
        if len(target['ear_points']): colors[cKDTree(target['ear_points']).query(contours['xy'])[0]<=10]=[150,150,150]
        for point,color in zip(contours['xy'],colors): draw_points(image,point[None],tuple(int(c) for c in color))
    else:
        draw_points(image,contours['xy'],(20,25,240))
        draw_points(image,target['points'][target['labels']==1],(0,190,25))
        draw_points(image,target['points'][target['labels']==0],(240,80,0))
    x0,y0,x1,y1=crop_box(target)
    crop=cv2.resize(image[y0:y1,x0:x1],None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
    label(crop,f'{title} frame {target["frame"]}')
    return crop


def padded_stack(rows):
    width=max(image.shape[1] for image in rows); result=[]
    for image in rows:
        canvas=np.full((image.shape[0],width,3),248,np.uint8); canvas[:,:image.shape[1]]=image; result.append(canvas)
    return np.vstack(result)


def sideburns(model,poses,targets):
    images=[]
    for target in targets:
        image=image_for(target['frame']); pose=poses[str(target['frame'])]
        contour=surface_contour(model['vertices'],model['triangles'],pose)
        draw_points(image,contour['xy'],(20,25,240))
        draw_points(image,target['points'][target['labels']==0],(240,80,0))
        draw_points(image,target['points'][target['labels']==1],(0,190,25))
        count,labels,stats,centers=cv2.connectedComponentsWithStats(target['ear'],8)
        components=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>20]
        for i in components:
            x,y=centers[i]; x+=15*np.sign(360-x)
            x,y=np.clip(np.rint([x,y]).astype(int),50,669)
            crop=cv2.resize(image[y-50:y+50,x-50:x+50],(200,200),interpolation=cv2.INTER_NEAREST)
            label(crop,f'frame {target["frame"]}'); images.append(crop)
    return np.hstack(images)


def ict_models():
    cameras,shared,sref=context()
    v,tri,_,masks,faces=template()
    egg,eggtri,_=compact_surface(v,tri[faces['head']])
    original=load_surface('surface-original-a')
    a=original['vertices']; atri=original['triangles']
    source_a=np.load(FIT/'results/fit2d/a-state.npz')['vertices']
    used=np.unique(tri[faces['head']])
    w3=np.load(FIT/'results/fit2d/b3-state-w3.npz')['vertices'][used]
    models={
        'A':{'vertices':a,'triangles':atri,'poses':cameras,'t':shared['thickness_px']},
        'B3-w3':{'vertices':w3,'triangles':atri,'poses':cameras,'t':shared['thickness_px'],
                 'base_vertices':a}}
    stats,curvature,excluded=curvature_stats(a*sref,atri,(w3-a)*sref,original['ear_excluded'])
    models['B3-w3'].update(curvature=curvature,excluded=excluded,curvature_stats=stats)
    for number in (1,2):
        model=load_surface(f'b3-step{number}-state')
        summary=read_json(RESULTS/f'b3-step{number}-residuals.json')
        if summary.get('valid_for_spec') is False:
            raise RuntimeError('Corrected step2 must be solved before final sheets')
        model.update(poses=cameras,t=shared['thickness_px'],
                     curvature_stats=summary['curvature'])
        models[f'B3-step{number}']=model
    return (egg,eggtri),models


def tripo_models():
    models={}
    for prefix,title in [('T1','T1-A'),('T2','T2-A')]:
        path=TOUT/(prefix+'-a-state.npz')
        if not path.exists(): continue
        with np.load(path) as source: model={k:source[k].copy() for k in source.files}
        model.update(poses=read_json(TOUT/(prefix+'-poses.json'))['frames'],
                     t=read_json(TOUT/(prefix+'-alignment.json'))['pass2']['thickness_px'])
        models[title]=model
    if (TOUT/'T1-b3-state.npz').exists():
        with np.load(TOUT/'T1-b3-state.npz') as source: model={k:source[k].copy() for k in source.files}
        summary=read_json(TOUT/'T1-b3-residuals.json')
        model.update(poses=models['T1-A']['poses'],t=models['T1-A']['t'],curvature_stats=summary['curvature'])
        models['T1-B3']=model
    return models


def front_features(model):
    """Frame the shaded front on canonical art using head bbox only, not features."""
    from PIL import Image
    from fit_core import project
    import torch
    art=Image.open(ROOT/'source/character/still.png').convert('RGBA')
    art.thumbnail((1280,1280),Image.Resampling.LANCZOS)
    padded=Image.new('RGBA',(1280,1280),'white')
    padded.alpha_composite(art,((1280-art.width)//2,(1280-art.height)//2))
    image=cv2.cvtColor(np.array(padded.convert('RGB').resize((720,720))),cv2.COLOR_RGB2BGR)
    mask=np.zeros((1280,1280),np.uint8)
    for name in ('head','face','ears'):
        alpha=np.array(Image.open(TRIPO/f'inputs/{name}.png').convert('RGBA'))[:,:,3]
        mask=np.maximum(mask,alpha)
    mask=cv2.resize(mask,(720,720),interpolation=cv2.INTER_NEAREST)>0
    ys,xs=np.where(mask); target_lo=np.array([xs.min(),ys.min()]); target_hi=np.array([xs.max(),ys.max()])
    v=model['vertices']; tri=model['triangles']
    q=project(torch.tensor(v),torch.tensor([0.,0.,0.,0.,0.,0.])).numpy()[:,:2]
    scale=float(np.sqrt(np.prod((target_hi-target_lo)/np.ptp(q,axis=0))))
    trans=(target_lo+target_hi)/2-scale*(q.min(0)+q.max(0))/2
    pose=[0.,0.,0.,np.log(scale),*trans]
    gray,visible,_=shade(v,tri,pose,cull_backfaces=True)
    image[visible]=(.5*image[visible]+.5*gray[visible]).astype(np.uint8)
    lo=np.maximum(target_lo-30,0); hi=np.minimum(target_hi+31,720)
    crop=cv2.resize(image[lo[1]:hi[1],lo[0]:hi[0]],None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
    label(crop,'T1-A / canonical art / shaded alpha 0.5')
    write_json(TOUT/'front-features.json',{'source':'source/character/still.png','display_pose':list(map(float,pose)),
                 'framing':'uniform scale and translation from bald head/face/ears layer bounding box, no feature landmark fitting',
                 'crop_720':[lo.tolist(),hi.tolist()]})
    return crop


def tripo_sheets(egg,models):
    rows=[]
    for title in ['EGG V5','A','B3-w3','B3-step2','T1-A','T1-B3','T2-A']:
        if title=='EGG V5': rows.append((title,*egg,None))
        elif title in models:
            m=models[title]; rows.append((title,m['vertices'],m['triangles'],None))
    for title in ['B3-step2','T1-B3']:
        if title not in models: continue
        m=models[title]; stats=m['curvature_stats']
        rows.append((f'{title} p99={stats["p99"]:.4f} max={stats["max"]:.4f}',m['vertices'],m['triangles'],
                     heat(m['curvature'],.05,m['excluded'])))
    output=[save(TRIPO/'sheets/tripo-shape.png',shape_rows(rows))]
    keys=frame_sets()[0][:9]; rows=[]
    for frame in keys:
        images=[]
        for title in ['B3-step2','T1-A','T1-B3']:
            if title not in models: continue
            m=models[title]; target=load_targets_f([frame],m['t'])[0]
            images.append(panel(m,target,m['poses'][str(frame)],title))
        rows.append(np.hstack(images))
    output.append(save(TRIPO/'sheets/tripo-keys.png',padded_stack(rows)))
    if 'T1-A' in models: output.append(save(TRIPO/'sheets/tripo-front-features.png',front_features(models['T1-A'])))
    return output


def all_sheets(tripo_only=False):
    started=time.perf_counter(); manifest=[]
    egg,models=ict_models(); remesh=load_surface('surface-a')
    cameras,shared,sref=context(); frames=[3,42,264]
    targets=load_targets_f(frames,shared['thickness_px'])
    keys=frame_sets()[0][:9]; evaluation=load_targets_f(keys,shared['thickness_px'])
    report=read_json(TOUT/'compare.json')['models'] if tripo_only else {}
    for name,model in models.items():
        if tripo_only: continue
        if name in ('B3-step1','B3-step2'):
            rows=read_json(RESULTS/(name.lower()+'-residuals.json'))['frames']
        else: rows=measure_surface(model['vertices'],model['triangles'],[model['poses'][str(f)] for f in keys],evaluation)
        report[name]={'frames':rows,**({'curvature':model['curvature_stats']} if 'curvature_stats' in model else {})}
    for number in (1,2):
        if tripo_only: break
        name=f'B3-step{number}'; model=models[name]; stats=model['curvature_stats']
        rows=[('EGG V5',*egg,None),('A',models['A']['vertices'],models['A']['triangles'],None),
              ('REMESH A',remesh['vertices'],remesh['triangles'],None),
              (name,model['vertices'],model['triangles'],None),
              (f'CURV p99={stats["p99"]:.4f} max={stats["max"]:.4f}',model['vertices'],model['triangles'],heat(model['curvature'],.05,model['excluded'])),
              ('DIHEDRAL CHANGE 0-10 DEG',model['vertices'],model['triangles'],heat(dihedral_change(model['base_vertices'],model['vertices'],model['triangles']),10.))]
        manifest.append(save(FIT/f'sheets/fit2f-shape-step{number}.png',shape_rows(rows)))
        for mode in ('jaw','err'):
            images=[panel(model,t,cameras[str(t['frame'])],name,mode=='err') for t in targets]
            manifest.append(save(FIT/f'sheets/fit2f-{mode}-step{number}.png',padded_stack(images)))
        manifest.append(save(FIT/f'sheets/fit2f-sideburn-step{number}.png',sideburns(model,cameras,targets)))
    # Authentication may explicitly stop part 2. Preserve the same table schema
    # and make unavailable models visible instead of filling fake measurements.
    status=read_json(TOUT/'status.json') if (TOUT/'status.json').exists() else {'status':'not_run'}
    generated=tripo_models(); models.update(generated)
    for name in ('T1-A','T1-B3','T2-A'):
        if name not in generated:
            report[name]={'available':False,'reason':status}; continue
        model=generated[name]
        if name=='T1-B3': rows=read_json(TOUT/'T1-b3-residuals.json')['frames']
        else:
            rows=measure_surface(model['vertices'],model['triangles'],[model['poses'][str(f)] for f in keys],load_targets_f(keys,model['t']))
        report[name]={'available':True,'frames':rows,**({'curvature':model['curvature_stats']} if 'curvature_stats' in model else {})}
    if generated:
        manifest.extend(tripo_sheets(egg,models))
        write_json(TOUT/'sheet-sha256.json',{'sheets':manifest,'elapsed_seconds':time.perf_counter()-started})
    write_json(TOUT/'compare.json',{'definition':'2f label-free full silhouette, ear10px and open-edge exclusions; directed skin/hair, tapered head IoU',
                'frames':keys,'models':report,'tripo_status':status})
    with (TOUT/'compare.csv').open('w',encoding='utf-8',newline='') as file:
        writer=csv.writer(file)
        writer.writerow(['model','frame','skin_median_px','skin_max_px','hair_median_px','hair_max_px','head_iou','curvature_p99','curvature_max'])
        for name,record in report.items():
            for row in record.get('frames',[]):
                curv=record.get('curvature',{})
                writer.writerow([name,row['frame'],row['skin']['median_px'],row['skin']['max_px'],
                                 row['hair']['median_px'],row['hair']['max_px'],row['head_iou'],curv.get('p99',''),curv.get('max','')])
    write_json(RESULTS/'residuals.json',{'models':{k:v for k,v in report.items() if k.startswith(('A','B3'))}})
    if not tripo_only: write_json(RESULTS/'sheet-sha256.json',{'sheets':manifest,'elapsed_seconds':time.perf_counter()-started})
    print('SHEETS',manifest,flush=True)


if __name__=='__main__':
    import argparse
    import torch
    parser=argparse.ArgumentParser(); parser.add_argument('--tripo-only',action='store_true'); args=parser.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1); cv2.setNumThreads(2)
    all_sheets(args.tripo_only)
