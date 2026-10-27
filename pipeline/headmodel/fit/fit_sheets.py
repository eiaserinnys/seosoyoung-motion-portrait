"""Fixed-light z-buffer renderings of measured geometry, without image synthesis."""
from pathlib import Path
import cv2
import numpy as np
import torch

from fit_core import (FIT, ROOT, WORKSPACE, SIZE, read_json, write_json, load_model,
                      define_regions, load_targets, project, rasterize, boundary,
                      lower_boundary, eroded_target)

LIGHT = np.array([-.32,.52,.79]); LIGHT /= np.linalg.norm(LIGHT)


def shade(vertices, tri, pose, size=720, colors=None, cull_backfaces=False):
    q = project(torch.tensor(vertices), torch.tensor(pose,dtype=torch.float32)).numpy()
    _, owner = rasterize(q,tri,size,cull_backfaces=cull_backfaces)
    normals = np.cross(vertices[tri[:,1]]-vertices[tri[:,0]],vertices[tri[:,2]]-vertices[tri[:,0]])
    normals /= np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-8)
    # One fixed world-space light and constant ambient, shared by every view.
    brightness = .30+.70*np.maximum(normals@LIGHT,0)
    facecolors = np.full((len(tri),3),205.) if colors is None else colors[tri].mean(1)
    facecolors = np.clip(facecolors*brightness[:,None],0,255).astype(np.uint8)
    image = np.full((size,size,3),248,np.uint8)
    mask = owner>=0; image[mask] = facecolors[owner[mask]]
    return image,mask,q


def label(image,text,origin=(12,25)):
    cv2.putText(image,text,origin,cv2.FONT_HERSHEY_SIMPLEX,.58,(25,25,25),1,cv2.LINE_AA)


def source_image(target):
    path = WORKSPACE / f'.local/artifacts/20261025-ponytail-seedance-reg/frames/frame_{target["frame"]:04d}.png'
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.resize(image,(720,720),interpolation=cv2.INTER_CUBIC)
    return cv2.warpAffine(image,np.array(target['affine']),(720,720),borderValue=(255,255,255))


def save_image(name,image):
    path = FIT/'sheets'/name; path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),image):
        raise RuntimeError(path)
    print('sheet',path,flush=True)


def regions_sheet(v,tri,regions):
    colors = np.full_like(v,205.)
    colors[regions['neck']] = (80,145,240)  # BGR orange
    colors[regions['face_lower']] = (255,110,50)
    colors[regions['ear-left']] = (0,230,240)
    colors[regions['ear-right']] = (0,230,240)
    cell = 600; scale = 15.
    panels=[]
    for name,yaw in [('FRONT',0.),('SIDE',90.),('BACK',180.)]:
        pose=[yaw,0.,0.,np.log(scale),cell/2,cell/2-25]
        image,_,_=shade(v,tri,pose,cell,colors)
        label(image,name)
        label(image,'orange: excluded neck',(12,cell-50))
        label(image,'blue: lower Face; yellow: ears',(12,cell-25))
        panels.append(image)
    save_image('regions.png',np.hstack(panels))


def overlay_sheet(v,tri,poses,targets,t,name):
    regions,faces = define_regions(load_model()[0],tri)
    rows=[]
    for target,pose in zip(targets,poses):
        original = source_image(target)
        gray,mask,q = shade(v,tri[faces['head']],pose)
        overlay = original.copy()
        overlay[mask] = (.48*original[mask]+.52*gray[mask]).astype(np.uint8)
        outlines = original.copy()
        def draw(points,color):
            cv2.polylines(outlines,[np.rint(points).astype(np.int32)],False,color,2,cv2.LINE_AA)
        head = eroded_target(target['head'],t)
        cv2.drawContours(outlines,[boundary(head).astype(np.int32)],-1,(0,210,0),2,cv2.LINE_AA)
        draw(target['lower'],(255,70,0))
        for ear in target['ears'].values():
            cv2.drawContours(outlines,[boundary(ear).astype(np.int32)],-1,(0,230,255),2,cv2.LINE_AA)
        cv2.drawContours(outlines,[boundary(mask).astype(np.int32)],-1,(0,0,240),2,cv2.LINE_AA)
        label(overlay,f'frame {target["frame"]}: shaded mesh, alpha .52')
        label(outlines,'H - t green / lower S blue / E yellow / mesh red')
        rows.append(np.hstack([overlay,outlines]))
    save_image(name,np.vstack(rows))


def shape_sheet(v,neutral,tri):
    views=[(0,0),(-45,0),(45,0),(-67.5,0),(67.5,0),(-78.75,0),(78.75,0),
           (0,30),(0,-30),(45,-30)]
    # A common world origin and scale: neither row is fitted into its own box.
    cell=400
    both=np.r_[v,neutral]
    radius=np.linalg.norm(both,axis=1).max()
    scale=(cell-50)/(2*radius)
    rows=[]
    for name,vertices in [('FITTED',v),('ICT NEUTRAL',neutral)]:
        panels=[]
        for yaw,pitch in views:
            image,_,_=shade(vertices,tri,[yaw,pitch,0,np.log(scale),cell/2,cell/2],cell)
            label(image,f'{name}  y {yaw:g} p {pitch:g}')
            panels.append(image)
        rows.append(np.hstack(panels))
    save_image('falsify-shape.png',np.vstack(rows))


def contour_residual(a,b):
    # Exact pixel DT measurements, independent of the training approximation.
    def distances(points,other):
        field=np.ones((720,720),np.uint8)
        xy=np.rint(other).astype(int); field[xy[:,1],xy[:,0]]=0
        dt=cv2.distanceTransform(field,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        return dt[np.rint(points[:,1]).astype(int),np.rint(points[:,0]).astype(int)]
    ab=distances(a,b); ba=distances(b,a)
    return {'mean_px':float((ab.mean()+ba.mean())/2),'max_px':float(max(ab.max(),ba.max())),
            'model_to_target_mean_px':float(ab.mean()),'target_to_model_mean_px':float(ba.mean())}


def iou(a,b):
    return float(np.logical_and(a,b).sum()/np.logical_or(a,b).sum())


def measure(v,tri,faces,poses,targets,t,cull_backfaces=False):
    result=[]
    for pose,target in zip(poses,targets):
        q=project(torch.tensor(v),torch.tensor(pose,dtype=torch.float32)).numpy()
        _,owner=rasterize(q,tri[faces['head']],cull_backfaces=cull_backfaces); model=owner>=0
        global_owner=np.full_like(owner,-1); global_owner[model]=faces['head'][owner[model]]
        _,faceowner=rasterize(q,tri[faces['face']],cull_backfaces=cull_backfaces); skin=faceowner>=0
        h=eroded_target(target['head'],t)
        ears={}
        for key,wanted in target['ears'].items():
            visible=np.isin(global_owner,faces[key])
            # Also retain the isolated projection: occlusion is not silently ignored.
            _,eo=rasterize(q,tri[faces[key]],cull_backfaces=cull_backfaces)
            ears[key]={'visible_iou':iou(visible,wanted),'isolated_iou':iou(eo>=0,wanted),
                       'visible_area_px':int(visible.sum()),'target_area_px':int(wanted.sum())}
        head_res=contour_residual(boundary(model),boundary(h))
        skin_res=contour_residual(lower_boundary(skin),target['lower'])
        cutoff=int(np.argmax(target['skin'].sum(1)))
        mp=boundary(model); hp=boundary(h)
        skull_res=contour_residual(mp[mp[:,1]<cutoff],hp[hp[:,1]<cutoff])
        result.append({'frame':target['frame'],'head_iou':iou(model,h),'head':head_res,
                       'skin_lower':skin_res,'skull_upper':skull_res,'ears':ears,
                       'required_projected_displacement_estimate_px':max(skull_res['max_px'],skin_res['max_px']),
                       'skin_over_10px':skin_res['max_px']>10.,
                       'failure_candidate_over_26px':max(skull_res['max_px'],skin_res['max_px'])>26.})
    return result
