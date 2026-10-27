"""Depth-tested, culled diagnostic sheets for the fixed-camera 2d experiment."""
import cv2
import numpy as np
from fit_core import ROOT, WORKSPACE, read_json
from fit_sheets import shade, label, save_image


def image_for(frame):
    manifest=read_json(ROOT/f'pipeline/headmodel/data/frame_{frame:04d}/manifest.json')
    path=WORKSPACE/f'.local/artifacts/20261025-ponytail-seedance-reg/frames/frame_{frame:04d}.png'
    image=cv2.imread(str(path))
    if image is None: raise FileNotFoundError(path)
    return cv2.warpAffine(image,np.asarray(manifest['to720'])[:2],(720,720),borderValue=(255,255,255))


def init_sheet(v,tri,faces,poses,targets):
    panels=[]; checks=[]
    for pose,target in zip(poses,targets):
        image=image_for(target['frame'])
        gray,mask,q=shade(v,tri[faces['head']],pose,cull_backfaces=True)
        image[mask]=(.48*image[mask]+.52*gray[mask]).astype(np.uint8)
        ys,xs=np.where(target['head']); my,mx=np.where(mask)
        target_center=np.array([(xs.min()+xs.max())/2,(ys.min()+ys.max())/2])
        model_center=np.array([(mx.min()+mx.max())/2,(my.min()+my.max())/2])
        error=float(np.linalg.norm(model_center-target_center))
        for center,color in ((target_center,(255,100,0)),(model_center,(0,0,255))):
            cv2.drawMarker(image,tuple(np.rint(center).astype(int)),color,cv2.MARKER_CROSS,18,2)
        label(image,f'frame {target["frame"]} / center {error:.2f}px / NO FIT')
        panels.append(image)
        checks.append({'frame':target['frame'],'target_center':target_center.tolist(),
                       'model_center':model_center.tolist(),'center_error_px':error})
    save_image('fit2d-init.png',np.vstack([np.hstack(panels[:3]),np.hstack(panels[3:])]))
    return checks


def shape_sheet(rows,tri,head_faces,name):
    """2c's shared ten-view sheet, generalized to a list of named surfaces."""
    views=[(0,0),(-45,0),(45,0),(-67.5,0),(67.5,0),(-78.75,0),(78.75,0),(0,30),(0,-30),(45,-30)]
    visible=np.unique(tri[head_faces]); template=rows[0][1]
    center=(template[visible].min(0)+template[visible].max(0))/2
    radius=max(np.linalg.norm(v[visible]-center,axis=1).max() for _,v in rows)
    cell=400; scale=(cell-50)/(2*radius); images=[]
    for title,v in rows:
        panels=[]
        for yaw,pitch in views:
            image,_,_=shade(v-center,tri[head_faces],[yaw,pitch,0,np.log(scale),cell/2,cell/2],cell,
                            cull_backfaces=True)
            label(image,f'{title} y {yaw:g} p {pitch:g}'); panels.append(image)
        images.append(np.hstack(panels))
    save_image(name,np.vstack(images))


def center_checks(v,tri,faces,poses,targets):
    checks=[]
    for pose,target in zip(poses,targets):
        from fit_core import project, rasterize
        import torch
        q=project(torch.tensor(v),torch.tensor(pose)).numpy()
        _,owner=rasterize(q,tri[faces['head']],cull_backfaces=True)
        ys,xs=np.where(target['head']); my,mx=np.where(owner>=0)
        tc=np.array([(xs.min()+xs.max())/2,(ys.min()+ys.max())/2])
        mc=np.array([(mx.min()+mx.max())/2,(my.min()+my.max())/2])
        checks.append({'frame':target['frame'],'target_center':tc.tolist(),
                       'model_center':mc.tolist(),'center_error_px':float(np.linalg.norm(tc-mc))})
    return checks


def b3_sheets(vertices,tri,masks,faces,poses,targets):
    from scipy.spatial import cKDTree
    from fit_core import boundary
    from fit2d_core import model_contours
    overlays=[]; crops=[]; errors=[]
    for pose,target in zip(poses,targets):
        image=image_for(target['frame'])
        gray,mask,_=shade(vertices,tri[faces['head']],pose,cull_backfaces=True)
        overlay=image.copy(); overlay[mask]=(.48*image[mask]+.52*gray[mask]).astype(np.uint8)
        label(overlay,f'frame {target["frame"]} / B3 shared surface'); overlays.append(overlay)
        curves=model_contours(vertices,tri,masks,faces,pose)
        jaw=image.copy(); err=(image*.4+255*.6).astype(np.uint8)
        pairs=[('head',target['head_boundary']),('face',target['face_points'])]
        pairs += [(key,boundary(mask)) for key,mask in target['ears'].items()]
        for key,wanted in pairs:
            predicted=curves[key]['xy']; distance=cKDTree(wanted).query(predicted)[0]
            for point,value in zip(predicted,distance):
                fraction=min(float(value)/3.,1.)
                color=(0,int(220*(1-fraction)),int(255*fraction))
                cv2.circle(err,tuple(np.rint(point).astype(int)),1,color,-1,cv2.LINE_AA)
        for points,color in ((target['face_points'],(255,70,0)),(curves['face']['xy'],(0,0,240))):
            for point in points:
                cv2.circle(jaw,tuple(np.rint(point).astype(int)),1,color,-1,cv2.LINE_AA)
        joint=np.r_[target['face_points'],curves['face']['xy']]
        low=np.maximum(np.floor(joint.min(0)).astype(int)-18,0)
        high=np.minimum(np.ceil(joint.max(0)).astype(int)+18,719)
        x0,y0=low; x1,y1=high
        crop=cv2.resize(jaw[y0:y1+1,x0:x1+1],None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
        label(crop,f'frame {target["frame"]} / used face target blue / model red / 2x'); crops.append(crop)
        label(err,f'frame {target["frame"]} / 0px green -> 3px+ red'); errors.append(err)
    save_image('fit2d-overlay-b3.png',np.hstack(overlays))
    save_image('fit2d-err-b3.png',np.hstack(errors))
    width=max(c.shape[1] for c in crops)
    padded=[cv2.copyMakeBorder(c,0,0,0,width-c.shape[1],cv2.BORDER_CONSTANT,value=(248,248,248)) for c in crops]
    save_image('fit2d-jaw-b3.png',np.vstack(padded))
