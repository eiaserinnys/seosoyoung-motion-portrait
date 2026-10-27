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
