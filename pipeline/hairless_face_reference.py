"""Shared reference registration for independent face-only measurements."""
import json, subprocess
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
FILES={'front':'head_front_ref.png','left':'left_45_ref_720.png','right':'right_45_ref_720.png',
       **{n:n.replace('-','_')+'_ref_720.png' for n in ['up-left','up','up-right','down-left','down','down-right']}}
def geometry():
    return json.loads(subprocess.check_output(['node',str(ROOT/'pipeline/hairless_face_geometry.mjs')],text=True))
@lru_cache(maxsize=9)
def raw_mask(name):
    if name=='front':
        mask=np.zeros((720,720),np.uint8)
        a=np.array(Image.open(ROOT/'parts/face-skin.png'))[:,:,3]
        mask[198:198+a.shape[0],154:154+a.shape[1]]=a
    else:mask=np.array(Image.open(ROOT/'sam3_masks/references'/f'{name}-face.png'))
    _,labels,stats,_=cv2.connectedComponentsWithStats((mask>127).astype(np.uint8))
    return np.uint8(labels==1+np.argmax(stats[1:,cv2.CC_STAT_AREA]))*255
def matrix(pose):
    r=pose['feature'].get('registration')
    if not r:return np.array([[1,0,152],[0,1,20]],float)
    scale=r['scale'];mid=r['referenceEyeMid'];eye=r['eyeMid']
    return np.array([[scale,0,eye[0]+(152-mid[0])*scale],[0,scale,eye[1]+(20-mid[1])*scale]])
def registered_point(pose,point):
    r=pose['feature'].get('registration')
    return np.array(point) if not r else np.array(r['eyeMid'])+(np.array(point)-r['referenceEyeMid'])*r['scale']
def mask_interval(pose,canvas_y):
    m=raw_mask(pose['name']);a=matrix(pose);y=(canvas_y-a[1,2])/a[1,1]
    y=int(round(y))
    if y<0 or y>=m.shape[0]:return None
    xs=np.flatnonzero(m[y]>127)
    if not len(xs):return None
    return a[0,2]+a[0,0]*np.array([xs[0],xs[-1]])
