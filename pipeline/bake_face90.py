"""Bake de-occluded 67.5/90 degree RGB onto the unchanged face-skin UV mesh.

The four tiles share rest-canvas UVs. Each triangle samples the reference at
P_solve / 1024; the renderer keeps the original skin alpha and depth unchanged.
"""
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
CENTERS={
    'left67':[(309,430),(404,429)],'left90':[(313,430),(382,429)],
    'right67':[(600,431),(695,428)],'right90':[(622,431),(691,428)],
}


def clean_skin(rgb,mask,name):
    """Interpolate only occluded skin; keep the lower nose/lip/cheek artwork."""
    skin=(mask>127)&(rgb[:,:,0]>225)&(rgb[:,:,1]>165)&(rgb[:,:,2]<240)
    skin&=(rgb[:,:,0].astype(float)-rgb[:,:,2]>15)
    for cx,cy in CENTERS[name]:
        x,y=round(cx-152),round(cy-20)
        skin[max(0,y-65):y+64,max(0,x-58):x+58]=False
        skin[max(0,y-116):y-65,max(0,x-64):x+64]=False
    # Reconstruct a smooth 2D skin field, rather than stretching a row's last
    # brown hair sample over the whole forehead/eye opening.
    if not skin.any():raise ValueError('reference contains no skin samples')
    weights=cv2.GaussianBlur(skin.astype(float),(0,0),22)
    sums=cv2.GaussianBlur(rgb.astype(float)*skin[:,:,None],(0,0),22)
    median=np.median(rgb[skin],axis=0)
    out=(sums+median*.02)/(weights[:,:,None]+.02)
    # Only the measured anterior nose/lip windows retain dark linework.
    # The smoothed SAM silhouette is geometry, not a safe RGB-detail mask.
    detail=np.zeros_like(skin)
    nose={'left67':(134,462,156,497),'left90':(122,458,155,508),
          'right67':(566,462,588,497),'right90':(566,458,600,508)}[name]
    x0,y0,x1,y1=nose;detail[y0:y1,x0:x1]=True
    mouth=(316,548) if name=='left67' else (308,548) if name=='left90' else (688,548) if name=='right67' else (696,548)
    x,y=mouth[0]-152,mouth[1]-20
    detail[y-12:y+14,x-24:x+24]=True
    interior=cv2.erode((mask>127).astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
    detail&=interior&(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)<190)
    # Feather the window so its edge cannot become a rectangular decal seam.
    blend=cv2.GaussianBlur(detail.astype(float),(0,0),.8)*detail
    out=out*(1-blend[:,:,None])+rgb*blend[:,:,None]
    return out.clip(0,255).astype(np.uint8)


def bake_uv(source,rest,solved,triangles):
    mapx=np.full((1024,1024),-1,np.float32);mapy=mapx.copy()
    rest=np.asarray(rest,float);solved=np.asarray(solved,float)
    for tri in triangles:
        a,b,c=rest[tri,:2];target=solved[tri,:2]
        lo=np.maximum(0,np.floor(np.min([a,b,c],axis=0)).astype(int))
        hi=np.minimum(1023,np.ceil(np.max([a,b,c],axis=0)).astype(int))
        if np.any(hi<lo):continue
        xx,yy=np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1))
        det=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        u=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/det
        v=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/det
        w=1-u-v;inside=np.minimum(np.minimum(u,v),w)>=-1e-7
        for d,dst in enumerate([mapx,mapy]):
            patch=dst[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
            patch[inside]=(u*target[0,d]+v*target[1,d]+w*target[2,d])[inside]
    baked=cv2.remap(source,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    # Extend RGB through the 1px raster seam; original alpha owns coverage.
    valid=(mapx>=0).astype(np.uint8)
    expanded=cv2.dilate(baked,np.ones((3,3),np.uint8))
    baked[valid==0]=expanded[valid==0]
    return baked


def profile_canvas(name):
    # The approved inset nose/mouth centers are exact mirrors around x502.
    # Use one clean left-side detail source so neither the right reference's
    # +20px registration nor background hair can create an asymmetric decal.
    left=name.replace('right','left');file=left.replace('67','_67').replace('90','_90')
    rgb=np.array(Image.open(ROOT/f'source/references/{file}_no_rimlight.png').convert('RGB'))
    mask=np.array(Image.open(ROOT/f'sam3_masks/references/{file}-face.png').convert('L'))
    clean=clean_skin(rgb,mask,left)
    canvas=np.full((1024,1024,3),[255,222,196],np.uint8);canvas[20:740,152:872]=clean
    if name.startswith('right'):
        canvas=cv2.warpAffine(canvas,np.array([[-1.,0.,1004.],[0.,1.,0.]]),(1024,1024),borderMode=cv2.BORDER_REPLICATE)
    return canvas


def build():
    geometry=json.loads(subprocess.check_output(['node',str(ROOT/'pipeline/profile_geometry.mjs')],timeout=60))
    atlas=Image.new('RGB',(2048,2048))
    for i,pose in enumerate(geometry['poses']):
        name=('left' if pose['yaw']<0 else 'right')+('67' if abs(pose['yaw'])==67.5 else '90')
        canvas=profile_canvas(name)
        baked=bake_uv(canvas,geometry['rest'],pose['positions'],geometry['triangles'])
        atlas.paste(Image.fromarray(baked),((i%2)*1024,(i//2)*1024))
    path=ROOT/'bundle/assets/face-profile.png';atlas.save(path)
    print('PASS baked four clean profile RGB tiles on unchanged skin triangles:',path.relative_to(ROOT))


if __name__=='__main__':build()
