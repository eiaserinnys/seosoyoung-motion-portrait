"""Fill hidden crown inter-part gaps; retain detached wisps outside the solid locks."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
def build():
    parts=json.loads((ROOT/'parts/parts_index.json').read_text())
    selected=[p for p in parts if p['id'] in ['hair-top','side-hair-left','side-hair-right','back-hair-center']]
    # Immutable source assets make rebuilding idempotent.
    original={p['id']:np.array(Image.open(ROOT/'source/front-hair-inputs'/(p['id']+'.png')).convert('RGBA')) for p in selected}
    union=np.zeros((1536,1024),dtype=np.uint8)
    for p in selected:
        if p['id']=='back-hair-center':continue
        x,y,w,h=p['bbox'];union[y:y+h,x:x+w]=np.maximum(union[y:y+h,x:x+w],original[p['id']][:,:,3])
    for p in selected:
        x,y,w,h=p['bbox'];a=original[p['id']].copy();domain=a[:,:,3]>128
        for row in range(h):
            if not 60<=row+y<=(400 if p['id']=='back-hair-center' else 350):continue
            line=union[row+y] if p['id']=='hair-top' else a[row,:,3]
            edges=np.diff(np.pad(line>128,(1,1)).astype(int));runs=[(l,r) for l,r in zip(np.where(edges==1)[0],np.where(edges==-1)[0]) if r-l>=20]
            if not runs:continue
            l,r=runs[0][0],runs[-1][1]
            if p['id']=='hair-top':l-=x;r-=x
            domain[row,max(0,l):min(w,r)]=True
        repair=domain & (a[:,:,3]<255)
        rgb=cv2.inpaint(a[:,:,:3].copy(),repair.astype(np.uint8)*255,5,cv2.INPAINT_TELEA)
        a[repair,:3]=rgb[repair];a[domain,3]=255
        Image.fromarray(a).save(ROOT/p['file'])
    print('PASS hidden front crown gaps filled; detached outer strands retained')
def fill_seam():
    """The smoothed seam must land inside opaque artwork, even at local notches."""
    rows=json.loads((ROOT/'source/viewer/dome-fit.json').read_text())['frontSeam']
    for part in json.loads((ROOT/'parts/parts_index.json').read_text()):
        if part['id'] not in ['hair-top','side-hair-left','side-hair-right']:continue
        x,y,w,h=part['bbox'];a=np.array(Image.open(ROOT/part['file']).convert('RGBA'));domain=a[:,:,3]>128
        for yy in range(max(y,60),min(y+h,351)):
            edge=np.interp(yy,np.array(rows)[:,0],np.array(rows)[:,1 if part['id'].endswith('left') else 2])
            xs=np.flatnonzero(domain[yy-y])
            if not len(xs):continue
            if part['id']=='hair-top':
                left=np.interp(yy,np.array(rows)[:,0],np.array(rows)[:,1]);right=np.interp(yy,np.array(rows)[:,0],np.array(rows)[:,2])
                lo,hi=int(np.floor(left-x))-1,int(np.ceil(right-x))+2
            elif part['id'].endswith('left'):lo,hi=int(np.floor(edge-x))-1,int(xs[-1])+1
            else:lo,hi=int(xs[0]),int(np.ceil(edge-x))+2
            domain[yy-y,max(0,lo):min(w,hi)]=True
        repair=domain&(a[:,:,3]<255)
        rgb=cv2.inpaint(a[:,:,:3].copy(),repair.astype(np.uint8)*255,5,cv2.INPAINT_TELEA)
        a[repair,:3]=rgb[repair];a[domain,3]=255
        if part['id']=='hair-top':
            # Only the added fill fades into the original lower crop edge.
            # Otherwise its rectangular last row becomes a visible dark shelf.
            original=np.array(Image.open(ROOT/'source/front-hair-inputs/hair-top.png').convert('RGBA'))[:,:,3]
            t=np.clip((h-1-np.arange(h))/24,0,1)
            weight=(t*t*(3-2*t))[:,None]
            a[:,:,3]=np.rint(original+(a[:,:,3].astype(float)-original)*weight).astype(np.uint8)
        Image.fromarray(a).save(ROOT/part['file'])
if __name__=='__main__':build()
