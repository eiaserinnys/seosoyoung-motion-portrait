"""Compile independently segmented SAM 3 cranium pixels into atlas parts."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
ROOT=Path(__file__).resolve().parents[1]
def clean_white(rgb,alpha):
    # Remove isolated mask specks; keep the main hair component with its strands.
    n,labels=cv2.connectedComponents((alpha>127).astype(np.uint8));counts=np.bincount(labels.ravel());counts[0]=0
    core=labels==counts.argmax()
    alpha=np.where(core,alpha,0).astype(np.float32)/255
    # Remove the white matte from fractional-alpha boundary pixels.
    edge=rgb.min(axis=2)>220
    alpha=np.where(edge,alpha*np.clip((255-rgb.min(axis=2))/35,0,1),alpha)
    corrected=(rgb.astype(float)-255*(1-alpha[...,None]))/np.maximum(alpha[...,None],.01)
    out=np.dstack([np.clip(corrected,0,255),alpha*255]).astype(np.uint8)
    return Image.fromarray(out)
def build():
    entries=[]
    for side,file in [('right','left_45_ref_720.png'),('left','right_45_ref_720.png')]:
        rgb=np.array(Image.open(ROOT/'source/references'/file).convert('RGB'))
        alpha=np.array(Image.open(ROOT/'sam3_masks/references'/f'{side}-cranium.png').convert('L'))
        source=clean_white(rgb,alpha)
        a=np.array(source.getchannel('A'),dtype=float)
        ys,xs=np.where(a>0);top=int(ys.min())
        for y in range(top,a.shape[0]):
            row=np.flatnonzero(a[y]>0)
            if not len(row):continue
            inner=int(row.min() if side=='right' else row.max())
            join=np.clip((np.arange(a.shape[1])-inner)*(1 if side=='right' else -1)/18,0,1)
            cap=min(1,(y-top)/80);cap=cap*cap*(3-2*cap)
            a[y]*=join*cap
        source.putalpha(Image.fromarray(a.astype(np.uint8)))
        box=source.getchannel('A').getbbox();crop=source.crop(box)
        key='side-cranium-'+side;path='parts/'+key+'.png';crop.save(ROOT/path)
        entries.append(dict(id=key,label=('왼쪽' if side=='left' else '오른쪽')+' 측후두부',file=path,
                            bbox=[box[0]+152,box[1]+20,box[2]-box[0],box[3]-box[1]],reference=file))
    path=ROOT/'parts/parts_index.json';parts=json.loads(path.read_text())
    parts=[p for p in parts if not p['id'].startswith('side-cranium')]+entries
    path.write_text(json.dumps(parts,ensure_ascii=False,indent=2)+'\n')
    print('PASS compiled two independent SAM 3 cranium masks with white despill')
if __name__=='__main__':build()
