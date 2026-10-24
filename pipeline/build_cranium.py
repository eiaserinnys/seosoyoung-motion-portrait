"""Compile full-height SAM hair into opaque temporal surfaces with filled ear notches."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from hair_albedo import interior_tone
ROOT=Path(__file__).resolve().parents[1]
def build():
    front=np.array(Image.open(ROOT/'source/front-hair-inputs/hair-top.png').convert('RGBA'))
    target_mean,target_std=interior_tone(front)
    entries=[]
    for side,file,crown in [('right','left_45_ref_720.png','left-crown'),('left','right_45_ref_720.png','right-crown')]:
        rgb=np.array(Image.open(ROOT/'source/references'/file).convert('RGB'))
        mask=np.array(Image.open(ROOT/'sam3_masks/references'/f'{crown}.png').convert('L'))>127
        n,labels=cv2.connectedComponents(mask.astype(np.uint8));counts=np.bincount(labels.ravel());counts[0]=0
        mask=labels==counts.argmax()
        assert mask.sum()>50000, f'{crown}: full head segmentation required'
        interior=cv2.erode(mask.astype(np.uint8),np.ones((5,5),np.uint8))>0
        color=rgb.astype(np.float32);light=color.mean(axis=2)
        valid=interior & (light>=28) & (light<105) & (color[:,:,0]<140)
        # Skin/white antialias fringes around the ear are not hair samples.
        valid[360:] &= light[360:]<105
        repair=(~valid).astype(np.uint8)*255
        seed=rgb.copy();seed[~valid]=0
        texture=cv2.inpaint(seed,repair,7,cv2.INPAINT_TELEA)
        texture[valid]=rgb[valid]
        # Geometry, not a rectangular alpha cutout, owns the spherical boundary.
        # Keep an opaque near-half texture including hair-filled ear/inner regions.
        box=(350,0,680,521) if side=='right' else (30,0,351,521)
        source=Image.fromarray(np.dstack([texture,np.full(mask.shape,255,np.uint8)]))
        crop=np.array(source.crop(box))
        lab=cv2.cvtColor(crop[:,:,:3].copy(),cv2.COLOR_RGB2LAB).astype(np.float32)
        mean=lab.reshape(-1,3).mean(0);std=lab.reshape(-1,3).std(0)
        ratio=np.clip(target_std/np.maximum(std,1e-6),.85,1.0)
        lab=target_mean+(lab-mean)*ratio
        crop[:,:,:3]=cv2.cvtColor(np.uint8(np.clip(lab,0,255)),cv2.COLOR_LAB2RGB)
        crop=Image.fromarray(crop)
        key='side-cranium-'+side;path='parts/'+key+'.png';crop.save(ROOT/path)
        entries.append(dict(id=key,label=('왼쪽' if side=='left' else '오른쪽')+' 측후두부',file=path,
                            bbox=[box[0]+152,box[1]+20,box[2]-box[0],box[3]-box[1]],reference=file))
    path=ROOT/'parts/parts_index.json';parts=json.loads(path.read_text())
    parts=[p for p in parts if not p['id'].startswith('side-cranium')]+entries
    path.write_text(json.dumps(parts,ensure_ascii=False,indent=2)+'\n')
    print('PASS full-height opaque SAM cranium; ear notches filled from hair texels')
if __name__=='__main__':build()
