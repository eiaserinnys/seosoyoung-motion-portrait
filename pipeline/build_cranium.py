"""Compile full-height SAM hair into opaque temporal surfaces with filled ear notches."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from hair_albedo import interior_tone, smoothstep
ROOT=Path(__file__).resolve().parents[1]
def build():
    front=np.array(Image.open(ROOT/'source/front-hair-inputs/hair-top.png').convert('RGBA'))
    target_mean,target_std=interior_tone(front)
    entries=[]
    for side,file,crown in [('right','left_45_no_rimlight.png','left-crown'),('left','right_45_no_rimlight.png','right-crown')]:
        rgb=np.array(Image.open(ROOT/'source/references'/file).convert('RGB'))
        mask=np.array(Image.open(ROOT/'sam3_masks/references'/f'{crown}.png').convert('L'))>127
        n,labels=cv2.connectedComponents(mask.astype(np.uint8));counts=np.bincount(labels.ravel());counts[0]=0
        mask=labels==counts.argmax()
        assert mask.sum()>50000, f'{crown}: full head segmentation required'
        interior=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8))>0
        color=rgb.astype(np.float32);light=color.mean(axis=2)
        valid=interior & (light<150) & (color[:,:,0]<175)
        # Skin/white antialias fringes around the ear are not hair samples.
        valid[340:] &= (light[340:]<105) & (color[340:,:,0]<135)
        lab_full=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(np.float32)
        w_mask=valid.astype(np.float32)
        # OpenCV squeezes a singleton channel, so restore the channel axis on
        # the normalized weights explicitly before dividing the LAB image.
        low=cv2.GaussianBlur(lab_full*w_mask[:,:,None],(0,0),18)/np.maximum(cv2.GaussianBlur(w_mask,(0,0),18)[:,:,None],1e-4)
        high=lab_full-low
        leveled=lab_full.copy()
        leveled[:,:,0]=np.clip(target_mean[0]+.85*high[:,:,0],12,95)
        leveled[:,:,1:]=np.clip(target_mean[1:]+.5*high[:,:,1:],0,255)
        rgb_leveled=cv2.cvtColor(np.uint8(np.clip(leveled,0,255)),cv2.COLOR_LAB2RGB)
        seed_valid=(cv2.erode(valid.astype(np.uint8),np.ones((5,5),np.uint8))>0)&(light>=30)
        repair=(~seed_valid).astype(np.uint8)*255
        seed=rgb_leveled.copy();seed[~seed_valid]=0
        texture=cv2.inpaint(seed,repair,7,cv2.INPAINT_TELEA)
        inner_core=cv2.erode(valid.astype(np.uint8),np.ones((7,7),np.uint8))>0
        keep_detail=inner_core|(valid&(light>=32))
        texture[keep_detail]=rgb_leveled[keep_detail]
        yy_ref,xx_ref=np.indices(mask.shape)
        outer_mask=cv2.GaussianBlur((cv2.dilate(mask.astype(np.uint8),np.ones((9,9),np.uint8))>0).astype(np.float32),(0,0),3)
        inner_keep=(1-smoothstep(490,545,xx_ref)) if side=='right' else smoothstep(155,210,xx_ref)
        lower_blend=smoothstep(415,445,yy_ref)
        alpha_f=1-lower_blend*(1-(inner_keep+(1-inner_keep)*outer_mask))
        alpha_f*=1-smoothstep(465,505,yy_ref)
        alpha=np.uint8(np.clip(np.rint(alpha_f*255),0,255))
        box=(350,0,680,521) if side=='right' else (30,0,351,521)
        source=Image.fromarray(np.dstack([texture,alpha]))
        crop=np.array(source.crop(box))
        lab=cv2.cvtColor(crop[:,:,:3].copy(),cv2.COLOR_RGB2LAB).astype(np.float32)
        valid_crop=valid[box[1]:box[3],box[0]:box[2]]
        mean=lab[valid_crop].mean(0);std=lab[valid_crop].std(0)
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
    print('PASS leveled SAM cranium detail, valid-only tone match and soft lower alpha')
if __name__=='__main__':build()
