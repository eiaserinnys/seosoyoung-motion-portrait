"""Rebuild clean, de-lit foreground hair from immutable source pixels."""
import json,shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from hair_albedo import delight,neutralize_outline,transfer_redrawn_hair,smoothstep
ROOT=Path(__file__).resolve().parents[1]
IDS={'hair-top','bangs-left','bangs-right','side-hair-left','side-hair-right','back-hair-center','back-hair-left','back-hair-right'}

def copy_bangs_rgb(a,part,parts,images):
 x,y,w,h=part['bbox'];seed=np.zeros((h,w),bool)
 for key in ['bangs-left','bangs-right']:
  bx,by,bw,bh=parts[key]['bbox'];left=max(x,bx);top=max(y,by);right=min(x+w,bx+bw);bottom=min(y+h,by+bh)
  if left>=right or top>=bottom:continue
  source=images[key][top-by:bottom-by,left-bx:right-bx];target=a[top-y:bottom-y,left-x:right-x]
  mask=source[:,:,3]>200;target[mask,:3]=source[mask,:3];seed[top-y:bottom-y,left-x:right-x]|=mask
 return seed

def taper_added(a,original,canvas_y):
 added=original[:,:,3]==0
 fade=np.broadcast_to(1-smoothstep(230,250,np.arange(a.shape[0])[:,None]+canvas_y),added.shape)
 a[:,:,3]=np.where(added,np.rint(a[:,:,3]*fade),a[:,:,3]).astype(np.uint8)

def build():
 parts={p['id']:p for p in json.loads((ROOT/'parts/parts_index.json').read_text())}
 source=ROOT/'source/front-hair-inputs';source.mkdir(exist_ok=True)
 for key in ['bangs-left','bangs-right']:
  path=source/(key+'.png')
  if not path.exists():shutil.copyfile(ROOT/parts[key]['file'],path)
 original={}
 for key in IDS:
  a=np.array(Image.open(source/(key+'.png')).convert('RGBA'))
  if key=='back-hair-center':
   # The immutable crop ends at x690. Pad only the working image so the
   # requested x700 backing is not silently clipped by its old bbox width.
   width=max(a.shape[1],700-parts[key]['bbox'][0])
   a=np.pad(a,((0,0),(0,width-a.shape[1]),(0,0)))
   parts[key]['bbox'][2]=width
  original[key]=a
 images={key:neutralize_outline(a) for key,a in original.items()}
 still=np.array(Image.open(ROOT/'source/character/still.png').convert('RGB'))
 redrawn=np.array(Image.open(ROOT/'source/references/still_no_rimlight.png').convert('RGB'))
 redrawn[20:740,152:872]=np.array(Image.open(ROOT/'source/references/head_front_no_rimlight.png').convert('RGB'))
 union=np.zeros((1536,1024),np.uint8)
 for key in IDS-{'back-hair-center','back-hair-left','back-hair-right'}:
  x,y,w,h=parts[key]['bbox'];union[y:y+h,x:x+w]=np.maximum(union[y:y+h,x:x+w],images[key][:,:,3])
 for key in sorted(IDS):
  part=parts[key];x,y,w,h=part['bbox'];a=images[key];domain=a[:,:,3]>128
  if not key.startswith('bangs') and key not in {'back-hair-left','back-hair-right'}:
   for row in range(h):
    limit=250 if key=='hair-top' else 407 if key=='back-hair-center' else 406 if key=='side-hair-right' else 350
    if not 60<=row+y<=limit:continue
    line=union[row+y] if key=='hair-top' else a[row,:,3]
    edges=np.diff(np.pad(line>128,(1,1)).astype(int));runs=[(l,r) for l,r in zip(np.where(edges==1)[0],np.where(edges==-1)[0]) if r-l>=20]
    if not runs:continue
    left,right=runs[0][0],runs[-1][1]
    if key=='hair-top':left-=x;right-=x
    # Restore the pin cutout below the accessory, stopping before the ear.
    if key=='side-hair-right' and 340<=row+y<=406:left=min(left,634-x)
    if key=='back-hair-center' and 350<=row+y<=407:right=max(right,min(w,700-x))
    domain[row,max(0,left):min(w,right)]=True
   if key=='back-hair-center':
    expanded=cv2.dilate(domain.astype(np.uint8),np.ones((5,5),np.uint8))>0
    upper=(np.arange(h)+y>=200)&(np.arange(h)+y<=400);domain[upper]=expanded[upper]
   seed=copy_bangs_rgb(a,part,parts,images) if key=='hair-top' else np.zeros(domain.shape,bool)
   repair=domain&(a[:,:,3]==0 if key=='hair-top' else a[:,:,3]<255)&~seed
   rgb=cv2.inpaint(a[:,:,:3].copy(),repair.astype(np.uint8)*255,5,cv2.INPAINT_TELEA)
   a[repair,:3]=rgb[repair];a[domain,3]=255
   if key=='hair-top':taper_added(a,original[key],y)
  fallback=delight(a,y,key.startswith('back-hair'))
  # Visible redrawn texels retain their authored tone; correction only supplies
  # hidden backing and the soft inward edge of the safe transfer mask.
  images[key],_=transfer_redrawn_hair(fallback,original[key],still[y:y+h,x:x+w],redrawn[y:y+h,x:x+w],x,y)
 # Seed real RGB before inpainting, then retain identical overlap after tone correction.
 copy_bangs_rgb(images['hair-top'],parts['hair-top'],parts,images)
 for key,a in images.items():Image.fromarray(a).save(ROOT/parts[key]['file'])
 (ROOT/'parts/parts_index.json').write_text(json.dumps(list(parts.values()),ensure_ascii=False,indent=2)+'\n')
 print('PASS eight immutable hair inputs, intact fringe alpha and redrawn hair RGB')

def fill_seam():
 """Fill only the crown backing; side-hair never expands toward frontSeam."""
 rows=np.array(json.loads((ROOT/'source/viewer/dome-fit.json').read_text())['frontSeam'])
 parts={p['id']:p for p in json.loads((ROOT/'parts/parts_index.json').read_text())}
 part=parts['hair-top'];x,y,w,h=part['bbox'];a=np.array(Image.open(ROOT/part['file']).convert('RGBA'))
 original=np.array(Image.open(ROOT/'source/front-hair-inputs/hair-top.png').convert('RGBA'))
 domain=a[:,:,3]>128
 for yy in range(max(y,60),min(y+h,251)):
  left=np.interp(yy,rows[:,0],rows[:,1]);right=np.interp(yy,rows[:,0],rows[:,2])
  domain[yy-y,max(0,int(np.floor(left-x))-1):min(w,int(np.ceil(right-x))+2)]=True
 images={key:np.array(Image.open(ROOT/parts[key]['file']).convert('RGBA')) for key in ['bangs-left','bangs-right']}
 seed=copy_bangs_rgb(a,part,parts,images);repair=domain&(a[:,:,3]==0)&~seed
 rgb=cv2.inpaint(a[:,:,:3].copy(),repair.astype(np.uint8)*255,5,cv2.INPAINT_TELEA)
 a[repair,:3]=rgb[repair];a[domain,3]=255;taper_added(a,original,y)
 copy_bangs_rgb(a,part,parts,images);Image.fromarray(a).save(ROOT/part['file'])

if __name__=='__main__':build()
