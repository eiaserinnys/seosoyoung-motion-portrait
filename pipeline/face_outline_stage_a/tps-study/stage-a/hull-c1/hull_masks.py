"""Restore verified left v2 and union a local convex hull into six far eyes."""
import hashlib,json,shutil,sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
V=Path(__file__).resolve().parent;A=V.parent;O=A.parent
sys.path.insert(0,str(A));from input_masks import NAMES
from fit_inputs import fit_pose
from c1_sheet import draw_pose
FAR={'right':1,'down':1,'up-left':0,'up-right':1,'down-left':0,'down-right':1}
def read(p):return np.array(Image.open(p))>127
def eye_hull(face,center):
 y,x=np.indices(face.shape);cx,cy=center;roi=(x-cx)**2+(y-cy)**2<=50**2
 yy,xx=np.where(face&roi);assert len(xx)>=3
 polygon=cv2.convexHull(np.column_stack((xx,yy)).astype(np.int32));hull=np.zeros(face.shape,np.uint8);cv2.fillPoly(hull,[polygon],1)
 out=face|(roi&(hull>0));assert not ((out!=face)&~roi).any()
 return out,roi,(hull>0)&roi

def main():
 cv2.setNumThreads(1);snap=A/'c1-before-v2-hull';assert not snap.exists();snap.mkdir()
 for p in ['inputs','tps','sheets','reference-contours']:shutil.copytree(A/p,snap/p,symlinks=True)
 for p in ['design.md','C1-input-audit.json']:shutil.copyfile(A/p,snap/p)
 d=A/'inputs/left';d.joinpath('tps.json').unlink();copies={}
 for source,dest in [(O/'v2/tps.json',A/'tps/left.json'),(O/'v2/reference-mask.png',d/'reference-mask.png'),(O/'v2/reference-ear-mask.png',d/'ear-right-mask.png'),(O/'v2/hairband.png',d/'hairband.png')]:
  shutil.copyfile(source,dest);assert source.read_bytes()==dest.read_bytes();copies[str(dest.relative_to(A))]=hashlib.sha256(dest.read_bytes()).hexdigest()
 d.joinpath('tps.json').symlink_to('../../tps/left.json')
 # Green display is the preserved union minus red approved ear, not a TPS refit.
 face=read(d/'reference-mask.png')&~read(d/'ear-right-mask.png');Image.fromarray(np.uint8(face)*255).save(d/'skin-mask.png')
 m=json.loads((d/'manifest.json').read_text());m.update(maskRevision='verified v2 inputs restored',skinPixels=int(face.sum()),unionPixels=int(read(d/'reference-mask.png').sum()),
  skinBoundaryMethod='display: verified v2 silhouette minus approved ear; TPS and union copied byte-exact',TPSProvenance='verified v2/tps.json reused without refit',v2SHA256=copies)
 (d/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
 rows=[];failed=[];overlap_alert=[]
 for n in NAMES:
  d=A/'inputs'/n;m=json.loads((d/'manifest.json').read_text());attempts=[];added=overlap=added_overlap=0
  before=read(d/'skin-mask.png');oldunion=read(d/'reference-mask.png')
  if n in FAR:
   center=np.array(m['registeredLandmarks'])[FAR[n]];face,roi,hull=eye_hull(before,center);band=read(d/'hairband.png')
   extra=face&~before;added=int(extra.sum());overlap=int((hull&band).sum());added_overlap=int((extra&band).sum())
   if overlap>200:overlap_alert.append(n)
   union=face.copy()
   for s,e in m['ears'].items():
    if e['visible']:union|=read(d/f'ear-{s}-mask.png')
   for title,q in [('skin-mask',face),('reference-mask',union)]:Image.fromarray(np.uint8(q)*255).save(d/f'{title}.png')
   m.update(skinPixels=int(face.sum()),unionPixels=int(union.sum()),maskRevision='far-eye radius50 local convex hull',farEyeHull={'center':center.tolist(),'eyeIndex':FAR[n],'radius':50,'addedPixels':added,'outsideCircleChanges':0,'hullHairbandOverlap':overlap,'addedPixelsHairbandOverlap':added_overlap})
   (d/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
   if not np.array_equal(union,oldunion):
    d.joinpath('tps.json').unlink();fit=fit_pose(n);attempts.append(fit)
    if not fit['success'] and fit['nfev']==120:
     shutil.copyfile(A/'tps'/f'{n}.json',V/f'{n}-tps-120.json');d.joinpath('tps.json').unlink();fit=fit_pose(n,max_nfev=200);attempts.append(fit)
    if not fit['success']:failed.append(n)
  face=read(d/'skin-mask.png');contours,_=cv2.findContours(np.uint8(face),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);c=max(contours,key=cv2.contourArea)[:,0,:]
  (A/'reference-contours'/f'{n}.json').write_text(json.dumps(c.tolist())+'\n')
  fit=json.loads((A/'tps'/f'{n}.json').read_text())['fit']
  rows.append({'name':n,'addedPixels':added,'outsideCircleChanges':0,'hullHairbandOverlap':overlap,'addedPixelsHairbandOverlap':added_overlap,'TPSAttempts':attempts,'TPS':fit,'TPSReused':not attempts})
  print('V2/HULL C1',n,'added',added,'hull-band',overlap,'added-band',added_overlap,'TPS',fit['success'],fit['nfev'],flush=True)
 sheet=Image.new('RGB',(2800,1345),(30,36,48));dr=ImageDraw.Draw(sheet);font=ImageFont.load_default(size=17)
 dr.text((8,8),'C1: LEFT verified v2 | other six far-eye radius50 hull | GREEN skin / RED visible ears / BLUE hair band',fill='white',font=font)
 for i,row in enumerate(rows):
  im,m=draw_pose(row['name']);x=i%4*700;y=i//4*650+45;sheet.paste(im.crop((150,150,850,720)),(x,y+75))
  dr.text((x+6,y+4),row['name']+(' VERIFIED V2' if row['name']=='left' else f" | +{row['addedPixels']}px; outside0; hull-band {row['hullHairbandOverlap']}px"),fill='white',font=font)
  ear=' / '.join(s[0].upper()+':'+str(e['maskPixels'])+' '+('visible' if e['visible'] else 'hidden') for s,e in m['ears'].items())
  dr.text((x+6,y+30),ear,fill='white',font=font)
 path=A/'sheets/C1-eight-pose-v2-hull-overlays.png';sheet.save(path)
 report={'poses':rows,'leftCopiedSHA256':copies,'leftRefit':False,'TPSRefitPoses':[r['name'] for r in rows if r['TPSAttempts']],
  'failedTPSPoses':failed,'hullHairbandOverlapAbove200':overlap_alert,'sheet':str(path),'earFittingStarted':False,'ASolveStarted':False}
 (V/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');(A/'C1-input-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print('HULL C1 RESULT failedTPS',failed,'hull overlap>200',overlap_alert,flush=True)
if __name__=='__main__':main()
