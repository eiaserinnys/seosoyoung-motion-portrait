"""Coordinator C1 mechanical corrections; no new SAM or visibility inference."""
import json,shutil,sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import binary_closing,binary_opening,binary_fill_holes,label
V=Path(__file__).resolve().parent;A=V.parent
sys.path.insert(0,str(A));from input_masks import NAMES
from fit_inputs import fit_pose
from c1_sheet import draw_pose
VISIBLE={'left':['right'],'right':['left'],'up':['left','right'],'down':['left','right'],'up-left':['right'],'up-right':['left'],'down-left':['right'],'down-right':['left']}
def disk(r):
 y,x=np.mgrid[-r:r+1,-r:r+1];return x*x+y*y<=r*r

def correct_face(face,centers):
 y,x=np.indices(face.shape);roi=np.zeros(face.shape,bool)
 for cx,cy in centers:roi|=(x-cx)**2+(y-cy)**2<=45**2
 closed=binary_fill_holes(binary_closing(face,structure=disk(9)))
 out=face|(roi&closed)
 assert not np.any((face!=out)&~roi)
 return out,roi

def correct_ear(ear):
 opened=binary_opening(ear,structure=disk(3));labs,n=label(opened,structure=np.ones((3,3)))
 assert n>0,'ear disappeared during 3px opening'
 counts=np.bincount(labs.ravel());counts[0]=0
 return binary_fill_holes(labs==counts.argmax())

def main():
 cv2.setNumThreads(1);snap=A/'c1-before-mechanical';assert not snap.exists();snap.mkdir()
 for n in ['inputs','tps','sheets','reference-contours']:shutil.copytree(A/n,snap/n,symlinks=True)
 for n in ['design.md','C1-input-audit.json']:shutil.copyfile(A/n,snap/n)
 rows=[];failed=[];changes=[];losses=[]
 for n in NAMES:
  d=A/'inputs'/n;m=json.loads((d/'manifest.json').read_text());before=np.array(Image.open(d/'skin-mask.png'))>127
  face,roi=correct_face(before,np.array(m['registeredLandmarks'])[:2]);union=face.copy();ears={}
  for side,e in m['ears'].items():
   visible=side in VISIBLE[n];candidate=np.array(Image.open(A/'point-c1'/f'{n}-ear-{side}-candidate.png'))>127
   if visible:
    ear=candidate if (n,side)==('left','right') else correct_ear(candidate)
   else:ear=np.zeros_like(face)
   loss=1-int(ear.sum())/int(candidate.sum()) if visible else None
   if visible and loss>.25:losses.append({'pose':n,'side':side,'before':int(candidate.sum()),'after':int(ear.sum()),'fraction':loss})
   if visible:union|=ear
   record={**e,'visible':visible,'maskPixels':int(ear.sum()),'preCleanupPixels':int(candidate.sum()),'cleanupLossFraction':loss,
    'cleanup':'approved unchanged' if (n,side)==('left','right') else ('disk radius3 opening / largest8connected / fill holes' if visible else 'hidden zero mask; identity transform'),
    'visibilitySource':'coordinator C1 approved table','geometryPreviouslyVisible':e['visible'],'geometryDiffersFromParent':visible!=e['visible'],'hiddenTransform':None if visible else 'identity'}
   ears[side]=record;Image.fromarray(np.uint8(ear)*255).save(d/f'ear-{side}-mask.png')
  oldunion=np.array(Image.open(d/'reference-mask.png'))>127
  for title,q in [('skin-mask',face),('reference-mask',union)]:Image.fromarray(np.uint8(q)*255).save(d/f'{title}.png')
  contours,_=cv2.findContours(np.uint8(face),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);c=max(contours,key=cv2.contourArea)[:,0,:]
  (A/'reference-contours'/f'{n}.json').write_text(json.dumps(c.tolist())+'\n')
  m.update(ears=ears,skinPixels=int(face.sum()),unionPixels=int(union.sum()),maskRevision='coordinator mechanical C1 accepted',
   eyeRefinement={'method':'radius9 disk closing plus holes filled; changes unioned only inside radius45 eye circles','changedPixels':int((before!=face).sum()),'outsideCircleChanges':0,'openingMasksUsed':False},manualJawCut=False)
  (d/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
  oldm=json.loads((snap/'inputs'/n/'manifest.json').read_text())
  boxes=lambda es:sorted(tuple(e['box']) for e in es.values() if e['visible'])
  changed=not np.array_equal(union,oldunion) or boxes(ears)!=boxes(oldm['ears']);attempts=[]
  if changed:
   changes.append(n);(d/'tps.json').unlink();fit=fit_pose(n);attempts.append(fit)
   if not fit['success'] and fit['nfev']==120:
    shutil.copyfile(A/'tps'/f'{n}.json',V/f'{n}-tps-120.json');(d/'tps.json').unlink();fit=fit_pose(n,max_nfev=200);attempts.append(fit)
   if not fit['success']:failed.append(n)
  else:fit=json.loads((A/'tps'/f'{n}.json').read_text())['fit']
  rows.append({'name':n,'eyeChangedPixels':int((face!=before).sum()),'ears':ears,'TPSAttempts':attempts,'TPS':fit})
  print('MECHANICAL C1',n,'eye fill',rows[-1]['eyeChangedPixels'],'TPS',fit['success'],fit['nfev'],'ears',[(s,e['maskPixels']) for s,e in ears.items()],flush=True)
 sheet=Image.new('RGB',(2800,1385),(30,36,48));draw=ImageDraw.Draw(sheet);font=ImageFont.load_default(size=16)
 draw.text((8,10),'C1 corrected: GREEN skin | RED 10 approved-visible ears | BLUE unchanged band | hidden ears = identity',fill='white',font=font)
 for i,row in enumerate(rows):
  n=row['name'];im,m=draw_pose(n);x=i%4*700;y=i//4*670+45;sheet.paste(im.crop((150,150,850,720)),(x,y+95))
  draw.text((x+8,y+4),f"{n}: eye fill {row['eyeChangedPixels']}px; jaw manual: NO",fill='white',font=font)
  for j,s in enumerate(['left','right']):
   e=m['ears'][s];draw.text((x+8,y+27+23*j),f"{s}: {e['method']} {e['maskPixels']}px "+('visible' if e['visible'] else 'hidden / identity'),fill='white',font=font)
 out=A/'sheets/C1-eight-pose-corrected-overlays.png';sheet.save(out)
 report={'poses':rows,'changedTPSPoses':changes,'failedTPSPoses':failed,'earLossAbove25Percent':losses,'visibleEarCount':sum(e['visible'] for row in rows for e in row['ears'].values()),'sheet':str(out),
  'C1Status':'mechanical corrections accepted by coordinator' if not failed else 'TPS failed after authorized retry; stop before C2','earFitStarted':False,'ASolveStarted':False}
 assert report['visibleEarCount']==10
 (V/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 (A/'C1-input-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print('C1 CORRECTIONS FINISHED',json.dumps({'failedTPS':failed,'areaLossAbove25':losses}),flush=True)
if __name__=='__main__':main()
