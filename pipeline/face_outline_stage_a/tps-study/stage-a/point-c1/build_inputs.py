"""C1 point-mask integration only; registered ear-center geometry is auditable."""
import hashlib,json,shutil,sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
V=Path(__file__).resolve().parent;A=V.parent;O=A.parent
sys.path.insert(0,str(A));from input_masks import NAMES
from fit_inputs import fit_pose
from c1_sheet import draw_pose
from requests_points import KEEP,EAR_MANUAL

def geometry(face,center):
 x,y=np.rint(center).astype(int)
 return not bool(face[y,x])

def mask(path):return np.array(Image.open(path))>127

def tps_boxes(ears):
 return sorted(tuple(e['box']) for e in ears.values() if e['visible'])

def main():
 cv2.setNumThreads(1)
 snapshot=A/'c1-rejected-sam-text';assert not snapshot.exists();snapshot.mkdir()
 for p in ['inputs','tps','sheets','reference-contours']:shutil.copytree(A/p,snapshot/p,symlinks=True)
 for p in ['design.md','C1-input-audit.json']:shutil.copyfile(A/p,snapshot/p)
 (snapshot/'README.md').write_text('# C1 SAM 텍스트 불통과 보존\n\n목과 머리를 포함한 얼굴 및 오인 귀 출력이다.\n')
 summary=[];changed=[]
 for n in NAMES:
  dest=A/'inputs'/n;m=json.loads((dest/'manifest.json').read_text());oldm=json.loads((A/'c1-rejected-color/inputs'/n/'manifest.json').read_text())
  raw=mask(V/'raw'/n/'face.png');band=mask(dest/'hairband.png');face=raw&~band
  # No manual jaw clipping: inspect every pose after point SAM and hair-band subtraction.
  former=mask(dest/'reference-mask.png');union=face.copy();ears={}
  for side in ['left','right']:
   center=np.mean(oldm['ears'][side]['registeredInputContour'],0);visible=geometry(face,center);expected=side in KEEP[n]
   if n=='left' and side=='right':
    candidate=mask(O/'v2/reference-ear-mask.png');method='approved';box=[625,408,714,537];score=None
   elif (n,side) in EAR_MANUAL:
    candidate=mask(V/'raw'/n/f'ear-{side}.png');method='SAM point';m2=json.loads((V/'raw'/n/f'ear-{side}.json').read_text());score=m2['score'];yy,xx=np.where(candidate);box=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
   else:
    candidate=mask(A/'sam3-c1/raw'/n/f'ear-{side}.png');method='SAM text';m2=json.loads((A/'sam3-c1/raw'/n/f'ear-{side}.json').read_text());score=m2['score'];box=m['ears'][side]['box']
   # Hidden geometry is not an observation. Keep candidate for diagnosis, not in union.
   Image.fromarray(np.uint8(candidate)*255).save(V/f'{n}-ear-{side}-candidate.png')
   used=candidate if visible else np.zeros_like(face)
   Image.fromarray(np.uint8(used)*255).save(dest/f'ear-{side}-mask.png')
   if visible:union|=used
   ears[side]={'visible':visible,'method':method,'source':method,'maskPixels':int(used.sum()),'candidatePixels':int(candidate.sum()),'score':score,'box':box,
    'registeredContourCenter':center.tolist(),'centerInsideFace':not visible,'parentExpectedVisible':expected,'geometryDiffersFromParent':visible!=expected,
    'hiddenTransform':'identity' if not visible else None,'candidatePreserved':str(V/f'{n}-ear-{side}-candidate.png'),'selectionRule':'registered earContour center outside new face mask; no score/area visibility gate'}
  for title,q in [('skin-mask',face),('reference-mask',union)]:Image.fromarray(np.uint8(q)*255).save(dest/f'{title}.png')
  contours,_=cv2.findContours(np.uint8(face),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);contour=max(contours,key=cv2.contourArea)[:,0,:]
  (A/'reference-contours'/f'{n}.json').write_text(json.dumps(contour.tolist())+'\n')
  fm=json.loads((V/'raw'/n/'face.json').read_text())
  altered=not np.array_equal(union,former) or tps_boxes(m['ears'])!=tps_boxes(ears)
  m.update(ears=ears,faceSAMScore=fm['score'],faceSAMMetadata=fm,skinPixels=int(face.sum()),unionPixels=int(union.sum()),
   skinBoundaryMethod='SAM3 face positive/negative points; subtract unchanged hairband; no manual jaw cut',maskRevision='C1 point-prompt revision',manualJawCut=False,
   faceHairBandPixelsRemoved=int((raw&band).sum()),TPSInputChanged=altered)
  (dest/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
  if altered:
   changed.append(n);(dest/'tps.json').unlink();fit=fit_pose(n)
  else:fit=json.loads((A/'tps'/f'{n}.json').read_text())['fit']
  assert (dest/'hairband.png').read_bytes()==(snapshot/'inputs'/n/'hairband.png').read_bytes()
  assert (dest/'reference-frame.png').read_bytes()==(snapshot/'inputs'/n/'reference-frame.png').read_bytes()
  summary.append({'name':n,'faceScore':fm['score'],'manualJawCut':False,'ears':ears,'TPS':fit,'TPSRefit':altered})
  print('POINT C1',n,'ears',[(s,e['method'],e['maskPixels'],e['visible'],e['geometryDiffersFromParent']) for s,e in ears.items()],flush=True)
 w,h=700,670;sheet=Image.new('RGB',(w*4,h*2+45),(30,36,48));d=ImageDraw.Draw(sheet);f=ImageFont.load_default(size=16)
 d.text((8,10),'C1 POINT: GREEN face | RED used ears | BLUE hair band | geometry visibility; ! = coordinator table differs',fill='white',font=f)
 for i,n in enumerate(NAMES):
  im,m=draw_pose(n);x=i%4*w;y=i//4*h+45;sheet.paste(im.crop((150,150,850,720)),(x,y+95))
  d.text((x+8,y+4),f"{n}: face {m['faceSAMScore']:.3f}; manual jaw cut: NO",fill='white',font=f)
  for j,s in enumerate(['left','right']):
   e=m['ears'][s];label='visible' if e['visible'] else 'hidden / T=identity';warn=' ! TABLE DIFF' if e['geometryDiffersFromParent'] else ''
   d.text((x+8,y+25+j*23),f"{s}: {e['method']}, {e['maskPixels']}px, {label}{warn}",fill='white',font=f)
 path=A/'sheets/C1-eight-pose-point-overlays.png';sheet.save(path)
 # Detail comparison for all candidates even when geometric visibility excluded them.
 cand=Image.new('RGB',(700*4,620*2),(30,36,48));dc=ImageDraw.Draw(cand)
 for i,n in enumerate(NAMES):
  im=np.array(Image.open(A/'inputs'/n/'reference-frame.png').convert('RGB'));m=json.loads((A/'inputs'/n/'manifest.json').read_text())
  for s,e in m['ears'].items():
   q=mask(V/f'{n}-ear-{s}-candidate.png');cont,_=cv2.findContours(np.uint8(q),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
   if s in KEEP[n]:cv2.drawContours(im,cont,-1,(255,65,65),2)
   c=tuple(np.rint(e['registeredContourCenter']).astype(int));cv2.circle(im,c,5,(0,255,255),-1)
  x=i%4*700;y=i//4*620;cand.paste(Image.fromarray(im).crop((150,150,850,720)),(x,y+40));dc.text((x+5,y+5),n+' | RED proposed visible-ear masks, CYAN registered centers',fill='white',font=f)
 cand.save(A/'sheets/C1-point-ear-candidates-centers.png')
 report={'poses':summary,'TPSRefitPoses':changed,'sheet':str(path),'logicalSAMCalls':12,'snapshot':str(snapshot),
  'hairBandsUnchanged':True,'registeredFramesUnchanged':True,'manualJawCuts':[],
  'geometryDifferences':[{'pose':r['name'],'side':s,**e} for r in summary for s,e in r['ears'].items() if e['geometryDiffersFromParent']],
  'earFitting':False,'ASolve':False,'checkpoint':'C1 awaiting coordinator review'}
 (V/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');(A/'C1-input-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print('POINT C1 COMPLETE changed',changed,'geometry differences',len(report['geometryDifferences']),flush=True)
if __name__=='__main__':main()
