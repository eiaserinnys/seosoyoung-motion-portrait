from pathlib import Path as _ProjectPath
PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[5]
"""Authorized C1 redo: 8 face point requests and 4 optional ear point requests."""
import concurrent.futures,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
V=Path(__file__).resolve().parent;A=V.parent;O=A.parent;R=PROJECT_ROOT
sys.path.insert(0,str(R/'pipeline'))
from stage1_sam3_segment import call_sam3_on_crop
NAMES=['left','right','up','down','up-left','up-right','down-left','down-right']
# Actual image ear locations reviewed on registered frames, not inferred color scores.
EAR_MANUAL={('up','right'):[665,480],('down','left'):[310,410],('down','right'):[650,410],('up-left','right'):[660,515]}
KEEP={'left':['right'],'right':['left'],'up':['left','right'],'down':['left','right'],'up-left':['right'],'up-right':['left'],'down-left':['right'],'down-right':['left']}

def plans():
 jobs=[];centers={}
 for n in NAMES:
  m=json.loads((A/'c1-rejected-color/inputs'/n/'manifest.json').read_text());centers[n]={}
  for side,e in m['ears'].items():
   center=np.mean(e['registeredInputContour'],0)
   if (n,side) in EAR_MANUAL:center=np.array(EAR_MANUAL[n,side])
   elif side in KEEP[n]:
    q=np.array(Image.open(A/'sam3-c1/raw'/n/f'ear-{side}.png'))>127
    if n=='left' and side=='right':q=np.array(Image.open(O/'v2/reference-ear-mask.png'))>127
    yy,xx=np.where(q);center=np.array([xx.mean(),yy.mean()])
   centers[n][side]=np.rint(center).astype(int).tolist()
  pts=np.array(m['registeredLandmarks']);eyes=pts[:2];mid=eyes.mean(0)
  points=[np.rint(p).astype(int).tolist()+[1] for p in [eyes[0]+[0,40],eyes[1]+[0,40],mid+[0,-60],pts[4]]]
  points += [[int(round(pts[6,0])),int(round(pts[6,1]+60)),0],[int(round(mid[0])),max(10,int(round(mid[1]-260))),0]]
  points += [[*centers[n][s],0] for s in ['left','right']]
  jobs.append({'id':n+'-face','pose':n,'part':'face','prompt':'face','crop':[150,0,850,780],'points':points,'boxes':None})
 for (n,s),center in EAR_MANUAL.items():
  x,y=center;cheek=[x+(-65 if s=='right' else 65),y+5];hair=[x,y-55]
  jobs.append({'id':n+'-ear-'+s,'pose':n,'part':'ear-'+s,'prompt':'ear','crop':[x-90,y-90,x+90,y+90],
   'points':[[x,y,1],[*cheek,0],[*hair,0]],'boxes':None})
 for j in jobs:
  j['frame']=str(A/'inputs'/j['pose']/'reference-frame.png');j['frameSHA256']=hashlib.sha256(Path(j['frame']).read_bytes()).hexdigest()
  assert all(j['crop'][0]<=x<j['crop'][2] and j['crop'][1]<=y<j['crop'][3] for x,y,_ in j['points'])
 return jobs,centers

def call(job,key,api=call_sam3_on_crop):
 im=Image.open(job['frame']).convert('RGBA')
 try:mask,meta=api(key,im,tuple(job['crop']),job['prompt'],points=job['points'],boxes=None,upscale=1)
 except ValueError as e:
  if str(e)!='SAM 3 returned an empty mask list':raise
  mask=np.zeros((im.height,im.width),np.uint8);meta={'score':0.,'area_px':0,'emptyMaskResponse':True}
 return mask,meta

def execute(j,key):
 out=V/'raw'/j['pose'];out.mkdir(parents=True,exist_ok=True);p=out/(j['part']+'.json');png=p.with_suffix('.png')
 fp=hashlib.sha256(json.dumps(j,sort_keys=True).encode()).hexdigest()
 if p.exists():
  result=json.loads(p.read_text());assert result['requestFingerprint']==fp and png.exists();return result
 start=time.time();mask,meta=call(j,key);Image.fromarray(mask).save(png)
 result={**meta,**j,'requestFingerprint':fp,'elapsedSeconds':round(time.time()-start,2)}
 p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print('POINT SAM',j['id'],'score',meta['score'],'pixels',int((mask>127).sum()),flush=True);return result

def main():
 jobs,centers=plans();(V/'request-plan.json').write_text(json.dumps({'jobs':jobs,'pointEarCenters':centers},ensure_ascii=False,indent=2)+'\n')
 sys.path.insert(0,str(Path(os.environ['AGENT_COMMON_FILES_DIR'])/'lib'))
 from secret_loader import get_secret
 from scratch_workspace import resolve_scratch_dir
 resolve_scratch_dir();key=get_secret('shared/fal','api_key');assert bool(key)
 done=[execute(jobs[0],key)];print('ONE IMAGE POINT SMOKE COMPLETE',flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for result in pool.map(lambda j:execute(j,key),jobs[1:]):done.append(result)
 (V/'call-manifest.json').write_text(json.dumps(done,ensure_ascii=False,indent=2)+'\n')
 print('POINT SAM COMPLETE',len(done),'calls; no TPS or renderer',flush=True)
if __name__=='__main__':main()
