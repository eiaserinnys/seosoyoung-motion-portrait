"""Least squares ellipse from SAM 3 crown/temporal outer silhouettes."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
PITCH_ROWS=[70,150,230,310,380]
def measure_pitch(fit):
 """Registered solid crown (excluding flyaways) and blue bead-pair centers.

 Keep raw measurements alongside the constrained crown target: down poses
 retain at least 25px headroom, the requested solid-crown silhouette limit.
 """
 features=json.loads((ROOT/'source/viewer/face-feature-fit.json').read_text())
 poses=[]
 for feature in features['poses']:
  if not feature['pitch']:continue
  name=feature['name'];path=ROOT/'source/references'/f'{name.replace("-","_")}_ref_720.png'
  rgb=np.array(Image.open(path).convert('RGB'));hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
  mask=((rgb.max(2)>45)&(hsv[:,:,1]>25)).astype(np.uint8)
  solid=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((7,7),np.uint8))
  count,labels,stats,_=cv2.connectedComponentsWithStats(solid)
  largest=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));solid=labels==largest
  raw_top=int(np.where(solid[:180])[0].min())
  delta=feature['registration']['delta'];offset=[152+delta[0],20+delta[1]]
  measured_top=raw_top+offset[1]
  target_top=max(25.,measured_top) if feature['pitch']<0 else measured_top
  beads={}
  for side,(x0,x1) in {'left':(50,240),'right':(470,650)}.items():
   blue=(hsv[:,:,0]>85)&(hsv[:,:,0]<130)&(hsv[:,:,1]>110)&(hsv[:,:,2]>65)
   roi=np.zeros_like(blue);roi[440:600,x0:x1]=True;blue&=roi
   yy,xx=np.where(blue)
   if len(xx)<30:raise ValueError(f'{name} {side}: missing bead measurement')
   # Extent midpoint represents the attachment/pair center, unlike a color
   # centroid biased toward one highlighted bead. Trim isolated color pixels.
   center=[float((np.percentile(v,2)+np.percentile(v,98))/2) for v in [xx,yy]]
   beads[side]=dict(reference=center,registeredObservation=[center[i]+offset[i] for i in [0,1]],pixels=len(xx),roi=[x0,440,x1,600])
  poses.append(dict(name=name,yaw=feature['yaw'],pitch=feature['pitch'],source=str(path.relative_to(ROOT)),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),registration=delta,
                    crown=dict(referenceTopY=raw_top,measuredY=measured_top,targetY=target_top),beads=beads))
 # Evaluate only Stage 1. Rebuilding cannot feed the previous residual back
 # into its own fit, and no face/outline generator is invoked here.
 script="""
 import {readFileSync} from 'node:fs';
 import {fringeRotation,craniumRotation} from './bundle/hair-shell.mjs';
 import {domeSection,craniumFacing} from './bundle/hair-surface.mjs';
 import {beadWorld} from './bundle/stereo-attachments.mjs';
 import {CORRECTION_PRESETS} from './bundle/geometry.mjs';
 import {DOME} from './bundle/dome-data.mjs';
 const input=JSON.parse(readFileSync(0,'utf8'));
 DOME.depthPrior=input.depthPrior;
 const rig=JSON.parse(readFileSync('./bundle/assets/seosoyoung-rig.json'));
 const defaults=Object.fromEntries(rig.parameters.map(p=>[p.id,p.default]));
 const rear=rig.meshes.filter(m=>m.id.startsWith('side-cranium'));
 console.log(JSON.stringify(input.poses.map(pose=>{
  const p={...defaults,...CORRECTION_PRESETS.recommended,CorrTurn:1,ParamAngleX:pose.yaw,ParamAngleY:pose.pitch,ParamAngleZ:0};
  const sample=y=>{
   const {cx,rx}=domeSection(y);
   const samples=Array.from({length:41},(_,i)=>{
    const point=[502+cx+rx*(i/20-1),y];return Object.assign(fringeRotation(point,p),{point,kind:'front'});
   });
   for(const sign of [-1,1])for(let i=0;i<=40;i++){
    const pt=[0,y,sign*(88+i/40*92)*Math.PI/180];
    const side=sign<0?'left':'right';
    if(craniumFacing(pt,p)>1e-6)samples.push(Object.assign(craniumRotation(pt,side,p),{point:pt,kind:'rear',side}));
   }
   return samples;
  };
  const candidates=Array.from({length:35},(_,i)=>70+i*5).flatMap(y=>sample(y).map(q=>({y,worldY:q[1],point:q.point,kind:q.kind,side:q.side})));
  for(const m of rear)for(const point of m.positions)if(point[1]<=240&&craniumFacing(point,p)>1e-6){
   const side=m.id.endsWith('left')?'left':'right';
   candidates.push({y:point[1],worldY:craniumRotation(point,side,p)[1],point,kind:'rear',side});
  }
  const crown=candidates.sort((a,b)=>a.worldY-b.worldY)[0];
  return {top:fringeRotation([502,70],p),silhouette:crown,rows:[70,150,230,310,380].map(y=>{
   const samples=sample(y),{cx,rx}=domeSection(y);
   const xs=samples.map(q=>q[0]);
   const edges=[-1,1].map(s=>fringeRotation([502+cx+s*rx,y],p));
   return {span:[Math.min(...xs),Math.max(...xs)],y:(edges[0][1]+edges[1][1])/2};
  }),beads:Object.fromEntries(['left','right'].map(side=>[side,beadWorld(side,p)]))};
 })));
 """
 baseline=json.loads(subprocess.check_output(['node','--input-type=module','-e',script],input=json.dumps({'poses':poses,'depthPrior':fit['depthPrior']}),text=True,cwd=ROOT))
 for pose,base in zip(poses,baseline):
  rgb=np.array(Image.open(ROOT/pose['source']).convert('RGB'));hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
  mask=((rgb.max(2)>45)&(hsv[:,:,1]>25)).astype(np.uint8)
  mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
  c=pose['crown'];c['baseline']=base['top'];c['silhouette']=base['silhouette'];c['deltaY']=c['targetY']-base['silhouette']['worldY'];c['rows']=[];c['measurements']=[]
  for y,edge in zip(PITCH_ROWS,base['rows']):
   if y==380:c['rows'].append([y,0.,0.]);continue
   t=float(np.clip((y-240)/140,0,1));fade=1-t*t*(3-2*t)
   target_y=edge['y']+c['deltaY']*fade
   row=int(np.clip(round(target_y-20-pose['registration'][1]),0,330))
   xs=np.flatnonzero(mask[row,40:680])+40
   if not len(xs):raise ValueError(f'{pose["name"]}: missing crown row {row}')
   left,right=[float(v+152+pose['registration'][0]) for v in [xs[0],xs[-1]]]
   lo,hi=edge['span'];center=(lo+hi)/2
   measured_scale=(right-left)/(hi-lo)-1
   # Close-up crop widths are observations, not permission to expand the
   # approved head silhouette. Only inward interior corrections are fitted.
   scale=min(0.,measured_scale)
   # Interpolate affine coefficients, never a weighted pivot. A missing
   # neutral row then means exactly the zero displacement at every X.
   c['rows'].append([y,-center*scale,scale])
   c['measurements'].append(dict(sourceY=y,referenceRow=row,targetSpan=[left,right],measuredScale=measured_scale,baseline=edge))
  # Reference silhouette centers are not the approved face's head center.
  # Fit only width; the pole's apex row is not homologous to a full 3D ring.
  # Extend the first upper-temple width to the pole instead of collapsing it.
  scale=c['rows'][1][2];lo,hi=base['rows'][0]['span']
  c['rows'][0][1]=-(lo+hi)/2*scale;c['rows'][0][2]=scale
  for side in ['left','right']:
   # Generated references keep blue ties at almost fixed screen Y. Facial
   # registration therefore separates them from the rotated skull. Retain
   # that observation for inspection; never fit it as an attachment offset.
   b=pose['beads'][side];b['baseline']=base['beads'][side]
   b['observedDelta']=[b['registeredObservation'][i]-b['baseline'][i] for i in [0,1]]
   b['attachmentPolicy']='measured-observation-only'
 fit['pitchRows']=PITCH_ROWS;fit['pitchPoses']=poses
def write_fit(fit):
 (ROOT/'source/viewer/dome-fit.json').write_text(json.dumps(fit,ensure_ascii=False,indent=2)+'\n')
 (ROOT/'bundle/dome-data.mjs').write_text('// Generated by pipeline/fit_dome.py.\nexport const DOME='+json.dumps(fit,ensure_ascii=False)+';\n')
 print(json.dumps([dict(name=p['name'],measuredY=p['crown']['measuredY'],targetY=p['crown']['targetY'],beadObservations={s:b['registeredObservation'] for s,b in p['beads'].items()}) for p in fit['pitchPoses']]))
def update_pitch(fit):
 from hair_depth import build_depth_prior
 fit['depthPrior']=build_depth_prior(infer='--infer-depth' in sys.argv)
 measure_pitch(fit);write_fit(fit)
def build(refit_yaw=False):
 # The approved yaw calibration is an input to the pitch fit. Recomputing
 # its seam changes neutral-pitch vertices, so yaw refitting is explicit.
 if not refit_yaw:
  fit=json.loads((ROOT/'source/viewer/dome-fit.json').read_text())
  update_pitch(fit);return
 samples=[]
 sources=[('sam3_masks/hair-crown-all_mask.png',0,0,'both'),
          ('sam3_masks/references/left-crown.png',152,20,'both'),
          ('sam3_masks/references/right-crown.png',152,20,'both')]
 for path,ox,oy,side in sources:
  mask=np.array(Image.open(ROOT/path).convert('L'))
  for y in range(70,341,10):
   xs=np.flatnonzero(mask[y-oy]>127)
   if not len(xs):continue
   for x in ([xs[0],xs[-1]] if side=='both' else [xs[0]]):samples.append([float(x+ox),float(y)])
 # x is centred at the common stage head axis, not mirrored between references.
 pts=np.array(samples);yy=pts[:,1]-326;xx=pts[:,0]-502
 coef=np.linalg.lstsq(np.column_stack([np.ones(len(yy)),yy,yy*yy]),xx*xx,rcond=None)[0]
 a,b,c=coef;cy=326-b/(2*c);rx=np.sqrt(a-b*b/(4*c));ry=np.sqrt(-(a-b*b/(4*c))/c)
 predicted=rx*np.sqrt(np.maximum(0,1-((pts[:,1]-cy)/ry)**2))
 error=predicted-np.abs(xx)
 fit=dict(depthRatio=1.0,cx=502,cy=float(cy),rx=float(rx),ry=float(ry),sampleCount=len(samples),
          rmsPixels=float(np.sqrt(np.mean(error*error))),maxPixels=float(np.max(np.abs(error))),sources=[x[0] for x in sources])
 # Lower skull: solve its posterior centre and depth radius jointly. The
 # ear attachment lies on the front surface; the observed 45-degree outer
 # silhouette is the support function of the rotated ellipse.
 crowns=[np.array(Image.open(ROOT/'sam3_masks/references'/f'{side}-crown.png').convert('L')) for side in ['left','right']]
 outer=[]
 for y in range(450,481,5):
  l=np.flatnonzero(crowns[0][y-20]>127)+152
  r=np.flatnonzero(crowns[1][y-20]>127)+152
  outer.extend([float(l[-1]-502),float(502-r[0])])
 rx_lower=175.;attachment_x=123.5;attachment_z=(19.3+38.1)/2
 extent=float(np.median(outer));factor=np.sqrt(1-(attachment_x/rx_lower)**2)
 lo,hi=0.,500.
 for _ in range(60):
  rz=(lo+hi)/2;cz=attachment_z-rz*factor
  if (np.hypot(rx_lower,rz)-cz)/np.sqrt(2)<extent:lo=rz
  else:hi=rz
 fit['lower']=dict(rx=rx_lower,rz=float(rz),cz=float(cz),extent45=extent,
                   attachmentX=attachment_x,attachmentZ=attachment_z,samples=outer)
 # The front artwork is the seam contract, not the least-squares ellipse.
 parts=json.loads((ROOT/'parts/parts_index.json').read_text())
 alpha=np.zeros((1536,1024),dtype=np.uint8)
 for part in parts:
  if part['id'] not in ['hair-top','side-hair-left','side-hair-right']:continue
  x,y,w,h=part['bbox'];a=np.array(Image.open(ROOT/part['file']).convert('RGBA'))[:,:,3]
  alpha[y:y+h,x:x+w]=np.maximum(alpha[y:y+h,x:x+w],a)
 rows=[]
 for y in range(60,351,2):
  edges=np.diff(np.pad(alpha[y]>128,(1,1)).astype(int))
  runs=[(l,r) for l,r in zip(np.where(edges==1)[0],np.where(edges==-1)[0]) if r-l>=20]
  rows.append([y,float(runs[0][0]+1),float(runs[-1][1]-2)])
 values=np.array(rows,dtype=float)
 values[:,1:]=cv2.GaussianBlur(values[:,1:],(1,0),sigmaY=4,sigmaX=0)
 values[:,1]+=2;values[:,2]-=2
 fit['frontSeam']=values.tolist()
 update_pitch(fit)
if __name__=='__main__':build(refit_yaw='--refit-yaw' in sys.argv)
