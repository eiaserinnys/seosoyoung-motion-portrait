"""Deterministic color/ink measurements inside explicit reference-space ROIs.

The neutral yaw row is the approved r11 calibration. New pitch keyforms use
iris color extents, eyebrow ink, nose highlight and mouth ink centroids.
"""
import hashlib,json,subprocess
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
# Each seed bounds one visible feature, in the 720 square reference canvas.
SEEDS={
 'up-left':[(183,335),(330,334),(188,256),(330,249),(210,376),(220,451)],
 'up':[(269,294),(450,294),(271,208),(443,208),(361,322),(359,405)],
 'up-right':[(411,335),(550,338),(423,249),(552,261),(544,369),(524,450)],
 'down-left':[(183,437),(336,458),(174,355),(322,374),(225,503),(246,561)],
 'down':[(269,483),(439,481),(273,421),(437,420),(357,539),(354,583)],
 'down-right':[(397,493),(546,473),(421,419),(570,399),(515,540),(478,595)],
}
KEYS=['left','right','browLeft','browRight','nose','mouth']
def symmetrize(front,poses):
 for pose in poses:
  pose['registeredCenters']={k:v[:] for k,v in pose['centers'].items()}
 for pitch in sorted({p['pitch'] for p in poses}):
  group={p['yaw']:p for p in poses if p['pitch']==pitch};center=group[0]
  for key in ['nose','mouth']:center['centers'][key][0]=front[key][0]
  for left,right in [('left','right'),('browLeft','browRight')]:
   l=center['centers'][left];r=center['centers'][right]
   dx=((r[0]-front[right][0])-(l[0]-front[left][0]))/2
   dy=((l[1]-front[left][1])+(r[1]-front[right][1]))/2
   center['centers'][left]=[front[left][0]-dx,front[left][1]+dy]
   center['centers'][right]=[front[right][0]+dx,front[right][1]+dy]
  scale=[(center['scales']['left'][i]+center['scales']['right'][i])/2 for i in [0,1]]
  center['scales']={'left':scale[:],'right':scale[:]}
  neg,pos=group[-45],group[45]
  for key in ['nose','mouth']:
   n,p=neg['centers'][key][:],pos['centers'][key][:];f=front[key]
   dx=((p[0]-f[0])-(n[0]-f[0]))/2;dy=(n[1]+p[1])/2-f[1]
   neg['centers'][key]=[f[0]-dx,f[1]+dy];pos['centers'][key]=[f[0]+dx,f[1]+dy]
  for left,right in [('left','right'),('browLeft','browRight')]:
   for a,b in [(left,right),(right,left)]:
    n,p=neg['centers'][a][:],pos['centers'][b][:];fa,fb=front[a],front[b]
    dx=((p[0]-fb[0])-(n[0]-fa[0]))/2;dy=((n[1]-fa[1])+(p[1]-fb[1]))/2
    neg['centers'][a]=[fa[0]-dx,fa[1]+dy];pos['centers'][b]=[fb[0]+dx,fb[1]+dy]
  for a,b in [('left','right'),('right','left')]:
   scale=[float(np.sqrt(neg['scales'][a][i]*pos['scales'][b][i])) for i in [0,1]]
   neg['scales'][a]=scale[:];pos['scales'][b]=scale[:]
def measure(rgb,seed,kind):
 x,y=seed;rx,ry=(36,44) if kind in ['left','right'] else (29,10) if kind.startswith('brow') else (13,10) if kind=='nose' else (28,9)
 crop=rgb[y-ry:y+ry+1,x-rx:x+rx+1];hsv=cv2.cvtColor(crop,cv2.COLOR_RGB2HSV)
 if kind in ['left','right']:
  mask=(hsv[:,:,0]>20)&(hsv[:,:,0]<145)&(hsv[:,:,1]>75)&(hsv[:,:,2]>75)
  connected=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
  count,labels,stats,centers=cv2.connectedComponentsWithStats(connected)
  candidates=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>20 and np.linalg.norm(centers[i]-[rx,ry])<30]
  mask=np.isin(labels,candidates)
 elif kind=='nose':mask=(crop.min(2)>215)&(hsv[:,:,1]<65)
 else:mask=crop.mean(2)<(145 if kind.startswith('brow') else 165)
 if kind=='mouth':
  count,labels,stats,centers=cv2.connectedComponentsWithStats(mask.astype(np.uint8))
  keep=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=3 and np.linalg.norm(centers[i]-[rx,ry])<23]
  mask=np.isin(labels,keep)
 yy,xx=np.where(mask)
 if len(xx)<3:raise ValueError(f'No feature at {seed} ({kind})')
 # Medians resist individual highlights and single-pixel ink tails.
 center=[(np.percentile(xx,2)+np.percentile(xx,98))/2,(np.percentile(yy,2)+np.percentile(yy,98))/2] if kind in ['left','right','mouth'] else [np.median(xx),np.median(yy)]
 return dict(center=[float(center[0]+x-rx+152),float(center[1]+y-ry+20)],
             extent=[int(np.percentile(xx,98)-np.percentile(xx,2)+1),int(np.percentile(yy,98)-np.percentile(yy,2)+1)],roi=[x-rx,y-ry,x+rx,y+ry])
def build():
 front={'left':[415.7,432.2],'right':[588.1,430.8],'nose':[508,497],'mouth':[504.5,554.5]}
 rig=json.loads((ROOT/'bundle/assets/seosoyoung-rig.json').read_text())
 for key,side in [('browLeft','left'),('browRight','right')]:
  m=next(m for m in rig['meshes'] if m['id']==f'eye-{side}-brow')
  a=np.array(Image.open(ROOT/'parts'/f'eye-{side}-brow.png'))[:,:,3].astype(float);yy,xx=np.indices(a.shape)
  front[key]=[float((xx*a).sum()/a.sum()+m['bbox'][0]),float((yy*a).sum()/a.sum()+m['bbox'][1])]
 poses=[]
 for name,yaw in [('left',-45),('front',0),('right',45)]:
  centers={k:v[:] for k,v in front.items()}
  neg={'left':[362.4,429.3],'right':[525.1,429.2],'nose':[406.9,497.1],'mouth':[434.1,556.2]}
  pos={'left':[514.2,430.7],'right':[670.1,428.9],'nose':[629.9,490.5],'mouth':[598.7,553.3]}
  if yaw:centers.update(neg if yaw<0 else pos)
  # r11 places brows in their eye's center-relative coordinate frame.
  widths=[36/44,47/43.9] if yaw<0 else [45/44,.72] if yaw>0 else [1,1]
  for key,side,i in [('browLeft','left',0),('browRight','right',1)]:
   centers[key]=[front[key][j]+centers[side][j]-front[side][j] for j in [0,1]]
  poses.append(dict(name=name,yaw=yaw,pitch=0,centers=centers,scales={'left':[widths[0],1],'right':[widths[1],1]},source='approved r11'))
 front_rgb=np.array(Image.open(ROOT/'source/references/head_front_ref.png').convert('RGB'))
 extents={k:measure(front_rgb,(round(front[k][0]-152),round(front[k][1]-20)),k)['extent'] for k in ['left','right']}
 for name,seeds in SEEDS.items():
  file=name.replace('-','_')+'_ref_720.png';src=ROOT/'source/references'/file;rgb=np.array(Image.open(src).convert('RGB'))
  measured={k:measure(rgb,seed,k) for k,seed in zip(KEYS,seeds)}
  scales={k:[measured[k]['extent'][i]/extents[k][i] for i in [0,1]] for k in ['left','right']}
  measured_scales={k:v[:] for k,v in scales.items()}
  for value in scales.values():value[1]=float(np.clip(value[1],.88,1.02))
  if name in ['up','down']:
   height=float(np.clip(max(v[1] for v in measured_scales.values()),.94,1.0))
   for value in scales.values():value[1]=height
  poses.append(dict(name=name,yaw=-45 if name.endswith('-left') else 45 if name.endswith('-right') else 0,pitch=30 if name.startswith('up') else -30,
                    centers={k:v['center'] for k,v in measured.items()},scales=scales,measuredScales=measured_scales,scalePolicy='height clamp [0.88,1.02]; yaw=0 paired eyes use unoccluded height clamped [0.94,1.0]',measurements=measured,source=file,sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
 # Symmetrize the approved row before deriving its shared registration frame.
 neutral=[p for p in poses if not p['pitch']]
 symmetrize(front,neutral)
 registration=json.loads(subprocess.check_output(['node',str(ROOT/'pipeline/reference_registration.mjs')],input=json.dumps(neutral,ensure_ascii=False),text=True))
 for pose in poses:
  if not pose['pitch']:continue
  reg=next(r for r in registration if r['yaw']==pose['yaw'] and r['pitch']==pose['pitch'])
  midpoint=[(pose['centers']['left'][i]+pose['centers']['right'][i])/2 for i in [0,1]]
  delta=[reg['eyeMid'][i]-midpoint[i] for i in [0,1]]
  pose['rawCenters']={k:v[:] for k,v in pose['centers'].items()}
  pose['centers']={k:[v[i]+delta[i] for i in [0,1]] for k,v in pose['centers'].items()}
  eye_dist=float(np.linalg.norm(np.array(pose['rawCenters']['right'])-pose['rawCenters']['left']))
  pose['registration']={**reg,'delta':delta,'referenceEyeMid':midpoint,'binocularRatio':eye_dist/reg['r11EyeDist'],'method':'translation: approved r11 yaw midpoint X plus relative child-eye pitch Y; no distance scaling'}
 symmetrize(front,poses)
 row={p['yaw']:p for p in poses if p['pitch']==0}
 landmarks={key:dict(front=value,neg=row[-45]['centers'][key],pos=row[45]['centers'][key]) for key,value in front.items()}
 for side in ['left','right']:landmarks[side]['width']=[row[-45]['scales'][side][0],row[45]['scales'][side][0]]
 data=dict(method=__doc__,front=front,poses=poses,landmarks=landmarks,symmetry='rest-relative mirrored center displacements; geometric paired eye scales')
 (ROOT/'source/viewer/face-feature-fit.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
 (ROOT/'bundle/feature-data.mjs').write_text('// Generated by pipeline/fit_features.py.\nexport const FEATURE_FIT='+json.dumps(data,ensure_ascii=False)+';\n')
 print('PASS measured',len(poses),'feature keyforms')
if __name__=='__main__':build()
