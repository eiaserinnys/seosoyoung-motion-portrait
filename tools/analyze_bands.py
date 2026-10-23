"""Private spatial diagnostics: crown/jaw bands and triangle distortion, not semantic labels."""
import gzip,json,sys
from pathlib import Path
import numpy as np
from deformation_metrics import metrics
sf,cf,out=sys.argv[1:];s=json.load(gzip.open(sf,'rt'));c=json.loads(Path(cf).read_text());ids=s['meshIds']
def points(p,i):return np.array(s['poses'][p]['vertices'][i]).reshape(-1,2)*[1,-1]
f=ids.index(c['face']);face=points('neutral',f);lo=face.min(axis=0);hi=face.max(axis=0);width=hi[0]-lo[0]
selections={'jaw_lower_15_percent':[(f,np.flatnonzero(face[:,1]>=hi[1]-.15*(hi[1]-lo[1])))],
 'face_screen_left':[(f,np.flatnonzero(face[:,0]<(lo[0]+hi[0])/2))],
 'face_screen_right':[(f,np.flatnonzero(face[:,0]>=(lo[0]+hi[0])/2))]}
selections['upper_hair_band']=[]
for region in ['hair_front','hair_side','hair_back']:
 for id in c['regions'][region]:
  i=ids.index(id);a=points('neutral',i);jj=np.flatnonzero(a[:,1]<=lo[1]+.2*(hi[1]-lo[1]))
  if len(jj):selections['upper_hair_band'].append((i,jj))
report={}
for key,selection in selections.items():
 a=np.concatenate([points('neutral',i)[jj] for i,jj in selection]);report[key]={'vertices':len(a),'poses':{}}
 for pose in ['xy_-30_0','xy_30_0','xy_0_-30','xy_0_30','xy_-30_-30','xy_-30_30','xy_30_-30','xy_30_30','roll_30']:
  b=np.concatenate([points(pose,i)[jj] for i,jj in selection]);report[key]['poses'][pose]=metrics(a/width,b/width)
# Geometry quality of the visible face mesh alone; excludes almost-zero rest triangles.
tri=np.array(s['indices'][f]).reshape(-1,3)
def area(p):
 a=p[tri[:,1]]-p[tri[:,0]];b=p[tri[:,2]]-p[tri[:,0]];return a[:,0]*b[:,1]-a[:,1]*b[:,0]
base=area(face);valid=np.abs(base)>width*width*1e-10
report['face_triangles']={}
for pose in ['xy_30_0','xy_0_30','xy_30_30']:
 ratio=area(points(pose,f))[valid]/base[valid];report['face_triangles'][pose]={'valid':int(valid.sum()),'reversed':int((ratio<0).sum()),'area_ratio_p05_p50_p95':np.quantile(ratio,[.05,.5,.95]).tolist()}
Path(out).write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k=='face_triangles'}))
