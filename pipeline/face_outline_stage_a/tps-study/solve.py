from pathlib import Path as _ProjectPath
PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[3]
import json,subprocess,sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from tps_distance import TPS,fit
P=Path(__file__).resolve().parents[1];O=P/'tps-study';R=PROJECT_ROOT
sys.path.insert(0,str(R/'pipeline'))
from optimize_3d_2d import laplacian,determinants
from ponytail_observations import similarity
from ponytail_mesh import registration_matrix

def read_data(root):
 p=root/'bundle/face-solve-data.mjs';s=p.read_text();mark='export const FACE_SOLVE='
 return p,s.split(mark)[0]+mark,json.loads(s.split(mark)[1].strip().rstrip(';'))

def outer_loop(mesh):
 counts={}
 for tri in mesh['triangles']:
  for a,b in zip(tri,np.roll(tri,-1)):
   k=tuple(sorted((int(a),int(b))));counts[k]=counts.get(k,0)+1
 adj={}
 for (a,b),n in counts.items():
  if n==1:adj.setdefault(a,[]).append(b);adj.setdefault(b,[]).append(a)
 loops=[];unused=set(adj)
 while unused:
  first=next(iter(unused));loop=[first];prev=-1;cur=first
  while True:
   nxt=next(i for i in adj[cur] if i!=prev)
   if nxt==first:break
   loop.append(nxt);prev,cur=cur,nxt
  unused.difference_update(loop);loops.append(loop)
 return max(loops,key=len)

def correspondence():
 _,_,data=read_data(R);rig=json.loads((R/'bundle/assets/seosoyoung-rig.json').read_text());m=next(m for m in rig['meshes'] if m['id']=='face-skin')
 x,y,w,h=m['bbox'];ax,ay,_,_=m['atlas'];texture=np.array(Image.open(R/'bundle/assets/texture_00.png').convert('RGBA'))
 source=np.zeros((1024,1024),bool);source[y:y+h,x:x+w]=texture[ay:ay+h,ax:ax+w,3]>127
 q=np.array(json.loads((P/'reference-contours/left.json').read_text()));target=np.zeros((1024,1024),np.uint8);cv2.fillPoly(target,[np.round(q).astype(np.int32)],1);target=target>0
 front=np.array(data['points']);reg=json.loads((R/'source/viewer/ponytail-seedance-registration.json').read_text());key=next(k for k in reg['keys'] if k['name']=='left')
 features=json.loads((R/'source/viewer/face-feature-fit.json').read_text())['poses'];f=next(k for k in features if k['name']=='left')
 a=registration_matrix(key,f);registered=np.array(key['landmarks'])@a[:,:2].T+a[:,2]
 srcEyes=front[[data['anchors']['left'],data['anchors']['right']]]
 linear,shift=similarity(srcEyes,registered[:2])
 brow=int(np.ceil(max(front[data['anchors'][k],1] for k in ['browLeft','browRight'])))
 bottom=int(np.ceil(np.where(source)[0].max()+40))
 model,info=fit(source,target,brow,bottom,linear,shift,(registered[1,0]+10,registered[:2,1].mean()))
 Image.fromarray(np.uint8(source)*255).save(O/'source-mask.png');Image.fromarray(np.uint8(target)*255).save(O/'reference-mask.png')
 info.update(sourceBrowY=brow,sourceBottomY=bottom,referenceBrowY=float(registered[2:4,1].max()),referenceChinY=float(registered[6,1]),frame=42)
 (O/'tps.json').write_text(json.dumps({'controls':model.controls.tolist(),'coefficients':model.coefficients.tolist(),'fit':info,'referenceLandmarks':registered.tolist()},indent=2)+'\n')
 print('TPS FIT',info,flush=True)

def solve(root):
 path,prefix,data=read_data(root);rig=json.loads((root/'bundle/assets/seosoyoung-rig.json').read_text());mesh=next(m for m in rig['meshes'] if m['id']=='face-skin')
 inputs=json.loads((O/(root.name+'-model.json')).read_text());points=np.array(data['points']);old=np.array(inputs['poses']['left']['world'])
 saved=json.loads((O/'tps.json').read_text());model=TPS(saved['controls'],saved['coefficients'])
 boundary=outer_loop(mesh);anchors=set(data['anchors'].values());boundary=[i for i in boundary if points[i,1]>=saved['fit']['sourceBrowY'] and i not in anchors]
 # Only observations and the six anchors are Dirichlet conditions. The
 # neutral pose is retained unchanged; no source-height lock is added.
 pinned=anchors|set(boundary)
 delta=np.zeros((len(points),2));delta[boundary]=model(points[boundary])-old[boundary,:2]
 held=np.array(sorted(pinned),int);free=np.array(sorted(set(range(len(points)))-pinned),int)
 lap=laplacian(points,data['triangles'],data['meshCount'])
 if len(free):delta[free]=np.linalg.solve(lap[np.ix_(free,free)],-lap[np.ix_(free,held)]@delta[held])
 pose=next(p for p in data['poses'] if p['name']=='left');beforeZ=list(data['Z'])
 pose['du']=(np.array(pose['du'])+delta).tolist()
 assert data['Z']==beforeZ and np.max(np.abs(delta[list(anchors)]))==0
 det=determinants(points,old[:,:2]+delta,data['triangles'])
 report={'variant':root.name,'boundaryGoals':len(boundary),'freeVertices':len(free),'ZChanged':0,'anchorMaxDelta':0,
  'maxDisplacement':float(np.linalg.norm(delta,axis=1).max()),'F4BelowQuarter':int((det<.25).sum()),'negativeDeterminants':int((det<=0).sum()),'minDeterminant':float(det.min()),
  'belowQuarterTriangles':np.flatnonzero(det<.25).tolist(),'barrierUsed':False,'halfStepLimiterUsed':False}
 path.write_text(prefix+json.dumps(data,separators=(',',':'))+';\n')
 (O/(root.name+'-solve.json')).write_text(json.dumps(report,indent=2)+'\n')
 (O/(root.name+'-goals.json')).write_text(json.dumps({'indices':boundary,'target':model(points[boundary]).tolist(),'delta':delta.tolist()}))
 print('B SOLVE',report,flush=True)

if __name__=='__main__':
 if sys.argv[1]=='correspondence':correspondence()
 else:solve(Path(sys.argv[1]))
