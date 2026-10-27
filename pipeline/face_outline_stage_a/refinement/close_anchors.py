"""Close exact anchor observations against the actual JS runtime after du recovery."""
from pathlib import Path
import copy,json,math,subprocess,sys

def read_face(root):
 return json.loads((root/'bundle/face-solve-data.mjs').read_text().split('export const FACE_SOLVE=',1)[1].strip().rstrip(';'))
def write_face(root,face):
 (root/'bundle/face-solve-data.mjs').write_text('export const FACE_SOLVE='+json.dumps(face,ensure_ascii=False,separators=(',',':'))+';\n')
def runtime(root,mode='anchors'):
 return json.loads(subprocess.check_output(['node',str(Path(__file__).with_name('runtime_nodes.mjs')),str(root),mode],timeout=120))
def close(root,inputs,report_path):
 face=read_face(root);before=copy.deepcopy(face);history=[]
 poses=[*face['poses'],*face.get('family45',{}).get('poses',[])];old_poses=[*before['poses'],*before.get('family45',{}).get('poses',[])];observations=[*inputs['poses'],*inputs.get('family45',{}).get('poses',[])]
 for iteration in range(2):
  actual=runtime(root);changes=[]
  for pose in poses:
   obs=next(q for q in observations if q['name']==pose['name'])['obs'];row=next(q for q in actual if q['name']==pose['name'])
   for key,index in face['anchors'].items():
    delta=[obs[index][d]-row['anchors'][key][d] for d in [0,1]]
    if math.hypot(*delta)<1e-12:continue
    for d in [0,1]:pose['du'][index][d]+=delta[d]
    changes.append(dict(pose=pose['name'],anchor=key,index=index,delta=delta))
  maximum=max((math.dist(p['du'][i],old['du'][i]) for p,old in zip(poses,old_poses) for i in face['anchors'].values()),default=0)
  history.append(dict(iteration=iteration+1,changes=changes,maxDeltaDu=maximum))
  report_path.write_text(json.dumps(dict(history=history,maxDeltaDu=maximum),indent=2)+'\n')
  if maximum>1e-11:raise RuntimeError('Anchor correction exceeds the instructed 1e-11 scale; stop')
  if not changes:break
  write_face(root,face)
 actual=runtime(root);residuals=[]
 for pose in poses:
  obs=next(q for q in observations if q['name']==pose['name'])['obs'];row=next(q for q in actual if q['name']==pose['name'])
  for key,index in face['anchors'].items():residuals.append(dict(pose=pose['name'],anchor=key,error=math.dist(obs[index],row['anchors'][key][:2])))
 report=dict(history=history,maxDeltaDu=maximum,maxError=max(q['error'] for q in residuals),residuals=residuals)
 report_path.write_text(json.dumps(report,indent=2)+'\n')
 if report['maxError']>=1e-12:raise RuntimeError('Runtime anchor closure did not reach the unchanged 1e-12 gate')
 return report
if __name__=='__main__':
 root=Path(sys.argv[1]).resolve();print(json.dumps(close(root,json.loads(Path(sys.argv[2]).read_text()),Path(sys.argv[3])),indent=2))
