"""7.18 A fallback: retain the approved 45 keys and blend absolute 6dd profile positions."""
from pathlib import Path
import hashlib,json,re,subprocess,sys

BASELINE='6dd12cfe12b89a3b67b23938996a62c70e281173'
def restore_b(root):
 p=root/'bundle/face-solve.mjs';s=p.read_text()
 if "from './profile-baseline/face-solve.mjs';" not in s:return
 for line in ["import {faceSolveWorld as baselineProfileWorld} from './profile-baseline/face-solve.mjs';\n","import {faceSolveWorld as baselineProfileWorld,faceSolveAnchorWorld as baselineProfileAnchorWorld} from './profile-baseline/face-solve.mjs';\n","import {FRAME_CACHE as BASELINE_FRAME_CACHE} from './profile-baseline/pose-grid.mjs';\n","import {smoothstep} from './face-profile.mjs';\n"]:s=s.replace(line,'')
 s=s.replace('function face45Prior(point,p,family,relief,weights,useBaseline=true){','function face45Prior(point,p,family,relief,weights){').replace('face45Prior(FACE_SOLVE.points[i],flat,family,0,[[i,1]],false)','face45Prior(FACE_SOLVE.points[i],flat,family,0,[[i,1]])')
 start=s.index(' const current=artWorld(');end=s.index('\n}',start)
 s=s[:start]+" return artWorld([q[0]-502,q[1]-326,q[2]+shifted0[2]-z-relief],{...p,ParamAngleX:0,ParamAngleY:0});"+s[end:];p.write_text(s)
def install(root,out):
 restore_b(root)
 target=root/'bundle/profile-baseline';target.mkdir(exist_ok=True);pending=['face-solve.mjs'];seen=set();records=[]
 while pending:
  name=pending.pop()
  if name in seen:continue
  seen.add(name);raw=subprocess.check_output(['git','-C',str(root),'show',BASELINE+':bundle/'+name]);dest=target/name;dest.parent.mkdir(exist_ok=True,parents=True);dest.write_bytes(raw)
  records.append(dict(path=name,sha256=hashlib.sha256(raw).hexdigest()))
  for child in re.findall(r"from ['\"]\./([^'\"]+)['\"]",raw.decode()):pending.append(child)
 (target/'manifest.json').write_text(json.dumps(dict(commit=BASELINE,files=sorted(records,key=lambda q:q['path'])),indent=2)+'\n')
 p=root/'bundle/face-solve.mjs';s=p.read_text();needle="import {FACE_SOLVE} from './face-solve-data.mjs';";s=s.replace(needle,needle+"\nimport {faceSolveWorld as baselineProfileWorld,faceSolveAnchorWorld as baselineProfileAnchorWorld} from './profile-baseline/face-solve.mjs';\nimport {FRAME_CACHE as BASELINE_FRAME_CACHE} from './profile-baseline/pose-grid.mjs';\nimport {smoothstep} from './face-profile.mjs';")
 start=s.index('function face45Prior(');end=s.index('const fixedSideNodes=',start);piece=s[start:end]
 piece=piece.replace('function face45Prior(point,p,family,relief,weights){','function face45Prior(point,p,family,relief,weights,useBaseline=true){')
 piece=piece.replace('face45Prior(FACE_SOLVE.points[i],flat,family,0,[[i,1]])','face45Prior(FACE_SOLVE.points[i],flat,family,0,[[i,1]],false)')
 old=" return artWorld([q[0]-502,q[1]-326,q[2]+shifted0[2]-z-relief],{...p,ParamAngleX:0,ParamAngleY:0});"
 new=""" const current=artWorld([q[0]-502,q[1]-326,q[2]+shifted0[2]-z-relief],{...p,ParamAngleX:0,ParamAngleY:0});
 if(!useBaseline)return current;
 // Beyond 67.5 degrees the authored 6dd absolute profile is retained. At
 // 45 degrees the weight and its first derivative are both zero.
 const baselinePose=frameValue(p,'profile-baseline-pose',()=>({...p,[BASELINE_FRAME_CACHE]:new Map()}));
 const anchorName=weights.length===1&&weights[0][1]===1?Object.keys(FACE_SOLVE.anchors).find(name=>FACE_SOLVE.anchors[name]===weights[0][0]):undefined;
 const old=anchorName===undefined?baselineProfileWorld(point,baselinePose,relief):baselineProfileAnchorWorld(anchorName,baselinePose),w=smoothstep(0,22.5,Math.abs(yaw)-45);
 return current.map((v,d)=>v*(1-w)+old[d]*w);"""
 assert piece.count(old)==1;piece=piece.replace(old,new);s=s[:start]+piece+s[end:];p.write_text(s)
 out.mkdir(exist_ok=True,parents=True);(out/'profile-a.json').write_text(json.dumps(dict(baseline=BASELINE,files=records,cost='67.5°부터 떠 있는 귀와 직선으로 잘린 볼이 돌아온다'),ensure_ascii=False,indent=2)+'\n')
 print('A baseline dependency snapshot',len(records),'files',flush=True)
if __name__=='__main__':install(*(Path(a).resolve() for a in sys.argv[1:3]))
