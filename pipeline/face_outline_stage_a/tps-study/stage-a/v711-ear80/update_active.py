import json,copy
import numpy as np
from PIL import Image
from config import V,A,T,near_sides
import stage_a
from active import restore
from earmath import inner_line
W=A/'v711-filled'
inputs=json.loads((V/'a-inputs.json').read_text());face=json.loads((W/'face-active.json').read_text());old=copy.deepcopy(face);fits=json.loads((V/'ear-fit.json').read_text());bases=json.loads((V/'ear-bases.json').read_text())['poses'];goals=json.loads((W/'goals.json').read_text());c2=json.loads((W/'C2.json').read_text());_,systems=stage_a.prepare(inputs);p=np.array(inputs['points']);z=np.array(face['Z']);tri=np.array(inputs['triangles']);changed=set(json.loads((V/'changed-poses.json').read_text()));reports=[]
for pose in face['poses']:
 n=pose['name']
 if n not in changed:continue
 system=next(s for s in systems if s['pose']['name']==n);tface=np.array(pose['t']);t,du0=stage_a.recover(system,z);assert np.max(abs(t-tface))<1e-9
 lines=[];corners=[]
 for side in near_sides(pose['yaw']):
  fit=next(e for e in fits if e['pose']==n and e['side']==side);q=fit['parameters'];theta=[*q['t'],q['sx'],q['s'],q['theta']];alpha=np.array(Image.open(A/'ear-fit'/f'front-{side}-ear-alpha.png'));line,yc,corner=inner_line(alpha,bases[n]['sides'][side],theta,side);lines.append((side,line,yc));corners.append(dict(side=side,yc=yc,corner=corner))
 bnd=next(g['boundaryIndices'] for g in goals if g['name']==n);du,r=restore(system,z,tface,du0,p,bnd,lines,3);pose['du']=du.tolist();world=system['a']+z[:,None]*system['b']+tface+du;det=stage_a.determinants(p,world,tri);pose['stats'].update(duRms=float(np.sqrt(np.mean(np.sum(du*du,axis=1)))),duMax=float(np.linalg.norm(du,axis=1).max()),minDetJ=float(det.min()),reversedTriangles=int((det<=0).sum()));reports.append(dict(name=n,**r,corners=corners))
print('ACTIVE UPDATE',[(r['name'],r['hiddenPinCount']) for r in reports],flush=True)
for pose in face['poses']:
 oldreport=next(r for r in c2['poses'] if r['name']==pose['name']);oldreport.update(pose['stats'])
c2['reversedTotal']=sum(r['reversedTriangles'] for r in c2['poses'])
c2['C2StrictPass']=False;c2['renderAllowedWithEarExceptions']=False;c2['renderAuthorizedRegardlessOfMetrics']=True
c2['activeSets']=[next((r for r in reports if r['name']==q['name']),q) for q in c2['activeSets']];c2['earFailures']=[q['pose']+'-'+q['side'] for q in fits if q['finalRenderer']['wholeScreenIoU']<.7];c2['earF1bUnjudgedPoses']=sorted(set(q['pose'] for q in fits if q['finalRenderer']['wholeScreenIoU']<.7));c2['diagnosticOnly']=True
assert face['Z']==old['Z'] and all(a['t']==b['t'] for a,b in zip(face['poses'],old['poses']))
assert all(a==b for a,b in zip(face['poses'],old['poses']) if a['name'] not in changed)
(V/'face-active.json').write_text(json.dumps(face,separators=(',',':'))+'\n');(T/'bundle/face-solve-data.mjs').write_text('export const FACE_SOLVE='+json.dumps(face,separators=(',',':'))+';\n');(V/'C2.json').write_text(json.dumps(c2,indent=2)+'\n');(V/'active-audit.json').write_text(json.dumps(dict(ZExact=True,tExact=True,unchangedPosesExact=True,changedPoses=sorted(changed),activeSets=reports),indent=2)+'\n')
