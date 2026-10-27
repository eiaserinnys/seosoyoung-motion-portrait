"""Hidden-row du restoration and 7.11 C2 criteria; no further depth solve."""
import json,sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from config import V,A,O,T,near_sides
import stage_a
from active import restore
from earmath import inner_line
sys.path.insert(0,str(O));from tps_distance import TPS

def main():
 inputs=json.loads((V/'a-inputs.json').read_text());face=json.loads((V/'face-observed.json').read_text());before=json.loads((V/'runtime-current.json').read_text())['prior'];bases=json.loads((V/'ear-bases.json').read_text())['poses'];fits=json.loads((V/'ear-fit.json').read_text());goals=json.loads((V/'goals.json').read_text());invent=json.loads((V/'pin-inventory.json').read_text());p=np.array(inputs['points']);tri=np.array(inputs['triangles']);z=np.array(face['Z']);z_before=z.copy();_,systems=stage_a.prepare(inputs)
 poseReports=[];positions=[];activeReports=[];hasSeam=False;hasOther=False;allFarBelow=True
 for pose in face['poses']:
  name=pose['name']
  if name=='front':poseReports.append({'name':name,**pose['stats'],'belowQuarter':0,'seamReversed':0,'farBelow410Reversed':0});continue
  system=next(s for s in systems if s['pose']['name']==name);tface=np.array(pose['t']);t,du0=stage_a.recover(system,z);assert np.max(abs(t-tface))<1e-10 and np.max(abs(du0-pose['du']))<1e-9
  bnd=next(g['boundaryIndices'] for g in goals if g['name']==name);lines=[];corners=[]
  for side in near_sides(pose['yaw']):
   fit=next(e for e in fits if e['pose']==name and e['side']==side);pars=fit['parameters'];theta=[*pars['t'],pars['sx'],pars['s'],pars['theta']];alpha=np.array(Image.open(A/'ear-fit'/f'front-{side}-ear-alpha.png'));line,yc,corner=inner_line(alpha,bases[name]['sides'][side],theta,side);lines.append((side,line,yc));corners.append({'side':side,'yc':yc,'corner':corner})
  du,active=restore(system,z,tface,du0,p,bnd,lines,3);pose['du']=du.tolist();activeReports.append({'name':name,**active,'corners':corners})
  actual=system['a']+z[:,None]*system['b']+tface+du;det=stage_a.determinants(p,actual,tri);w=np.array(system['pose']['weights']);mask=w>0;disp=system['d'];explained=1-float(np.sum(w[mask,None]*du[mask]**2)/np.sum(w[mask,None]*disp[mask]**2));err=actual-np.array(system['pose']['obs']);stats={'rms':float(np.sqrt(np.mean(np.sum(err[mask]**2,axis=1)))),'observationRms':float(np.sqrt(np.mean(np.sum(err[mask]**2,axis=1)))),'explained3D':explained,'duRms':float(np.sqrt(np.mean(np.sum(du*du,axis=1)))),'duMax':float(np.linalg.norm(du,axis=1).max()),'minDetJ':float(det.min()),'reversedTriangles':int((det<=0).sum())};pose['stats']=stats
  saved=json.loads((A/'tps'/f'{name}.json').read_text());model=TPS(saved['controls'],saved['coefficients']);mean=p[tri].mean(1);target=model(mean);rounded=np.rint(target).astype(int);band=np.array(Image.open(A/'inputs'/name/'hairband.png'))>127;valid=(rounded[:,0]>=0)&(rounded[:,0]<1024)&(rounded[:,1]>=0)&(rounded[:,1]<1024);covered=np.zeros(len(tri),bool);covered[valid]=band[rounded[valid,1],rounded[valid,0]]
  seam=(mean[:,1]<410)
  far=(mean[:,0]<502 if pose['yaw']<0 else mean[:,0]>502) if pose['yaw'] else np.zeros(len(tri),bool)
  farbelow=far&(mean[:,1]>=410);rev=det<=0;seamcount=int((rev&seam).sum());farcount=int((rev&farbelow).sum());hasSeam|=seamcount>0
  if rev.any() and not np.all(farbelow[rev]):allFarBelow=False
  hasOther|=bool((rev&~seam&~farbelow).any())
  cv=next(x for x in face['crossValidation'] if x['heldOut']==name);poseReports.append({'name':name,**stats,'belowQuarter':int((det<.25).sum()),'seamReversed':seamcount,'farBelow410Reversed':farcount,'otherReversed':int((rev&~seam&~farbelow).sum()),'CVsilhouetteRms':cv['silhouetteRms'],'CVfeatureRms':cv['featureRms']})
  positions.append({'name':name,'rowsAndSidesDefinition':'correction1 seam = rest triangle mean y<410; band recorded separately','triangles':[{'id':int(i),'vertices':tri[i].tolist(),'det':float(det[i]),'restMean':mean[i].tolist(),'TPSMean':target[i].tolist(),'worldMean':actual[tri[i]].mean(0).tolist(),'du':du[tri[i]].tolist(),'restYBelow410':bool(mean[i,1]<410),'TPSInHairband':bool(covered[i]),'farSide':bool(far[i]),'seam':bool(seam[i]),'farBelow410':bool(farbelow[i])} for i in np.flatnonzero(det<.25)]})
 assert np.array_equal(z,z_before)
 face['family45']=stage_a.solve_family45(inputs,z);(V/'face-active.json').write_text(json.dumps(face,separators=(',',':'))+'\n');(T/'bundle/face-solve-data.mjs').write_text('export const FACE_SOLVE='+json.dumps(face,separators=(',',':'))+';\n')
 earfails=[e['pose']+'-'+e['side'] for e in fits if e['earFailed']];metricfails=[r['name'] for r in poseReports if r['name']!='front' and (r['explained3D']<.85 or r['CVsilhouetteRms']>35)];reversedTotal=sum(r['reversedTriangles'] for r in poseReports);geometryPass=reversedTotal==0;renderAllowed=geometryPass and not metricfails
 dz=z-np.array(before['Z']);Zreport={'min':float(dz.min()),'max':float(dz.max()),'RMS':float(np.sqrt(np.mean(dz*dz))),'yAtLeast410RMS':float(np.sqrt(np.mean(dz[p[:,1]>=410]**2)))}
 report={'poses':poseReports,'pinInventory':invent,'activeSets':activeReports,'earFailures':earfails,'earF1bUnjudgedPoses':sorted(set(e['pose'] for e in fits if e['earFailed'])),'C2StrictPass':renderAllowed and not earfails,'renderAllowedWithEarExceptions':renderAllowed,'metricFailures':metricfails,'seamDesignFailure':hasSeam,'onlyFarBelow410Reversals':bool(reversedTotal and allFarBelow),'otherReversedPresent':hasOther,'reversedTotal':reversedTotal,'ZUnchangedDuringActive':True,'ZDelta':Zreport,'faceTranslationFrozenDuringActive':True,'lambda':stage_a.LAMBDA,'previous710ReversedTotal':159,'TPSGatesWithdrawn':True}
 (V/'C2.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');(V/'C2-locations.json').write_text(json.dumps(positions,indent=2)+'\n')
 # Same offline Z map as previous, with no new runtime render.
 im=Image.new('RGB',(1050,850),(30,36,48));d=ImageDraw.Draw(im);font=ImageFont.load_default(size=18);bound=max(abs(dz.min()),abs(dz.max()),1.)
 for q,v in zip(p,dz):
  x=130+(q[0]-250)*1.6;y=75+(q[1]-180)*1.3;t=min(1,abs(v)/bound);c=tuple(int((1-t)*220+t*u) for u in ([255,70,80] if v>=0 else [70,135,255]));d.ellipse((x-3,y-3,x+3,y+3),fill=c)
 d.line((100,75+(410-180)*1.3,950,75+(410-180)*1.3),fill='white');d.text((100,5),'7.11 depth change: red positive / blue negative; rest y=410 line',font=font,fill='white');d.text((100,35),f"min {dz.min():.3f} max {dz.max():.3f} RMS {Zreport['RMS']:.3f}",font=font,fill='white');(V/'sheets').mkdir(exist_ok=True);im.save(V/'sheets/Z-change.png')
 print('711 C2',json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
