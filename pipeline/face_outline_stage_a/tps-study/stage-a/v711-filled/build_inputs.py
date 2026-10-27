import json,sys
import numpy as np
from PIL import Image
from config import V,A,O,T,near_sides
from policy import observed_targets,reference_geometry
from hair_mass import make_hair_mass
import hashlib
sys.path.insert(0,str(O));from solve import outer_loop
from tps_distance import TPS

def main():
 inputs=json.loads((V/'base-inputs.json').read_text());run=json.loads((V/'runtime-current.json').read_text());old=run['prior'];p=np.array(inputs['points']);meshCount=inputs['meshCount']
 mesh=next(m for m in json.loads((T/'bundle/assets/seosoyoung-rig.json').read_text())['meshes'] if m['id']=='face-skin')
 boundary=[i for i in outer_loop(mesh) if p[i,1]>=348 and i not in inputs['anchors'].values()];rim=[s[side] for s in inputs['samples'] if s['y']>=348 for side in ['left','right']];candidates=sorted(set(boundary+rim));assert (len(boundary),len(rim),len(candidates))==(97,36,133)
 cache={};massReports=[];(V/'hair-mass').mkdir()
 for pose in inputs['poses']:
  n=pose['name']
  if n=='front':continue
  meta=json.loads((A/'inputs'/n/'manifest.json').read_text());saved=json.loads((A/'tps'/f'{n}.json').read_text());model=TPS(saved['controls'],saved['coefficients']);band,hm=make_hair_mass(np.array(Image.open(A/'inputs'/n/'reference-frame.png').convert('RGB')));Image.fromarray(np.uint8(band)*255).save(V/'hair-mass'/f'{n}.png');hm.update(name=n,existingBandSHA256=hashlib.sha256((A/'inputs'/n/'hairband.png').read_bytes()).hexdigest(),condition2Only=True);massReports.append(hm);silhouette=np.array(Image.open(A/'inputs'/n/'reference-mask.png'))>127;geo=reference_geometry(silhouette);target=model(p[candidates]);boxes=[meta['ears'][s]['box'] for s in near_sides(pose['yaw'])];cache[n]=(meta,target,boxes,band,geo)
 # Only this authorized comparison selects the global 3px or 6px threshold.
 v7=json.loads((O/'seeds/left-chin-boundary.json').read_text());expected=[i for i,q in zip(v7['indices'],v7['rest']) if q[0]>502 and q[1]>=523];lookup={i:j for j,i in enumerate(candidates)};meta,target,boxes,band,geo=cache['left'];checks=[];threshold=3
 for limit in [3,6]:
  measured,flags=observed_targets(target,band,boxes,geo,limit);missing=[i for i in expected if not measured[lookup[i]]]
  checks.append({'threshold':limit,'expectedCount':len(expected),'retainedCount':len(expected)-len(missing),'retainedIds':[i for i in expected if i not in missing],'missingIds':missing,'expectedSignedDistances':{str(i):float(flags['signedDistance'][lookup[i]]) for i in expected},'missingReasons':{str(i):{'sd':float(flags['signedDistance'][lookup[i]]),'nearestBoundary':flags['nearestBoundaryPixel'][lookup[i]].tolist(),'nearestInBand':bool(flags['hairband'][lookup[i]]),'nearEarBox':bool(flags['nearEarBox'][lookup[i]])} for i in missing}})
  if not missing:threshold=limit;break
  threshold=6
 selection={'selectedThreshold':threshold,'selectionBasis':'left v7 near-side boundary restY>=523; try3, if any missing use6','checks':checks,'allRetainedAtFinalThreshold':not checks[-1]['missingIds']}
 (V/'threshold-selection.json').write_text(json.dumps(selection,indent=2)+'\n');print('CORRECTION2 LEFT CHIN CHECK',json.dumps(selection),flush=True)
 inventories=[];goals=[];original_goals=json.loads((A/'v711/goals.json').read_text())
 for pose in inputs['poses']:
  name=pose['name'];current=np.array(run['current'][name]);obs=current.copy();w=np.zeros(len(p));kind=np.array(['free-interior']*len(p),object)
  if name=='front':pose.update(obs=obs.tolist(),weights=w.tolist(),pins=[]);inventories.append({'name':name,'fixedIdentity':True});continue
  meta,target,boxes,band,geo=cache[name];measured,flags=observed_targets(target,band,boxes,geo,threshold)
  previous=next(g for g in original_goals if g['name']==name)
  assert previous['candidateIndices']==candidates
  if name=='left':assert np.allclose(previous['TPSGoals'],target,atol=1e-9,rtol=0)
  flags['originalObserved']=np.array(previous['observed'],bool)
  upper=np.flatnonzero((np.arange(len(p))<meshCount)&(p[:,1]<348));w[upper]=4;kind[upper]='upper-mesh-current';origw=np.array(pose['weights'])
  for sample in inputs['samples']:
   c=sample['center'];w[c]=origw[c];kind[c]='sample-center-current'
   for side in ['left','right']:
    i=sample[side]
    if p[i,1]<348:w[i]=4;kind[i]='upper-rim-current'
  obs[candidates]=target
  for j,i in enumerate(candidates):w[i]=8 if measured[j] else 0;kind[i]=('observed-' if measured[j] else 'free-')+('boundary' if i in boundary else 'rim')
  for k,i in inputs['anchors'].items():obs[i]=meta['anchorCenters'][k];w[i]=8;kind[i]='anchor-observed'
  pins=np.flatnonzero(w>0).tolist();pose.update(obs=obs.tolist(),weights=w.tolist(),pins=pins);assert np.array_equal(w[candidates]>0,measured)
  released=[i for j,i in enumerate(candidates) if flags['originalObserved'][j] and not measured[j]]
  inventories.append({'name':name,'pinnedCount':len(pins),'freeCount':len(p)-len(pins),'candidateCount':133,'observedCandidates':int(measured.sum()),'freeCandidates':int((~measured).sum()),'releasedOriginalPinIds':released,'releasedOriginalPinCount':len(released),'vertex120Released':120 in released,'signedDistanceThreshold':threshold,'nearestBoundaryInBandCount':int(flags['hairband'].sum()),'farFromBoundaryCount':int((~flags['nearBoundary']).sum()),'nearEarBoxCount':int(flags['nearEarBox'].sum()),'nearBoxes':boxes,'candidatePolicyVerified':True})
  goals.append({'name':name,'boundaryIndices':boundary,'lowerRimIndices':rim,'candidateIndices':candidates,'rest':p[candidates].tolist(),'TPSGoals':target.tolist(),'observed':measured.tolist(),'signedDistance':flags['signedDistance'].tolist(),'nearestBoundaryPixel':flags['nearestBoundaryPixel'].tolist(),'nearestBoundaryInBand':flags['hairband'].tolist(),'nearEarBox':flags['nearEarBox'].tolist(),'originalObserved':flags['originalObserved'].tolist()})
 (V/'hair-mass-audit.json').write_text(json.dumps(massReports,indent=2)+'\n')
 inputs['prior']=old['Z'];(V/'a-inputs.json').write_text(json.dumps(inputs,separators=(',',':'))+'\n');(V/'pin-inventory.json').write_text(json.dumps(inventories,indent=2)+'\n');(V/'goals.json').write_text(json.dumps(goals,indent=2)+'\n');print('CORRECTION2 PIN INVENTORY',json.dumps(inventories),flush=True)
if __name__=='__main__':main()
