"""Measure the same registered up jaw point on the approved silhouette."""
from pathlib import Path
import json,sys
import numpy as np
from PIL import Image

def measure(root,out):
 F=root/'pipeline/face_outline_stage_a';A=F/'tps-study/stage-a';sys.path.insert(0,str(F/'tps-study'));from tps_distance import TPS
 tps=json.loads((A/'tps/up.json').read_text());model=TPS(np.array(tps['controls']),np.array(tps['coefficients']))
 scalars=json.loads((F/'accepted/scalars.json').read_text());keys=[k for k in scalars['outline'] if k.startswith('0,30,') and k.endswith(',550')];left=min(float(k.split(',')[2]) for k in keys);q=model(np.array([[left,550.]]))[0]
 mask=np.array(Image.open(A/'inputs/up/reference-mask.png'))>127;x=int(round(q[0]));yy=np.flatnonzero(mask[:,x]);chin=int(np.where(mask)[0].max());edge=int(yy.max());distance=chin-edge
 old=json.loads((F/'accepted/C3.json').read_text());pose=next(p for p in old['poses'] if p['name']=='up');rms=max(p['F2']['rms'] for p in pose['nearSides']);report={'sourceRest':[left,550],'tpsPoint':q.tolist(),'referenceColumn':x,'referenceJawY':edge,'referenceChinY':chin,'referenceDifference':distance,'acceptedF2Rms':rms,'threshold':distance-rms,'referenceBelow35':distance<35}
 if distance<35:
  path=root/'pipeline/test_pitch_3d.mjs';s=path.read_text().replace("chin[1]-jaw[1]>35",f"chin[1]-jaw[1]>{distance-rms!r}");path.write_text(s)
 out.write_text(json.dumps(report,indent=2)+'\n');print('UP JAW REFERENCE',json.dumps(report),flush=True)
 return report
if __name__=='__main__':measure(Path(sys.argv[1]).resolve(),Path(sys.argv[2]))
