"""Sequential per-pose v2 distance TPS; unchanged objective/Jacobian/optimizer."""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image
from tps_distance_inputs import fit
from input_masks import V,O,R
sys.path.insert(0,str(O))
from solve import read_data
from ponytail_observations import similarity

def fit_pose(name,max_nfev=120):
    dest=V/'inputs'/name;meta=json.loads((dest/'manifest.json').read_text());_,_,data=read_data(R)
    front=np.array(data['points']);registered=np.array(meta['registeredLandmarks'])
    linear,shift=similarity(front[[data['anchors']['left'],data['anchors']['right']]],registered[:2])
    source=np.array(Image.open(O/'original/front-skin.png'))[:,:,3]>127
    target=np.array(Image.open(dest/'reference-mask.png'))>127;band=np.array(Image.open(dest/'hairband.png'))>127
    boxes=[e['box'] for e in meta['ears'].values() if e['visible']]
    # Same v2 broad ear half-weight region, reflected for the near side.
    ey=float(registered[:2,1].mean());regions=[]
    if name in ['right','up-right','down-right','up','down']:regions.append(['left',float(registered[0,0]-10),ey])
    if name in ['left','up-left','down-left','up','down']:regions.append(['right',float(registered[1,0]+10),ey])
    brow=int(np.ceil(max(front[data['anchors'][k],1] for k in ['browLeft','browRight'])))
    bottom=int(np.ceil(np.where(source)[0].max()+40))
    model,report=fit(source,target,brow,bottom,linear,shift,regions,boxes,band,max_nfev=max_nfev)
    report.update(sourceBrowY=brow,sourceBottomY=bottom,referenceBrowY=float(registered[2:4,1].max()),referenceChinY=float(registered[6,1]),pose=name)
    output={'controls':model.controls.tolist(),'coefficients':model.coefficients.tolist(),'fit':report,'referenceLandmarks':registered.tolist()}
    (V/'tps').mkdir(exist_ok=True);(V/'tps'/f'{name}.json').write_text(json.dumps(output,indent=2)+'\n')
    (dest/'tps.json').symlink_to('../../tps/'+name+'.json')
    print('C1 TPS',name,json.dumps(report),flush=True)
    return report
