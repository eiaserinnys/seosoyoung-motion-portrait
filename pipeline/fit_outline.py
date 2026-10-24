"""Sample actual SAM 3 face silhouettes; compile independent left/right targets."""
import json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
ROOT=Path(__file__).resolve().parents[1]
# With no separate chin underside, compromise relative to the rest 61.5px span.
CHIN_CANVAS_SPAN={30:64.0,-30:50.0}
def interval(mask,y):
    xs=np.flatnonzero(mask[y]>127)
    if not len(xs):raise ValueError(f'Empty silhouette row {y}')
    return [int(xs[0]),int(xs[-1])]
def main_component(mask):
    n,labels=cv2.connectedComponents((mask>127).astype(np.uint8));counts=np.bincount(labels.ravel());counts[0]=0
    return np.where(labels==counts.argmax(),255,0)
def symmetrized_outline_rows(rows):
    result=[row[:] for row in rows]
    for row in result:
        _,l,r,ll,lr,rl,rr,_=row
        near=((l-ll)+(rr-r))/2;far=((r-lr)+(rl-l))/2
        row[3:7]=[l-near,r-far,l+far,r+near]
    return result
def symmetrize(rows,poses):
    rows[:]=symmetrized_outline_rows(rows)
    # The approved yaw row is shared by neutral poses and near-cheek bounds.
    for p in poses:
        if p['pitch']:continue
        i=1 if p['yaw']==0 else 3 if p['yaw']<0 else 5
        p['rows']=[[r[0],r[0] if not p['yaw'] else r[7],r[i],r[0] if not p['yaw'] else r[7],r[i+1]] for r in rows]
    for pitch in [-30,30]:
        group={p['yaw']:p for p in poses if p['pitch']==pitch}
        for p in group.values():p['measuredRows']=[r[:] for r in p['rows']]
        center=group[0]
        for row,base in zip(center['rows'],rows):
            _,l,r,*_=base;sy,yl,xl,yr,xr=row
            # Prevent a quartic endpoint crossing; retain the 11px rest chin.
            d=max(((l-xl)+(xr-r))/2,5.5-(r-l)/2)
            row[:]=[sy,(yl+yr)/2,l-d,(yl+yr)/2,r+d]
        neg,pos=group[-45],group[45]
        for left,right,base in zip(neg['rows'],pos['rows'],rows):
            sy,l,r,*_=base
            far=((l-left[2])+(right[4]-r))/2
            # SAM cuts the near cheek at side-hair. Preserve the approved r11
            # attachment width through the upper jaw, then mirror the less cut
            # observation so an occluded side cannot shrink its counterpart.
            if sy<=570:
                left[4]=max(left[4],base[4])
                right[2]=min(right[2],base[5])
            near=min(r-left[4],right[2]-l)
            yl=(left[1]+right[3])/2;yr=(left[3]+right[1])/2
            left[:]=[sy,yl,l-far,yr,r-near]
            right[:]=[sy,yr,l+near,yl,r+far]
    for p in poses:
        p['yTop']=(p['rows'][0][1]+p['rows'][0][3])/2
        p['yChin']=(p['rows'][-1][1]+p['rows'][-1][3])/2
def build():
    front=np.array(Image.open(ROOT/'parts/face-skin.png'))[:,:,3]
    masks=[np.array(Image.open(ROOT/'sam3_masks/references'/f'{side}-face.png')) for side in ['left','right']]
    masks=[main_component(m) for m in masks]
    rows=[]
    for y in [410,420,430,440,450,465,480,490,510,530,550,570,590,600,610,616]:
        f=[x+306 for x in interval(front,y-218)]
        ty=round(420+(y-420)*195/196) if y>=420 else y
        targets=[[x+152 for x in interval(mask,ty-20)] for mask in masks]
        rows.append([y,*f,*targets[0],*targets[1],ty])
    raw=[row[:] for row in rows]
    fit_rows=np.array([row for row in rows if row[0]>=465],float)
    errors=[]
    for column in [3,4,5,6]:
        coeff=np.polyfit((fit_rows[:,0]-540)/100,fit_rows[:,column],4)
        fitted=np.polyval(coeff,(fit_rows[:,0]-540)/100)
        errors.extend((fitted-fit_rows[:,column]).tolist())
        for row in rows:
            if row[0]>=465:row[column]=float(np.polyval(coeff,(row[0]-540)/100))
    data=dict(rawRows=raw,fitRmsPixels=float(np.sqrt(np.mean(np.square(errors)))),fitMaxPixels=float(np.max(np.abs(errors))),method='SAM 3 alpha >127; largest connected face component, outer row extent; independent quartic least-squares contours below y465; hair-occluded upper face uses boundary residual at y465; chin y registration 616 to 615',
              columns=['front_y','front_left','front_right','left45_left','left45_right','right45_left','right45_right','target_y'],rows=rows)
    features=json.loads((ROOT/'source/viewer/face-feature-fit.json').read_text())
    poses=[]
    samples=rows
    ss=np.array([(r[0]-410)/(616-410) for r in samples])
    fractions=json.loads(subprocess.check_output(['node',str(ROOT/'pipeline/outline_vertical_registration.mjs')],input=json.dumps(rows),text=True))
    for pose in features['poses']:
        name=pose['name'];yaw=pose['yaw'];pitch=pose['pitch']
        if not pitch:
            index=1 if yaw==0 else 3 if yaw<0 else 5
            target=[[r[0],r[0] if not yaw else r[7],r[index],r[0] if not yaw else r[7],r[index+1]] for r in samples]
            poses.append(dict(name=name,yaw=yaw,pitch=pitch,rows=target,fitRmsPixels=0,source='approved r11'))
            continue
        mask=main_component(np.array(Image.open(ROOT/'sam3_masks/references'/f'{name}-face.png')))
        chin=int(np.where(mask>127)[0].max())+20
        raw=pose['rawCenters'];canvas=pose['centers']
        s3d=fractions[str(pitch)]
        # Upward eye-edge projection enters hair-occluded temple rows. Start
        # about 15px below the eye (rest Y~442) rather than sample that cut.
        s_floor=0.06 if pitch>0 else 0.0
        mouth=canvas['mouth'][1]+(-12.0 if pitch>0 else 6.0)
        chin_canvas=mouth+CHIN_CANVAS_SPAN[pitch]
        raw_y={side:[raw[side][1],chin] for side in ['left','right']}
        canvas_y={side:[canvas[side][1],chin_canvas] for side in ['left','right']}
        raw_xy=np.array([[interval(mask,round(raw_y[side][0]+max(s_floor,s)*(chin-raw_y[side][0]))-20)[i]+152+pose['registration']['delta'][0] for i,side in enumerate(['left','right'])] for s in s3d],float)
        coeff=[np.polyfit(ss,raw_xy[:,i],4) for i in [0,1]]
        fitted=np.stack([np.polyval(c,ss) for c in coeff],axis=1)
        target=[]
        for r,s,fraction in zip(samples,ss,s3d):
            yl,yr=[canvas_y[side][0]+fraction*(chin_canvas-canvas_y[side][0]) for side in ['left','right']]
            target.append([r[0],yl,float(np.polyval(coeff[0],s)),yr,float(np.polyval(coeff[1],s))])
        poses.append(dict(name=name,yaw=yaw,pitch=pitch,rawYChin=chin,registration=pose['registration'],rows=target,
                          mapping=dict(restY=[431.5,616.0],s3d=s3d,sFloor=s_floor,clampedRows=[r[0] for r,s in zip(samples,s3d) if s<s_floor],rawYSpan=raw_y,canvasYSpan=canvas_y),
                          fitRmsPixels=float(np.sqrt(np.mean((fitted-raw_xy)**2))),coefficients=[c.tolist() for c in coeff],source=f'{name}-face.png'))
    data['measuredRows']=[r[:] for r in rows]
    symmetrize(rows,poses)
    data.update(poses=poses,symmetry='far cheek mirror mean; near cheek r11 ear attachment bound and less-occluded mirror envelope; mirrored independent edge heights',registration='shared feature translation; yaw-zero 3D eye-to-chin vertical fractions shared across yaw; hair-occlusion sampling floor; quartic X fit over rest y410..616')
    (ROOT/'source/viewer/outline-traces.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'bundle/outline-data.mjs').write_text('// Generated by pipeline/fit_outline.py from SAM 3 masks.\nexport const OUTLINE_ROWS='+json.dumps(rows)+';\nexport const OUTLINE_POSES='+json.dumps(poses)+';\n')
    print('PASS sampled',len(rows),'rows from independent face masks')
if __name__=='__main__':build()
