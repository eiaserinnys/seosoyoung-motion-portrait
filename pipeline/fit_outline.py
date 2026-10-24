"""Sample actual SAM 3 face silhouettes; compile independent left/right targets."""
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
ROOT=Path(__file__).resolve().parents[1]
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
def edge_depth(rows,y):
    if y<=rows[0][0]:row=rows[0]
    elif y>=rows[-1][0]:row=rows[-1]
    else:
        row=None
        for a,b in zip(rows,rows[1:]):
            if y<=b[0]:
                t=(y-a[0])/(b[0]-a[0]);row=[a[i]+(b[i]-a[i])*t for i in range(len(a))];break
    return ((row[5]-row[3])+(row[6]-row[4]))/(4*math.sin(math.radians(45)))
def rotated_span(rows,yaw,pitch):
    edge_delta=edge_depth(rows,616)-edge_depth(rows,465)
    return (616-465)*math.cos(math.radians(pitch))-edge_delta*math.cos(math.radians(yaw))*math.sin(math.radians(pitch))
def symmetrize(rows,poses):
    rows[:]=symmetrized_outline_rows(rows)
    source=[r for r in rows if r[0]>=465]
    for pitch in [-30,30]:
        group={p['yaw']:p for p in poses if p['pitch']==pitch}
        for p in group.values():p['measuredRows']=[r[:] for r in p['rows']]
        center=group[0]
        center['registration']['topEdgeY']=[center['yTop'],center['yTop']]
        for row,base in zip(center['rows'],source):
            _,l,r,*_=base;ty,tl,tr=row;d=((l-tl)+(tr-r))/2
            row[:]=[ty,l-d,r+d]
        neg,pos=group[-45],group[45]
        top=(neg['yTop']+pos['yTop'])/2;chin=(neg['yChin']+pos['yChin'])/2
        a=neg['registration']['topEdgeY'];b=pos['registration']['topEdgeY']
        edge=[(a[0]+b[1])/2,(a[1]+b[0])/2]
        neg['registration']['topEdgeY']=edge;pos['registration']['topEdgeY']=edge[::-1]
        for p in [neg,pos]:p.update(yTop=top,yChin=chin)
        for left,right,base in zip(neg['rows'],pos['rows'],source):
            _,l,r,*_=base;ty=(left[0]+right[0])/2
            near=((l-left[1])+(right[2]-r))/2
            # Both pitched far cheeks can be cut inward by side-hair. Never
            # narrow beyond the clear horizontal reference at this same row.
            far=min(r-left[2],right[1]-l,r-base[4]) if pitch==30 else ((r-left[2])+(right[1]-l))/2
            left[:]=[ty,l-near,r-far];right[:]=[ty,l+far,r+near]
        for p in [neg,pos]:
            p['farFit']='minimum inward displacement of both pitched samples and clear pitch-zero contour' if pitch==30 else 'mirror average'
    # The middle pitch row uses the same balanced targets as OUTLINE_ROWS.
    for p in poses:
        if p['pitch']:continue
        i=1 if p['yaw']==0 else 3 if p['yaw']<0 else 5
        p['rows']=[[r[0] if not p['yaw'] else r[7],r[i],r[i+1]] for r in source]
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
    samples=[r for r in rows if r[0]>=465]
    span_rows=symmetrized_outline_rows(rows)
    ss=np.linspace(0,1,41)
    for pose in features['poses']:
        name=pose['name'];yaw=pose['yaw'];pitch=pose['pitch']
        if not pitch:
            index=1 if yaw==0 else 3 if yaw<0 else 5
            target=[[r[0],r[index],r[index+1]] if not yaw else [r[7],r[index],r[index+1]] for r in samples]
            poses.append(dict(name=name,yaw=yaw,pitch=pitch,yTop=target[0][0],yChin=target[-1][0],rows=target,fitRmsPixels=0,source='approved r11'))
            continue
        mask=main_component(np.array(Image.open(ROOT/'sam3_masks/references'/f'{name}-face.png')))
        chin=int(np.where(mask>127)[0].max())+20
        sample_span=rotated_span(span_rows,yaw,pitch)
        top=round(chin-sample_span)
        raw_xy=np.array([interval(mask,round(top+(chin-top)*s)-20) for s in ss],float)+152
        coeff=[np.polyfit(ss,raw_xy[:,i],4) for i in [0,1]]
        fitted=np.stack([np.polyval(c,ss) for c in coeff],axis=1)
        target_span=(166.0 if yaw==0 else 158.0) if pitch>0 else (116.0 if yaw==0 else 128.0)
        top_y=float(pose['registration']['topY'])
        target=[]
        for r in samples:
            s=(r[0]-465)/(616-465)
            target.append([top_y+target_span*s,*[float(np.polyval(c,s))+pose['registration']['delta'][0] for c in coeff]])
        poses.append(dict(name=name,yaw=yaw,pitch=pitch,yTop=top_y,yChin=float(top_y+target_span),samplingSpan=sample_span,targetSpan=target_span,rawYTop=top,rawYChin=chin,registration=pose['registration'],rows=target,
                          fitRmsPixels=float(np.sqrt(np.mean((fitted-raw_xy)**2))),coefficients=[c.tolist() for c in coeff],source=f'{name}-face.png'))
    data['measuredRows']=[r[:] for r in rows]
    symmetrize(rows,poses)
    data.update(poses=poses,symmetry='rest-relative mirror averaging; up far-cheek inward displacement capped by both pitched samples and clear pitch-zero contour',registration='s=(front_y-465)/(616-465); y_k(s)=yTop_k+s*(yChin_k-yTop_k); independent quartic mask contours')
    (ROOT/'source/viewer/outline-traces.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'bundle/outline-data.mjs').write_text('// Generated by pipeline/fit_outline.py from SAM 3 masks.\nexport const OUTLINE_ROWS='+json.dumps(rows)+';\nexport const OUTLINE_POSES='+json.dumps(poses)+';\n')
    print('PASS sampled',len(rows),'rows from independent face masks')
if __name__=='__main__':build()
