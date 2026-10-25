"""Sample actual SAM 3 face silhouettes; compile independent left/right targets."""
import json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
ROOT=Path(__file__).resolve().parents[1]
CHIN_ROWS={600,610,616}
def interval(mask,y):
    xs=np.flatnonzero(mask[y]>127)
    if not len(xs):raise ValueError(f'Empty silhouette row {y}')
    return [int(xs[0]),int(xs[-1])]
def main_component(mask):
    n,labels=cv2.connectedComponents((mask>127).astype(np.uint8));counts=np.bincount(labels.ravel());counts[0]=0
    return np.where(labels==counts.argmax(),255,0)
def classify_boundary(img,y,x_edge,dx):
    h,w,_=img.shape
    vals=[img[y,max(0,min(w-1,x_edge+step))] for step in (dx,dx+(1 if dx>0 else -1))]
    px=np.median(vals,axis=0)
    if px.max()<40:return 'BG'
    if px[0]>170 and px[1]>120 and px[0]-px[2]>20 and px[0]-px[1]<80:return 'SKIN'
    return 'HAIR'
def boundary_sample(img,mask,y,side,reg):
    index=0 if side=='left' else 1;dx=-4 if index==0 else 4
    x=interval(mask,y)[index];kind=classify_boundary(img,y,x,dx)
    adjacent=[]
    for yy in [y-1,y+1]:
        if 0<=yy<mask.shape[0] and np.any(mask[yy]>127):
            adjacent.append(classify_boundary(img,yy,interval(mask,yy)[index],dx))
    return dict(referenceY=y,xMask=x,xCanvas=reg['eyeMid'][0]+(152+x-reg['referenceEyeMid'][0])*reg['scale'],
                **{'class':kind},neckBounded=kind=='SKIN',validBG=kind=='BG' and 'BG' in adjacent)
def qualify_background(samples):
    # A sampling floor can repeat one temple-gap row several times. Repeated
    # copies are one observation, not a continuous silhouette segment.
    for side in ['left','right']:
        i=0
        while i<len(samples):
            if samples[i][side]['class']!='BG':i+=1;continue
            j=i
            while j+1<len(samples) and samples[j+1][side]['class']=='BG':j+=1
            if samples[j]['restY']<480 and len({s[side]['referenceY'] for s in samples[i:j+1]})<2:
                for s in samples[i:j+1]:s[side].update(validBG=False,excluded='upper isolated background gap')
            i=j+1
def sphere_ease(t):return (1-np.cos(np.clip(t,0,1)*np.pi))*.5
def blend_profile(prior,measured,valid,offset):
    prior=np.asarray(prior,float)+offset;measured=np.asarray(measured,float)
    target=prior.copy();weight=np.zeros(len(prior));i=0
    while i<len(valid):
        if not valid[i]:i+=1;continue
        end=i
        while end+1<len(valid) and valid[end+1]:end+=1
        for k in range(max(0,i-2),min(len(valid),end+3)):
            w=min(float(sphere_ease((k-(i-2))/4)) if i else 1.,float(sphere_ease(((end+2)-k)/4)) if end<len(valid)-1 else 1.)
            if w>weight[k]:
                j=min(end,max(i,k));target[k]=prior[k]+w*(measured[j]-prior[j]);weight[k]=w
        i=end+1
    return target,weight
def monotone_profile(rows):
    values=np.asarray(rows,float);mid=(values[:,2]+values[:,4])/2
    half=np.maximum(5.5,(values[:,4]-values[:,2])/2)
    candidates=np.flatnonzero((values[:,0]>=440)&(values[:,0]<=490))
    peak=int(candidates[np.argmax(half[candidates])])
    half[peak:]=np.minimum.accumulate(half[peak:])
    values[:,2]=mid-half;values[:,4]=mid+half
    return values.tolist()
def correct_up_diagonal(p):
    if p['pitch']<=0 or not p['yaw']:return
    base_rows=np.asarray(p['rows'],float).copy()
    for row,sample in zip(p['rows'],p['boundarySamples']):
        if row[0]<480 or row[0] in CHIN_ROWS:continue
        ease=float(sphere_ease((row[0]-480)/(616-480)))
        for side,column in [('left',2),('right',4)]:
            if sample[side]['validBG']:continue
            base=row[column];meas=sample[side]['xCanvas']
            if sample[side]['class']=='SKIN':
                # Skin beyond the face mask is neck: it only bounds inward.
                inward=max(base,meas) if side=='left' else min(base,meas)
                row[column]=base+ease*(inward-base)
            elif sample[side]['class']=='HAIR':
                # Hair can trim a mask inward; an outward edge is face evidence.
                outward=min(base,meas) if side=='left' else max(base,meas)
                if abs(outward-base)>3.:row[column]=base+ease*(outward-base)
    # Each side owns its monotonic bound; a shifted midpoint must not pull
    # the opposite jaw inward through another symmetric half-width pass.
    rows=np.asarray(p['rows'],float)
    pl=int(np.argmin(rows[:,2]));pr=int(np.argmax(rows[:,4]))
    rows[pl:,2]=np.maximum.accumulate(rows[pl:,2])
    rows[pr:,4]=np.minimum.accumulate(rows[pr:,4])
    # Fade back to the original prior profile before the fixed chin rows.
    for i,row in enumerate(rows):
        w=float(sphere_ease((row[0]-570)/30))
        rows[i,2]=(1.-w)*rows[i,2]+w*base_rows[i,2]
        rows[i,4]=(1.-w)*rows[i,4]+w*base_rows[i,4]
    p['rows']=rows.tolist()
def symmetrized_outline_rows(rows):
    result=[row[:] for row in rows]
    for row in result:
        _,l,r,ll,lr,rl,rr,_=row
        near=((l-ll)+(rr-r))/2;far=((r-lr)+(rl-l))/2
        row[3:7]=[l-near,r-far,l+far,r+near]
    return result
def symmetrize(rows,poses):
    # The caller supplies the canonical r11 row, symmetrized exactly once.
    for p in poses:
        if p['pitch']:continue
        i=1 if p['yaw']==0 else 3 if p['yaw']<0 else 5
        p['rows']=[[r[0],r[0] if not p['yaw'] else r[7],r[i],r[0] if not p['yaw'] else r[7],r[i+1]] for r in rows]
    for pitch in [-30,30]:
        group={p['yaw']:p for p in poses if p['pitch']==pitch}
        for p in group.values():p['measuredRows']=[r[:] for r in p['rows']]
        center=group[0]
        for row in center['rows']:
            sy,yl,xl,yr,xr=row;half=max(5.5,(xr-xl)/2)
            row[:]=[sy,(yl+yr)/2,502-half,(yl+yr)/2,502+half]
        neg,pos=group[-45],group[45]
        for left,right in zip(neg['rows'],pos['rows']):
            sy=left[0];l=(left[2]+1004-right[4])/2;r=(left[4]+1004-right[2])/2
            yl=(left[1]+right[3])/2;yr=(left[3]+right[1])/2
            left[:]=[sy,yl,l,yr,r]
            right[:]=[sy,yr,1004-r,yl,1004-l]
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
    # This polynomial belongs only to the immutable approved r11 yaw row.
    # The six pitch profiles below never fit a polynomial to occluded masks.
    fit_rows=np.array([row for row in rows if row[0]>=465],float)
    errors=[]
    for column in [3,4,5,6]:
        coeff=np.polyfit((fit_rows[:,0]-540)/100,fit_rows[:,column],4)
        fitted=np.polyval(coeff,(fit_rows[:,0]-540)/100)
        errors.extend((fitted-fit_rows[:,column]).tolist())
        for row in rows:
            if row[0]>=465:row[column]=float(np.polyval(coeff,(row[0]-540)/100))
    data=dict(rawRows=raw,fitRmsPixels=float(np.sqrt(np.mean(np.square(errors)))),fitMaxPixels=float(np.max(np.abs(errors))),method='approved neutral r11 retains its original quartic yaw contours and 616-to-615 chin registration; pitch profiles use classified BG evidence, rotated r11 priors plus side translations, cosine seams and monotone half-widths; no pitch quartic fit',
              columns=['front_y','front_left','front_right','left45_left','left45_right','right45_left','right45_right','target_y'],rows=rows)
    measured_rows=[r[:] for r in rows]
    rows[:]=symmetrized_outline_rows(rows)
    features=json.loads((ROOT/'source/viewer/face-feature-fit.json').read_text())
    poses=[]
    samples=rows
    fractions=json.loads(subprocess.check_output(['node',str(ROOT/'pipeline/outline_vertical_registration.mjs')],input=json.dumps(rows),text=True))
    priors=json.loads(subprocess.check_output(['node',str(ROOT/'pipeline/outline_prior.mjs')],input=json.dumps(rows),text=True))
    for pose in features['poses']:
        name=pose['name'];yaw=pose['yaw'];pitch=pose['pitch']
        if not pitch:
            index=1 if yaw==0 else 3 if yaw<0 else 5
            target=[[r[0],r[0] if not yaw else r[7],r[index],r[0] if not yaw else r[7],r[index+1]] for r in samples]
            poses.append(dict(name=name,yaw=yaw,pitch=pitch,rows=target,fitRmsPixels=0,source='approved r11'))
            continue
        mask=main_component(np.array(Image.open(ROOT/'sam3_masks/references'/f'{name}-face.png')))
        img=np.array(Image.open(ROOT/'source/references'/pose['source']).convert('RGB'))
        chin=int(np.where(mask>127)[0].max())+20
        raw=pose['rawCenters'];canvas=pose['centers']
        vertical=fractions[str(pitch)];s3d=vertical['s3d']
        # Upward eye-edge projection enters hair-occluded temple rows. Start
        # at the derived restY440 fraction rather than sample that cut.
        s_floor=vertical['sFloor'];rim_offset=vertical['rimOffset']
        reg=pose['registration'];scale=reg['scale'];midpoint=reg['referenceEyeMid'];eye_mid=reg['eyeMid']
        chin_canvas=eye_mid[1]+(chin-midpoint[1])*scale
        raw_y={side:[raw[side][1]+rim_offset/scale,chin] for side in ['left','right']}
        canvas_y={side:[canvas[side][1]+rim_offset,chin_canvas] for side in ['left','right']}
        observations=[]
        for row,s in zip(samples,s3d):
            item=dict(restY=row[0])
            for side in ['left','right']:
                y=round(raw_y[side][0]+max(s_floor,s)*(chin-raw_y[side][0]))-20
                item[side]=boundary_sample(img,mask,y,side,reg)
            item['neckBounded']=any(item[side]['neckBounded'] for side in ['left','right'])
            observations.append(item)
        qualify_background(observations)
        prior=priors[name];offsets={}
        for j,side in enumerate(['left','right']):
            valid=[s[side]['xCanvas']-r[j+1] for s,r in zip(observations,prior) if s[side]['validBG'] and s['restY']<600]
            offsets[side]=float(np.mean(valid)) if valid else None
        target=[]
        for i,(r,fraction) in enumerate(zip(samples,s3d)):
            yl,yr=[canvas_y[side][0]+fraction*(chin_canvas-canvas_y[side][0]) for side in ['left','right']]
            target.append([r[0],yl,prior[i][1],yr,prior[i][2]])
        poses.append(dict(name=name,yaw=yaw,pitch=pitch,rawYChin=chin,registration=pose['registration'],rows=target,
                          mapping=dict(restY=[431.5,616.0],s3d=s3d,sFloor=s_floor,rimOffset=rim_offset,clampedRows=[r[0] for r,s in zip(samples,s3d) if s<s_floor],rawYSpan=raw_y,canvasYSpan=canvas_y),
                          priorRows=prior,boundarySamples=observations,measuredOffsets=offsets,source=f'{name}-face.png'))
    for p in poses:
        if not p['pitch']:continue
        mirror=next(q for q in poses if q['pitch']==p['pitch'] and q['yaw']==-p['yaw'])
        p['offsets']={side:(value if value is not None else -mirror['measuredOffsets'][other] if p['yaw'] and mirror['measuredOffsets'][other] is not None else 0.)
                      for side,other,value in [('left','right',p['measuredOffsets']['left']),('right','left',p['measuredOffsets']['right'])]}
        p['blendWeights']={};bg_errors=[]
        for j,side in enumerate(['left','right']):
            observed=p['boundarySamples'];measured=[s[side]['xCanvas'] for s in observed];valid=[s[side]['validBG'] for s in observed]
            target,weight=blend_profile([r[j+1] for r in p['priorRows']],measured,valid,p['offsets'][side])
            p['blendWeights'][side]=weight.tolist()
            for i,row in enumerate(p['rows']):row[2 if j==0 else 4]=float(target[i])
            bg_errors.extend(float(v-measured[i]) for i,v in enumerate(target) if valid[i])
        if not p['yaw'] and not any(s[side]['validBG'] for s in p['boundarySamples'] for side in ['left','right']):
            # A neck-bounded chin may supply an inward bound, never a hair cut.
            for row,sample in zip(p['rows'],p['boundarySamples']):
                if row[0]<480 or not sample['neckBounded']:continue
                mid=(row[2]+row[4])/2;half=(row[4]-row[2])/2
                measured=max(5.5,(sample['right']['xCanvas']-sample['left']['xCanvas'])/2)
                h=half+(min(half,measured)-half)*float(sphere_ease((row[0]-480)/(616-480)))
                row[2]=mid-h;row[4]=mid+h
        p['rows']=monotone_profile(p['rows'])
        correct_up_diagonal(p)
        p['fitRmsPixels']=float(np.sqrt(np.mean(np.square(bg_errors)))) if bg_errors else 0.
    data['measuredRows']=measured_rows
    symmetrize(rows,poses)
    data.update(poses=poses,symmetry='pitched profiles use absolute x502 mirror mean; approved neutral r11 rows unchanged',registration='shared feature eye-midpoint and yaw-zero zoom; model-rim-to-actual-chin vertical fractions; BG-only measurements plus rotated r11 prior and per-side translation; cosine seams; monotone half-width below cheek peak')
    (ROOT/'source/viewer/outline-traces.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'bundle/outline-data.mjs').write_text('// Generated by pipeline/fit_outline.py from SAM 3 masks.\nexport const OUTLINE_ROWS='+json.dumps(rows)+';\nexport const OUTLINE_POSES='+json.dumps(poses)+';\n')
    print('PASS sampled',len(rows),'rows from independent face masks')
if __name__=='__main__':build()
