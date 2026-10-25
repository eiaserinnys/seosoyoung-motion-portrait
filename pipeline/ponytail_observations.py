"""Image measurements: pose-bank NCC, forward/backward LK and edge-aware skin contours.

OpenCV (Apache-2.0) supplies matching, LK and GrabCut. No neural model or
third-party landmark topology is required for this illustrated face.
"""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['front','left','right','up','down','up-left','up-right','down-left','down-right']
ANGLES = [(0,0),(-45,0),(45,0),(0,30),(0,-30),(-45,30),(45,30),(-45,-30),(45,-30)]
FEATURES = ['left','right','browLeft','browRight','nose','mouth','chin','earLeft','earRight']
THETA = np.linspace(-np.pi,np.pi,96,endpoint=False)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def match_patch(template, gray, expected, radius):
    h,w=template.shape
    x,y=np.rint(expected).astype(int)
    x=int(np.clip(x,w//2,gray.shape[1]-(w-w//2)))
    y=int(np.clip(y,h//2,gray.shape[0]-(h-h//2)))
    x0=max(0,x-w//2-radius);y0=max(0,y-h//2-radius)
    x1=min(gray.shape[1],x+(w-w//2)+radius);y1=min(gray.shape[0],y+(h-h//2)+radius)
    correlation=cv2.matchTemplate(gray[y0:y1,x0:x1],template,cv2.TM_CCOEFF_NORMED)
    _,score,_,(dx,dy)=cv2.minMaxLoc(correlation)
    return np.array([x0+dx+w//2,y0+dy+h//2],float),float(score)


def iris_centers(rgb, previous=None):
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    # Chroma only proposes regions. Template NCC verifies their identities.
    mask=((hsv[:,:,0]>85)&(hsv[:,:,0]<160)&(hsv[:,:,1]>90)&(hsv[:,:,2]>80)).astype(np.uint8)
    mask[:240]=0;mask[590:]=0;mask[:,:65]=0;mask[:,660:]=0
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    n,labels,stats,centers=cv2.connectedComponentsWithStats(mask)
    candidates=[i for i in range(1,n) if 70<stats[i,4]<5000 and 8<stats[i,2]<90 and 10<stats[i,3]<100]
    if len(candidates)<2:raise ValueError('Two iris candidates required')
    pairs=[]
    for i in candidates:
        for j in candidates:
            if centers[j,0]-centers[i,0]<50:continue
            d=centers[j]-centers[i]
            if np.linalg.norm(d)>260 or abs(d[1])>85:continue
            pts=np.array([centers[i],centers[j]])
            score=stats[i,4]+stats[j,4]
            if previous is not None:score-=10*np.linalg.norm(pts-previous)
            pairs.append((score,pts))
    return max(pairs,key=lambda item:item[0])[1]


def similarity(source,target):
    a=source[1]-source[0];b=target[1]-target[0]
    scale=np.linalg.norm(b)/np.linalg.norm(a)
    c=np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b));s=(a[0]*b[1]-a[1]*b[0])/(np.linalg.norm(a)*np.linalg.norm(b))
    rotation=np.array([[c,-s],[s,c]])*scale
    return rotation,target.mean(0)-source.mean(0)@rotation.T


def patches(gray,points):
    result=[]
    for k,(x,y) in enumerate(points):
        rx,ry=(20,24) if k<2 else (27,9) if k<4 else (10,11) if k==4 else (22,10) if k<7 else (22,27)
        x,y=np.rint([x,y]).astype(int)
        result.append(gray[y-ry:y+ry+1,x-rx:x+rx+1].copy())
    return result


def skin_contour(rgb,points,chin_y=None):
    """GrabCut separates hair/background; ray gradients separate jaw from neck.

    Contours are measurements, not polynomial predictions. The known chin is
    used for reference images only; video chin is measured from the jaw ink.
    """
    h,w=rgb.shape[:2];left,right=points[:2];eye=(left+right)/2
    mouth=points[5];center=(eye+mouth)/2
    distance=np.linalg.norm(right-left)
    # Fit a skin Gaussian from forehead/cheek samples instead of fixed RGB cuts.
    samples=[]
    for x,y in [eye+[0,-60],eye+[0,35],mouth+[-35,-15],mouth+[35,-15]]:
        x,y=np.rint([x,y]).astype(int)
        samples.extend(rgb[max(0,y-5):y+6,max(0,x-5):x+6].reshape(-1,3))
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(float)
    samples=cv2.cvtColor(np.uint8(np.array(samples).reshape(1,-1,3)),cv2.COLOR_RGB2LAB).reshape(-1,3).astype(float)
    med=np.median(samples,axis=0);cov=np.cov(samples.T)+np.diag([120,25,25])
    delta=lab-med
    score=np.einsum('...i,ij,...j->...',delta,np.linalg.inv(cov),delta)
    yy,xx=np.indices((h,w));ellipse=((xx-center[0])/(distance*1.55))**2+((yy-center[1])/(distance*1.9))**2
    mask=np.where((score<22)&(ellipse<1),cv2.GC_PR_FGD,cv2.GC_PR_BGD).astype(np.uint8)
    mask[(ellipse>1.2)|(yy<80)|(yy>min(h-1,mouth[1]+130))]=cv2.GC_BGD
    for x,y in [eye+[0,30],mouth+[-25,-15],mouth+[25,-15]]:
        cv2.circle(mask,tuple(np.rint([x,y]).astype(int)),6,cv2.GC_FGD,-1)
    bg=np.zeros((1,65));fg=np.zeros((1,65))
    # Graph-cut at half resolution; ray/ink localization stays at 720px.
    small=cv2.resize(rgb,(360,360),interpolation=cv2.INTER_AREA)
    labels=cv2.resize(mask,(360,360),interpolation=cv2.INTER_NEAREST)
    cv2.grabCut(small,labels,None,bg,fg,2,cv2.GC_INIT_WITH_MASK)
    skin=cv2.resize(np.isin(labels,[cv2.GC_FGD,cv2.GC_PR_FGD]).astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)
    skin=cv2.morphologyEx(skin,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    # Mouth-to-chin ratio is only the search window; dark jaw ink chooses tip.
    if chin_y is None:
        x=int(round(mouth[0]));lo=int(mouth[1]+20);hi=min(h-5,int(mouth[1]+100))
        ink=cv2.GaussianBlur(gray,(3,3),0)[lo:hi,max(0,x-20):min(w,x+21)]
        row=np.percentile(ink,25,axis=1)
        chin_y=float(lo+np.argmin(row))
    skin[yy>chin_y]=0
    contours,_=cv2.findContours(skin,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    face=max(contours,key=cv2.contourArea)
    skin[:]=0
    # Eyes and brow strokes are holes in the skin classifier. The external
    # skin hull must surround them; they are never silhouette exits.
    cv2.fillConvexPoly(skin,cv2.convexHull(face),1)
    moments=cv2.moments(skin)
    center=np.array([moments['m10'],moments['m01']])/moments['m00']
    radii=[]
    for theta in THETA:
        ray=np.column_stack((center[0]+np.cos(theta)*np.arange(15,380),center[1]+np.sin(theta)*np.arange(15,380)))
        xy=np.rint(ray).astype(int);valid=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
        xy=xy[valid];r=np.arange(15,380)[valid]
        inside=skin[xy[:,1],xy[:,0]]>0
        # The central lower rays hit the neck; cap them by the jaw's observed Y.
        if np.sin(theta)>.3:
            inside &= xy[:,1]<=chin_y
        runs=np.flatnonzero(inside)
        if len(runs):
            # Ignore disconnected clothing after leaving the face.
            gaps=np.flatnonzero(np.diff(runs)>8)
            end=runs[gaps[0]] if len(gaps) else runs[-1]
            radius=float(r[end])
        else:radius=15.
        radii.append(radius)
    radii=np.array(radii)
    # Periodic median rejects eyelashes/individual antialias pixels.
    radii=np.median(np.array([np.roll(radii,k) for k in [-2,-1,0,1,2]]),axis=0)
    contour=center+np.column_stack((np.cos(THETA),np.sin(THETA)))*radii[:,None]
    # Ear arcs are the visible skin contour outside the central facial oval.
    ears=[]
    for sign in [-1,1]:
        selection=(contour[:,0]<left[0]-10) if sign<0 else (contour[:,0]>right[0]+10)
        selection &= (contour[:,1]>eye[1]-30)&(contour[:,1]<eye[1]+110)
        arc=contour[selection]
        ears.append(arc.tolist())
    chin=np.array([mouth[0],chin_y])
    ear_points=[np.mean(arc,axis=0) if len(arc) else contour[np.argmin(contour[:,0]) if i==0 else np.argmax(contour[:,0])] for i,arc in enumerate(ears)]
    return dict(center=center.tolist(),radii=radii.tolist(),contour=contour.tolist(),earContours=ears,extra=np.array([chin,*ear_points]).tolist())


def seedance_bank(inputs):
    """Use only selected video patches; no independently generated side view."""
    manifest=json.loads((ROOT/'source/viewer/seedance-reference-frames.json').read_text())
    bank=[]
    for row in manifest['frames']:
        path=inputs/'frames'/('frame_'+str(row['frame']).zfill(4)+'.png')
        if digest(path)!=row['sourceSha256']:raise ValueError('Seedance reference frame changed')
        gray=cv2.resize(cv2.imread(str(path),cv2.IMREAD_GRAYSCALE),(720,720),interpolation=cv2.INTER_CUBIC)
        points=np.array(row['landmarks'])
        bank.append(dict(frame=row['frame'],points=points,templates=patches(gray,points)))
    cv2.setNumThreads(1);cv2.setRNGSeed(42)
    fit=json.loads((ROOT/'source/viewer/face-feature-fit.json').read_text())
    points=np.array([fit['front'][k] for k in FEATURES[:6]])-[152,20]
    path=inputs/'front_ponytail_canonical.png'
    rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
    contour=skin_contour(rgb,points,596)
    contour.pop('extra')
    if digest(path)!=manifest['canonicalSha256']:raise ValueError('Canonical front image changed')
    rest=dict(name='front',yaw=0.,pitch=0.,landmarks=np.vstack((points,manifest['canonicalExtra'])).tolist(),sha256=digest(path),**contour)
    return rest,manifest['frames'],bank


def track_video(inputs,bank):
    """Bidirectional LK between inspected seeds, then local NCC verification.

    Each segment restarts at measured image patches in both directions. This
    prevents a full-clip LK drift or a global template switch to another feature.
    """
    cv2.setNumThreads(1);cv2.setRNGSeed(42)
    paths=sorted((inputs/'frames').glob('frame_*.png'))
    if len(paths)!=361:raise ValueError('Expected exactly 361 Seedance frames')
    gray=[cv2.resize(cv2.imread(str(p),cv2.IMREAD_GRAYSCALE),(720,720),interpolation=cv2.INTER_CUBIC) for p in paths]
    seeds={key['frame']-1:key['points'] for key in bank}
    # The clip returns to its frontal starting pose. Locate that endpoint from
    # its actual pixels before using it as a final optical-flow boundary.
    first=bank[0];eyes=iris_centers(cv2.cvtColor(cv2.resize(cv2.imread(str(paths[-1])),(720,720)),cv2.COLOR_BGR2RGB))
    linear,shift=similarity(first['points'][:2],eyes)
    guess=first['points']@linear.T+shift
    seeds[len(paths)-1]=np.array([match_patch(t,gray[-1],p,18)[0] for t,p in zip(first['templates'],guess)])
    anchors=sorted(seeds);tracks={i:[] for i in range(len(paths))}
    def advance(start,end,step):
        points=seeds[start].copy();origin=gray[start];templates=patches(origin,points)
        for i in range(start,end+step,step):
            if i!=start:
                previous=points.copy()
                flow,status,_=cv2.calcOpticalFlowPyrLK(gray[i-step],gray[i],np.float32(previous).reshape(-1,1,2),None,winSize=(31,31),maxLevel=3)
                back,ok,_=cv2.calcOpticalFlowPyrLK(gray[i],gray[i-step],flow,None,winSize=(31,31),maxLevel=3)
                flow=flow.reshape(-1,2);valid=(status.ravel()>0)&(ok.ravel()>0)&(np.linalg.norm(back.reshape(-1,2)-previous,axis=1)<2)
                points=np.where(valid[:,None],flow,previous)
                for k,template in enumerate(templates):
                    candidate,score=match_patch(template,gray[i],points[k],8)
                    if score>.8:points[k]=.9*points[k]+.1*candidate
            tracks[i].append((abs(i-start),points.copy()))
    for a,b in zip(anchors[:-1],anchors[1:]):advance(a,b,1);advance(b,a,-1)
    result=[]
    for index,path in enumerate(paths):
        if index in seeds:points=seeds[index].copy()
        else:
            weights=np.array([1/(d+1)**2 for d,_ in tracks[index]])
            points=np.average([p for _,p in tracks[index]],axis=0,weights=weights)
        rgb=cv2.cvtColor(cv2.resize(cv2.imread(str(path)),(720,720),interpolation=cv2.INTER_CUBIC),cv2.COLOR_BGR2RGB)
        # Irises have a unique chromatic patch and give an independent absolute
        # drift correction; the other features retain their own material track.
        eyes=iris_centers(rgb,points[:2]);points[:2]=eyes
        contour=skin_contour(rgb,points,points[6,1]);contour.pop('extra')
        result.append(dict(index=index,frame=index+1,landmarks=points.tolist(),**contour))
        if index%60==0:print('tracked',index+1,'/361',flush=True)
    return result
