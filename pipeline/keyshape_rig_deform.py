"""방향별 메시와 부위 변환 함수."""
import math
import cv2
import numpy as np
from keyshape_rig_parts import *
def grid_mesh(part,cell):
    x,y,w,h=part['bbox']; a=part['img'][...,3]>0
    cols=max(1,math.ceil(w/cell)); rows=max(1,math.ceil(h/cell)); pos=[];tri=[]
    xs=[int(round(i*w/cols)) for i in range(cols+1)]; ys=[int(round(j*h/rows)) for j in range(rows+1)]   # 5판: 정면 정점은 정수 화소 자리(GPU 부분 화소 격자에서 정면 표본이 어긋나지 않게)
    for j in range(rows+1):
        for i in range(cols+1): pos.append([x+xs[i],y+ys[j]])
    for j in range(rows):
        for i in range(cols):
            r0,r1=ys[j],ys[j+1]; c0,c1=xs[i],xs[i+1]
            if not a[r0:r1,c0:c1].any(): continue
            p=j*(cols+1)+i; q=p+1; r=p+cols+1; s=r+1; tri+=[[p,q,r],[q,s,r]]
    return np.array(pos,float),np.array(tri,np.int32)
def cell_for(pid):
    if pid.startswith('eye-') or pid.startswith('closed') or pid in ('nose-detail','mouth-detail'): return 16
    if pid in ('face-skin','face-scalp','skull-plate','hair-plate','neck','face-backing') or pid.startswith('still-fill'): return 20
    return 40
def side_of(pid): return 'l' if '-left' in pid or pid.endswith('-left') else 'r'
def group_of(pid):
    if pid.startswith('still-fill'): return 'fill'
    if pid in ('sweater','skirt','necklace'): return 'body'
    if pid=='neck': return 'neck'
    if pid.startswith('tail-'): return 'tail'
    if pid in ('bangs-left','bangs-right','hairpin'): return 'bangs'
    if pid.startswith('side-hair'): return 'side'
    if pid in ('hair-top','back-hair-center','back-hair-left','back-hair-right','skull-plate','hair-plate'): return 'skull'
    if pid.startswith('ear-'): return 'ear'
    if pid.startswith('eye-') and pid.endswith('brow'): return 'brow'
    if pid.startswith('eye-') or pid.startswith('closed-eyes'): return 'eye'
    if pid=='nose-detail': return 'nose'
    if pid=='mouth-detail': return 'mouth'
    return 'field'    # face-skin, skull-plate, hair-plate, still-fill
HAIR_CLIP_PARTS=('bangs-left','bangs-right')      # 머리핀은 자르지 않는다(위 방향에서 실루엣 밖에 놓여 사라졌다)
def part_transform(pid,rest,tk,field,hidden):
    """파츠 하나의 2×3 변환(또는 None=변위장)."""
    g=group_of(pid); s=side_of(pid); sname={'l':'left','r':'right'}[s]
    if g=='eye': return affine_fit([rest[k] for k in EYE(s)],[tk[k] for k in EYE(s)])
    if g=='brow': return similarity_fit([rest[k] for k in BROW(s)],[tk[k] for k in BROW(s)])
    if g=='mouth': return similarity_fit([rest[k] for k in ['mouth_l','mouth_c','mouth_r']],[tk[k] for k in ['mouth_l','mouth_c','mouth_r']])
    if g=='nose': return translation([rest['nose']],[tk['nose']])
    if g=='ear':
        return similarity_fit([rest[k] for k in EAR(s)],[tk[k] for k in EAR(s)])
    if g=='skull': return similarity_fit([rest[k] for k in SK],[tk[k] for k in SK])
    if g=='bangs':  # 2판과 같이 두 눈썹 꼭대기 이동의 평균으로 앞머리·머리핀이 한 덩어리로 움직인다(3판의 관자 점 추종은 가르마 단면, 4판의 세로 두상 섞기는 윗머리 모서리·살색 쐐기를 냈다). 위 방향의 뿔은 hairClip이 막는다
        d=(np.array(tk['brow_l_arch'])-rest['brow_l_arch']+np.array(tk['brow_r_arch'])-rest['brow_r_arch'])/2; return np.array([[1,0,d[0]],[0,1,d[1]]],float)
    if g=='side': return translation([rest[f'ct_{s}_eye']],[tk[f'ct_{s}_eye']])
    if g=='tail':   # 4판: 묶음머리·머리끈은 같은 쪽 귀(4점 유사변환)를 따른다. 머리끈은 귀 바로 아래에 묶여 있다. 3판의 두상 추종은 위 방향에서 머리끈을 볼 높이로 올렸다.
        if hidden==s: return translation([rest[f'ear_{s}_c']],[tk[f'ear_{s}_c']])     # 가려진 먼 귀는 자리만(가로 누름·회전은 쓰지 않는다)
        return similarity_fit([rest[k] for k in EAR(s)],[tk[k] for k in EAR(s)])
    if g=='body': return np.array([[1,0,0],[0,1,0]],float)
    return None
OUTSIDE_SHRINK=1.0                            # 4판: 얼굴 다각형 바깥 오프셋을 줄이지 않는다(실루엣 마스크가 잘라 낸다). 3판 0.3·0.15에서 접힘이 늘었다
def outside_field(pos,rest,field,far=None):
    """6판: 가까운 쪽은 밖 150px까지 TPS를 쓰고 최근접 변위장으로 부드럽게 전환한다. 먼 쪽은 최근접 규칙을 쓴다."""
    poly=np.array([rest[k] for k in FACE_POLY],np.float32); d=np.array([cv2.pointPolygonTest(poly,(float(x),float(y)),True) for x,y in pos])
    out=d<0; delta=tps_eval(field,pos)-pos
    if out.any():
        P=pos[out]; Q=np.zeros_like(P)
        for i,pt in enumerate(P):   # 다각형 변 위 최근접점
            best=None;bd=1e18
            for a,b in zip(poly,np.roll(poly,-1,axis=0)):
                ab=b-a; t=np.clip(np.dot(pt-a,ab)/max(1e-6,np.dot(ab,ab)),0,1); q=a+t*ab; dd=np.sum((pt-q)**2)
                if dd<bd: bd=dd;best=q
            Q[i]=best
        dq=tps_eval(field,Q)-Q; near=Q+dq+(P-Q)*OUTSIDE_SHRINK-P
        t=np.clip(-d[out]/OUTSIDE_BLEND_PX,0,1)
        if far: t=np.where((P[:,0]<rest['nose'][0]) if far=='l' else (P[:,0]>rest['nose'][0]),1.0,t)
        weight=(t*t*(3-2*t))[:,None]; delta[out]=delta[out]*(1-weight)+near*weight
    return delta
def forehead_weight(pos,rest):
    y_brow=(rest['ct_l_brow'][1]+rest['ct_r_brow'][1])/2; t2=np.clip((y_brow-pos[:,1])/FOREHEAD_BLEND_PX,0,1); return t2*t2*(3-2*t2)
def blink_curves(parts):
    """5판 깜빡임 곡선. 쪽마다 x(정수, 속눈썹 파츠 x 범위) → top(x) 흰자 위 가장자리, bot(x) 흰자 아래 가장자리, closed(x) 감은 눈 부품 선의 알파 가중 중심,
    lashTop/lashBot(x) 위 속눈썹 파츠의 세로 범위, lowerTop/lowerBot(x) 아래 선 파츠의 세로 범위, closedTop/closedBot(x) 감은 눈 부품의 세로 범위(모두 알파≥128).
    흰자가 없는 열(눈꼬리 밖 속눈썹)은 가장 가까운 열의 값을 쓴다. 감은 선이 없는 열도 같다."""
    out={}
    for side in ('left','right'):
        sc=parts[f'eye-{side}-sclera']; sx,sy,sw,sh=sc['bbox']; a=sc['img'][...,3]>=128
        la=parts[f'eye-{side}-upper-lashes']['bbox']; lo=parts[f'eye-{side}-lower-line']['bbox']
        x0=min(la[0],lo[0],sx); x1=max(la[0]+la[2],lo[0]+lo[2],sx+sw)
        top=np.full(x1-x0,np.nan); bot=np.full(x1-x0,np.nan)
        for i in range(sw):
            ys=np.nonzero(a[:,i])[0]
            if len(ys): top[sx+i-x0]=sy+ys.min(); bot[sx+i-x0]=sy+ys.max()
        ce=parts[f'closed-eyes-{side}']; cx,cy,cw,ch=ce['bbox']; ca=ce['img'][...,3].astype(float); closed=np.full(x1-x0,np.nan)
        for i in range(cw):
            col=ca[:,i]
            if col.sum()>0 and cx+i-x0>=0 and cx+i-x0<x1-x0: closed[cx+i-x0]=cy+(col*np.arange(ch)).sum()/col.sum()
        def fillnan(v):
            idx=np.arange(len(v)); ok=~np.isnan(v); return np.interp(idx,idx[ok],v[ok])
        def extents(pid):      # 5판: 열마다 알파≥128 세로 범위(위, 아래)
            px,py,pw,ph=parts[pid]['bbox']; pa=parts[pid]['img'][...,3]>=128; t=np.full(x1-x0,np.nan); b=np.full(x1-x0,np.nan)
            for i in range(pw):
                ys=np.nonzero(pa[:,i])[0]; j=px+i-x0
                if len(ys) and 0<=j<x1-x0: t[j]=py+ys.min(); b[j]=py+ys.max()
            return t,b
        lt,lb=extents(f'eye-{side}-upper-lashes'); ot,ob=extents(f'eye-{side}-lower-line')
        runs=[]                # 감은 눈 부품 「선」의 두께: 열마다 알파 가중 중심을 품은 알파≥128 연속 구간의 길이(아래로 뻗은 속눈썹 가닥과 위 주름은 끊겨 있어 빠진다). 그 중앙값의 절반을 closed(x) 위아래로 둔다
        for i in range(cw):
            col=ca[:,i]
            if col.sum()<=0: continue
            c=int(round((col*np.arange(ch)).sum()/col.sum())); m=col>=128
            if not m[c]:
                idx=np.nonzero(m)[0]
                if not len(idx): continue
                c=int(idx[np.argmin(abs(idx-c))])
            t=c
            while t>0 and m[t-1]: t-=1
            b=c
            while b<ch-1 and m[b+1]: b+=1
            runs.append(b-t+1)
        half=float(np.median(runs))/2 if runs else 2.0; closed_f=fillnan(closed); ct,cb=closed_f-half,closed_f+half
        R=lambda v:np.round(fillnan(v),2).tolist()
        out[side]=dict(x0=int(x0),top=R(top),bot=R(bot),closed=R(closed),lashTop=R(lt),lashBot=R(lb),lowerTop=R(ot),lowerBot=R(ob),closedTop=R(ct),closedBot=R(cb),closedLineHalfPx=round(half,2))
    return out
