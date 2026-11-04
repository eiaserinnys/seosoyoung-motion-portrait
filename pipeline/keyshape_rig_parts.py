"""원화 점과 파츠를 방향별 데이터로 바꾸는 재현 가능한 단계."""
import json, math, os
from pathlib import Path
import numpy as np, cv2
from PIL import Image
from scipy import ndimage as ndi
from matplotlib.path import Path as _MplPath
ROOT = Path(__file__).resolve().parents[1]
HERE = str(ROOT)
R = str(ROOT)
K = str(ROOT / 'source/keyshapes')
P = str(ROOT / 'bundle/assets')
KEYS={'front':(0,0),'left':(-1,0),'right':(1,0),'up':(0,1),'down':(0,-1),'up-left':(-1,1),'up-right':(1,1),'down-left':(-1,-1),'down-right':(1,-1)}
HAIR=('hair-plate','hair-top','hairpin','bangs-left','bangs-right','side-hair-left','side-hair-right','back-hair-center','back-hair-left','back-hair-right','tail-left-hair','tail-right-hair','tail-left-tie','tail-right-tie')
EYE_CLOSED_Y={'left':445.6,'right':445.4}   # parts/closed-eyes-v1.png 선의 y 평균(측정)
GAZE_PX=11                                  # (흰자 폭 96 − 홍채 폭 74)/2
FAR_EAR_FADE=(0.45,0.60)                    # 먼 쪽 옆머리·묶음머리가 얼굴 뒤로 넘어가며 섞이는 슬라이더 구간(영상에서 먼 귀가 사라지는 |yaw| 17~21° = 끝값 36~38°의 0.45~0.6)
FAR_EAR_INSET=6                             # 먼 귀가 끝 자세에서 얼굴 윤곽 안쪽으로 들어가는 여유(px)
FAR_EAR_XSCALE=0.35                         # 먼 귀 끝 자세의 가로 누름
NECK_PIVOT=(502,568)
FAR_WIPE_FEATHER_PX=8                        # 먼 쪽 머리카락 앞 사본을 얼굴 안에서 닫는 마스크의 가장자리 부드러움(px). 근거 없음(trim_ears의 8px과 같은 값)
BLINK_SQUASH_END=0.29
BLINK_CLOSED_FADE=(0.27,0.29)
EDGE_AA_PX=1.0
EDGE_DARK_PX=4
EDGE_DARK_LUM=160
BARE_SKIN_LUM=160
BARE_EDGE_SIGMA=6
BARE_COVER={'face-skin':'face-scalp','ear-left':'ear-left-cover','ear-right':'ear-right-cover'}
OUTSIDE_BLEND_PX=150
SIL_FACE_PAD=3.0
BLINK_FADE=(0.22,0.30)                      # 뜬 정도가 이 구간을 지나며 눈 파츠(위 속눈썹·아래 선·흰자·홍채·동공·반사광)는 1→0, 감은 눈 부품은 0→1로 교차 페이드. 0.22는 감은 부품 전환점(2판), 0.30은 속눈썹이 감은 선 범위의 90%까지 눌린 뒤(s=0.9)라 두 그림이 같은 자리·두께에 겹친다
MASK_RASTER_MARGIN_PX=0.5
NECK_RAMP=(640,695)                         # 목·still-fill의 얼굴 추종 가중치: y≤640(턱 끝 616 아래, 얼굴 받침 아래 한계와 같음) 1 → y≥695(스웨터 불투명 시작 699 위) 0. 3판은 600~690이라 턱(616)이 비탈 안에 있어 턱 아래 목이 얼굴보다 덜 움직여 1~3px 틈(구멍 8~28px)이 났다
MIN_OWN_ALPHA=0.2
FILL_TILE=128                               # still-fill 조각 크기
BACKING_GROW=30
BACKING_MIN_Y=331                           # 얼굴 받침 위 한계(눈썹 꼭대기 y). 메울 틈(귀·볼·턱)은 전부 이 아래다
BACKING_MAX_Y=640                           # 얼굴 받침은 이 높이(턱 끝 616 위 56px)까지만. 목 쪽 받침이 턱 옆으로 삐져나오던 것(3판 1차) 방지                             # 얼굴 받침 확장 폭(px). 2판 14에서 귀·볼 사이 슬리버 구멍(최대 58px)이 남아 넓힘
NEAR_SIDE_TEMPLE_W=1.0                      # 가까운 쪽 옆머리 아래 끝이 관자 점을 따르는 비율. 4판: 2판과 같이 1.0(3판 0.5는 옆머리 안쪽의 곧은 가장자리(앞머리에 가려진 상자 경계)를 드러냈다)
FOREHEAD_BLEND_PX=140                       # 이마: 눈썹 높이에서 이 폭 위까지 두상 변환으로 넘어간다(좁으면 세로 압축이 음수가 되어 삼각형이 뒤집힌다)
HIDDEN_HAIR_ALPHA=0.2                       # 정면에서 이 알파 이상의 머리카락에 덮인 얼굴 화소는 매끈한 피부로 채운다
SIL_SUBDIV=6                                # 얼굴 다각형 스플라인 분할 수(점 사이)
CLIP_MARGIN_PX=None                         # 얼굴 자르기 마스크 여유. build()가 정면에서 잰다(rig2d.json clipMarginPx). None이면 아직 안 잼
FACE_POLY=['hl_l','hl_lm','hl_c','hl_rm','hl_r','ct_r_brow','ct_r_eye','ct_r_nose','ct_r_mouth','ct_r_jaw','chin','ct_l_jaw','ct_l_mouth','ct_l_nose','ct_l_eye','ct_l_brow']
EAR_ALPHA={}

# ---------- 보간·맞춤 도구 ----------
def tps_fit(src,dst,lam=1.0):
    src=np.asarray(src,float); dst=np.asarray(dst,float); n=len(src)
    d=np.linalg.norm(src[:,None]-src[None],axis=-1); Kmat=np.where(d>0,d*d*np.log(np.maximum(d,1e-9)),0.0)+lam*np.eye(n)
    Pm=np.hstack([np.ones((n,1)),src]); A=np.zeros((n+3,n+3)); A[:n,:n]=Kmat; A[:n,n:]=Pm; A[n:,:n]=Pm.T
    b=np.zeros((n+3,2)); b[:n]=dst; w=np.linalg.solve(A,b); return (src,w)
def tps_eval(model,pts):
    src,w=model; pts=np.asarray(pts,float); n=len(src)
    d=np.linalg.norm(pts[:,None]-src[None],axis=-1); U=np.where(d>0,d*d*np.log(np.maximum(d,1e-9)),0.0)
    return U@w[:n]+w[n]+pts@w[n+1:]
def similarity_fit(src,dst):
    src=np.asarray(src,float); dst=np.asarray(dst,float); ms,md=src.mean(0),dst.mean(0); s0=src-ms; d0=dst-md
    a=(s0[:,0]*d0[:,0]+s0[:,1]*d0[:,1]).sum(); b=(s0[:,0]*d0[:,1]-s0[:,1]*d0[:,0]).sum(); den=(s0**2).sum()
    if den<1e-9: return np.array([[1,0,md[0]-ms[0]],[0,1,md[1]-ms[1]]],float)
    ca,sa=a/den,b/den; t=md-np.array([ca*ms[0]-sa*ms[1],sa*ms[0]+ca*ms[1]])
    return np.array([[ca,-sa,t[0]],[sa,ca,t[1]]])
def affine_fit(src,dst):
    """최소제곱 아핀(2×3). 점 3개 이상."""
    src=np.asarray(src,float); dst=np.asarray(dst,float); A=np.hstack([src,np.ones((len(src),1))])
    X,_,_,_=np.linalg.lstsq(A,dst,rcond=None); return X.T
def translation(src,dst): src=np.asarray(src,float); dst=np.asarray(dst,float); t=(dst-src).mean(0); return np.array([[1,0,t[0]],[0,1,t[1]]],float)
def apply23(M,pts): pts=np.asarray(pts,float); return pts@M[:,:2].T+M[:,2]
def fit_ellipse(pts):
    x=np.array([p[0] for p in pts],float); y=np.array([p[1] for p in pts],float)
    D=np.stack([x*x,x*y,y*y,x,y,np.ones_like(x)],1); _,_,vt=np.linalg.svd(D); return vt[-1]
def ellipse_mask(coef,M,shape=(1536,1024)):
    """원뿔곡선 계수(영상 좌표)와 영상→원화 유사변환 M으로 원화 캔버스의 타원 안쪽 마스크"""
    a,b,c,d,e,f=coef; yy,xx=np.mgrid[0:shape[0],0:shape[1]].astype(float)
    Minv=cv2.invertAffineTransform(M.astype(np.float64)); X=Minv[0,0]*xx+Minv[0,1]*yy+Minv[0,2]; Y=Minv[1,0]*xx+Minv[1,1]*yy+Minv[1,2]
    val=a*X*X+b*X*Y+c*Y*Y+d*X+e*Y+f; cx,cy=(2*c*d-b*e)/(b*b-4*a*c),(2*a*e-b*d)/(b*b-4*a*c); sgn=np.sign(a*cx*cx+b*cx*cy+c*cy*cy+d*cx+e*cy+f)
    return val*sgn>0

# ---------- 제어점 ----------
SK=[f'sk{i:02d}' for i in range(13)]
SKULL_OUTLIER_PX=15      # 두상 점이 정면 두상의 유사변환에서 이보다 벗어나면 올림머리로 보고 뺀다(640 프레임 화소)
DROPPED_SKULL=[]
SKULL_RESID=[]
EYE=lambda s:[f'eye_{s}_out',f'eye_{s}_up1',f'eye_{s}_up2',f'eye_{s}_up3',f'eye_{s}_in',f'eye_{s}_lo3',f'eye_{s}_lo2',f'eye_{s}_lo1']
BROW=lambda s:[f'brow_{s}_a',f'brow_{s}_arch',f'brow_{s}_b']
EAR=lambda s:[f'ear_{s}_c',f'ear_{s}_top',f'ear_{s}_bot',f'ear_{s}_back']
def load_points():
    V=json.load(open(f'{K}/points_video.json')); A=json.load(open(f'{K}/points_art_auto.json'))
    return {d['name']:d['points'] for d in V.values()},A,{d['name']:int(k) for k,d in V.items()}
def complete_skull(pts,front):
    out=dict(pts); have=[k for k in SK if k in pts]
    if pts is front:
        coef=fit_ellipse([pts[k] for k in have]); a,b,c,d,e,f=coef; cx,cy=(2*c*d-b*e)/(b*b-4*a*c),(2*a*e-b*d)/(b*b-4*a*c)
        for i,k in enumerate(SK):
            if k in pts: continue
            ang=math.radians(180-15*i); dx,dy=math.cos(ang),-math.sin(ang)
            A2=a*dx*dx+b*dx*dy+c*dy*dy; B2=2*a*cx*dx+b*(cx*dy+cy*dx)+2*c*cy*dy+d*dx+e*dy; C2=a*cx*cx+b*cx*cy+c*cy*cy+d*cx+e*cy+f
            disc=max(0,B2*B2-4*A2*C2); t=max((-B2+math.sqrt(disc))/(2*A2),(-B2-math.sqrt(disc))/(2*A2)); out[k]=[cx+dx*t,cy+dy*t]
    else:
        use=list(have); hc=np.array(pts.get('head_c',[320,260]),float)
        for _ in range(4):
            M=similarity_fit([front[k] for k in use],[pts[k] for k in use])
            # 바깥쪽(두상 중심에서 멀어지는 방향)으로 벗어난 점만 올림머리로 본다
            rad={k:float(np.dot(np.array(pts[k],float)-apply23(M,[front[k]])[0],(np.array(pts[k],float)-hc)/max(1e-6,np.linalg.norm(np.array(pts[k],float)-hc)))) for k in use}
            worst=max(use,key=lambda k:rad[k])
            if rad[worst]<=SKULL_OUTLIER_PX or len(use)<=5: break
            use.remove(worst)
        SKULL_RESID.append({k:round(rad.get(k,0),1) for k in have})
        dropped=[k for k in have if k not in use]
        if dropped: DROPPED_SKULL.append((len(DROPPED_SKULL),dropped))
        M=similarity_fit([front[k] for k in use],[pts[k] for k in use])
        for k in SK:
            if k not in use: out[k]=[float(v) for v in apply23(M,[front[k]])[0]]
    return out
def far_side(name): return 'l' if 'left' in name else 'r' if 'right' in name else None
def hidden_ear(pts,name):
    far=far_side(name); return far if (far and f'ear_{far}_c' not in pts) else None
def build_targets():
    V,A,frames=load_points(); front=complete_skull(V['front'],V['front'])
    names=[k for k in front if k not in ('head_c','neck_base')]
    S=similarity_fit([front[k] for k in ['eye_l_iris','eye_r_iris','mouth_c','chin']],[A[k] for k in ['eye_l_iris','eye_r_iris','mouth_c','chin']])
    rest={k:(list(map(float,A[k])) if k in A else list(map(float,apply23(S,[front[k]])[0]))) for k in names}
    targets={}; hidden={}
    for name,pts in V.items():
        DROPPED_SKULL.clear(); SKULL_RESID.clear(); pts=complete_skull(pts,front); h=hidden_ear(pts,name); hidden[name]=h
        if SKULL_RESID: print('skull radial residual',name,SKULL_RESID[0],'dropped',DROPPED_SKULL[0][1] if DROPPED_SKULL else [])
        tk={k:list(map(float,np.array(rest[k])+S[:,:2]@(np.array(pts[k],float)-np.array(front[k],float)))) for k in names if k in pts}
        if h:   # 가려진 먼 쪽 귀: 귀 중심을 먼 쪽 윤곽(눈·코 높이 중점)에서 얼굴 안쪽으로 (귀 반폭×누름+FAR_EAR_INSET)만큼 들여놓고, 가로를 FAR_EAR_XSCALE로 누른다. 얼굴 아래 순서라 볼 실루엣에 가려진다
            c=np.array(rest[f'ear_{h}_c']); half=abs(rest[f'ear_{h}_back'][0]-c[0])+10
            mid=(np.array(tk[f'ct_{h}_eye'])+np.array(tk[f'ct_{h}_nose']))/2; inward=1 if h=='l' else -1
            dst=mid+np.array([inward*(half*FAR_EAR_XSCALE+FAR_EAR_INSET),0])
            for k in EAR(h):
                r=np.array(rest[k]); tk[k]=list(map(float,dst+np.array([(r[0]-c[0])*FAR_EAR_XSCALE,r[1]-c[1]])))
        targets[name]=tk
    coef=fit_ellipse([V['front'][k] for k in SK if k in V['front']])
    return dict(names=names,rest=rest,targets=targets,S=S,front=front,V=V,A=A,frames=frames,ellipse=coef,hidden=hidden)

# ---------- 파츠 ----------
ORDER=[('skull-plate',90),('hair-plate',91),('back-hair-center',100),('back-hair-left',110),('back-hair-right',120),('face-backing',150),('neck',200),('skirt',220),('sweater',230),('necklace',240),
 ('ear-left',280),('ear-left-cover',281),('ear-right',285),('ear-right-cover',286),('tail-left-hair',291),('tail-right-hair',292),('tail-left-tie',293),('tail-right-tie',294),('hair-top',295),('face-skin',300),('face-scalp',301),
 ('eye-left-sclera',400),('eye-right-sclera',405),('eye-left-iris',410),('eye-right-iris',415),('eye-left-pupil',420),('eye-right-pupil',425),
 ('eye-left-lower-line',440),('eye-right-lower-line',445),('eye-left-upper-lashes',450),('eye-right-upper-lashes',455),('eye-left-highlight',458),('eye-right-highlight',459),
 ('closed-eyes-left',460),('closed-eyes-right',461),('eye-left-brow',500),('eye-right-brow',510),('nose-detail',520),('mouth-detail',530),
 ('side-hair-left',700),('side-hair-right',710),('bangs-left',720),('bangs-right',730),('hairpin',740)]
SWAP={'closed-eyes-left','closed-eyes-right'}
CLIPS={'eye-left-iris':'eye-left-sclera','eye-left-pupil':'eye-left-sclera','eye-left-highlight':'eye-left-sclera','eye-right-iris':'eye-right-sclera','eye-right-pupil':'eye-right-sclera','eye-right-highlight':'eye-right-sclera'}
def load_parts(T,still):
    idx={p['id']:p for p in json.load(open(f'{R}/parts/parts_index.json'))}; parts={}
    for pid,_ in ORDER:
        if pid in ('skull-plate','hair-plate','still-fill','still-fill-hair','face-backing','face-scalp','ear-left-cover','ear-right-cover') or pid in SWAP: continue
        p=idx[pid]; x,y,w,h=p['bbox']; im=np.asarray(Image.open(f"{R}/{p['file']}").convert('RGBA')).copy(); parts[pid]=dict(bbox=[x,y,w,h],img=im)
    p=idx['closed-eyes-v1']; x,y,w,h=p['bbox']; im=np.asarray(Image.open(f"{R}/{p['file']}").convert('RGBA')).copy()
    L=im.copy(); L[:,502-x:]=0; Rr=im.copy(); Rr[:,:502-x]=0
    parts['closed-eyes-left']=dict(bbox=[x,y,w,h],img=L); parts['closed-eyes-right']=dict(bbox=[x,y,w,h],img=Rr)
    for pid in ('neck','face-skin','ear-left','ear-right'):   # 복원 찌꺼기(머리끈 구슬 파랑, 머리카락 검정) 투명
        im=parts[pid]['img'].astype(int); junk=((im[...,2]>im[...,0]+15)|(im[...,:3].sum(-1)<250))&(im[...,3]>0); parts[pid]['img'][junk,3]=0
    parts['face-skin']['orig']=parts['face-skin']['img'].copy()
    global EAR_ALPHA
    EAR_ALPHA={k:(parts[k]['bbox'],parts[k]['img'][...,3]>=128) for k in ('ear-left','ear-right')}
    parts['skull-plate']=silhouette_plate(T,parts['face-skin'])
    parts['hair-plate']=plate(T,parts['face-skin'],parts['back-hair-center'],skin=False)
    smooth_hidden_skin(parts,T)
    trim_ears(parts)
    smooth_hidden_hairtop(parts)
    parts['face-backing']=face_backing(parts['face-skin'],BACKING_GROW,[parts[k] for k in ('neck','ear-left','ear-right','skull-plate')])
    return parts
def smooth_hidden_skin(parts,T=None):
    """K. 정면에서 머리카락(앞머리·윗머리·옆머리·머리핀)에 알파≥HIDDEN_HAIR_ALPHA로 덮인 얼굴 피부 화소(속눈썹 위)를, 보이는 피부의 저주파 색(정규화 가우스, σ20)으로 바꾼다.
    정면에서 보이지 않는 화소만 바꾸므로 정면 차이는 0이다(머리카락이 반투명한 자리는 recut이 새 바탕색으로 다시 푼다)."""
    H,W=1536,1024; cover=np.zeros((H,W),np.float32)
    for pid in ('bangs-left','bangs-right','hair-top','side-hair-left','side-hair-right','hairpin'):
        x,y,w,h=parts[pid]['bbox']; a=parts[pid]['img'][...,3]/255.0; cover[y:y+h,x:x+w]=np.maximum(cover[y:y+h,x:x+w],a)
    face=parts['face-skin']; x,y,w,h=face['bbox']; img=face['img'].astype(np.float32); fa=img[...,3]>0
    lash_top=min(parts['eye-left-upper-lashes']['bbox'][1],parts['eye-right-upper-lashes']['bbox'][1])
    hidden=fa&(cover[y:y+h,x:x+w]>=HIDDEN_HAIR_ALPHA); hidden[lash_top-y:]=False      # 4판: 속눈썹 위(이마·관자)만. 옆머리 아래 볼의 복원 윤곽선은 그대로 둔다(3판 「평평한 피부판」)
    visible=fa&~(cover[y:y+h,x:x+w]>=HIDDEN_HAIR_ALPHA)
    rgb=img[...,:3]; wgt=visible.astype(np.float32)
    num=cv2.GaussianBlur(rgb*wgt[...,None],(0,0),20); den=cv2.GaussianBlur(wgt,(0,0),20)
    low=num/np.maximum(den,1e-4)[...,None]
    img[hidden,:3]=low[hidden]
    edge=fa&~ndi.binary_erosion(fa,iterations=EDGE_DARK_PX); luma=img[...,:3]@np.array([.299,.587,.114])
    edge_dark=edge&(cover[y:y+h,x:x+w]>=HIDDEN_HAIR_ALPHA)&(luma<EDGE_DARK_LUM); img[edge_dark,:3]=low[edge_dark]; face['edge_dark_px']=int(edge_dark.sum())
    zone=fa&(cover[y:y+h,x:x+w]>0); zone[lash_top-y:]=False
    bare=img.copy(); bare[zone,:3]=low[zone]; face['bare']=np.clip(bare+0.5,0,255).astype(np.uint8); face['bare_px']=int(zone.sum())
    if T is not None:   # 4판: 숨은 화소 가운데 얼굴 다각형(머리선·관자 윤곽) 밖은 두피·옆머리 자리이므로 얼굴판에서 뺀다(켬: 앞머리가 들리면 살색 쐐기 대신 머리카락 받침, 끔: 실루엣 받침이 채운다)
        poly=np.array([T['rest'][k] for k in FACE_POLY],np.int32); inside=np.zeros((1536,1024),np.uint8); cv2.fillPoly(inside,[poly],255); inside=inside[y:y+h,x:x+w]>0
        scalp=(fa&(cover[y:y+h,x:x+w]>=HIDDEN_HAIR_ALPHA))&~inside; img[scalp,3]=0; face['bare'][scalp,3]=0; face['scalp_removed_px']=int(scalp.sum())
    face['img']=np.clip(img+0.5,0,255).astype(np.uint8); face['hidden_px']=int(hidden.sum())
EAR_KEEP_PX=24                               # 귀 파츠 가운데 얼굴에 가려진 복원 영역은 얼굴 가장자리에서 이만큼 안쪽까지만 남긴다(L: 평평한 피부 조각 방지, 틈 메움)
def trim_ears(parts):
    """L. 귀는 얼굴 아래 순서라, 정면에서 얼굴 피부(알파≥200)에 가려진 귀 파츠의 복원 영역이 얼굴이 움직이면 평평한 조각으로 드러난다.
    가려진 영역은 얼굴 가장자리에서 EAR_KEEP_PX 안쪽까지만 남긴다(얼굴·귀 사이 틈을 메우는 띠)."""
    face=parts['face-skin']; fx,fy,fw,fh=face['bbox']; cover=np.zeros((1536,1024),bool); cover[fy:fy+fh,fx:fx+fw]=face['img'][...,3]>=200
    dist=ndi.distance_transform_edt(cover)      # 가려진 안쪽에서 가려지지 않은 곳까지 거리
    face128=np.zeros((1536,1024),bool); face128[fy:fy+fh,fx:fx+fw]=face['img'][...,3]>=128; cover128=face128.copy()
    for q in ('side-hair-left','side-hair-right','bangs-left','bangs-right','hair-top','back-hair-left','back-hair-right','tail-left-hair','tail-right-hair'):
        bx,by,bw,bh=parts[q]['bbox']; cover128[by:by+bh,bx:bx+bw]|=parts[q]['img'][...,3]>=int(HIDDEN_HAIR_ALPHA*255)
    for pid in ('ear-left','ear-right'):
        x,y,w,h=parts[pid]['bbox']; im=parts[pid]['img']; d=dist[y:y+h,x:x+w]; fade=np.clip((EAR_KEEP_PX-d)/8,0,1)
        f=im.astype(np.float32); lum=f[...,:3]@np.array([.299,.587,.114]); solid=f[...,3]>=128; edge=solid&~ndi.binary_erosion(solid,iterations=EDGE_DARK_PX)
        hidden=edge&cover128[y:y+h,x:x+w]&(lum<EDGE_DARK_LUM)
        wgt=(solid&(lum>=BARE_SKIN_LUM)).astype(np.float32)
        low=cv2.GaussianBlur(f[...,:3]*wgt[...,None],(0,0),BARE_EDGE_SIGMA)/np.maximum(cv2.GaussianBlur(wgt,(0,0),BARE_EDGE_SIGMA),1e-3)[...,None]
        im[hidden,:3]=np.clip(low[hidden]+0.5,0,255).astype(np.uint8); parts[pid]['dark_hidden_px']=int(hidden.sum()); parts[pid]['orig']=im.copy()
        im[...,3]=np.round(im[...,3]*fade).astype(np.uint8); parts[pid]['trim_fade']=fade
HAIR_STACK=['back-hair-center','back-hair-left','back-hair-right','hair-top','tail-left-hair','tail-right-hair','tail-left-tie','tail-right-tie','side-hair-left','side-hair-right','bangs-left','bangs-right','hairpin']
def smooth_hidden_hairtop(parts):
    """O. 머리카락 파츠(윗머리·옆머리·앞머리)에서 정면에 그 위의 머리카락이나 머리핀에 가려진(알파≥0.2) 살색 화소(밝기>140, 복원 자국)를,
    같은 파츠의 보이는 어두운 머리카락 화소의 저주파 색으로 바꾼다. 정면 불변. 파츠들이 서로 다르게 움직여 벌어질 때 살색 조각이 드러나지 않는다."""
    for k,pid in enumerate(HAIR_STACK):
        if pid!='hair-top': continue      # 4판: 윗머리만. 옆머리·앞머리에 적용하면 복원 조각이 어두운 네모가 되어 관자에 드러났다(review-3 2)
        top=parts[pid]; x,y,w,h=top['bbox']; img=top['img'].astype(np.float32); a=img[...,3]>0
        cover=np.zeros((1536,1024),np.float32)
        for q in HAIR_STACK[k+1:]:
            bx,by,bw,bh=parts[q]['bbox']; cover[by:by+bh,bx:bx+bw]=np.maximum(cover[by:by+bh,bx:bx+bw],parts[q]['img'][...,3]/255.0)
        luma=img[...,:3]@np.array([.299,.587,.114]); hidden=a&(cover[y:y+h,x:x+w]>=0.2)&(luma>140)
        wgt=(a&(luma<=140)).astype(np.float32)
        if wgt.sum()<100 or not hidden.any(): continue
        low=cv2.GaussianBlur(img[...,:3]*wgt[...,None],(0,0),15)/np.maximum(cv2.GaussianBlur(wgt,(0,0),15),1e-3)[...,None]
        img[hidden,:3]=low[hidden]; top['img']=np.clip(img+0.5,0,255).astype(np.uint8); top['pale_hidden_px']=int(hidden.sum())
def bare_cover(parts):
    """6판: 다시 자르기로 달라진 얼굴·귀 화소를 원본 색으로 되돌리는 끔 전용 층."""
    for src,dst in BARE_COVER.items():
        part=parts[src]; ref=part.get('bare',part['orig']); original=ref.astype(int); recut=part['img'].astype(int)
        changed=(original[...,3]>0)&((np.abs(original[...,:3]-recut[...,:3]).max(-1)>0)|(original[...,3]!=recut[...,3]))
        solid=original[...,3]>=128; lum=original[...,:3]@np.array([.299,.587,.114]); weight=(solid&(lum>BARE_SKIN_LUM)).astype(np.float32)
        low=cv2.GaussianBlur(original[...,:3].astype(np.float32)*weight[...,None],(0,0),BARE_EDGE_SIGMA)/np.maximum(cv2.GaussianBlur(weight,(0,0),BARE_EDGE_SIGMA),1e-3)[...,None]
        image=np.zeros_like(part['img']); image[changed]=ref[changed]
        edge=changed&~solid; image[edge,:3]=np.clip(low[edge]+0.5,0,255).astype(np.uint8)
        image[changed,3]=np.maximum(original[changed,3],recut[changed,3]).astype(np.uint8)
        if 'trim_fade' in part: image[...,3]=np.round(image[...,3]*part['trim_fade']).astype(np.uint8)
        parts[dst]=dict(bbox=list(part['bbox']),img=image); part['cover_px']=int((image[...,3]>0).sum())

def face_backing(face,grow=14,clip_parts=()):
    """얼굴 받침: 얼굴 피부 알파를 grow px 넓힌 피부색 층. 얼굴과 같은 변위장으로 움직여 귀·목과 얼굴 사이 틈을 메운다. 정면에서는 다른 파츠 아래라 보이지 않는다."""
    x,y,w,h=face['bbox']; a=np.zeros((1536,1024),bool); a[y:y+h,x:x+w]=face['img'][...,3]>127
    m=ndi.distance_transform_edt(~a)<=grow; m[BACKING_MAX_Y:]=False; m[:BACKING_MIN_Y]=False      # 눈썹 위에서는 받침을 두지 않는다(머리카락 사이로 살색이 비치던 흰 조각: 3판)
    sil=a.copy()                                                # 쉬는 자세의 피부 부위(얼굴·목·귀·두상 받침) 실루엣 안(8px 여유)으로 제한: 목 옆으로 삐져나오지 않게
    for cp in clip_parts:
        bx,by,bw,bh=cp['bbox']; sil[by:by+bh,bx:bx+bw]|=cp['img'][...,3]>=128
    m&=ndi.binary_dilation(sil,iterations=8); rgb=forehead_color(face)
    img=np.zeros((1536,1024,4),np.uint8); img[m,:3]=rgb.astype(np.uint8); img[m,3]=255
    ys,xs=np.nonzero(m); bx,by,bw,bh=xs.min(),ys.min(),xs.max()-xs.min()+1,ys.max()-ys.min()+1
    return dict(bbox=[int(bx),int(by),int(bw),int(bh)],img=img[by:by+bh,bx:bx+bw].copy())
def catmull_closed(P,n):
    """닫힌 점열 P(k×2)를 Catmull-Rom 스플라인으로 점 사이 n개씩 나눠 k·n개로 돌려준다(항상 같은 수, 키 사이 보간용)."""
    P=np.asarray(P,float); k=len(P); out=[]
    for i in range(k):
        p0,p1,p2,p3=P[(i-1)%k],P[i],P[(i+1)%k],P[(i+2)%k]
        for j in range(n):
            t=j/n; t2=t*t; t3=t2*t
            out.append(0.5*((2*p1)+(-p0+p2)*t+(2*p0-5*p1+4*p2-p3)*t2+(-p0+3*p1-3*p2+p3)*t3))
    return np.array(out)
def ellipse_point(coef,ang_deg):
    a,b,c,d,e,f=coef; cx,cy=(2*c*d-b*e)/(b*b-4*a*c),(2*a*e-b*d)/(b*b-4*a*c)
    dx,dy=math.cos(math.radians(ang_deg)),-math.sin(math.radians(ang_deg))
    A2=a*dx*dx+b*dx*dy+c*dy*dy; B2=2*a*cx*dx+b*(cx*dy+cy*dx)+2*c*cy*dy+d*dx+e*dy; C2=a*cx*cx+b*cx*cy+c*cy*cy+d*cx+e*cy+f
    disc=max(0,B2*B2-4*A2*C2); t=max((-B2+math.sqrt(disc))/(2*A2),(-B2-math.sqrt(disc))/(2*A2)); return np.array([cx+dx*t,cy+dy*t])
SIL_N=160                                    # 실루엣 곡선 표본 수(호 길이 등분, 꼭대기에서 시계 방향)
SIL_SMOOTH=3                                 # 실루엣 표본의 이동 평균 반폭(표본 수). 3이면 일곱 점(둘레의 약 4%, 약 80px 호): 타원과 얼굴 다각형이 만나는 먼 쪽 관자 모서리를 둥글린다
def silhouette_curve(T,key):
    """4판 실루엣 곡선(SIL_N점, 원화 좌표). (정면 두상 타원 전체를 두상 13점으로 맞춘 유사변환으로 옮긴 것 ∪ 얼굴 다각형)에서,
    먼 쪽 아래(얼굴 중심선 너머, 먼 쪽 눈썹 윤곽점보다 아래, 얼굴 다각형 밖)를 뺀 영역의 바깥 윤곽을 호 길이로 SIL_N등분한다. 시작점은 꼭대기(sk06) 최근접, 시계 방향.
    먼 쪽 = 귀가 가려지는 쪽. 정면·위·아래는 빼는 곳이 없다. 영상 두상 점은 윗부분에만 있고 아래 호는 영상에 없어 정면 타원을 쓴다."""
    rest,tk=T['rest'],T['targets'][key]; S=T['S']; coef=T['ellipse']; far=T['hidden'][key]
    Msk=similarity_fit([rest[k] for k in SK],[tk[k] for k in SK])
    ell=[]
    for ang in range(0,360,6):
        p=ellipse_point(coef,ang); q=S[:,:2]@p+S[:,2]; ell.append(apply23(Msk,[q])[0])
    fp=offset_outward(catmull_closed(face_polygon_points(T,key),SIL_SUBDIV),SIL_FACE_PAD)
    m=fill_mask([np.array(ell),fp])       # 두상 13점 꺾은선은 넣지 않는다(15° 간격 직선과 ±13px 잡음이 관자에 모서리를 만든다). 점은 유사변환 추정에만 쓴다
    for side in 'lr':
        if far==side or not EAR_ALPHA: continue
        (bx,by,bw,bh),ear_alpha=EAR_ALPHA['ear-left' if side=='l' else 'ear-right']; full=np.zeros((1536,1024),np.uint8); full[by:by+bh,bx:bx+bw]=ear_alpha.astype(np.uint8)*255
        matrix=similarity_fit([rest[k] for k in EAR(side)],[tk[k] for k in EAR(side)])
        moved=cv2.warpAffine(full,matrix.astype(np.float64),(1024,1536),flags=cv2.INTER_NEAREST)
        m=np.maximum(m,cv2.dilate(moved,np.ones((5,5),np.uint8)))
    if far:
        inside=fill_mask([fp]); yb=tk[f'ct_{far}_brow'][1]; xc=tk['nose'][0]
        yy,xx=np.mgrid[0:1536,0:1024]; cut=(inside==0)&(yy>yb)&((xx<xc) if far=='l' else (xx>xc)); m[cut]=0
    yb=min(tk['ct_l_eye'][1],tk['ct_r_eye'][1])
    cs,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); c0=max(cs,key=cv2.contourArea).reshape(-1,2)
    up=c0[c0[:,1]<=yb]
    if len(up)>=3: hull=cv2.convexHull(up.astype(np.int32)); cv2.fillPoly(m,[hull],255)
    cs,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); c=max(cs,key=cv2.contourArea).reshape(-1,2).astype(float)
    top=np.array(tk['sk06'],float); i0=int(np.argmin(((c-top)**2).sum(1))); c=np.roll(c,-i0,0)
    if cv2.contourArea(c.astype(np.float32),oriented=True)<0: c=np.roll(c[::-1],1,0)    # 시계 방향(화면 좌표)으로 통일
    seg=np.linalg.norm(np.roll(c,-1,0)-c,axis=1); L=np.concatenate([[0],np.cumsum(seg)]); t=np.linspace(0,L[-1],SIL_N,endpoint=False)
    cc=np.vstack([c,c[:1]]); s=np.stack([np.interp(t,L,cc[:,0]),np.interp(t,L,cc[:,1])],1)
    k=SIL_SMOOTH; return sum(np.roll(s,i,0) for i in range(-k,k+1))/(2*k+1)      # 둘레 방향 이동 평균: 타원과 얼굴 다각형이 만나는 먼 쪽 관자 모서리를 둥글린다
def face_polygon_points(T,key):
    tk=T['targets'][key]; return np.array([tk[k] for k in FACE_POLY],float)
def offset_outward(P,m):
    """닫힌 곡선 점열을 바깥 법선으로 m만큼 민다(시계 방향·반시계 모두 처리: 넓이 부호로 바깥을 정한다)."""
    P=np.asarray(P,float); n=len(P); d=np.roll(P,-1,0)-np.roll(P,1,0); nrm=np.stack([d[:,1],-d[:,0]],1); nrm/=np.maximum(np.linalg.norm(nrm,axis=1,keepdims=True),1e-6)
    def area(Q): return abs(0.5*np.sum(Q[:,0]*np.roll(Q[:,1],-1)-np.roll(Q[:,0],-1)*Q[:,1]))
    A,B=P+nrm*m,P-nrm*m
    return A if (area(A)>=area(B))==(m>=0) else B      # 5판: 넓이가 커지는 쪽이 바깥(4판은 부호 판정이 거꾸로라 안쪽으로 밀었다: 세라피엘 4판 §3). m<0이면 안쪽
def fill_mask(curves,shape=(1536,1024)):
    """6판: 화소 (X,Y)의 중심 (X+0.5,Y+0.5)가 다각형 안이면 255."""
    m=np.zeros(shape,np.uint8)
    for c in curves:
        c=np.asarray(c,float); x0=max(int(math.floor(c[:,0].min())),0); x1=min(int(math.ceil(c[:,0].max()))+1,shape[1]); y0=max(int(math.floor(c[:,1].min())),0); y1=min(int(math.ceil(c[:,1].max()))+1,shape[0])
        if x1<=x0 or y1<=y0: continue
        yy,xx=np.mgrid[y0:y1,x0:x1]; pts=np.stack([xx.ravel()+0.5,yy.ravel()+0.5],1)
        m[y0:y1,x0:x1][_MplPath(c).contains_points(pts).reshape(y1-y0,x1-x0)]=255
    return m
def band_triangles(curve,m):
    """실루엣 c와 바깥으로 m 민 곡선 o 사이의 띠 삼각형."""
    c=np.asarray(curve,float); o=offset_outward(c,m); n=len(c); out=[]
    for i in range(n):
        j=(i+1)%n; out.extend((np.array([c[i],o[i],o[j]]),np.array([c[i],o[j],c[j]])))
    return out
def fill_dilated(curve,m,shape=(1536,1024)):
    """6판 마스크: 실루엣 채움과 바깥 띠 삼각형의 합집합."""
    mask=fill_mask([curve],shape)>0
    if m>0:
        for triangle in band_triangles(curve,m): mask|=fill_mask([triangle],shape)>0
    return mask
def signed_edge_alpha(curve,shape=(1536,1024)):
    """알파 = clip(EDGE_AA_PX + signed distance, 0, 1), 화소 중심에서 잰 거리."""
    hard=fill_mask([curve],shape)>0; band=ndi.binary_dilation(hard,iterations=3)&~ndi.binary_erosion(hard,iterations=2)
    out=hard.astype(np.float32); ys,xs=np.nonzero(band)
    if len(ys):
        points=np.stack([xs+0.5,ys+0.5],1).astype(np.float64); c=np.asarray(curve,float); a=c; b=np.roll(c,-1,0); ab=b-a; length2=np.maximum((ab**2).sum(1),1e-9); best=np.full(len(points),1e9)
        for i in range(len(a)):
            t=np.clip(((points-a[i])@ab[i])/length2[i],0,1); q=a[i]+t[:,None]*ab[i]; d=np.sqrt(((points-q)**2).sum(1)); best=np.minimum(best,d)
        out[ys,xs]=np.clip(EDGE_AA_PX+np.where(hard[ys,xs],best,-best),0,1)
    return out
def forehead_color(face):
    """끔 받침 색: 얼굴 피부 가운데 눈썹 위(y<BACKING_MIN_Y) 불투명 화소의 평균. 머리선 경계에서 이마와 받침의 색 단차를 줄인다(얼굴 전체 평균은 볼 그늘이 섞여 더 어둡다)."""
    x,y,w,h=face['bbox']; a=face['img'][...,3]>127; a[max(0,BACKING_MIN_Y-y):]=False
    return face['img'][...,:3][a].astype(float).mean(0)
def silhouette_plate(T,face):
    """끔 받침층(skull-plate): 정면 실루엣 ∪ 얼굴 다각형을 피부색으로 채운 것. 메시는 build()가 실루엣 곡선 점으로 따로 만든다(silhouette 메시); 이 그림은 색과 아틀라스용."""
    m=fill_mask([silhouette_curve(T,'front')]); rgb=forehead_color(face)
    img=np.zeros((1536,1024,4),np.uint8); img[m>0,:3]=rgb.astype(np.uint8); img[m>0,3]=255
    ys,xs=np.nonzero(img[...,3]); bx,by,bw,bh=xs.min(),ys.min(),xs.max()-xs.min()+1,ys.max()-ys.min()+1
    return dict(bbox=[int(bx),int(by),int(bw),int(bh)],img=img[by:by+bh,bx:bx+bw].copy())
def plate(T,face,back,skin):
    """두피 받침층: 정면 두상 타원 전체(영상 점 맞춤, 원화로 옮김). 피부색 또는 뒷머리색. 두상과 함께 통째로 움직인다. 아래쪽 둥근 끝(y≈571)은 얼굴·목 뒤에 숨는다."""
    m=ellipse_mask(T['ellipse'],T['S'])
    src=face if skin else back; a=src['img'][...,3]>127; rgb=src['img'][...,:3][a].astype(float).mean(0)
    img=np.zeros((1536,1024,4),np.uint8); alpha=cv2.GaussianBlur((m*255).astype(np.float32),(3,3),0); img[...,3]=np.clip(alpha,0,255).astype(np.uint8); img[img[...,3]>0,:3]=rgb.astype(np.uint8)
    ys,xs=np.nonzero(img[...,3]); bx,by,bw,bh=xs.min(),ys.min(),xs.max()-xs.min()+1,ys.max()-ys.min()+1
    return dict(bbox=[int(bx),int(by),int(bw),int(bh)],img=img[by:by+bh,bx:bx+bw].copy())
def recut_front(parts,still):
    """정면 소유권(2판).
    - 화소마다 맨 위의 알파≥MIN_OWN_ALPHA 파츠가 소유자. 그 위의 흐린(알파<문턱) 파츠는 원래 색·알파를 그대로 둔다(가는 머리카락·속눈썹 가장자리가 점선이 되지 않게).
    - 소유자의 색은 「원화 = 위 흐린 층들 over (소유자 색×알파 + 아래 합성×(1−알파))」가 되도록 역산한다. 알파는 유지한다(8비트로 양자화).
      역산한 색이 0..255 밖이면 소유자의 알파를 필요한 만큼 올린 뒤 다시 역산한다(원화 불투명 안쪽에서만).
    - 합성과 같은 float32 순서로 후보 색(±3)을 검사해 반올림 뒤에도 원화와 같은 정수를 고른다.
    - 못 맞춘 화소와 원화의 반투명 가장자리는 소유자가 원화 화소를 통째로 갖고(알파=원화 알파) 그 화소의 다른 파츠는 투명(자르기 기준 흰자는 불투명 유지).
    - 어느 파츠도 없는 원화 화소는 still-fill."""
    H,W=1536,1024; order=[pid for pid,_ in ORDER if pid in parts and pid not in SWAP]; n=len(order)
    full=np.zeros((n,H,W,4),np.float32)
    for i,pid in enumerate(order):
        x,y,w,h=parts[pid]['bbox']; full[i,y:y+h,x:x+w]=parts[pid]['img']
    outside=still[...,3]==0; full[:,outside,3]=0
    for hid in ('face-scalp','skull-plate'):                                  # 머리카락 끔 전용 층: 정면(켬) 소유권 계산에 넣지 않는다(4판: 실루엣 받침이 턱 옆 묶음머리 화소를 소유해 정면 177px이 비었다)
        if hid in order: full[order.index(hid)][...,3]=0
    for child,mask in CLIPS.items():       # 자르기 기준(흰자)은 홍채·동공·반사광이 있는 화소에서 불투명
        ci,mi=order.index(child),order.index(mask); full[mi][full[ci][...,3]>0,3]=255
    maskparts={order.index(m) for m in set(CLIPS.values())}
    opaque0=still[...,3]==255; covered=np.zeros((H,W),bool)
    fb=order.index('face-backing'); full[fb][~opaque0,3]=0        # 얼굴 받침은 원화 가장자리에 없고, 화소를 소유하지 않는다(아래 깔리기만)
    hi_=order.index('hair-plate'); full[hi_][opaque0&(full[hi_][...,3]>0),3]=255
    for i in range(n):            # 아래에서 위로: 원화가 불투명한데 아직 불투명 파츠가 없으면 맨 아래 파츠를 불투명으로
        if i==fb: continue
        has=full[i][...,3]>0; make=opaque0&has&~covered; full[i][make,3]=255; covered|=has&opaque0
    alpha=full[...,3]/255
    owner=np.full((H,W),-1,np.int32)
    for i in range(n-1,-1,-1):
        if i!=fb: owner[(owner<0)&(alpha[i]>=MIN_OWN_ALPHA)]=i
    full[fb][(owner<fb),3]=0; alpha[fb]=full[fb][...,3]/255      # 소유자가 받침 아래(목·뒷머리·받침층)이거나 없는(still-fill) 화소에서는 얼굴 받침을 지운다
    faint_only=(owner<0)&(alpha.max(0)>0)
    for i in range(n-1,-1,-1):
        if i!=fb: owner[faint_only&(owner<0)&(alpha[i]>0)]=i
    Sc=still[...,:3].astype(np.float32); Sa=still[...,3].astype(np.float32)/255; opaque=still[...,3]==255
    stat=dict(solved=0,raised=0,whole=0)
    below=np.zeros((H,W,3),np.float32)      # 미리 곱한 아래 합성(float32, 합성과 같은 순서)
    def q(a): return np.round(a*255)/255
    for i in range(n):
        mine=owner==i; idx=np.nonzero(mine)
        if len(idx[0]):
            a=alpha[i][idx]; bc=below[idx]; S=Sc[idx]; sa=Sa[idx]
            solvable=opaque[idx]&(a>=MIN_OWN_ALPHA)
            # 위 흐린 층: 순서대로 over 하기 위한 목록(이 화소에서 알파>0인 j>i)
            above=[(full[j][...,:3][idx]*(alpha[j][idx][:,None]),alpha[j][idx]) for j in range(i+1,n) if (alpha[j][idx]>0).any()]
            def composite(c,aa):
                X=c*aa[:,None]+bc*(1-aa)[:,None]
                for pj,aj in above: X=pj+X*(1-aj)[:,None]
                return X
            # 위 층을 벗긴 목표값 T: S = A + (1−Aa)·X  →  X = (S − A)/(1−Aa)
            A=np.zeros_like(S); Aa=np.zeros_like(a)
            for pj,aj in above: A=pj+A*(1-aj)[:,None]; Aa=aj+Aa*(1-aj)
            T=np.where((1-Aa)[:,None]>1e-4,(S-A)/np.maximum(1-Aa,1e-4)[:,None],S)
            aq=q(a)
            c=(T-bc*(1-aq)[:,None])/np.maximum(aq,1e-6)[:,None]
            # 범위 밖이면 알파 올리기(불투명 안쪽에서만)
            need=np.zeros_like(a)
            lo=c<0; hi=c>255
            with np.errstate(divide='ignore',invalid='ignore'):
                a_lo=np.where(lo,1-T/np.maximum(bc,1e-6),0).max(1); a_hi=np.where(hi,(T-bc)/np.maximum(255-bc,1e-6),0).max(1)
            need=np.clip(np.maximum(a_lo,a_hi),0,1); raise_=need>aq; aq2=np.where(raise_,q(np.minimum(1,need+1/255)),aq)
            c=(T-bc*(1-aq2)[:,None])/np.maximum(aq2,1e-6)[:,None]
            c0=np.clip(np.round(c),0,255); best=c0.copy(); err=np.full(c0.shape,999.0)
            for k in range(-3,4):
                cc=np.clip(c0+k,0,255); X=composite(cc.astype(np.float32),aq2.astype(np.float32)); e=np.abs(np.floor(X+0.5)-S)
                better=e<err; best[better]=cc[better]; err[better]=e[better]
            ok=solvable&(err.max(1)==0)&((aq2<1)|(aq<1)|True)
            # 결과 기록
            col=np.where(ok[:,None],best,S); al=np.where(ok,aq2,sa)
            full[i][...,:3][idx]=col; full[i][...,3][idx]=al*255; alpha[i][idx]=al
            stat['solved']+=int(ok.sum()); stat['raised']+=int((ok&raise_).sum()); stat['whole']+=int((~ok).sum())
            bad=~ok
            if bad.any():
                bm=np.zeros((H,W),bool); bm[idx[0][bad],idx[1][bad]]=True; bm_edge=bm&~opaque
                for j in range(n):
                    if j==i: continue
                    if j>i: full[j][bm,3]=0; alpha[j][bm]=0                                  # 위 파츠는 투명
                    elif j in maskparts: full[j][bm_edge,3]=255; alpha[j][bm_edge]=1.0       # 아래의 자르기 기준은 불투명 유지
                    else: full[j][bm_edge,3]=0; alpha[j][bm_edge]=0                          # 반투명 가장자리에서만 아래도 투명
            stat.setdefault('whole_by_part',{})[order[i]]=int(bad.sum())
            if bad.any() and 'whole_sample' not in stat:
                k=int(np.nonzero(bad)[0][0]); stat['whole_sample']=dict(part=order[i],xy=(int(idx[1][k]),int(idx[0][k])),a=float(a[k]),aq2=float(aq2[k]),S=S[k].tolist(),T=T[k].tolist(),bc=bc[k].tolist(),c=c[k].tolist(),err=err[k].tolist(),above=len(above),opaque=bool(opaque[idx][k]))
        a_i=alpha[i][...,None]; below=full[i][...,:3]*a_i+below*(1-a_i)
    gap=(owner<0)&(still[...,3]>0)
    solid=np.zeros((H,W),bool)
    for j,pid in enumerate(order):
        if pid in HAIR or pid.startswith('still-fill'): continue
        solid|=alpha[j]>=0.5
    solid=ndi.binary_dilation(solid,iterations=3)
    gap_hair=gap&((still[...,3]<255)|~solid); gap_solid=gap&~gap_hair
    for i,pid in enumerate(order):
        if pid in ('face-scalp','skull-plate'): continue       # 끔 전용 층은 소유권 계산에서 알파를 0으로 두었으므로 되쓰지 않는다
        x,y,w,h=parts[pid]['bbox']; parts[pid]['img']=np.clip(full[i][y:y+h,x:x+w]+0.5,0,255).astype(np.uint8)
    for name,g in (('still-fill',gap_solid),('still-fill-hair',gap_hair)):   # 흩어진 화소를 FILL_TILE 칸으로 나눠 작은 파츠 여럿으로(한 장이면 683×1435 상자가 아틀라스 2048에 안 들어간다)
        n_t=0
        for ty in range(0,1536,FILL_TILE):
            for tx in range(0,1024,FILL_TILE):
                sub=g[ty:ty+FILL_TILE,tx:tx+FILL_TILE]
                if not sub.any(): continue
                ys,xs=np.nonzero(sub); bx,by,bw,bh=tx+xs.min(),ty+ys.min(),xs.max()-xs.min()+1,ys.max()-ys.min()+1
                img=np.zeros((bh,bw,4),np.uint8); gg=g[by:by+bh,bx:bx+bw]; img[gg]=still[by:by+bh,bx:bx+bw][gg]
                parts[f'{name}-{n_t:03d}']=dict(bbox=[int(bx),int(by),int(bw),int(bh)],img=img); n_t+=1
        stat[name]=int(g.sum()); stat[name+'_tiles']=n_t
    return stat

# ---------- 메시와 변형 ----------
