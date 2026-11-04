"""시제품과 비교하기 위한 기준 렌더링 함수."""
import json, math
import cv2, numpy as np
from PIL import Image
from scipy import ndimage as ndi
from keyshape_rig_parts import *
from keyshape_rig_deform import *
class Rig:
    def __init__(self):
        self.rig=json.load(open(f'{P}/seosoyoung-rig.json')); self.meshes=sorted(self.rig['meshes'],key=lambda m:m['drawOrder'])
        for m in self.meshes:
            im=np.asarray(Image.open(f"{P}/{m['file']}").convert('RGBA')).astype(np.float32)
            m['img']=im; m['pre']=np.concatenate([im[...,:3]*(im[...,3:4]/255),im[...,3:4]],-1)   # CPU 비교 렌더 전용. 배포 아틀라스는 straight RGBA를 그대로 쓴다
            m['pos']=np.array(m['positions'],float); m['tri']=np.array(m['triangles'],np.int32).reshape(-1,3); m['D']={k:np.array(v,float) for k,v in m['deltas'].items()}
        self.inv={tuple(v):k for k,v in self.rig['keys'].items()}
    def delta(self,m,u,v):
        wu={-1:max(0,-u),0:1-abs(u),1:max(0,u)}; wv={-1:max(0,-v),0:1-abs(v),1:max(0,v)}; d=np.zeros_like(m['pos'])
        for i,a in wu.items():
            for j,b in wv.items():
                if a*b>0: d+=a*b*m['D'][self.inv[(i,j)]]
        return d
    def eye_matrix(self,m,u,v):
        """눈 묶음의 방향 변환: 네 이웃 키의 2×3 아핀을 쌍선형 가중치로 섞는다(정면은 항등)."""
        wu={-1:max(0,-u),0:1-abs(u),1:max(0,u)}; wv={-1:max(0,-v),0:1-abs(v),1:max(0,v)}; I=np.array([[1,0,0],[0,1,0]],float); M=np.zeros((2,3))
        for i,a in wu.items():
            for j,b in wv.items():
                if a*b>0:
                    key=self.inv[(i,j)]; M+=a*b*(I if key=='front' else np.array(m['transforms'][key],float))
        return M
    def positions(self,m,p):
        pos=m['pos'].copy(); pid=m['id']; g=m['group']
        if g=='eye':
            # 깜빡임과 시선은 정면(쉬는) 좌표에서 먼저 하고, 그 결과에 방향별 눈 변환 M(u,v)(네 이웃 키의 아핀을 쌍선형으로 섞음)을 건다.
            # 정면 변위를 나중에 더하면 먼 눈의 세로 배율이 닫힘에 안 걸려 메시가 뒤집혔다(세라피엘 2판 결함 1~3).
            side='left' if ('-left' in pid) else 'right'; yc=self.rig['eyeClosedY'][side]; op=p['openL'] if side=='left' else p['openR']; kind=pid.split('-')[-1]
            if not pid.startswith('closed'):
                s=min(1.0,(1-op)/(1-self.rig['blinkSquashEnd']))                        # 7판: 0.29에서 1이 되어 감은 눈 부품이 켜지기 전에 자리가 이어진다
                bc=self.rig['blinkCurves'][side]; xi=np.clip(np.round(pos[:,0]).astype(int)-bc['x0'],0,len(bc['top'])-1)
                C=lambda k:np.array(bc[k])[xi]; closed=C('closed')
                if kind in ('lashes','line'):                                              # 5판: 열마다 파츠의 세로 범위[t0,b0]를 감은 눈 부품의 범위[ct,cb]로 옮긴다(s=1에서 두께·모양이 감은 부품과 같다 → 단차·눈꼬리 갈고리 없음)
                    t0,b0=(C('lashTop'),C('lashBot')) if kind=='lashes' else (C('lowerTop'),C('lowerBot')); ct,cb=C('closedTop'),C('closedBot')
                    k=np.clip((cb-ct)/np.maximum(b0-t0,1.0),0,1)                           # 감은 범위가 더 얇은 열만 누른다(1 이하). 더 두꺼운 열(속눈썹 끝, 아래 선)은 중심만 옮긴다
                    y1=(ct+cb)/2+(pos[:,1]-(t0+b0)/2)*k; pos[:,1]+=s*(y1-pos[:,1])
                else:
                    pos[:,1]=closed+(pos[:,1]-closed)*(1-s)                                # 흰자(자르기 기준)·홍채·동공·반사광은 열마다 감은 선 기준 세로 축소
                    if kind in ('iris','pupil','highlight'): pos+=[p.get('gazeX',0)*self.rig['gazePx'],p.get('gazeY',0)*self.rig['gazePx']]
            pos=apply23(self.eye_matrix(m,p['u'],p['v']),pos)
        else: pos=pos+self.delta(m,p['u'],p['v'])
        if p.get('roll') and g!='body':
            c=np.array(self.rig['neckPivot'],float); a=math.radians(p['roll']); Rm=np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])
            w=np.clip((NECK_RAMP[1]-m['pos'][:,1])/(NECK_RAMP[1]-NECK_RAMP[0]),0,1) if g in ('neck','fill') else np.ones(len(pos)); rot=(pos-c)@Rm.T+c; pos=pos+(rot-pos)*w[:,None]
        return pos
    def opacity(self,m,p):
        pid=m['id']; u=p['u']
        if (pid in self.rig['hair'] or pid.startswith('still-fill-hair')) and not p.get('hair',True): return 0.0
        if pid in self.rig['bareOnlyParts'] and p.get('hair',True): return 0.0
        a,b=self.rig['blinkFade']
        if m['group']=='eye' and not pid.startswith('closed'):              # 5판: 0.30→0.22 교차 페이드(눈 파츠 1→0)
            op=p['openL'] if '-left-' in pid else p['openR']; return float(np.clip((op-a)/(b-a),0,1))
        if pid.startswith('closed-eyes'):                                   # 7판: 감은 눈은 0.29→0.27에서 0→1로 페이드한다
            op=p['openL'] if pid.endswith('left') else p['openR']; c0,c1=self.rig['blinkClosedFade']; return float(1-np.clip((op-c0)/(c1-c0),0,1))
        return 1.0
    def order(self,p):
        """(순서, 메시, 불투명도 배율). 먼 쪽 묶음머리·옆머리는 farEarFade 구간에서 앞(원래 순서)과 뒤(95)에 같이 그려 섞는다."""
        u=p['u']; out=[]; a,b=self.rig['farEarFade']
        for m in self.meshes:
            o=m['drawOrder']; pid=m['id']; far=('-left' in pid and u<0) or ('-right' in pid and u>0)
            if (pid.startswith('tail-') or pid.startswith('side-hair')) and far:
                t=float(np.clip((abs(u)-a)/(b-a),0,1)); t=t*t*(3-2*t)
                if t>0: out.append((95,m,1.0,None))      # 뒤 사본은 늘 불투명
                if t<1: out.append((o,m,1.0,t))          # 5판: 앞 사본도 불투명. t>0이면 얼굴 다각형 안쪽을 깊이 e(t)부터 닫는 마스크(farWipe)로 잘린다
            else: out.append((o,m,1.0,None))
        return sorted(out,key=lambda t:t[0])
    def hair_clip_mask(self,p,hairtop):
        """5판 앞머리 자르기 마스크: 머리선(얼굴 다각형의 hl 5점) 위에서만 (실루엣 채움 ∪ 윗머리 알파/hairClipAlpha)로 자른다. 머리선 아래는 1. 위 방향에서 앞머리 꼭대기가 윗머리 위로 솟지 않게."""
        m=fill_mask([self.curve('silhouette',p)]).astype(np.float32)/255
        if hairtop is not None:
            ha,(mx,my)=hairtop; full=np.zeros((1536,1024),np.float32); full[my:my+ha.shape[0],mx:mx+ha.shape[1]]=ha/255; m=np.maximum(m,np.clip(full/self.rig['hairClipAlpha'],0,1))
        fp=self.curve('facePolygon',p); y_hl=float(fp[:5*SIL_SUBDIV,1].max()); m[int(math.ceil(y_hl)):]=1.0
        return m
    def face_depth(self,p):
        """얼굴 다각형 안쪽 깊이(px, 바깥 0): 5판 farWipe용."""
        m=fill_mask([self.curve('facePolygon',p)]); return ndi.distance_transform_edt(m>0).astype(np.float32)
    def far_wipe_mask(self,p,side,t):
        """먼 쪽 머리카락 앞 사본의 보이는 비율(0..1). 깊이 D가 e(t) 아래면 1, e(t)+F 위면 0. e(t)는 farWipeDepth[side](섞임 시작에서 머리카락이 얼굴 안에 들어간 최대 깊이)에서 −F까지 t에 따라 선형."""
        D=self.face_depth(p); F=self.rig['farWipeFeatherPx']; E0=self.rig['farWipeDepth'][side]; e=E0+(-F-E0)*t
        return np.clip((e+F-D)/F,0,1).astype(np.float32)
    def warp(self,m,pos):
        """화소 중심 +0.5 표본을 만들고 겹친 삼각형을 GPU와 같은 순서로 over 합성한다."""
        x,y,w,h=m['bbox']; src=m['pos']-[x,y]; tri=m['tri']
        x0=max(int(math.floor(pos[:,0].min()))-1,0); y0=max(int(math.floor(pos[:,1].min()))-1,0); x1=min(int(math.ceil(pos[:,0].max()))+2,1024); y1=min(int(math.ceil(pos[:,1].max()))+2,1536)
        if x1<=x0 or y1<=y0: return None
        W,Hh=x1-x0,y1-y0; idx=np.full((Hh,W),-1,np.int32); A=np.zeros((len(tri),2,3),np.float64); yy,xx=np.mgrid[0:Hh,0:W]; cx=xx+0.5; cy=yy+0.5; overlaps=[]
        for t,(a,b,c) in enumerate(tri):
            d=np.array([pos[a],pos[b],pos[c]],np.float64)-[x0,y0]; s=np.array([src[a],src[b],src[c]],np.float64)
            Mi=cv2.getAffineTransform(d.astype(np.float32),s.astype(np.float32))
            if Mi is None: continue
            A[t]=Mi; bx0=max(int(math.floor(d[:,0].min())),0); bx1=min(int(math.ceil(d[:,0].max()))+1,W); by0=max(int(math.floor(d[:,1].min())),0); by1=min(int(math.ceil(d[:,1].max()))+1,Hh)
            if bx1<=bx0 or by1<=by0: continue
            px=cx[by0:by1,bx0:bx1]; py=cy[by0:by1,bx0:bx1]
            if (d[1,0]-d[0,0])*(d[2,1]-d[0,1])-(d[1,1]-d[0,1])*(d[2,0]-d[0,0])<0: d=d[[0,2,1]]
            def edge(p,q): return (q[0]-p[0])*(py-p[1])-(q[1]-p[1])*(px-p[0])
            def top_left(p,q): return (q[1]-p[1])<0 or ((q[1]-p[1])==0 and (q[0]-p[0])>0)
            edges=[(edge(d[i],d[(i+1)%3]),top_left(d[i],d[(i+1)%3])) for i in range(3)]
            inside=np.ones(px.shape,bool)
            for value,is_top_left in edges: inside&=(value>0)|((value==0)&is_top_left)
            sub=idx[by0:by1,bx0:bx1]; again=inside&(sub>=0)
            if again.any():
                oy,ox=np.nonzero(again); overlaps.append((t,(oy+by0,ox+bx0)))
            sub[inside&(sub<0)]=t
        ok=idx>=0; Tn=np.where(ok,idx,0)
        mapx=(A[Tn,0,0]*cx+A[Tn,0,1]*cy+A[Tn,0,2]-0.5).astype(np.float32); mapy=(A[Tn,1,0]*cx+A[Tn,1,1]*cy+A[Tn,1,2]-0.5).astype(np.float32)
        out=cv2.remap(m['pre'],mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))   # 미리 곱한 알파로 보간
        out[~ok]=0
        for t,(oy,ox) in overlaps:
            px_=ox+0.5; py_=oy+0.5
            mx=(A[t,0,0]*px_+A[t,0,1]*py_+A[t,0,2]-0.5).astype(np.float32)[None]
            my=(A[t,1,0]*px_+A[t,1,1]*py_+A[t,1,2]-0.5).astype(np.float32)[None]
            source=cv2.remap(m['pre'],mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))[0]
            destination=out[oy,ox]
            out[oy,ox]=source+destination*(1-source[:,3:4]/255)
        m['overlap_px']=sum(len(indices[0]) for _,indices in overlaps)
        return out,(x0,y0)
    def curve(self,name,p):
        """실루엣(name='silhouette') 또는 얼굴 다각형('facePolygon') 곡선의 현재 자리: 9방향 자리를 사분면 쌍선형으로 섞는다."""
        wu={-1:max(0,-p['u']),0:1-abs(p['u']),1:max(0,p['u'])}; wv={-1:max(0,-p['v']),0:1-abs(p['v']),1:max(0,p['v'])}; C=self.rig[name]; out=None
        for i,a_ in wu.items():
            for j,b_ in wv.items():
                if a_*b_>0:
                    q=a_*b_*np.array(C[self.inv[(i,j)]],float); out=q if out is None else out+q
        return out
    def clip_mask(self,p):
        """6판 얼굴 자르기 마스크: 실루엣 채움과 clipMarginPx+0.5px 띠 삼각형의 합집합."""
        m=self.rig['clipMarginPx']+self.rig['maskRasterMarginPx']; return fill_dilated(self.curve('silhouette',p),m).astype(np.float32)
    def silhouette_layer(self,p):
        """6판 끔 받침: 실루엣 안은 불투명, 바깥 1px 띠는 부호 거리 알파."""
        a=signed_edge_alpha(self.curve('silhouette',p))*255; rgb=np.array(self.rig['skinColor'],np.float32)
        out=np.zeros((1536,1024,4),np.float32); out[...,:3]=rgb*(a/255)[...,None]; out[...,3]=a; return out
    def render(self,p):
        canvas=np.zeros((1536,1024,4),np.float32); masks={}; face_clip=None; hair_clip=None; wipe={}
        for _,m,mul,wipe_t in self.order(p):
            op=self.opacity(m,p)*mul
            if op<=0: continue
            if m['id']=='skull-plate':      # 끔 받침은 실루엣 곡선 채움(파츠 그림 대신). 뷰어는 실루엣 부채꼴 메시
                layer=self.silhouette_layer(p); a=layer[...,3:4]/255; canvas[...]=layer+canvas*(1-a); continue
            r=self.warp(m,self.positions(m,p))
            if r is None: continue
            layer,(x0,y0)=r; h,w=layer.shape[:2]
            if m['id'] in self.rig['clipParts']:                       # 얼굴 피부·이마 덮개·얼굴 받침: 실루엣(바깥으로 clipMarginPx) 마스크
                if face_clip is None: face_clip=self.clip_mask(p)
                layer=layer*face_clip[y0:y0+h,x0:x0+w][...,None]
            if m['id'] in self.rig['hairClipParts']:                   # 5판 앞머리·머리핀: (실루엣 채움 ∪ 윗머리 알파≥hairClipAlpha) 마스크. 위 방향에서 앞머리가 윗머리 위로 솟지 않는다
                if hair_clip is None: hair_clip=self.hair_clip_mask(p,masks.get('hair-top'))
                layer=layer*hair_clip[y0:y0+h,x0:x0+w][...,None]
            if wipe_t is not None and wipe_t>0:                         # 5판 farWipe: 먼 쪽 머리카락 앞 사본을 얼굴 안쪽에서 닫는다
                side=side_of(m['id'])
                if side not in wipe: wipe[side]=self.far_wipe_mask(p,side,wipe_t)
                layer=layer*wipe[side][y0:y0+h,x0:x0+w][...,None]
            if m['id'].endswith('sclera') or m['id']=='hair-top': masks[m['id']]=(layer[...,3].copy(),(x0,y0))
            clip_name=self.rig['clips'].get(m['id'])
            if clip_name and clip_name in masks:
                ma,(mx,my)=masks[clip_name]; full=np.zeros((1536,1024),np.float32); full[my:my+ma.shape[0],mx:mx+ma.shape[1]]=ma; layer=layer*(full[y0:y0+h,x0:x0+w]/255)[...,None]
            layer=layer*op; a=layer[...,3:4]/255; dst=canvas[y0:y0+h,x0:x0+w]; dst[...]=layer+dst*(1-a)        # 미리 곱한 over
        a=canvas[...,3:4]/255; a=np.where(a>=254.5/255,1.0,a); rgb=np.where(a>0,canvas[...,:3]/np.maximum(a,1e-6),0)   # 불투명은 나누지 않는다(반올림 오차 방지)
        return np.clip(np.concatenate([rgb,np.minimum(canvas[...,3:4],255)],-1)+0.5,0,255).astype(np.uint8)
def pose(u=0,v=0,openL=1,openR=1,hair=True,roll=0,gazeX=0,gazeY=0): return dict(u=u,v=v,openL=openL,openR=openR,hair=hair,roll=roll,gazeX=gazeX,gazeY=gazeY)
def holes(rgba):
    """머리카락을 뺀 단단한 부분의 실루엣 안쪽 구멍(배경이 비치는 화소) 수. 알파≥128을 실루엣으로 보고 채운 뒤 뺀다. 1~2px 잡티는 제외.
    머리카락 사이 틈은 원화에도 있으므로 세지 않는다: 호출하는 쪽이 머리카락 끔 렌더를 준다."""
    m=rgba[...,3]>=128; filled=ndi.binary_fill_holes(m); h=filled&~m; h[700:]=False      # 머리·목 영역(y<700)만. 몸 쪽 틈은 원화에도 있다
    lab,n=ndi.label(h); sizes=ndi.sum(h,lab,range(1,n+1)) if n else []
    big=[int(s) for s in sizes if s>=3]; return int(sum(big)),len(big)
