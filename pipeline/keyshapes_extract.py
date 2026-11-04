#!/usr/bin/env python3
"""저장된 대표 프레임에서 방향별 제어점을 다시 추출한다."""
import json, math, sys, os, copy
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
ROOT=Path(__file__).resolve().parents[1]
K5=str(ROOT/'source/keyshapes')
FR=str(ROOT/'source/references/seedance_frames640/frame_%04d.png')
REF=json.load(open(f'{K5}/seedance-reference-frames.json'))
# v2: 오른쪽을 240번으로 바꾼다(담당 결정). 랜드마크는 60번 것을 유사변환 없이 그대로 두되 눈·코·입·귀 점은 자동 추출이 다시 찾는다.
import copy
_f240=copy.deepcopy(next(f for f in REF['frames'] if f['frame']==60)); _f240['frame']=240
# 240번의 9점(눈·눈썹·코·입·턱·귀)은 손으로 읽은 값(640 좌표)에 1.125를 곱해 둔다(아래 overrides와 같은 기준)
_f240['landmarks']=[[round(x*1.125),round(y*1.125)] for x,y in [[348,362],[493,360],[330,290],[490,286],[430,408],[443,463],[383,520],[205,395],[480,395]]]
# v2: 왼쪽 아래를 309번으로 바꾼다(담당 결정: 오른쪽 아래 254번과 위아래 각도 짝 맞춤. 298번은 −32°, 309번은 −17°)
_f309=copy.deepcopy(next(f for f in REF['frames'] if f['frame']==298)); _f309['frame']=309
_f309['landmarks']=[[round(x*1.125),round(y*1.125)] for x,y in [[148,368],[280,370],[135,301],[260,300],[172,442],[198,483],[215,530],[105,368],[405,385]]]
REF['frames']=[f for f in REF['frames'] if f['frame'] not in (60,298)]+[_f240,_f309]
RED=json.load(open(f'{K5}/director_red_frame_coords.json'))
OVR=json.load(open(f'{K5}/overrides.json')) if len(sys.argv)>1 and sys.argv[1]=='--with-overrides' else {}

def masks(im):
    rgb=im.astype(int); s=rgb.sum(-1)
    bg=(np.abs(rgb-[250,243,241]).max(-1)<14)|(rgb.min(-1)>=252)
    sil=~bg; sil=ndi.binary_fill_holes(sil)
    lab,n=ndi.label(sil)
    if n>1:
        sizes=ndi.sum(sil,lab,range(1,n+1)); sil=lab==(1+int(np.argmax(sizes)))
    hair=(s<330)&sil
    skin=(rgb[...,0]>200)&(rgb[...,0]-rgb[...,2]>25)&(rgb[...,0]-rgb[...,2]<110)&sil   # 살색: R 높고 B 낮음
    mx0=rgb.max(-1);mn0=rgb.min(-1)
    white=(mn0>200)&((mx0-mn0)<40)&sil&~bg
    mx=rgb.max(-1);mn=rgb.min(-1); sat=(mx-mn)/np.maximum(mx,1)
    iris=(sat>0.45)&(mx>120)&sil&~hair
    return dict(bg=bg,sil=sil,hair=hair,skin=skin,white=white,iris=iris)

def ray(mask,cx,cy,ang,rmax=400,want=True):
    """(cx,cy)에서 ang(도, 화면 기준 x→, y↓, 각도는 수학 좌표계: 90=위)로 나가며 mask가 want에서 not want로 바뀌는 첫 점"""
    dx,dy=math.cos(math.radians(ang)),-math.sin(math.radians(ang))
    last=None
    for r in range(0,rmax):
        x=int(round(cx+dx*r)); y=int(round(cy+dy*r))
        if not(0<=x<mask.shape[1] and 0<=y<mask.shape[0]): return last
        if bool(mask[y,x])==want: last=(x,y)
        else:
            if last is not None: return last
    return last

def eye_points(m,c,side,w=58):
    cx,cy=c; x0,x1,y0,y1=cx-w,cx+w,cy-40,cy+36
    rgb=m['rgb']; sub=slice(y0,y1),slice(x0,x1)
    s=rgb[sub].astype(int).sum(-1)
    dark=(s<300)   # 속눈썹·눈꺼풀 선
    region=(m['white'][sub]|m['iris'][sub]|dark)
    region=ndi.binary_closing(region,iterations=4); region=ndi.binary_fill_holes(region)
    lab,n=ndi.label(region)
    if n==0: return {}
    # 눈 중심을 품은 성분
    l=lab[cy-y0,cx-x0]
    if l==0:
        d=ndi.distance_transform_edt(lab==0,return_indices=True)[1]; l=lab[d[0][cy-y0,cx-x0],d[1][cy-y0,cx-x0]]
    reg=lab==l
    ys,xs=np.nonzero(reg)
    xl,xr=xs.min(),xs.max()
    pts={}
    pts[f'eye_{side}_out' if side=='l' else f'eye_{side}_in']=(x0+xl,y0+int(np.median(ys[xs==xl])))
    pts[f'eye_{side}_in' if side=='l' else f'eye_{side}_out']=(x0+xr,y0+int(np.median(ys[xs==xr])))
    for k,t in [('1',.25),('2',.5),('3',.75)]:
        x=int(xl+(xr-xl)*t); col=np.nonzero(reg[:,x])[0]
        pts[f'eye_{side}_up{k}']=(x0+x,y0+int(col.min())); pts[f'eye_{side}_lo{k}']=(x0+x,y0+int(col.max()))
    ir=m['iris'][sub]&reg
    if ir.sum()>20:
        iy,ix=np.nonzero(ir); pts[f'eye_{side}_iris']=(x0+int(ix.mean()),y0+int(iy.mean()))
    return pts

def brow_points(m,eye_top,side):
    cx,cy=eye_top; x0,x1,y0,y1=max(0,cx-70),min(639,cx+70),max(0,cy-75),cy-8
    rgb=m['rgb'][y0:y1,x0:x1].astype(int); s=rgb.sum(-1)
    brown=(s<470)&(s>150)&((rgb[...,0]-rgb[...,2])>20)
    hairD=ndi.binary_dilation(m['hair'][y0:y1,x0:x1],iterations=4)
    brown&=~hairD
    lab,n=ndi.label(brown)
    best=None;bw=0
    for i in range(1,n+1):
        reg=lab==i; ys,xs=np.nonzero(reg); w=xs.max()-xs.min(); h=ys.max()-ys.min()
        if w<25 or h>40: continue
        if w>bw: bw=w;best=reg
    if best is None: return {}
    reg=best; ys,xs=np.nonzero(reg); xl,xr=xs.min(),xs.max()
    top=int(ys.min()); xa=int(np.median(xs[ys==top]))
    return {f'brow_{side}_a':(x0+xl,y0+int(np.median(ys[xs==xl]))),f'brow_{side}_arch':(x0+xa,y0+top),f'brow_{side}_b':(x0+xr,y0+int(np.median(ys[xs==xr])))}

def mouth_points(m,c):
    cx,cy=c; x0,x1,y0,y1=cx-45,cx+45,cy-18,cy+18
    rgb=m['rgb'][y0:y1,x0:x1].astype(int)
    dark=(rgb.sum(-1)<560)&(rgb[...,0]-rgb[...,2]>40)
    lab,n=ndi.label(dark)
    if n==0: return {'mouth_c':c}
    sizes=ndi.sum(dark,lab,range(1,n+1)); reg=lab==(1+int(np.argmax(sizes)))
    ys,xs=np.nonzero(reg); ym=int(np.median(ys))
    band=dark.copy(); band[:max(0,ym-5)]=False; band[ym+6:]=False          # 같은 높이의 어두운 화소 전부(끊긴 입 선 오른쪽 끝까지)
    ys,xs=np.nonzero(band); xl,xr=xs.min(),xs.max()
    return {'mouth_l':(x0+xl,y0+int(np.median(ys[xs==xl]))),'mouth_c':(x0+int((xl+xr)/2),y0+ym),'mouth_r':(x0+xr,y0+int(np.median(ys[xs==xr])))}

def frame_points(f):
    im=np.asarray(Image.open(FR%f['frame']).convert('RGB'))
    m=masks(im); m['rgb']=im
    L=[(round(x/1.125),round(y/1.125)) for x,y in f['landmarks']]
    eL,eR,bL,bR,nose,mouth,chin,earL,earR=L
    pts={}
    # 두상 중심: 두 눈 중점에서 위로. 반지름은 실루엣 폭으로
    ex,ey=(eL[0]+eR[0])//2,(eL[1]+eR[1])//2
    row=m['sil'][ey]; xs=np.nonzero(row)[0]; half=(xs.max()-xs.min())/2 if len(xs) else 200
    hc=(int((xs.min()+xs.max())/2) if len(xs) else ex, ey-int(half*0.45))
    pts['head_c']=hc
    for i,ang in enumerate(range(180,-15,-15)):
        p=ray(m['sil'],hc[0],hc[1],ang)
        if p: pts[f'sk{i:02d}']=p
    # 머리선: 눈썹 위에서 위로 쏘아 머리카락(어두움) 첫 화소
    for name,x in [('hl_l',bL[0]-20),('hl_lm',bL[0]+ (bR[0]-bL[0])//4),('hl_c',(bL[0]+bR[0])//2),('hl_rm',bR[0]-(bR[0]-bL[0])//4),('hl_r',bR[0]+20)]:
        x=int(np.clip(x,0,639)); y=min(bL[1],bR[1])-14
        yy=y
        while yy>0 and not m['hair'][yy,x] and m['sil'][yy,x]: yy-=1
        pts[name]=(x,yy)
    # 얼굴 윤곽: 실루엣 경계(먼 쪽) 또는 피부 경계. 코·입·눈 높이에서 좌우로
    far='l' if 'left' in f['name'] else 'r' if 'right' in f['name'] else None
    s_all=im.astype(int).sum(-1)
    for lvl,y in [('brow',min(bL[1],bR[1])),('eye',ey),('nose',nose[1]),('mouth',mouth[1]),('jaw',(mouth[1]+chin[1])//2)]:
        for side in ['l','r']:
            row=m['sil'][y]; xs=np.nonzero(row)[0]
            if not len(xs): continue
            if side==far or (far is None and lvl in ('mouth','jaw')):
                edge=xs.min() if side=='l' else xs.max(); step=1 if side=='l' else -1
                x=edge
                while 0<=x<640 and not m['skin'][y,x]: x+=step
                pts[f'ct_{side}_{lvl}']=(x,y)
            else:
                # 눈 바깥쪽에서 귀 중심 쪽으로 가며 가장 어두운 골(귀 앞 주름 또는 피부 끝)
                ear=earL if side=='l' else earR
                a=min(eL[0],ear[0]) if side=='l' else max(eR[0],ear[0]); b=ear[0] if side=='l' else ear[0]
                lo,hi=(ear[0],eL[0]-20) if side=='l' else (eR[0]+20,ear[0])
                lo=max(0,lo);hi=min(639,hi)
                if hi<=lo: continue
                seg=s_all[y,lo:hi]; skinline=np.where(m['skin'][y,lo:hi],seg,9999)
                # 피부가 끝나는(실루엣 밖 또는 머리카락) 첫 지점 또는 가장 어두운 피부 골
                k=int(np.argmin(seg))
                pts[f'ct_{side}_{lvl}']=(lo+k,y)
    pts['chin']=chin; pts['nose']=nose
    pts.update(eye_points(m,eL,'l')); pts.update(eye_points(m,eR,'r'))
    for s_,c in (('l',eL),('r',eR)):                                       # 아래 선 점은 홍채 바닥보다 아래여야 한다(세라피엘 2판 검증: 34, 89, 254에서 홍채 안으로 올라감)
        for k in '123':
            key=f'eye_{s_}_lo{k}'
            if key not in pts: continue
            x_,y_=pts[key]; col=m['iris'][c[1]-40:c[1]+40,max(0,x_-2):x_+3]; ys_=np.nonzero(col)[0]     # 그 열(±2)의 홍채 바닥
            if len(ys_):
                ib=c[1]-40+int(ys_.max())
                if y_<ib+3: pts[key]=(x_,ib+3)
    if 'eye_l_up2' in pts: pts.update(brow_points(m,pts['eye_l_up2'],'l'))
    if 'eye_r_up2' in pts: pts.update(brow_points(m,pts['eye_r_up2'],'r'))
    pts.update(mouth_points(m,mouth))
    pts['ear_l_c']=earL; pts['ear_r_c']=earR
    # 어깨선: 아래쪽에서 실루엣 폭이 급히 넓어지는 y
    widths=[np.nonzero(m['sil'][y])[0] for y in range(chin[1],640)]
    pts['neck_base']=(int(np.nonzero(m['sil'][min(639,chin[1]+40)])[0].mean()) if m['sil'][min(639,chin[1]+40)].any() else ex, min(639,chin[1]+40))
    red=RED.get(str(f['frame']),{}).get('overlay',[])
    if red:                                                                # 눈·코·입 높이의 윤곽 점은 디렉터 선(±2행, 15px 안)에 맞춘다
        red=np.array(red)
        for k in list(pts):
            if k.startswith('ct_') and k.split('_')[-1] in ('eye','nose','mouth'):
                x,y=pts[k]; cand=red[np.abs(red[:,1]-y)<=2]
                if len(cand):
                    j=int(np.argmin(np.abs(cand[:,0]-x)))
                    if abs(cand[j,0]-x)<=15: pts[k]=(int(cand[j,0]),y)
    for k,v in OVR.get(str(f['frame']),{}).items():
        if v is None: pts.pop(k,None)
        else: pts[k]=tuple(v)
    return {k:[int(v[0]),int(v[1])] for k,v in pts.items()}, m

def draw_check(f,pts,m,out):
    im=Image.open(FR%f['frame']).convert('RGB').resize((1280,1280),Image.LANCZOS)
    dr=ImageDraw.Draw(im)
    for g in range(0,640,20):
        col=(200,200,255) if g%100 else (120,120,255)
        dr.line([(g*2,0),(g*2,1280)],fill=col,width=1); dr.line([(0,g*2),(1280,g*2)],fill=col,width=1)
        if g%100==0: dr.text((g*2+2,2),str(g),fill=(0,0,200)); dr.text((2,g*2+2),str(g),fill=(0,0,200))
    for x,y in RED.get(str(f['frame']),{}).get('overlay',[]): dr.point((x*2,y*2),fill=(255,0,0)); dr.point((x*2+1,y*2),fill=(255,0,0))
    colors={'sk':(0,200,0),'hl':(255,140,0),'ct':(0,120,255),'eye':(255,0,255),'brow':(160,80,0),'mouth':(255,0,0),'ear':(0,160,160)}
    for k,(x,y) in pts.items():
        c=next((v for p,v in colors.items() if k.startswith(p)),(0,0,0))
        dr.ellipse([x*2-3,y*2-3,x*2+3,y*2+3],outline=c,width=2); dr.text((x*2+4,y*2-10),k,fill=c)
    im.save(out)

if __name__=='__main__':
    import os; os.makedirs(f'{K5}/check',exist_ok=True)
    allp={}
    for f in REF['frames']:
        pts,m=frame_points(f); allp[str(f['frame'])]=dict(name=f['name'],points=pts)
        draw_check(f,pts,m,f"{K5}/check/{f['frame']:04d}_{f['name']}_auto.png")
        print(f['frame'],f['name'],len(pts),'points')
    out='points_video.json' if OVR else 'points_video_auto.json'
    json.dump(allp,open(f'{K5}/{out}','w'),ensure_ascii=False,indent=0)
