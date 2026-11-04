#!/usr/bin/env python3
"""원화 그림의 파츠 알파에서 기준점을 다시 추출한다."""
import json, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
ROOT=Path(__file__).resolve().parents[1]
HERE=str(ROOT); K=ROOT/'source/keyshapes'; R=str(ROOT)
K.mkdir(parents=True,exist_ok=True)
(K/'check').mkdir(exist_ok=True)
parts={p['id']:p for p in json.load(open(f'{R}/parts/parts_index.json'))}
def alpha(pid):
    p=parts[pid]; x,y,w,h=p['bbox']; a=np.zeros((1536,1024),bool)
    a[y:y+h,x:x+w]=np.asarray(Image.open(f"{R}/{p['file']}").convert('RGBA'))[...,3]>127
    return a
pts={}
# 눈: 흰자 파츠(눈 개구부)의 윤곽
for s in 'lr':
    name='left' if s=='l' else 'right'
    a=alpha(f'eye-{name}-sclera'); a=ndi.binary_fill_holes(a)
    ys,xs=np.nonzero(a); xl,xr=xs.min(),xs.max()
    outer,inner=(xl,xr) if s=='l' else (xr,xl)
    pts[f'eye_{s}_out']=(int(outer),int(np.median(ys[xs==outer]))); pts[f'eye_{s}_in']=(int(inner),int(np.median(ys[xs==inner])))
    for k,t in [('1',.25),('2',.5),('3',.75)]:
        x=int(xl+(xr-xl)*t); col=np.nonzero(a[:,x])[0]; pts[f'eye_{s}_up{k}']=(x,int(col.min())); pts[f'eye_{s}_lo{k}']=(x,int(col.max()))
    ir=alpha(f'eye-{name}-iris'); iy,ix=np.nonzero(ir); pts[f'eye_{s}_iris']=(int(ix.mean()),int(iy.mean()))
    b=alpha(f'eye-{name}-brow'); by,bx=np.nonzero(b); bl,br=bx.min(),bx.max(); top=by.min(); xa=int(np.median(bx[by==top]))
    a_x,b_x=(bl,br) if s=='l' else (br,bl)
    pts[f'brow_{s}_a']=(int(a_x),int(np.median(by[bx==a_x]))); pts[f'brow_{s}_arch']=(xa,int(top)); pts[f'brow_{s}_b']=(int(b_x),int(np.median(by[bx==b_x])))
    e=alpha(f'ear-{name}'); ey,ex=np.nonzero(e)
    pts[f'ear_{s}_c']=(int(ex.mean()),int(ey.mean()))
    topy=ey.min(); pts[f'ear_{s}_top']=(int(np.median(ex[ey==topy])),int(topy)); boty=ey.max(); pts[f'ear_{s}_bot']=(int(np.median(ex[ey==boty])),int(boty))
    bk=ex.min() if s=='l' else ex.max(); pts[f'ear_{s}_back']=(int(bk),int(np.median(ey[ex==bk])))
n=alpha('nose-detail'); ny,nx=np.nonzero(n); pts['nose']=(int(nx.mean()),int(ny.max()))
m=alpha('mouth-detail'); my,mx=np.nonzero(m); pts['mouth_l']=(int(mx.min()),int(np.median(my[mx==mx.min()]))); pts['mouth_r']=(int(mx.max()),int(np.median(my[mx==mx.max()]))); pts['mouth_c']=(int(mx.mean()),int(my.mean()))
# 얼굴 윤곽: face-skin 알파. 턱 끝 = 중앙 열의 최하단(목 제외: 좌우 폭이 급히 넓어지기 전)
f=alpha('face-skin')
# 피부 파츠는 목까지 이어진다. 윤곽은 원화에서 보이는 볼·턱선: 머리카락 아래(y>=465)에서만 잰다.
for lvl,y in [('mouth',555),('jaw',590)]:
    xs=np.nonzero(f[y])[0]; pts[f'ct_l_{lvl}']=(int(xs.min()),y); pts[f'ct_r_{lvl}']=(int(xs.max()),y)
# 턱 끝: still.png 알파 기준으로 중앙 열에서 턱선(피부 끝) — face-skin 알파가 목까지 가므로 원화를 눈으로 잰 값 사용
pts['chin']=(502,616)
pts['nose']=(507,497)
json.dump({k:[int(v[0]),int(v[1])] for k,v in pts.items()},open(K/'points_art_auto.json','w'),ensure_ascii=False,indent=0)
im=Image.open(f'{R}/source/character/still.png').convert('RGB').crop((250,200,780,700)).resize((1060,1000),Image.LANCZOS); dr=ImageDraw.Draw(im)
for k,(x,y) in pts.items():
    X,Y=(x-250)*2,(y-200)*2; dr.ellipse([X-3,Y-3,X+3,Y+3],outline=(255,0,255),width=2); dr.text((X+4,Y-10),k,fill=(0,0,255))
for g in range(260,780,20):
    X=(g-250)*2; dr.line([(X,0),(X,1000)],fill=(200,200,255),width=1)
    if g%100==0: dr.text((X+2,2),str(g),fill=(0,0,200))
for g in range(200,700,20):
    Y=(g-200)*2; dr.line([(0,Y),(1060,Y)],fill=(200,200,255),width=1)
    if g%100==0: dr.text((2,Y+2),str(g),fill=(0,0,200))
im.save(K/'check/art_points_auto.png'); print(json.dumps(pts))
