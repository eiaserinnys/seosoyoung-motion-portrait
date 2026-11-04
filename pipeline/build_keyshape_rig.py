#!/usr/bin/env python3
"""원화 자료에서 방향별 얼굴 화면의 데이터를 다시 만든다."""
import json, math, os
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from keyshape_rig_parts import *
from keyshape_rig_deform import *
from keyshape_rig_runtime import Rig, pose
SIZE = 2048

def pack(items,size=SIZE):
    """가장자리 여유를 두고 크기 조정 없이 사각형을 배치한다."""
    for order in [lambda a:a[1]*a[2],lambda a:max(a[1:]),lambda a:a[2],lambda a:a[1]]:
        free=[(0,0,size,size)]; result={}
        for key,w,h in sorted(items,key=order,reverse=True):
            ww,hh=w+8,h+8
            candidates=[(min(fw-ww,fh-hh),max(fw-ww,fh-hh),x,y) for x,y,fw,fh in free if ww<=fw and hh<=fh]
            if not candidates: break
            _,_,x,y=min(candidates); result[key]=[x+4,y+4,w,h]
            remaining=[]
            for fx,fy,fw,fh in free:
                if x>=fx+fw or x+ww<=fx or y>=fy+fh or y+hh<=fy:
                    remaining.append((fx,fy,fw,fh));continue
                if x>fx: remaining.append((fx,fy,x-fx,fh))
                if x+ww<fx+fw: remaining.append((x+ww,fy,fx+fw-x-ww,fh))
                if y>fy: remaining.append((fx,fy,fw,y-fy))
                if y+hh<fy+fh: remaining.append((fx,y+hh,fw,fy+fh-y-hh))
            free=[a for a in set(remaining) if not any(a!=b and a[0]>=b[0] and a[1]>=b[1] and a[0]+a[2]<=b[0]+b[2] and a[1]+a[3]<=b[1]+b[3] for b in remaining)]
        if len(result)==len(items): return result
    raise ValueError('parts do not fit the atlas without resizing')
def build():
    T=build_targets(); rest,targets=T['rest'],T['targets']
    still=np.asarray(Image.open(f'{R}/source/character/still.png').convert('RGBA'))
    parts=load_parts(T,still)
    # 4판 실루엣 자르기: 정면 마스크(실루엣 ∪ 얼굴 다각형)를 잰 여유만큼 넓혀 얼굴 피부·이마 덮개·얼굴 받침 알파에 미리 곱한다(정면 동일 유지)
    global CLIP_MARGIN_PX
    sil0=silhouette_curve(T,'front'); m0=fill_mask([sil0])>0; fp0=catmull_closed(face_polygon_points(T,'front'),SIL_SUBDIV)
    cover=np.zeros((1536,1024),np.float32)
    for pid in HAIR:
        if pid in parts: bx,by,bw,bh=parts[pid]['bbox']; cover[by:by+bh,bx:bx+bw]=np.maximum(cover[by:by+bh,bx:bx+bw],parts[pid]['img'][...,3]/255.0)
    fx,fy,fw,fh=parts['face-skin']['bbox']; fa=np.zeros((1536,1024),bool); fa[fy:fy+fh,fx:fx+fw]=parts['face-skin']['img'][...,3]>=128
    vis_out=fa&~m0&(cover<HIDDEN_HAIR_ALPHA); dist=ndi.distance_transform_edt(~m0)
    CLIP_MARGIN_PX=float(math.ceil(dist[vis_out].max())) if vis_out.any() else 0.0
    print('clip margin px',CLIP_MARGIN_PX,'visible face px outside silhouette',int(vis_out.sum()))
    clip0=fill_dilated(sil0,CLIP_MARGIN_PX).astype(np.uint8)*255
    for pid in ('face-skin','face-backing'):
        bx,by,bw,bh=parts[pid]['bbox']; im=parts[pid]['img']; before=int((im[...,3]>0).sum()); im[...,3]=np.round(im[...,3]*(clip0[by:by+bh,bx:bx+bw]/255.0)).astype(np.uint8); print('clip bake',pid,before,'->',int((im[...,3]>0).sum()))
    # 5판 앞머리·머리핀 자르기(hairClip): 정면 마스크 = 실루엣 채움 ∪ 윗머리 알파≥HIDDEN_HAIR_ALPHA. 정면에서 잘리는 보이는 화소가 있으면 미리 곱한다(정면 동일 유지)
    hx,hy,hw,hh=parts['hair-top']['bbox']; ht=np.zeros((1536,1024),np.float32); ht[hy:hy+hh,hx:hx+hw]=parts['hair-top']['img'][...,3]/255.0
    hclip0=np.maximum(fill_mask([sil0]).astype(np.float32)/255,np.clip(ht/HIDDEN_HAIR_ALPHA,0,1)); hclip0[int(math.ceil(fp0[:5*SIL_SUBDIV,1].max())):]=1.0
    for pid in HAIR_CLIP_PARTS:
        bx,by,bw,bh=parts[pid]['bbox']; im=parts[pid]['img']; before=int((im[...,3]>0).sum()); im[...,3]=np.round(im[...,3]*hclip0[by:by+bh,bx:bx+bw]).astype(np.uint8); print('hair clip bake',pid,before,'->',int((im[...,3]>0).sum()))
    stat=recut_front(parts,still); print('recut',stat)
    bare_cover(parts)
    bx,by,bw,bh=parts['face-scalp']['bbox']; im=parts['face-scalp']['img']; before=int((im[...,3]>0).sum()); im[...,3]=np.round(im[...,3]*(clip0[by:by+bh,bx:bx+bw]/255.0)).astype(np.uint8); print('clip bake face-scalp',before,'->',int((im[...,3]>0).sum()))
    os.makedirs(f'{P}/parts',exist_ok=True)
    anchors=[[x,690] for x in (300,400,502,600,700)]+[[x,780] for x in (350,502,650)]
    fields={}
    for key,tk in targets.items():
        ks=[k for k in T['names'] if k in tk]; fields[key]=tps_fit([rest[k] for k in ks]+anchors,[tk[k] for k in ks]+anchors,lam=1.0)
    meshes=[]
    order_list=[(k,80) for k in sorted(parts) if k.startswith('still-fill-') and not k.startswith('still-fill-hair-')]+[(k,81) for k in sorted(parts) if k.startswith('still-fill-hair-')]+list(ORDER)
    for pid,order in order_list:
        if pid not in parts: continue
        part=parts[pid]; pos,tri=grid_mesh(part,cell_for(pid)); deltas={}; xforms={}
        for key in KEYS:
            if key=='front': deltas[key]=np.zeros_like(pos).tolist(); continue
            M=part_transform(pid,rest,targets[key],fields[key],T['hidden'][key])
            if pid=='hair-top':
                d_sk=apply23(M,pos)-pos; d_f=tps_eval(fields[key],pos)-pos; t=np.clip((pos[:,1]-120)/110,0,1); w=t*t*(3-2*t); d=d_sk+(d_f-d_sk)*w[:,None]; deltas[key]=np.round(d,3).tolist(); continue
            if pid in ('face-skin','face-backing','face-scalp'):   # J. 얼굴 본체 다각형 안은 변위장. 바깥(복원된 먼 볼·관자·이마 위)은 윤곽에서 가장 가까운 점의 변위를 따르고 바깥 오프셋을 OUTSIDE_SHRINK로 줄인다(먼 쪽 윤곽이 안으로 밀려도 접히지 않는다). 눈썹 위는 두상 변환으로 섞는다
                Msk=similarity_fit([rest[k] for k in SK],[targets[key][k] for k in SK]); d_sk=apply23(Msk,pos)-pos
                d_f=outside_field(pos,rest,fields[key],T['hidden'][key]); w=forehead_weight(pos,rest); d=d_f+(d_sk-d_f)*w[:,None]; deltas[key]=np.round(d,3).tolist(); continue
            if group_of(pid)=='side':      # 옆머리: 위(y≤170)는 두상과 함께, 아래(y≥330)는 관자 점을 따라 통째 이동(먼 쪽) / 두상과 관자의 절반씩(가까운 쪽: 윗머리 복원 영역이 드러나지 않게). 사이는 부드럽게
                Msk=similarity_fit([rest[k] for k in SK],[targets[key][k] for k in SK]); d_sk=apply23(Msk,pos)-pos; d_t=apply23(M,pos)-pos
                far=KEYS[key][0]!=0 and ((side_of(pid)=='l')==(KEYS[key][0]<0)); wmax=1.0 if far else NEAR_SIDE_TEMPLE_W
                t=np.clip((pos[:,1]-170)/160,0,1); w=t*t*(3-2*t)*wmax; d=d_sk+(d_t-d_sk)*w[:,None]; deltas[key]=np.round(d,3).tolist(); continue
            if group_of(pid)=='ear' and T['hidden'][key]!=side_of(pid):       # 귀(보이는 쪽): 얼굴에 붙은 안쪽 가장자리는 얼굴(변위장)을, 바깥쪽은 귀 모양(유사변환)을 따른다
                d_sim=apply23(M,pos)-pos; d_f=tps_eval(fields[key],pos)-pos; x0,x1=part['bbox'][0],part['bbox'][0]+part['bbox'][2]
                t=np.clip(((x1-pos[:,0]) if side_of(pid)=='l' else (pos[:,0]-x0))/part['bbox'][2],0,1); w=t*t*(3-2*t); d=d_f+(d_sim-d_f)*w[:,None]; deltas[key]=np.round(d,3).tolist(); continue
            if M is None:
                d=tps_eval(fields[key],pos)-pos
                if group_of(pid) in ('neck','fill'): t=np.clip((NECK_RAMP[1]-pos[:,1])/(NECK_RAMP[1]-NECK_RAMP[0]),0,1); d*=(t*t*(3-2*t))[:,None]   # still-fill의 목 윤곽선은 목과 같은 가중치
            else:
                d=apply23(M,pos)-pos; xforms[key]=np.round(M,5).tolist()
            deltas[key]=np.round(d,3).tolist()
        Image.fromarray(part['img']).save(f'{P}/parts/{pid}.png')
        meshes.append(dict(id=pid,drawOrder=order,group=group_of(pid),bbox=part['bbox'],file=f'parts/{pid}.png',cell=cell_for(pid),positions=np.round(pos,3).tolist(),triangles=tri.tolist(),deltas=deltas,transforms=xforms))
    # 4판: 실루엣 곡선(24점×SIL_SUBDIV)과 얼굴 다각형 곡선(16점×SIL_SUBDIV)의 9방향 자리. 뷰어는 두 곡선을 부채꼴 메시로 그려 마스크(여유 clipMarginPx만큼 바깥으로 민 곡선)와 끔 받침(안 민 곡선, 피부색)을 만든다
    sil={k:np.round(silhouette_curve(T,k),3).tolist() for k in KEYS}
    fpoly={k:np.round(catmull_closed(face_polygon_points(T,k),SIL_SUBDIV),3).tolist() for k in KEYS}   # 5판: 먼 쪽 머리카락 앞 사본의 닫힘 마스크용
    blink=blink_curves(parts)
    index={p['id']:p for p in json.loads((Path(R)/'parts/parts_index.json').read_text())}
    atlas_boxes=pack([(m['id'],m['bbox'][2],m['bbox'][3]) for m in meshes])
    atlas=Image.new('RGBA',(SIZE,SIZE))
    for mesh in meshes:
        ax,ay,w,h=atlas_boxes[mesh['id']]
        image=Image.open(Path(P)/mesh['file']).convert('RGBA')
        atlas.paste(image,(ax,ay))
        mesh['label']=index.get(mesh['id'],{}).get('label',mesh['id'])
        mesh['atlas']=[ax,ay,w,h]
        bx,by=mesh['bbox'][:2]
        mesh['uvs']=[[(ax+max(0,min(w,x-bx)))/SIZE,(ay+max(0,min(h,y-by)))/SIZE] for x,y in mesh['positions']]
    atlas.save(Path(P)/'texture_00.png')
    group_labels={'fill':'바탕 조각','skull':'두상','neck':'목','body':'의상','ear':'귀','tail':'묶음머리','bangs':'앞머리','side':'옆머리','eye':'눈','brow':'눈썹','nose':'코','mouth':'입','field':'얼굴'}
    parts_ui=[dict(id=group,label=group_labels.get(group,group),meshes=[m['id'] for m in meshes if m['group']==group]) for group in dict.fromkeys(m['group'] for m in meshes)]
    parameters=[
        dict(id='u',label='좌우',min=-1,max=1,default=0),
        dict(id='v',label='위아래',min=-1,max=1,default=0),
        dict(id='roll',label='기울임',min=-30,max=30,default=0),
        dict(id='openL',label='왼눈 뜨기',min=0,max=1,default=1),
        dict(id='openR',label='오른눈 뜨기',min=0,max=1,default=1),
        dict(id='gazeX',label='시선 X',min=-1,max=1,default=0),
        dict(id='gazeY',label='시선 Y',min=-1,max=1,default=0),
        dict(id='breath',label='호흡',min=0,max=1,default=0),
        dict(id='bodyAngleX',label='몸 기울임',min=-10,max=10,default=0),
    ]
    rig=dict(schema='keyshape-2d/6',silhouette=sil,facePolygon=fpoly,clipMarginPx=CLIP_MARGIN_PX,maskRasterMarginPx=MASK_RASTER_MARGIN_PX,clipParts=['face-skin','face-scalp','face-backing'],bareOnlyParts=['skull-plate']+list(BARE_COVER.values()),hairClipParts=list(HAIR_CLIP_PARTS),hairClipAlpha=HIDDEN_HAIR_ALPHA,farWipeFeatherPx=FAR_WIPE_FEATHER_PX,blinkCurves=blink,blinkFade=list(BLINK_FADE),blinkClosedOn=BLINK_CLOSED_ON,canvas=[1024,1536],atlasSize=SIZE,keys={k:list(v) for k,v in KEYS.items()},frames=T['frames'],eyeClosedY=EYE_CLOSED_Y,gazePx=GAZE_PX,farEarFade=FAR_EAR_FADE,neckPivot=NECK_PIVOT,neckRamp=list(NECK_RAMP),clips=CLIPS,hair=list(HAIR)+[k for k in parts if k.startswith('still-fill-hair-')],parts=parts_ui,parameters=parameters,
             skinColor=[float(v) for v in forehead_color(parts['face-skin'])],browTopY=float(min(parts['eye-left-brow']['bbox'][1],parts['eye-right-brow']['bbox'][1])),earCenterY=[float(rest['ear_l_c'][1]),float(rest['ear_r_c'][1])],
             eyeLashY={'left':float(rest['eye_l_up2'][1]),'right':float(rest['eye_r_up2'][1])},eyeLowerY={'left':float(rest['eye_l_lo2'][1]),'right':float(rest['eye_r_lo2'][1])},meshes=meshes)
    json.dump(rig,open(f'{P}/seosoyoung-rig.json','w'),separators=(',',':'))
    # 5판 farWipe 깊이: 섞임 시작(|u|=FAR_EAR_FADE[0])에서 먼 쪽 옆머리·묶음머리(알파≥0.5)가 얼굴 다각형 안으로 들어간 최대 깊이(px). 왼쪽·오른쪽 따로 잰다
    rg=Rig(); depth={}
    for side,u in (('l',-FAR_EAR_FADE[0]),('r',FAR_EAR_FADE[0])):
        p=pose(u,0); D=rg.face_depth(p); best=0.0
        for m in rg.meshes:
            if not ((m['id'].startswith('side-hair') or m['id'].startswith('tail-')) and side_of(m['id'])==side): continue
            r=rg.warp(m,rg.positions(m,p))
            if r is None: continue
            L,(x0,y0)=r; h,w=L.shape[:2]; a=L[...,3]>=128
            if a.any(): best=max(best,float(D[y0:y0+h,x0:x0+w][a].max()))
        depth[side]=round(best,1)
    rig['farWipeDepth']=depth; json.dump(rig,open(f'{P}/seosoyoung-rig.json','w'),separators=(',',':')); print('farWipeDepth',depth)
    json.dump(dict(names=T['names'],rest=rest,targets=targets,S=T['S'].tolist(),ellipse=[float(v) for v in T['ellipse']],frames=T['frames'],hidden=T['hidden']),open(f'{K}/points_art_keyshapes.json','w'),ensure_ascii=False,indent=0)
    print('meshes',len(meshes),'vertices',sum(len(m['positions']) for m in meshes),'triangles',sum(len(m['triangles']) for m in meshes))

if __name__=='__main__':
    build()

# ---------- 합성 ----------
