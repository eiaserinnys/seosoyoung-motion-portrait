#!/usr/bin/env python3
"""Build unchanged source crops, atlas UVs and an explicit custom web rig."""
import json
import math
import shutil
import numpy as np
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'source/character'
SIZE = 2048
GROUPS = [
 ('PartHairBack','뒷머리',[('back-hair-center',100),('back-hair-left',110),('back-hair-right',120)]),
 ('PartNeckAndBody','목과 의상',[('neck',200),('upper-body-short-sleeve',210),('skirt',220),('sweater',230),('necklace',240)]),
 ('PartEar','귀',[('ear-left',280),('ear-right',290)]),
 ('PartFace','얼굴',[('face-skin',300)]),
 ('PartEye','눈',[(f'eye-{side}-{kind}',order+offset) for kind,order in [('sclera',400),('iris',410),('pupil',420),('highlight',430),('lower-line',440),('upper-lashes',450)] for side,offset in [('left',0),('right',5)]]+[('closed-eyes-v1',460)]),
 ('PartBrow','눈썹',[('eye-left-brow',500),('eye-right-brow',510)]),
 ('PartNose','코',[('nose-detail',520)]),
 ('PartMouth','입',[('mouth-detail',530),('mouth-open-v1',540),('mouth-round-v1',550)]),
 ('PartHairTwinTails','양갈래',[('tail-left-hair',600),('tail-right-hair',610),('tail-left-tie',620),('tail-right-tie',630)]),
 ('PartHairCrownUnder','정수리',[('hair-top',295)]),
 ('PartHairSide','옆머리',[('side-hair-left',700),('side-hair-right',710)]),
 ('PartHairFront','앞머리',[('bangs-left',720),('bangs-right',730),('hairpin',740)])]


def pack(items,size=SIZE):
    """MaxRects, no rotation or rescaling, four pixels on every edge."""
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
    raise ValueError('Parts cannot fit the requested atlas without resizing')


def mesh(im,bbox,atlas):
    x,y,w,h=bbox; ax,ay,_,_=atlas
    cols=max(8,min(18,math.ceil(w/40))); rows=max(8,min(22,math.ceil(h/40)))
    positions=[];uvs=[];triangles=[]
    for j in range(rows+1):
        for i in range(cols+1):
            px=i*w/cols;py=j*h/rows
            positions.append([x+px,y+py]);uvs.append([(ax+px)/SIZE,(ay+py)/SIZE])
    alpha=im.getchannel('A')
    for j in range(rows):
        for i in range(cols):
            region=(math.floor(i*w/cols),math.floor(j*h/rows),math.ceil((i+1)*w/cols),math.ceil((j+1)*h/rows))
            if not alpha.crop(region).getbbox():continue
            a=j*(cols+1)+i;b=a+1;c=a+cols+1;d=c+1
            triangles.extend([[a,b,c],[b,d,c]])
    return dict(positions=positions,uvs=uvs,triangles=triangles,grid=[cols,rows])


def deformer_for(key):
    if key=='neck':return 'WarpNeck'
    if key in ['sweater','necklace','skirt','upper-body-short-sleeve']:return 'WarpBody'
    if key.startswith('tail-'):return 'RotTailL' if '-left-' in key else 'RotTailR'
    if key.startswith('back-'):return 'WarpHairBack'
    if key.startswith('side-hair'):return 'WarpHairSideL' if key.endswith('left') else 'WarpHairSideR'
    if key.startswith('bangs') or key=='hairpin':return 'WarpHairFront'
    if key.startswith('eye-left'):return 'WarpEyeL'
    if key.startswith('eye-right'):return 'WarpEyeR'
    if key.startswith('mouth'):return 'WarpMouth'
    return 'WarpFace'


def build_head_surface(parts, assets, originals=None):
    """Hair-only backing with a soft open face and jaw; never bake skin into the rear."""
    # Sample an unobstructed strip of our own crown drawing, not restored skin edges.
    source=np.asarray(Image.open(SOURCE/'still.png').convert('RGBA'),dtype=np.float32)
    w,h=1024,512
    lon=np.linspace(-np.pi,np.pi,w,dtype=np.float32)[None,:]
    lat=np.linspace(-np.pi/2,np.pi/2,h,dtype=np.float32)[:,None]
    px=np.broadcast_to(421+27*np.sin(lon*3),(h,w))
    py=np.broadcast_to(150+(lat+np.pi/2)/np.pi*135,(h,w))
    x=np.floor(px).astype(int);y=np.floor(py).astype(int)
    wx=(px-x)[...,None];wy=(py-y)[...,None]
    rgba=(source[y,x]*(1-wx)+source[y,x+1]*wx)*(1-wy)+(source[y+1,x]*(1-wx)+source[y+1,x+1]*wx)*wy
    # Front below the hairline is an opening. Rear stops above the jaw, also in pitch.
    front=np.clip((np.cos(lon)+.2)/.6,0,1)
    limit=.45-.62*front
    rgba[:,:,3]=255*np.clip((limit-lat)/.18,0,1)
    rgba[:,:,:3]*=.86
    Image.fromarray(np.clip(rgba,0,255).astype(np.uint8)).save(assets/('head-surface-before.png' if originals is not None else 'head-surface.png'))


def build():
    parts=json.loads((ROOT/'parts/parts_index.json').read_text())
    manifest=json.loads((SOURCE/'manifest.json').read_text()); clothing=json.loads((SOURCE/'clothing.json').read_text())
    old={p['id']:p for p in manifest['leaves']+manifest['expressions']+clothing['parts']}
    boxes=pack([(p['id'],*p['bbox'][2:]) for p in parts])
    assets=ROOT/'bundle/assets';assets.mkdir(parents=True,exist_ok=True)
    after=Image.new('RGBA',(SIZE,SIZE));before=Image.new('RGBA',(SIZE,SIZE))
    mapping={key:(group,draw) for group,_,keys in GROUPS for key,draw in keys}
    meshes=[]; originals={}
    for p in parts:
        key=p['id'];box=boxes[key];x,y,w,h=p['bbox']
        im=Image.open(ROOT/p['file']).convert('RGBA')
        if im.size!=(w,h):raise ValueError(key+' bbox mismatch')
        src=old[key];bounds=src['stageBounds']
        full=Image.new('RGBA',(1024,1536))
        raw=Image.open(SOURCE/src['src']).convert('RGBA').resize((bounds['width'],bounds['height']),Image.Resampling.LANCZOS)
        full.paste(raw,(bounds['x'],bounds['y']))
        original=full.crop((x,y,x+w,y+h))
        if key=='upper-body-short-sleeve':
            alpha=original.getchannel('A')
            alpha.putdata([round(a*max(0,min(1,(y+i//w-680)/12))) for i,a in enumerate(alpha.tobytes())])
            original.putalpha(alpha)
        originals[key]=original
        after.paste(im,tuple(box[:2]));before.paste(original,tuple(box[:2]))
        group,draw=mapping[key]
        masks=[f"eye-{'left' if '-left-' in key else 'right'}-sclera"] if any(key.endswith('-'+s) for s in ['iris','pupil','highlight']) else []
        # Union topology preserves both versions with identical positions and UVs.
        union=im.copy();union.putalpha(Image.frombytes('L',im.size,bytes(max(a,b) for a,b in zip(im.getchannel('A').tobytes(),original.getchannel('A').tobytes()))))
        meshes.append(dict(id=key,label=p.get('label',key),partId=group,drawOrder=draw,deformerId=deformer_for(key),bbox=p['bbox'],atlas=box,masks=masks,**mesh(union,p['bbox'],box)))
    after.save(assets/'texture_00.png');before.save(assets/'texture_before_00.png')
    build_head_surface(parts,assets)
    build_head_surface(parts,assets,originals)
    deformers=[]
    structure=[('RotRoot',None,[502,1400]),('WarpBody','RotRoot',[502,850]),('WarpNeck','WarpBody',[502,650]),('WarpHead','WarpNeck',[502,435]),('RotHeadZ','WarpHead',[502,568]),('WarpFace','RotHeadZ',[502,435]),('WarpEyeL','WarpFace',[410,435]),('WarpEyeR','WarpFace',[590,435]),('WarpMouth','WarpFace',[502,552]),('WarpHairFront','RotHeadZ',[502,220]),('WarpHairSideL','RotHeadZ',[308,300]),('WarpHairSideR','RotHeadZ',[688,300]),('RotTailL','RotHeadZ',[352,504]),('RotTailR','RotHeadZ',[646,504]),('WarpHairBack','RotHeadZ',[502,220])]
    for key,parent,pivot in structure:
        channels = {
            'RotRoot': {}, 'WarpBody': {'ParamBreath': [0, 1], 'ParamBodyAngleX': [-10, 0, 10]},
            'WarpNeck': {'ParamAngleX': [-50, 0, 50], 'ParamAngleY': [-40, 0, 40], 'ParamAngleZ': [-30, 0, 30], 'ParamBreath': [0, 1]},
            'WarpHead': {'ParamAngleX': [-50, 0, 50], 'ParamAngleY': [-40, 0, 40]},
            'RotHeadZ': {'ParamAngleZ': [-30, 0, 30]},
            'WarpFace': {}, 'WarpEyeL': {'ParamEyeLOpen': [0, 1, 1.2], 'ParamBrowLY': [-1, 0, 1]},
            'WarpEyeR': {'ParamEyeROpen': [0, 1, 1.2], 'ParamBrowRY': [-1, 0, 1]},
            'WarpMouth': {'ParamMouthOpenY': [0, 1], 'ParamMouthForm': [-1, 0, 1]},
            'WarpHairFront': {'ParamHairFront': [-1, 0, 1]},
            'WarpHairSideL': {'ParamHairSide': [-1, 0, 1]}, 'WarpHairSideR': {'ParamHairSide': [-1, 0, 1]},
            'RotTailL': {'ParamHairSide': [-1, 0, 1]}, 'RotTailR': {'ParamHairSide': [-1, 0, 1]},
            'WarpHairBack': {'ParamHairBack': [-1, 0, 1]}}
        keyforms = {param: [{'value': value, 'output': value} for value in values] for param, values in channels[key].items()}
        deformers.append(dict(id=key,parentId=parent,type='Rotation' if key.startswith('Rot') else 'Warp',pivot=pivot,keyforms=keyforms))
    parameters=[dict(id='Param'+key,min=lo,max=hi,default=default) for key,lo,hi,default in [('AngleX',-50,50,0),('AngleY',-40,40,0),('AngleZ',-30,30,0),('EyeBallX',-1,1,0),('EyeBallY',-1,1,0),('EyeLOpen',0,1.2,1),('EyeROpen',0,1.2,1),('BrowLY',-1,1,0),('BrowRY',-1,1,0),('MouthOpenY',0,1,0),('MouthForm',-1,1,0),('Breath',0,1,0),('HairFront',-1,1,0),('HairSide',-1,1,0),('HairBack',-1,1,0),('BodyAngleX',-10,10,0)]]
    rig=dict(geometry={'projection':'orthographic','head':{'type':'sphere','center':[502,326,0],'radius':262,'closed':True,'texture':'head-surface.png','beforeTexture':'head-surface-before.png'},'eyes':[{'id':'left','parent':'head','radius':54,'centerXY':[406,438]},{'id':'right','parent':'head','radius':54,'centerXY':[597,438]}],'outsideMapping':'vertical-tangent-strip','neckYawWeight':0},physics={'type':'damped-second-order','stiffness':28,'damping':5,'outputs':['ParamHairFront','ParamHairSide','ParamHairBack']},expressions={'blinkThreshold':.22,'mouthThreshold':.12},schema='pas-web-rig/1',runtime='custom-webgl',nativeCubism=False,canvas=[1024,1536],atlasSize=SIZE,parts=[dict(id=g,label=l) for g,l,_ in GROUPS],deformers=deformers,parameters=parameters,meshes=sorted(meshes,key=lambda m:m['drawOrder']))
    (assets/'seosoyoung-rig.json').write_text(json.dumps(rig,ensure_ascii=False,separators=(',',':')))
    for filename in ['sam3_masks_grid.png','before_after_parts_comparison.png']:
        generated=ROOT/'debug'/filename
        shutil.copyfile(generated if generated.exists() else ROOT/'source/previews'/filename,assets/filename)
    print(json.dumps(dict(meshes=len(meshes),parts=len(GROUPS),deformers=len(deformers),triangles=sum(len(m['triangles']) for m in meshes),atlas=SIZE),ensure_ascii=False))

if __name__=='__main__':build()
