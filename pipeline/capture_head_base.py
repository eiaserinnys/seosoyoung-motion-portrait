"""Capture fitted head backing and compare it with the Seedance key masks."""
import base64
import functools
import http.server
import io
import json
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
FIT = ROOT/'pipeline/headmodel/fit'
sys.path.insert(0, str(FIT))
from fit2d_sheets import image_for
from fit2f_core import load_targets_f

KEYS = [
    ('up-left', 192, -45, 30), ('up', 206, 0, 30), ('up-right', 219, 45, 30),
    ('left', 42, -45, 0), ('front', 3, 0, 0), ('right', 240, 45, 0),
    ('down-left', 303, -45, -30), ('down', 270, 0, -30), ('down-right', 264, 45, -30),
]
INTERMEDIATE = [(f'mid-{yaw}-{pitch}', None, yaw, pitch) for pitch in [-15, 15] for yaw in [-22.5, 22.5]]
PROFILES = [(f'profile-{yaw}', None, yaw, 0) for yaw in [-90, -67.5, 67.5, 90]]
STRIP = [(f'strip-{yaw}-{pitch}', None, yaw, pitch) for pitch in [-20, 0, 20] for yaw in range(-90, 91, 15)]
BG = (30, 36, 48)
CROP = (152, 20, 872, 740)

class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def decode(data):
    image = Image.open(io.BytesIO(base64.b64decode(data))).convert('RGBA')
    return image.crop((38, 32, 1062, 1056))


def alpha(image):
    return np.asarray(image)[:, :, 3] > 127


def contour(mask):
    lines, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not lines:
        raise RuntimeError('Empty rendered silhouette')
    return np.vstack([line[:, 0, :] for line in lines]).astype(float)


def residual(target, rendered):
    values = cKDTree(rendered).query(target)[0]
    return {'median_px': float(np.median(values)), 'max_px': float(values.max()),
            'p95_px': float(np.percentile(values, 95))}


def protrusion(head, full, skin):
    outside = head & ~full & ~skin
    depths = cv2.distanceTransform((~full).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    value = float(depths[outside].max()) if outside.any() else 0.
    return {'max_depth_px': value, 'pixels': int(outside.sum()), 'passed': value <= 1}


def n3_pass(value):
    return value['head']['median_px'] <= 1.5 and value['head']['max_px'] <= 6 and value['face']['median_px'] <= 2 and value['face']['p95_px'] <= 5


def jacobians(rest, current, triangles):
    chosen = triangles[np.all((rest[triangles, 1] >= 218) & (rest[triangles, 1] <= 347), axis=1)]
    source = np.stack([rest[chosen[:, 1]]-rest[chosen[:, 0]], rest[chosen[:, 2]]-rest[chosen[:, 0]]], axis=-1)
    deformed = np.stack([current[chosen[:, 1]]-current[chosen[:, 0]], current[chosen[:, 2]]-current[chosen[:, 0]]], axis=-1)
    matrices = deformed @ np.linalg.inv(source)
    determinant = np.linalg.det(matrices)
    singular = np.linalg.svd(matrices, compute_uv=False)
    anisotropy = singular[:, 0]/singular[:, 1]
    return {'min_det_j': float(determinant.min()), 'max_anisotropy': float(anisotropy.max()), 'triangles': len(chosen)}


def reference_image(frame, eye):
    bgr = image_for(frame)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    dx, dy = eye[0]-501.9, eye[1]-431.5
    matrix = np.array([[1., 0., 152+dx], [0., 1., 20+dy]])
    return Image.fromarray(cv2.warpAffine(rgb, matrix, (1024, 1024), borderValue=BG))


def tile(image, label, size):
    if image.mode == 'RGBA':
        background = Image.new('RGBA', image.size, BG+(255,)); background.alpha_composite(image)
        image = background.convert('RGB')
    result = image.crop(CROP).resize((size, size))
    ImageDraw.Draw(result).text((5, 5), label, fill=(255, 255, 255))
    return result


def save_sheets(renders, metrics):
    dest = ROOT/'docs/captures'; dest.mkdir(parents=True, exist_ok=True)
    hairless = Image.new('RGB', (4*440, 4*220), BG)
    for i, (name, frame, yaw, pitch) in enumerate(KEYS+PROFILES):
        data = renders[(yaw, pitch)]; x, y = (i % 4)*440, (i // 4)*220
        hairless.paste(tile(data['hairless'], name+' rig', 220), (x, y))
        if frame is not None:
            hairless.paste(tile(reference_image(frame, data['eye']), name+' Seedance', 220), (x+220, y))
    hairless.save(dest/'head_base_hairless_13.png')
    full = Image.new('RGB', (3*360, 3*360), BG)
    for i, (name, _, yaw, pitch) in enumerate(KEYS):
        full.paste(tile(renders[(yaw, pitch)]['full'], name, 360), ((i % 3)*360, (i // 3)*360))
    full.save(dest/'head_base_full_9.png')
    strip = Image.new('RGB', (13*160, 3*160+28), BG)
    ImageDraw.Draw(strip).text((5, 5), 'pitch -30: yaw 25 -> 45 reverses model yaw 27.97 -> 22.84', fill='white')
    for i, (_, _, yaw, pitch) in enumerate(STRIP):
        strip.paste(tile(renders[(yaw, pitch)]['hairless'], f'{yaw:+d} / {pitch:+d}', 160), ((i % 13)*160, 28+(i // 13)*160))
    strip.save(dest/'head_base_turn_strip.png')
    overlays = Image.new('RGB', (3*360, 3*360), BG)
    for i, (name, frame, yaw, pitch) in enumerate(KEYS):
        data = renders[(yaw, pitch)]; image = np.array(reference_image(frame, data['eye']))
        for mask, color in [(alpha(data['head']),(40,255,80)), (data['skin_mask'],(255,50,50))]:
            lines, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.polylines(image, lines, True, color, 2)
        target = data['skin_target'].round().astype(int)
        valid = (target[:, 0]>=0)&(target[:, 0]<1024)&(target[:, 1]>=0)&(target[:, 1]<1024)
        for x, y in target[valid]:cv2.circle(image, (x, y), 1, (40, 100, 255), -1)
        overlays.paste(tile(Image.fromarray(image), name+' blue target / red skin / green base', 360), ((i % 3)*360, (i // 3)*360))
    overlays.save(dest/'head_base_jaw_temple.png')


def main():
    rig = json.loads((ROOT/'bundle/assets/seosoyoung-rig.json').read_text())
    skin = next(m for m in rig['meshes'] if m['id']=='face-skin')
    rest = np.asarray(skin['positions'], float); triangles = np.asarray(skin['triangles'], int)
    r24 = {p['name']:p['minDetJ'] for p in json.loads((ROOT/'docs/face_hairless_metrics.json').read_text())['poses']}
    thickness = json.loads((FIT/'results/fit2f/shared-n3.json').read_text())['thickness_px']
    targets = {target['frame']:target for target in load_targets_f([item[1] for item in KEYS], thickness)}
    tasks = {}
    for item in KEYS+INTERMEDIATE+PROFILES+STRIP:
        tasks[(item[2],item[3])] = item
    metric_positions = {(p[2],p[3]) for p in KEYS+INTERMEDIATE+PROFILES if abs(p[2])<=67.5}
    full_positions = {(p[2],p[3]) for p in KEYS}
    server = http.server.ThreadingHTTPServer(('127.0.0.1',0), functools.partial(Handler,directory=str(ROOT/'bundle')))
    thread = threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    errors=[]; renders={}
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--use-angle=swiftshader'])
            page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None)
            page.on('response',lambda r:errors.append(str(r.status)+' '+r.url) if r.status>=400 else None)
            page.goto(f'http://127.0.0.1:{server.server_port}',wait_until='networkidle')
            page.wait_for_function('window.studio?.ready',timeout=30000)
            page.evaluate('''async()=>{
              const {Renderer}=await import('./renderer.mjs');
              const canvas=document.createElement('canvas'),overlay=document.createElement('canvas');
              Object.assign(canvas.style,{width:'1100px',height:'1600px',position:'fixed',left:'-2000px',top:'0'});
              document.body.append(canvas);window.headCapture=new Renderer(canvas,overlay,studio.rig,studio.renderer.images);
              window.headModule=await import('./head-base.mjs');window.poseModule=await import('./pose-grid.mjs');
              window.rigModule=await import('./rig.mjs');window.featureModule=await import('./face-feature-prior.mjs');
              window.featureData=await import('./feature-data.mjs');window.solveData=await import('./face-solve-data.mjs');
            }''')
            page.locator('#head-base-toggle').click(); assert page.evaluate("studio.state.hidden.has('head-base')")
            page.locator('#head-base-toggle').click(); assert not page.evaluate("studio.state.hidden.has('head-base')")
            page.locator('#hair-hide-toggle').click(); assert page.evaluate("studio.rig.meshes.filter(m=>m.id==='hair-top'||m.id==='hairpin'||/^(bangs-|side-hair-|back-hair-|side-cranium-|tail-)/.test(m.id)).every(m=>studio.state.hidden.has(m.id))")
            page.locator('#hair-hide-toggle').click(); assert not page.evaluate("studio.state.hidden.has('hair-top')")
            core=page.evaluate('''()=>{
              const meta=headModule.getHeadBase(),skin=studio.rig.meshes.find(m=>m.id==='face-skin');
              const make=(yaw,pitch)=>({...studio.p,CorrHeadBase:1,CorrOutline:1,CorrTurn:1,ParamAngleX:yaw,ParamAngleY:pitch,ParamAngleZ:0,[poseModule.FRAME_CACHE]:new Map()});
              const p=make(0,0),points=[...skin.positions,...solveData.FACE_SOLVE.points];let rest=0;
              for(const point of points){const q=headModule.headSolveWorld(point,p);rest=Math.max(rest,Math.hypot(q[0]-point[0],q[1]-point[1]));}
              const edges=[];for(const sign of [-1,1])for(const pitch of [-30,0,30]){
                const a=make(sign*44.99,pitch),b=make(sign*45.01,pitch),va=headModule.headBaseVertices(a).surface,vb=headModule.headBaseVertices(b).surface;
                let head=0,face=0;for(let i=0;i<va.length;i+=3)head=Math.max(head,Math.hypot(va[i]-vb[i],va[i+1]-vb[i+1]));
                for(const point of skin.positions){const x=headModule.headSolveWorld(point,a),y=headModule.headSolveWorld(point,b);face=Math.max(face,Math.hypot(x[0]-y[0],x[1]-y[1]));}
                edges.push({sign,pitch,head_px:head,face_px:face});
              }
              return {rest_max_px:rest,edges};
            }''')
            for (yaw,pitch), item in tasks.items():
                data=page.evaluate('''([yaw,pitch,metric,full])=>{
                  const p={...studio.p,CorrHeadBase:1,CorrOutline:1,CorrTurn:1,ParamAngleX:yaw,ParamAngleY:pitch,ParamAngleZ:0,ParamBodyAngleX:0,ParamBreath:0,ParamEyeLOpen:1,ParamEyeROpen:1,ParamEyeBallX:0,ParamEyeBallY:0};
                  const hairs=studio.rig.meshes.filter(m=>m.id==='hair-top'||m.id==='hairpin'||/^(bangs-|side-hair-|back-hair-|side-cranium-|tail-)/.test(m.id)).map(m=>m.id);
                  const state={...studio.state,mode:'all',wire:false,grid:false,before:false,surfaceOnly:false,masks:false,identify:null};
                  const capture=(hidden,identify=null)=>{headCapture.render(p,{...state,hidden:new Set(hidden),identify});const error=headCapture.gl.getError();if(error)throw Error('WebGL error '+error);return headCapture.canvas.toDataURL('image/png').split(',')[1];};
                  const result={hairless:capture(hairs)};
                  if(full)result.full=capture([]);
                  if(metric){
                    result.head=capture(studio.rig.meshes.map(m=>m.id));
                    result.skin=capture([...studio.rig.meshes.filter(m=>m.id!=='face-skin').map(m=>m.id),'head-base'],'face-skin');
                    result.nohead=capture(['head-base']);
                    const q={...p,[poseModule.FRAME_CACHE]:new Map()},skin=studio.rig.meshes.find(m=>m.id==='face-skin');
                    result.skinPositions=skin.positions.map(v=>rigModule.deform3D(skin,v,q).slice(0,2));
                    const left=featureModule.registeredFeatureWorld('eye-left-sclera',featureData.FEATURE_FIT.front.left,q);
                    const right=featureModule.registeredFeatureWorld('eye-right-sclera',featureData.FEATURE_FIT.front.right,q);
                    result.eye=[(left[0]+right[0])/2,(left[1]+right[1])/2];
                    const attached=headModule.headSolveWorld(headModule.getHeadBase().rest_eye,q);
                    result.eyeError=Math.hypot(attached[0]-result.eye[0],attached[1]-result.eye[1]);
                  }
                  return result;
                }''',[yaw,pitch,(yaw,pitch) in metric_positions,(yaw,pitch) in full_positions])
                result={'hairless':decode(data['hairless'])}
                if 'full' in data:result['full']=decode(data['full'])
                if 'head' in data:
                    result.update(head=decode(data['head']),skin=decode(data['skin']),nohead=decode(data['nohead']),
                                  eye=data['eye'],eye_error=data['eyeError'],skin_positions=data['skinPositions'])
                renders[(yaw,pitch)]=result
            browser.close()
            if errors:raise RuntimeError('Browser errors: '+str(errors))
    finally:
        server.shutdown();server.server_close();thread.join()
    rows=[]
    for name, frame, yaw, pitch in KEYS+INTERMEDIATE+PROFILES:
        if (yaw,pitch) not in metric_positions:continue
        data=renders[(yaw,pitch)];headmask=alpha(data['head']);headedge=contour(headmask)
        skin_pixels=np.asarray(data['skin']);skinmask=(skin_pixels[:,:,0]>200)&(skin_pixels[:,:,1]>200)&(skin_pixels[:,:,2]>200)&(skin_pixels[:,:,3]>127)
        skinedge=contour(skinmask);data['skin_mask']=skinmask
        base=alpha(data['nohead'])
        row={'name':name,'yaw':yaw,'pitch':pitch,'N4':jacobians(rest,np.asarray(data['skin_positions']),triangles),
             'N5':protrusion(headmask,base,skinmask),
             'N8':{'error_px':data['eye_error'],'passed':data['eye_error']<=1e-3},'r24_min_det_j':r24.get(name)}
        row['N4']['passed']=row['N4']['min_det_j']>=.35 and row['N4']['max_anisotropy']<=2.
        if frame is not None:
            target=targets[frame];shift=np.array([152,20])+np.array(data['eye'])-[501.9,431.5]
            hair=target['points'][target['labels']==1]+shift
            skin_target=target['skin_all'][target['skin_keep']]+shift;data['skin_target']=skin_target
            row['N3']={'head':residual(hair,headedge),'face':residual(skin_target,skinedge)}
            value=row['N3'];value['passed']=n3_pass(value)
        rows.append(row)
    n6max=max(max(x['head_px'],x['face_px']) for x in core['edges'])
    metrics={'N1':{'max_error_px':core['rest_max_px'],'passed':core['rest_max_px']<=1e-3},
             'N2':{'passed':True,'source':'pipeline/test_head_base.mjs: main hashes'},
             'N6':{'max_difference_px':n6max,'passed':n6max<=.5,'edges':core['edges']},
             'N7':{'passed':True,'source':'pipeline/test_head_base.mjs: 100 vertices x 13 poses'},
             'poses':rows,'r24_reference':r24,
             'angle_note':'pitch -30: effective yaw 25 to 45 maps model yaw 27.97 to 22.84 degrees'}
    (ROOT/'docs/head_base_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
    save_sheets(renders,metrics)
    failures=[{'name':r['name'],'depth_px':r['N5']['max_depth_px']} for r in rows if not r['N5']['passed']]
    print(json.dumps({'poses':len(rows),'N1':metrics['N1'],'N6':metrics['N6']['max_difference_px'],'N5_failures':failures},ensure_ascii=False),flush=True)
    if failures:raise SystemExit('STOP: head base protrudes more than 1 px')

if __name__=='__main__':main()
