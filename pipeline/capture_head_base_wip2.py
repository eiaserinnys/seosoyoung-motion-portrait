"""Capture the head and face against Seedance with a fixed rest translation."""
import base64
import functools
import http.server
import io
import json
import math
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
FIT = ROOT / 'pipeline/headmodel/fit'
sys.path.insert(0, str(FIT))
from fit2d_sheets import image_for
from fit2f_core import load_targets_f

KEYS = [
    ('up-left', 192, -45, 30), ('up', 206, 0, 30), ('up-right', 219, 45, 30),
    ('left', 42, -45, 0), ('front', 3, 0, 0), ('right', 240, 45, 0),
    ('down-left', 303, -45, -30), ('down', 270, 0, -30), ('down-right', 264, 45, -30),
]
PROFILES = [(-90, 0), (-67.5, 0), (67.5, 0), (90, 0)]
STRIP = [(yaw, pitch) for pitch in (-20, 0, 20) for yaw in range(-90, 91, 15)]
BG = (30, 36, 48)
CANVAS_WIDTH = 1500
X_OFFSET = (CANVAS_WIDTH - 1024) // 2
DEST = ROOT / 'docs/captures/wip2-head-base'


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def decode(data):
    image = Image.open(io.BytesIO(base64.b64decode(data))).convert('RGBA')
    return image.crop((0, 32, CANVAS_WIDTH, 1056))


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


def transform(frame, poses, translation):
    front, pose = poses[3], poses[frame]
    scale = math.exp(front[3] - pose[3])
    shift = np.asarray(translation) - scale * np.asarray(pose[4:6]) + [X_OFFSET, 0]
    return scale, shift


def reference(frame, poses, translation):
    scale, shift = transform(frame, poses, translation)
    matrix = np.array([[scale, 0, shift[0]], [0, scale, shift[1]]])
    rgb = cv2.cvtColor(image_for(frame), cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2.warpAffine(rgb, matrix, (CANVAS_WIDTH, 1024), borderValue=BG))


def composite(image):
    background = Image.new('RGBA', image.size, BG + (255,))
    background.alpha_composite(image)
    return background.convert('RGB')


def make_crop(renders):
    bounds = np.asarray([item['bounds'] for item in renders.values()])
    left, top = bounds[:, :2].min(axis=0) + [X_OFFSET, 0]
    right, bottom = bounds[:, 2:].max(axis=0) + [X_OFFSET, 0]
    if left < 0 or right >= CANVAS_WIDTH or top < 0 or bottom >= 1024:
        raise RuntimeError(f'Head or face extends outside capture: {[left, top, right, bottom]}')
    size = math.ceil(max(right - left + 80, bottom - top + 160))
    center = (left + right) / 2
    return (math.floor(center - size / 2), math.floor(top - 40),
            math.floor(center - size / 2) + size, math.floor(top - 40) + size)


def tile(image, crop, size, label):
    if image.mode == 'RGBA':
        image = composite(image)
    result = image.crop(crop).resize((size, size), Image.Resampling.LANCZOS)
    ImageDraw.Draw(result).text((7, 7), label, fill='white', stroke_width=2, stroke_fill=BG)
    return result


def draw_targets(image, hair, skin):
    output = np.asarray(composite(image)).copy()
    for points, color in ((hair, (50, 235, 80)), (skin, (50, 120, 255))):
        for x, y in np.rint(points).astype(int):
            if 0 <= x < output.shape[1] and 0 <= y < output.shape[0]:
                cv2.circle(output, (x, y), 2, color, -1)
    return Image.fromarray(output)


def save_sheets(renders, rows, poses, translation, crop):
    DEST.mkdir(parents=True, exist_ok=True)
    size = 360
    sheet = Image.new('RGB', (6 * size, 3 * size + 28), BG)
    ImageDraw.Draw(sheet).text((8, 7), 'Green: Seedance head target   Blue: Seedance skin target', fill='white')
    for i, (name, frame, yaw, pitch) in enumerate(KEYS):
        x, y = (i % 3) * 2 * size, (i // 3) * size + 28
        item, row = renders[(yaw, pitch)], rows[i]
        overlay = draw_targets(item['portrait'], row['hair_points'], row['skin_points'])
        sheet.paste(tile(overlay, crop, size, name + ' / model'), (x, y))
        sheet.paste(tile(reference(frame, poses, translation), crop, size, name + ' / Seedance'), (x + size, y))
    sheet.save(DEST / 'head_base_seedance_9.png')

    size = 280
    strip = Image.new('RGB', (13 * size, 3 * size + 28), BG)
    ImageDraw.Draw(strip).text((8, 7), 'Effective yaw -90 to +90; pitch -20 / 0 / +20', fill='white')
    for i, (yaw, pitch) in enumerate(STRIP):
        strip.paste(tile(renders[(yaw, pitch)]['portrait'], crop, size, f'{yaw:+g} / {pitch:+g}'),
                    ((i % 13) * size, (i // 13) * size + 28))
    strip.save(DEST / 'head_base_turn_strip.png')

    size = 500
    profiles = Image.new('RGB', (4 * size, size), BG)
    for i, (yaw, pitch) in enumerate(PROFILES):
        profiles.paste(tile(renders[(yaw, pitch)]['portrait'], crop, size, f'yaw {yaw:+g}'), (i * size, 0))
    profiles.save(DEST / 'head_base_profiles.png')


def main():
    meta = json.loads((ROOT / 'bundle/assets/head-base.json').read_text())
    pose_rows = json.loads((FIT / 'results/fit2e/poses-refined.json').read_text())['frames']
    poses = {row['frame']: row['refined_pose'] for row in pose_rows}
    thickness = json.loads((FIT / 'results/fit2f/shared-n3.json').read_text())['thickness_px']
    targets = {target['frame']: target for target in load_targets_f([key[1] for key in KEYS], thickness)}
    for _, frame, _, _ in KEYS:
        scale, shift = transform(frame, poses, meta['rest_translation'])
        assert np.allclose(scale * np.asarray(poses[frame][4:6]) + shift,
                           np.asarray(meta['rest_translation']) + [X_OFFSET, 0], atol=1e-9)
    tasks = list(dict.fromkeys([(key[2], key[3]) for key in KEYS] + STRIP + PROFILES))
    key_positions = {(key[2], key[3]) for key in KEYS}
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Handler, directory=str(ROOT / 'bundle')))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    errors, renders = [], {}
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=['--no-sandbox', '--use-angle=swiftshader'])
            page = browser.new_page(viewport={'width': 1440, 'height': 1000}, device_scale_factor=1)
            page.on('pageerror', lambda e: errors.append(str(e)))
            page.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)
            page.on('response', lambda r: errors.append(str(r.status) + ' ' + r.url) if r.status >= 400 else None)
            page.goto(f'http://127.0.0.1:{server.server_port}', wait_until='networkidle')
            page.wait_for_function('window.studio?.ready', timeout=30000)
            page.evaluate('''async()=>{
              const {Renderer}=await import('./renderer.mjs');
              const canvas=document.createElement('canvas'),overlay=document.createElement('canvas');
              Object.assign(canvas.style,{width:'1500px',height:'1600px',position:'fixed',left:'-2000px',top:'0'});
              document.body.append(canvas);window.headCapture=new Renderer(canvas,overlay,studio.rig,studio.renderer.images);
              window.headModule=await import('./head-base.mjs');window.poseModule=await import('./pose-grid.mjs');
              window.rigModule=await import('./rig.mjs');window.solveData=await import('./face-solve-data.mjs');
            }''')
            core = page.evaluate('''()=>{
              const make=(yaw,pitch)=>({...studio.p,CorrHeadBase:1,CorrOutline:1,CorrTurn:1,ParamAngleX:yaw,ParamAngleY:pitch,ParamAngleZ:0,[poseModule.FRAME_CACHE]:new Map()});
              const skin=studio.rig.meshes.find(m=>m.id==='face-skin'),p=make(0,0);let rest=0;
              for(const point of [...skin.positions,...solveData.FACE_SOLVE.points]){const q=headModule.headSolveWorld(point,p);rest=Math.max(rest,Math.hypot(q[0]-point[0],q[1]-point[1]));}
              const edges=[];for(const sign of [-1,1])for(const pitch of [-30,0,30]){
                const a=make(sign*44.99,pitch),b=make(sign*45.01,pitch),va=headModule.headBaseVertices(a).surface,vb=headModule.headBaseVertices(b).surface;
                let head=0,face=0;for(let i=0;i<va.length;i+=3)head=Math.max(head,Math.hypot(va[i]-vb[i],va[i+1]-vb[i+1]));
                for(const point of skin.positions){const x=headModule.headSolveWorld(point,a),y=headModule.headSolveWorld(point,b);face=Math.max(face,Math.hypot(x[0]-y[0],x[1]-y[1]));}
                edges.push({sign,pitch,head_px:head,face_px:face});
              }
              return {rest_max_px:rest,edges};
            }''')
            for i, (yaw, pitch) in enumerate(tasks):
                data = page.evaluate('''([yaw,pitch,key])=>{
                  const p={...studio.p,CorrHeadBase:1,CorrOutline:1,CorrTurn:1,ParamAngleX:yaw,ParamAngleY:pitch,ParamAngleZ:0,ParamBodyAngleX:0,ParamBreath:0,ParamEyeLOpen:1,ParamEyeROpen:1,ParamEyeBallX:0,ParamEyeBallY:0};
                  const meshes=studio.rig.meshes,skin=meshes.find(m=>m.id==='face-skin');
                  const hidden=meshes.filter(m=>m.id!=='face-skin'&&m.id!=='neck'&&m.deformerId!=='WarpBody').map(m=>m.id);
                  const state={...studio.state,mode:'all',wire:false,grid:false,before:false,surfaceOnly:false,masks:false,identify:null};
                  const capture=(ids,identify=null)=>{headCapture.render(p,{...state,hidden:new Set(ids),identify});const error=headCapture.gl.getError();if(error)throw Error('WebGL error '+error);return headCapture.canvas.toDataURL('image/png').split(',')[1];};
                  const portrait=capture(hidden),head=capture(meshes.map(m=>m.id));
                  const skinImage=key?capture([...meshes.filter(m=>m.id!=='face-skin').map(m=>m.id),'head-base'],'face-skin'):null;
                  const q={...p,[poseModule.FRAME_CACHE]:new Map()},points=headModule.headBaseVertices(q).inset;
                  let left=Infinity,top=Infinity,right=-Infinity,bottom=-Infinity;
                  const include=(x,y)=>{left=Math.min(left,x);top=Math.min(top,y);right=Math.max(right,x);bottom=Math.max(bottom,y);};
                  for(let j=0;j<points.length;j+=3)include(points[j],points[j+1]);
                  for(const point of skin.positions){const v=rigModule.deform3D(skin,point,q);include(v[0],v[1]);}
                  return {portrait,head,skinImage,bounds:[left,top,right,bottom],modelAngle:headModule.modelAngles(q),hidden};
                }''', [yaw, pitch, (yaw, pitch) in key_positions])
                renders[(yaw, pitch)] = {'portrait': decode(data['portrait']), 'head': decode(data['head']),
                                         'bounds': data['bounds'], 'model_angle': data['modelAngle'], 'hidden': data['hidden']}
                if data['skinImage'] is not None:
                    renders[(yaw, pitch)]['skin'] = decode(data['skinImage'])
                if (i + 1) % 10 == 0 or i + 1 == len(tasks):
                    print(f'Captured {i + 1}/{len(tasks)} poses', flush=True)
            browser.close()
            if errors:
                raise RuntimeError('Browser errors: ' + str(errors))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    rows = []
    for name, frame, yaw, pitch in KEYS:
        item, target = renders[(yaw, pitch)], targets[frame]
        scale, shift = transform(frame, poses, meta['rest_translation'])
        hair = target['points'][target['labels'] == 1] * scale + shift
        skin_target = target['skin_all'][target['skin_keep']] * scale + shift
        pixels = np.asarray(item['skin'])
        skinmask = (pixels[:, :, 0] > 200) & (pixels[:, :, 1] > 200) & (pixels[:, :, 2] > 200) & (pixels[:, :, 3] > 127)
        n3 = {'head': residual(hair, contour(alpha(item['head']))), 'face': residual(skin_target, contour(skinmask))}
        n3['passed'] = n3['head']['median_px'] <= 1.5 and n3['head']['max_px'] <= 6 and n3['face']['median_px'] <= 2 and n3['face']['p95_px'] <= 5
        rows.append({'name': name, 'frame': frame, 'yaw': yaw, 'pitch': pitch, 'model_angle': item['model_angle'],
                     'N3': n3, 'hair_points': hair, 'skin_points': skin_target})
    crop = make_crop(renders)
    n6max = max(max(edge['head_px'], edge['face_px']) for edge in core['edges'])
    metrics = {'N1': {'max_error_px': core['rest_max_px'], 'passed': core['rest_max_px'] <= 1e-3},
               'N6': {'max_difference_px': n6max, 'passed': n6max <= .5, 'edges': core['edges']},
               'poses': [{k: v for k, v in row.items() if k not in ('hair_points', 'skin_points')} for row in rows],
               'rest_translation': meta['rest_translation'], 'common_crop_pixels': crop,
               'hidden_meshes': renders[(0, 0)]['hidden'],
               'reference_transform': '(q - camera_translation) * exp(logscale_front - logscale_key) + rest_translation',
               'note': 'Angle-dependent displacement and outside-offset decay remain active.'}
    save_sheets(renders, rows, poses, meta['rest_translation'], crop)
    (DEST / 'head_base_metrics_wip2.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'poses': len(tasks), 'N1': metrics['N1'], 'N6': metrics['N6']['max_difference_px'],
                      'crop': crop, 'N3_passed': sum(row['N3']['passed'] for row in rows)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
