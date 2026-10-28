"""Compare the fixed dome with the merged r25 viewer in one capture run."""
import base64
import functools
import http.server
import io
import json
import subprocess
import tarfile
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'docs/captures'
BASE = '43d4eeaf1d3b5c5cfb9e22ab77cc5def8abc3ca5'
BACKGROUND = (30, 36, 48, 255)
FONT = ImageFont.truetype('/usr/share/fonts/truetype/nanum/NanumGothic.ttf', 17)
POSES = [('정면', 0, 0, False), ('디렉터 X 25 · Y 18', 25, 18, False),
         ('X -25 · Y 18', -25, 18, False), ('X 45 · Y 20', 45, 20, False),
         ('유효 67.5°', 67.5, 0, True), ('유효 90°', 90, 0, True),
         ('위 20°', 0, 20, False), ('아래 20°', 0, -20, False)]
EAR_POSES = [POSES[1], ('X -45', -45, 0, False), ('X +45', 45, 0, False), POSES[4]]


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def image(data):
    return Image.open(io.BytesIO(base64.b64decode(data))).convert('RGBA')


def rgb(frame):
    base = Image.new('RGBA', frame.size, BACKGROUND)
    base.alpha_composite(frame)
    return base.convert('RGB')


def panel(frame, box, width, height):
    result = rgb(frame).crop(box)
    result.thumbnail((width, height), Image.Resampling.LANCZOS)
    background = Image.new('RGB', (width, height), BACKGROUND[:3])
    background.paste(result, ((width-result.width)//2, (height-result.height)//2))
    return background


def sheet(name, poses, old, new, mode, box):
    width, height, gap, header = 360, 360, 12, 32
    page = Image.new('RGB', (2*(2*width+gap), ((len(poses)+1)//2)*(height+header)), (18, 22, 30))
    draw = ImageDraw.Draw(page)
    for i, pose in enumerate(poses):
        label = pose[0]
        x, y = (i % 2)*(2*width+gap), (i//2)*(height+header)
        draw.text((x+8, y+6), label+'  |  r25', font=FONT, fill='white')
        draw.text((x+width+8, y+6), '새 돔', font=FONT, fill='white')
        page.paste(panel(old[label][mode], box, width, height), (x, y+header))
        page.paste(panel(new[label][mode], box, width, height), (x+width, y+header))
    page.save(DEST / name, optimize=True)


def diff(a, b):
    return np.any(np.asarray(a) != np.asarray(b), axis=2)


def run():
    with tempfile.TemporaryDirectory(prefix='head-base-dome-main-') as directory:
        archive = subprocess.check_output(['git', 'archive', BASE, 'bundle'], cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(directory, filter='data')
        servers = []
        for root in [Path(directory)/'bundle', ROOT/'bundle']:
            server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Handler, directory=str(root)))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start(); servers.append((server, thread))
        errors = []
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True, args=['--no-sandbox', '--use-angle=swiftshader'])
                pages = []
                for server, _ in servers:
                    page = browser.new_page(viewport={'width': 1440, 'height': 1000}, device_scale_factor=1)
                    page.on('pageerror', lambda error: errors.append(str(error)))
                    page.goto(f'http://127.0.0.1:{server.server_port}', wait_until='networkidle')
                    page.wait_for_function('window.studio?.ready', timeout=30000)
                    page.evaluate('''async()=>{
                      const {Renderer}=await import('./renderer.mjs');
                      const canvas=document.createElement('canvas'),overlay=document.createElement('canvas');
                      Object.assign(canvas.style,{width:'1500px',height:'1600px',position:'fixed',left:'-2000px',top:'0'});
                      document.body.append(canvas);window.headCapture=new Renderer(canvas,overlay,studio.rig,studio.renderer.images);
                      window.viewer=await import('./viewer-pose.mjs');
                    }''')
                    pages.append(page)
                results = []
                all_poses = list(dict.fromkeys(POSES + EAR_POSES))
                for page in pages:
                    captures = {}
                    for label, x, y, effective in all_poses:
                        variants = page.evaluate('''([x,y,effective])=>{
                          const meshes=studio.rig.meshes;
                          const hair=meshes.filter(m=>m.id==='hair-top'||m.id==='hairpin'||/^(bangs-|side-hair-|back-hair-|side-cranium-|tail-)/.test(m.id)).map(m=>m.id);
                          const slider=effective?viewer.sliderForYaw(x,studio.p.CorrTurn):x;
                          const p={...studio.p,CorrHeadBase:1,ParamAngleX:slider,ParamAngleY:y,ParamAngleZ:0};
                          const state={...studio.state,mode:'all',wire:false,grid:false,before:false,surfaceOnly:false,masks:false};
                          const capture=(hidden,active=1,identify=null)=>{
                            headCapture.render(viewer.viewerPose({...p,CorrHeadBase:active}),{...state,hidden:new Set(hidden),identify});
                            if(headCapture.gl.getError())throw Error('WebGL capture failed');
                            return headCapture.canvas.toDataURL('image/png').split(',')[1];
                          };
                          return {full:capture([]),hairless:capture(hair),head:capture(meshes.map(m=>m.id)),
                                  legacy:x===0&&y===0?capture([],0):null};
                        }''', [x, y, effective])
                        captures[label] = {key: image(value) for key, value in variants.items() if value is not None}
                    results.append(captures)
                if errors:
                    raise RuntimeError('Browser errors: '+str(errors))
                browser.close()
            old, new = results
            DEST.mkdir(parents=True, exist_ok=True)
            box = (238+145, 32+70, 238+870, 32+795)
            sheet('head_base_dome_hairless.png', POSES, old, new, 'hairless', box)
            sheet('head_base_dome_hair.png', [POSES[i] for i in (0, 1, 3, 5)], old, new, 'full', box)
            sheet('head_base_dome_ears.png', EAR_POSES, old, new, 'hairless', (238+180,32+220,238+840,32+720))
            director = POSES[1][0]
            overlay = np.asarray(rgb(new[director]['hairless'])).copy()
            silhouette = np.asarray(old[director]['head'])[:, :, 3] > 127
            contours, _ = cv2.findContours(silhouette.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(overlay, contours, -1, (255, 50, 50), 2)
            annotated = panel(Image.fromarray(overlay).convert('RGBA'), box, 720, 720)
            annotated.save(DEST/'head_base_dome_overlay.png', optimize=True)
            rest_on = diff(old['정면']['full'], new['정면']['full'])
            rest_off = diff(old['정면']['legacy'], new['정면']['legacy'])
            old_head = np.asarray(old['정면']['head'])[:, :, 3] > 127
            new_head = np.asarray(new['정면']['head'])[:, :, 3] > 127
            result = {'baseline': BASE, 'rest_head_on_changed_pixels': int(rest_on.sum()),
                      'rest_head_off_changed_pixels': int(rest_off.sum()),
                      'rest_difference_outside_old_head_pixels': int(np.count_nonzero(rest_on & ~old_head)),
                      'rest_difference_outside_new_head_pixels': int(np.count_nonzero(rest_on & ~new_head))}
            if rest_on.any():
                visualization = np.asarray(rgb(new['정면']['full'])).copy()
                visualization[rest_on] = [255, 50, 50]
                Image.fromarray(visualization).save(DEST/'head_base_dome_rest_diff.png')
                y, x = np.where(rest_on)
                result['rest_change_bbox'] = [int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)]
            (ROOT/'docs/head_base_dome_capture.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
            print(json.dumps(result, ensure_ascii=False), flush=True)
        finally:
            for server, thread in servers:
                server.shutdown(); server.server_close(); thread.join()


if __name__ == '__main__':
    run()
