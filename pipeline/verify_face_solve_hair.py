"""Hair geometry stays exact; shared skin-mask pixel changes are diagnostic."""
import base64
import functools
import hashlib
import http.server
import io
import json
from pathlib import Path
import subprocess
import threading

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASELINE = 'c18cd691ddde3fb26e18d85bffb6eee3eafcec68'


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def changed_pixels(a, b):
    return int(np.count_nonzero(np.any(a != b, axis=2)))


def main():
    out = ROOT/'debug/face-solve/hair-baseline'
    out.mkdir(parents=True, exist_ok=True)
    files = subprocess.check_output(['git', 'ls-tree', '--name-only', BASELINE, 'bundle/'], text=True, timeout=60).splitlines()
    for name in files:
        if name.endswith('.mjs'):
            (out/Path(name).name).write_bytes(subprocess.check_output(['git', 'show', f'{BASELINE}:{name}'], timeout=60))
    # The measurement must reject a single changed pixel.
    fixture = np.zeros((2, 2, 4), np.uint8)
    mutated = fixture.copy();mutated[0, 0, 0] = 1
    assert changed_pixels(fixture, fixture) == 0 and changed_pixels(fixture, mutated) == 1
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Handler, directory=str(ROOT)))
    thread = threading.Thread(target=server.serve_forever, daemon=True);thread.start()
    errors, rows = [], []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=['--no-sandbox', '--use-angle=swiftshader'])
            page = browser.new_page(viewport={'width': 1440, 'height': 1000}, device_scale_factor=1)
            page.on('pageerror', lambda e: errors.append(str(e)))
            page.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)
            page.goto(f'http://127.0.0.1:{server.server_port}/bundle/', wait_until='networkidle')
            page.wait_for_function('window.studio?.ready', timeout=30000)
            page.evaluate('''async()=>{
              const {Renderer}=await import('./renderer.mjs');
              const {Renderer:Old}=await import('../debug/face-solve/hair-baseline/renderer.mjs');
              const create=Type=>{const c=document.createElement('canvas'),o=document.createElement('canvas');
               Object.assign(c.style,{width:'1100px',height:'1600px'});document.body.append(c);
               return new Type(c,o,structuredClone(studio.rig),studio.renderer.images);};
              window.hairRenderers=[create(Old),create(Renderer)];
              window.hairModules=await Promise.all([
               import('../debug/face-solve/hair-baseline/rig.mjs'),import('./rig.mjs'),
               import('../debug/face-solve/hair-baseline/pose-grid.mjs'),import('./pose-grid.mjs')]);
            }''')
            for pitch in [30, 0, -30]:
                for yaw in [-45, 0, 45]:
                    data = page.evaluate('''([yaw,pitch])=>{
                      studio.setPose(yaw,pitch,0,'Hair pixel verification');
                      Object.assign(studio.p,{CorrTurn:1,ParamBodyAngleX:0,ParamBreath:0,ParamHairFront:0,ParamHairSide:0,ParamHairBack:0});
                      const keep=id=>id.includes('hair')||id.includes('cranium')||id.startsWith('tail-')||id==='hairpin';
                      const hidden=new Set(studio.rig.meshes.filter(m=>!keep(m.id)).map(m=>m.id));
                      const state={...studio.state,hidden,mode:'normal',wire:false,grid:false,before:false,surfaceOnly:false,identify:null};
                      const urls=hairRenderers.map(r=>{r.render(studio.p,state);if(r.gl.getError())throw Error('hair WebGL error');return r.canvas.toDataURL();});
                      const oldP={...studio.p,[hairModules[2].FRAME_CACHE]:new Map()},newP={...studio.p,[hairModules[3].FRAME_CACHE]:new Map()};
                      let maxDisplacement=0;
                      for(const [i,m] of hairRenderers[0].rig.meshes.entries())if(keep(m.id))for(const pt of m.positions){
                       const a=hairModules[0].deform3D(m,pt,oldP),b=hairModules[1].deform3D(hairRenderers[1].rig.meshes[i],pt,newP);
                       maxDisplacement=Math.max(maxDisplacement,Math.hypot(...a.map((v,k)=>v-b[k])));
                      }
                      return {urls,maxDisplacement};
                    }''', [yaw, pitch])
                    arrays = [np.array(Image.open(io.BytesIO(base64.b64decode(s.split(',')[1]))).convert('RGBA')) for s in data['urls']]
                    count = changed_pixels(*arrays)
                    rows.append(dict(yaw=yaw, pitch=pitch, changedPixels=count,geometryMaxDisplacement=data['maxDisplacement'],
                                     baselineSha256=hashlib.sha256(arrays[0].tobytes()).hexdigest(),
                                     currentSha256=hashlib.sha256(arrays[1].tobytes()).hexdigest()))
            browser.close()
    finally:
        server.shutdown();server.server_close();thread.join()
    original_renderer = subprocess.check_output(['git', 'show', f'{BASELINE}:bundle/renderer.mjs'], timeout=60)
    renderer_unchanged = original_renderer == (ROOT/'bundle/renderer.mjs').read_bytes()
    result = dict(baseline=BASELINE, poses=rows, errors=errors,rendererUnchanged=renderer_unchanged,
                  pixelPolicy='Hair geometry is protected. Shared actual face-skin mask pixel changes are permitted; no dual-mask path.',
                  faceSolveSha256=hashlib.sha256((ROOT/'bundle/face-solve-data.mjs').read_bytes()).hexdigest(),
                  passed=not errors and renderer_unchanged and all(r['geometryMaxDisplacement'] < 1e-9 for r in rows))
    (ROOT/'docs/face-solve-hair-pixels.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False))
    assert result['passed']


if __name__ == '__main__':
    main()
