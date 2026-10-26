"""Render the current atlas and its preserved visible backing with the same viewer."""
import base64
import functools
import http.server
import io
import threading

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright


def neutral_metrics(before, after, still, gap):
    delta = np.abs(after.astype(int) - before.astype(int))
    error = lambda image: float(np.abs(image[:, :, :3].astype(float) - still)[gap].mean())
    return {
        'beforeMAE': error(before), 'afterMAE': error(after),
        'outsideMaxLSB': int(delta[~gap].max()),
        'outsideChangedPixels': int(np.count_nonzero(np.any(delta > 0, axis=2) & ~gap)),
    }


def render_neutral_pair(root, preserved_atlas):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(
        ('127.0.0.1', 0), functools.partial(Handler, directory=str(root / 'bundle')))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    buffer = io.BytesIO()
    Image.fromarray(preserved_atlas).save(buffer, format='PNG')
    baseline = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
    errors = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=['--no-sandbox', '--use-angle=swiftshader'])
            page = browser.new_page(viewport={'width': 1440, 'height': 1000}, device_scale_factor=1)
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(f'http://127.0.0.1:{server.server_port}', wait_until='networkidle')
            page.wait_for_function('window.studio?.ready', timeout=30000)
            shots = page.evaluate('''async baseline => {
                studio.state.auto=false;studio.state.track=false;studio.renderer.render=()=>{};
                const {Renderer}=await import('./renderer.mjs');
                const c=document.createElement('canvas'),o=document.createElement('canvas');
                Object.assign(c.style,{width:'1100px',height:'1600px'});document.body.append(c);
                const r=new Renderer(c,o,studio.rig,studio.renderer.images),gl=r.gl;
                studio.setPose(0,0,0,'Neutral hairline contract');
                Object.assign(studio.p,{CorrTurn:1,ParamBodyAngleX:0,ParamBreath:0,
                    ParamEyeLOpen:1,ParamEyeROpen:1,ParamEyeBallX:0,ParamEyeBallY:0,
                    ParamHairFront:0,ParamHairSide:0,ParamHairBack:0});
                const state={...studio.state,auto:false,track:false,hidden:new Set(),
                    mode:'normal',wire:false,grid:false,before:false,surfaceOnly:false,identify:null};
                const snap=()=>{r.render(studio.p,state);gl.finish();
                    if(gl.getError())throw Error('neutral GL error');return c.toDataURL();};
                const after=snap(),image=new Image();image.src=baseline;await image.decode();
                gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,r.textures[0]);
                gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,image);
                return [snap(),after];
            }''', baseline)
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert not errors, errors
    return [np.array(Image.open(io.BytesIO(base64.b64decode(s.split(',')[1])))
                     .crop((38, 32, 1062, 1056)).convert('RGBA')) for s in shots]
