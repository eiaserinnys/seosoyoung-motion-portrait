"""One-pose isolation of the existing alpha-mask fallback, without source edits."""
import functools
import http.server
import json
import threading
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
from capture_sam3_eyes import ROOT, OUT, SETUP, Handler, image

server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Handler,directory=str(ROOT)))
thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
try:
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--use-angle=swiftshader'])
        page=browser.new_page(viewport=dict(width=1440,height=1000),device_scale_factor=1)
        page.goto(f'http://127.0.0.1:{server.server_port}/bundle/',wait_until='networkidle')
        page.wait_for_function('window.studio?.ready',timeout=30000);page.evaluate(SETUP)
        raw=page.evaluate('''()=>{
          const draw=eyeCapture.draw.bind(eyeCapture),calls=[];
          eyeCapture.draw=(mesh,p,alpha,options={})=>{
            if(options.faceClip){
              calls.push({id:mesh.id,originalOptions:options});
              return draw(mesh,p,alpha,{...options,faceClip:false,clip:mesh.masks.length?options.clip:false});
            }
            return draw(mesh,p,alpha,options);
          };
          const result=captureEyes([0,0,false,false]);return {color:result.color,calls};
        }''')
        im=image(raw['color']);im.save(OUT/'neutral-original-clip-isolated.png')
        baseline=np.array(Image.open(OUT/'r24-neutral.png')).astype(int)
        current=np.array(Image.open(OUT/'v291-neutral.png')).astype(int)
        isolated=np.array(im).astype(int)
        # Artwork crop coordinates are normalized 720px in all three images.
        eye=np.s_[355:460,180:520,:]
        result=dict(originalMaxLSB=int(abs(current-baseline)[eye].max()),
                    isolatedMaxLSB=int(abs(isolated-baseline)[eye].max()),
                    originalChangedPixels=int(np.any(current!=baseline,axis=2)[355:460,180:520].sum()),
                    isolatedChangedPixels=int(np.any(isolated!=baseline,axis=2)[355:460,180:520].sum()),
                    isolatedUnmaskedDraws=raw['calls'],productionRendererChanged=False)
        (OUT/'faceclip-original-behavior-isolation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(result,ensure_ascii=False,indent=2))
        browser.close()
finally:
    server.shutdown();server.server_close();thread.join()
