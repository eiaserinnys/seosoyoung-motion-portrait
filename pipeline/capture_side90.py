"""Bounded WebGL before/after turns, closed lids and <=45 pixel equality."""
import base64
import functools
import http.server
import io
import json
import subprocess
import threading

import numpy as np
from PIL import Image,ImageDraw
from playwright.sync_api import sync_playwright
from capture_hairless_face import BG,Handler
from hairless_face_reference import ROOT

BASELINE='0b75a0211691965ca234ad9114073ba934568394'
ANGLES=[-90,-67.5,-45,0,45,67.5,90]


class BaselineHandler(Handler):
    def do_GET(self):
        if self.path.startswith('/baseline/'):
            name=self.path.removeprefix('/baseline/').split('?')[0]
            if '/' in name or not name.endswith('.mjs'):
                self.send_error(404);return
            data=subprocess.check_output(['git','show',f'{BASELINE}:bundle/{name}'],cwd=ROOT,timeout=10)
            self.send_response(200);self.send_header('Content-Type','text/javascript');self.end_headers();self.wfile.write(data)
        else:super().do_GET()


def decode(data):
    return Image.open(io.BytesIO(base64.b64decode(data))).convert('RGBA').crop((38,32,1062,1568))


def changed_pixels(before,after):
    return int(np.count_nonzero(np.any(np.array(before)!=np.array(after),axis=2)))


def tile(im,box,size,label):
    out=Image.new('RGBA',im.size,BG+(255,));out.alpha_composite(im)
    out=out.convert('RGB').crop(box).resize(size)
    ImageDraw.Draw(out).text((8,8),label,fill='white')
    return out


def main():
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(BaselineHandler,directory=str(ROOT/'bundle')))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    errors=[];results={};equal=[];clipping=[]
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--use-angle=swiftshader'])
            page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
            page.set_default_timeout(30000)
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None)
            page.on('response',lambda r:errors.append(str(r.status)+' '+r.url) if r.status>=400 else None)
            page.goto(f'http://127.0.0.1:{server.server_port}',wait_until='networkidle')
            page.wait_for_function('window.studio?.ready')
            page.evaluate('''async()=>{
              studio.setPose(0,0,0,'Profile verification');
              const [{Renderer},{Renderer:Old}]=await Promise.all([import('./renderer.mjs'),import('/baseline/renderer.mjs')]);
              window.captures={};
              for(const [key,Type]of [['current',Renderer],['old',Old]]){
                const c=document.createElement('canvas'),o=document.createElement('canvas');
                Object.assign(c.style,{width:'1100px',height:'1600px'});document.body.append(c);
                captures[key]=new Type(c,o,studio.rig,studio.renderer.images);
              }
            }''')
            def render(yaw,pitch=0,blink=1,face=False,only=None):
                data=page.evaluate('''([yaw,pitch,blink,face,only])=>{
                  const p={...studio.p,ParamAngleX:yaw,ParamAngleY:pitch,ParamAngleZ:0,CorrTurn:1,
                    ParamBodyAngleX:0,ParamBreath:0,ParamEyeLOpen:blink,ParamEyeROpen:blink,
                    ParamEyeBallX:0,ParamEyeBallY:0,ParamHairSide:0,ParamHairFront:0,ParamHairBack:0,
                    ParamMouthOpenY:blink===.5?.6:0};
                  const feature=id=>id.startsWith('eye-')||id==='closed-eyes-v1'||id==='nose-detail'||id.startsWith('mouth-');
                  const keep=id=>only==='skin'?id==='face-skin':only==='features'?feature(id):id==='face-skin'||feature(id);
                  const hidden=new Set(studio.rig.meshes.filter(m=>(face||only)&&!keep(m.id)).map(m=>m.id));
                  const state={...studio.state,hidden,mode:'all',wire:false,grid:false,before:false,surfaceOnly:false,identify:null};
                  return Object.fromEntries(Object.entries(captures).map(([key,r])=>{
                    r.render(p,state);if(r.gl.getError()!==0)throw Error(key+' WebGL error');
                    return [key,r.canvas.toDataURL('image/png').split(',')[1]];
                  }));
                }''',[yaw,pitch,blink,face,only])
                return {key:decode(value) for key,value in data.items()}
            for yaw in ANGLES:
                results[yaw]={'full':render(yaw),'face':render(yaw,face=True)}
            for yaw in [-45,0,45]:
                for pitch in [-30,0,30]:
                    ims=results[yaw]['full'] if not pitch else render(yaw,pitch)
                    pixels=changed_pixels(ims['old'],ims['current'])
                    equal.append(dict(yaw=yaw,pitch=pitch,changedPixels=pixels))
            # The actual 44.99/45.00 boundary, plus the inspector entry points.
            for yaw in [-44.99,44.99]:
                ims=render(yaw)
                equal.append(dict(yaw=yaw,pitch=0,changedPixels=changed_pixels(ims['old'],ims['current'])))
            blink={yaw:render(yaw,blink=0,face=True)['current'] for yaw in [-90,-67.5,67.5,90]}
            for yaw in [-90,-67.5,67.5,90]:
                skin=np.array(render(yaw,only='skin')['current'])[:,:,3]
                for eye_open in [1,0,.5]:
                    details=np.array(render(yaw,blink=eye_open,only='features')['current'])[:,:,3]
                    outside=int(np.count_nonzero((details>0)&(skin==0)))
                    clipping.append(dict(yaw=yaw,eyeOpen=eye_open,outsideFacePixels=outside))
            # Exercise the real app with its default recommended rotation gain;
            # inspect the parameters actually delivered to its renderer.
            page.evaluate('''()=>{
              studio.setCorrection('recommended');
              const render=studio.renderer.render.bind(studio.renderer);
              studio.renderer.render=(p,s)=>{window.lastViewerYaw=p.ParamAngleX*p.CorrTurn;return render(p,s);};
            }''')
            for yaw in [-90,90]:
                page.locator('#ParamAngleX').evaluate('''(el,value)=>{el.value=value;el.dispatchEvent(new Event('input',{bubbles:true}));}''',yaw)
                assert page.evaluate('lastViewerYaw')==yaw
            for name,yaw in [('좌 67.5°',-67.5),('우 67.5°',67.5),('좌 90°',-90),('우 90°',90)]:
                page.get_by_role('button',name=name,exact=True).click()
                assert abs(page.evaluate('lastViewerYaw')-yaw)<1e-8
            page.screenshot(path=str(ROOT/'debug/side90-viewer-wide.png'),full_page=False)
            page.set_viewport_size(dict(width=720,height=1000))
            page.screenshot(path=str(ROOT/'debug/side90-viewer-narrow.png'),full_page=False)
            for yaw in ANGLES:
                page.locator('#ParamAngleX').evaluate('''(el,value)=>{studio.p.CorrTurn=1;el.value=value;el.dispatchEvent(new Event('input',{bubbles:true}));}''',yaw)
                assert page.evaluate('studio.p.ParamAngleX*studio.p.CorrTurn')==yaw
            browser.close()
    finally:
        server.shutdown();server.server_close();thread.join()
    assert not errors,errors
    face=Image.new('RGB',(7*480,3*480),BG);full=Image.new('RGB',(7*384,384*2+576),BG)
    for col,yaw in enumerate(ANGLES):
        name=('left' if yaw<0 else 'right')+('_67' if abs(yaw)==67.5 else '_90' if abs(yaw)==90 else '')
        if abs(yaw)>45:
            ref=Image.open(ROOT/f'source/references/{name}_no_rimlight.png').convert('RGBA')
            canvas=Image.new('RGBA',(1024,1536),BG+(255,));canvas.alpha_composite(ref,(132 if yaw>0 else 152,20))
        else:canvas=results[yaw]['face']['old']
        for row,im in enumerate([canvas,results[yaw]['face']['old'],results[yaw]['face']['current']]):
            label=['Reference / baseline at <=45','Before: face only','After: face only'][row]
            face.paste(tile(im,(260,220,740,700),(480,480),f'{yaw:+g} | {label}'),(col*480,row*480))
        for row,key in enumerate(['old','current']):
            full.paste(tile(results[yaw]['full'][key],(152,20,872,740),(384,384),f'{yaw:+g} | {key} head'),(col*384,row*384))
        full.paste(tile(results[yaw]['full']['current'],(0,0,1024,1536),(384,576),f'{yaw:+g} | full portrait'),(col*384,768))
    face.save(ROOT/'docs/captures/face_turnaround_7angle.png');full.save(ROOT/'docs/captures/full_turnaround_7angle.png')
    sheet=Image.new('RGB',(4*480,2*480),BG)
    for col,(yaw,im) in enumerate(blink.items()):
        sheet.paste(tile(results[yaw]['face']['current'],(260,220,740,700),(480,480),f'{yaw:+g} | open'),(col*480,0))
        sheet.paste(tile(im,(260,220,740,700),(480,480),f'{yaw:+g} | closed'),(col*480,480))
    sheet.save(ROOT/'docs/captures/profile_blink.png')
    report=dict(baseline=BASELINE,invariance=equal,errors=errors,angles=ANGLES,clipping=clipping,
                viewerRecommended=dict(sliderEndpoints=[-90,90],presetAngles=[-90,-67.5,67.5,90]))
    (ROOT/'docs/side90-webgl.json').write_text(json.dumps(report,indent=2)+'\n')
    assert all(p['changedPixels']==0 for p in equal),equal
    assert all(p['outsideFacePixels']==0 for p in clipping),clipping
    print('PASS 7 face/full turns, 4 closed lids, 9 original poses + 44.99 pixel equality; 12 feature clips=0; recommended sliders/presets; errors 0')


if __name__=='__main__':main()
