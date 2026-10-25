"""Before/after WebGL sheet and continuous sweep, with no working-tree swaps."""
import base64
import functools
import hashlib
import http.server
import io
import json
import subprocess
import threading
from pathlib import Path
from urllib.parse import urlsplit
from PIL import Image,ImageDraw
from playwright.sync_api import sync_playwright
from probe_hair_only import ROOT

BASE='6915b2751ef94ff80d3b86456befda5a9a958f0c'
ANGLES=[(-22.5,15),(0,15),(22.5,15),(-22.5,-15),(0,-15),(22.5,-15),
        (-67.5,0),(-22.5,0),(22.5,0),(67.5,0)]


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self,*args):pass

    def do_GET(self):
        path=urlsplit(self.path).path.lstrip('/')
        if path.startswith('main/'):
            name=path[5:] or 'index.html'
            try:data=subprocess.check_output(['git','show',BASE+':bundle/'+name],cwd=ROOT,stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',self.guess_type(name))
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            return
        return super().do_GET()


def main():
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Handler,directory=str(ROOT/'bundle')))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    images=[];errors=[];sweeps=[];out=ROOT/'docs/captures';out.mkdir(exist_ok=True)
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--use-angle=swiftshader'])
            for variant in ['main','after']:
                page=browser.new_page(viewport={'width':1100,'height':1600},device_scale_factor=1)
                page.on('pageerror',lambda e:errors.append(str(e)))
                page.on('response',lambda response:errors.append(str(response.status)+' '+response.url) if response.status>=400 else None)
                page.goto(f'http://127.0.0.1:{server.server_port}/'+('main/' if variant=='main' else ''),wait_until='networkidle')
                page.wait_for_function('window.studio?.ready',timeout=30000)
                page.add_style_tag(content='#stage{position:fixed!important;inset:0!important;width:1100px!important;height:1600px!important;max-width:none!important;max-height:none!important;}')
                for yaw,pitch in ANGLES:
                    data=page.evaluate('''({yaw,pitch})=>{
                     studio.setPose(yaw,pitch,0,"3D 구면");
                     Object.assign(studio.p,{CorrTurn:1,ParamBodyAngleX:0,ParamBreath:0,ParamHairFront:0,ParamHairSide:0,ParamHairBack:0});
                     Object.assign(studio.state,{auto:false,track:false});
                     studio.renderer.render(studio.p,{...studio.state,grid:false,wire:false,before:false});
                     return {url:studio.renderer.canvas.toDataURL(),view:studio.renderer.view,error:studio.renderer.gl.getError()};
                    }''',dict(yaw=yaw,pitch=pitch))
                    assert data['error']==0,data
                    assert data['view']['scale']==1,data['view']
                    image=Image.open(io.BytesIO(base64.b64decode(data['url'].split(',')[1]))).convert('RGBA')
                    ox,oy=data['view']['ox'],data['view']['oy']
                    image=image.crop((ox+152,oy+20,ox+872,oy+740))
                    tile=Image.new('RGBA',image.size,'#24303c');tile.alpha_composite(image)
                    tile=tile.convert('RGB').resize((320,320))
                    images.append((variant,yaw,pitch,tile))
                # A persistent renderer and pose object traverse both neutral and
                # profile transitions; this catches state-dependent render faults.
                frames=page.evaluate('''()=>{
                 const frames=[];
                 for(const pitch of [-15,0,15])for(let yaw=-90;yaw<=90;yaw+=5){
                  studio.setPose(yaw,pitch,0,"3D 구면");studio.p.CorrTurn=1;
                  studio.renderer.render(studio.p,{...studio.state,grid:false,wire:false,before:false});
                  frames.push({yaw,pitch,error:studio.renderer.gl.getError()});
                 }
                 return frames;
                }''')
                assert all(f['error']==0 for f in frames)
                sweeps.append(dict(variant=variant,frames=len(frames),glErrors=sum(f['error']!=0 for f in frames)))
                page.close()
            browser.close()
    finally:server.shutdown();server.server_close()
    assert not errors,errors
    sheet=Image.new('RGB',(1600,1392),'#182330');draw=ImageDraw.Draw(sheet)
    for i,(variant,yaw,pitch,tile) in enumerate(images):
        index=i%10;row=(index//5)*2+(variant=='after');x=(index%5)*320;y=row*348
        sheet.paste(tile,(x,y+28));draw.text((x+10,y+8),f'{variant} | yaw {yaw:+g} / pitch {pitch:+g}',fill='white')
    target=out/'vdd_unobserved_angles_sheet.png';sheet.save(target)
    result=dict(baseline=BASE,angles=ANGLES,variants=['main','after'],webglSweeps=sweeps,pageErrors=errors,
                sheet=str(target.relative_to(ROOT)),sheetSha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                definition='same WebGL full portraits, frozen motion, scale=1, 720x720 crop resized equally to 320x320; rows main/after for each group of five angles')
    (ROOT/'docs/vdd-webgl.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
