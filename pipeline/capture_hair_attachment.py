"""Capture the 13 attachment poses once, with Stage 4 pixel gates and r24 sheet."""
import base64,functools,hashlib,http.server,io,json,shutil,subprocess,sys,threading
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from playwright.sync_api import sync_playwright
from ponytail_mesh import registration_matrix
from skin_fill_validation import measure_pixels

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'.local/forehead-temple/final-render'
BG=(30,36,48);FONT=ImageFont.load_default(size=16)
ORDER=['up-left','up','up-right','left','front','right','down-left','down','down-right']

class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_GET(self):
        if not self.path.startswith('/r24/'):
            return super().do_GET()
        path=self.path.split('?',1)[0][5:] or 'index.html'
        try:data=subprocess.check_output(['git','show','1458c77:bundle/'+path],cwd=ROOT,stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type',self.guess_type(path));self.send_header('Content-Length',str(len(data)))
        self.end_headers();self.wfile.write(data)

def make_sheet(poses,checks):
    references=json.loads((ROOT/'source/viewer/ponytail-seedance-registration.json').read_text())
    enlarged={'down-right','down-left','right','left','up-left','up-right','yaw-30','yaw+30'}
    tips={'left','up-left','down-left','down-right','yaw-30'}
    sheet=Image.new('RGB',(1080,360*13+180*(len(enlarged)+len(tips)+13)),BG);top=0
    geometry=json.loads((ROOT/'.local/forehead-temple/final-geometry.json').read_text())
    quality=json.loads((ROOT/'.local/forehead-temple/final-quality.json').read_text())
    def canvas(im):
        out=Image.new('RGBA',(1024,1024),BG+(255,));out.alpha_composite(im.convert('RGBA'));return out.convert('RGB')
    def label(im,text):
        draw=ImageDraw.Draw(im);box=draw.textbbox((6,6),text,font=FONT)
        draw.rectangle((2,2,box[2]+4,box[3]+4),fill=BG);draw.text((6,6),text,fill='white',font=FONT)
    def crop_strip(im,centers,label_text):
        strip=Image.new('RGB',(360,180),BG)
        for j,center in enumerate(centers):
            x,y=np.rint(center).astype(int);width=360//len(centers)
            crop=im.crop((x-width//2,y-90,x+width//2,y+90))
            label(crop,label_text);strip.paste(crop,(j*width,0))
        return strip
    for pose in poses:
        start=top;name=pose['name']
        # Intermediate yaw uses the nearest registered Seedance image without
        # inventing a reference frame or deforming it to the candidate.
        key=min(references['keys'],key=lambda r:(r['yaw']-pose['yaw'])**2+(r['pitch']-pose['pitch'])**2)
        ref=Image.fromarray(cv2.warpAffine(np.array(Image.open(ROOT/key['image']).convert('RGB')),registration_matrix(key,next(p for p in poses if p["name"]==key["name"])),(1024,1024),borderValue=BG))
        if name=='front':
            ref=Image.new('RGB',(1024,1024),BG);ref.paste(Image.open(ROOT/'source/references/head_front_ref.png').convert('RGB'),(152,20))
        images=[ref,canvas(Image.open(OUT/f'r24-{name}.png')),canvas(Image.open(OUT/f'{name}-full.png'))]
        for column,im in enumerate(images):
            tile=im.crop((152,20,872,740)).resize((360,360))
            title=['original' if name=='front' else 'Seedance '+key['name'],'r24','candidate'][column]
            label(tile,name+' | '+title);sheet.paste(tile,(column*360,top))
        top+=360
        row=next(p for p in geometry if p['name']==name)
        if name in enlarged:
            centers=[]
            for side,point in [('left',[350,380]),('right',[650,380])]:
                m=next(m for m in row['meshes'] if m['id']=='side-hair-'+side)
                k=int(np.argmin(np.square(np.array(m['rest'])-point).sum(axis=1)))
                centers.append(np.array(m['points']['rawCanopy'][k])[:2]+np.array(m['points']['flare'][k])[:2]+np.array(m['points']['residual'][k])[:2])
            for column,im in enumerate(images):sheet.paste(crop_strip(im,centers,'temple 2x'),(column*360,top))
            top+=180
        if name in tips:
            worst=next(p for p in quality if p['name']==name)['bangsWorst']
            m=next(m for m in row['meshes'] if m['id']==worst['mesh']);k=worst['vertex']
            center=np.array(m['points']['rawCanopy'][k])[:2]+np.array(m['points']['flare'][k])[:2]+np.array(m['points']['residual'][k])[:2]
            for column,im in enumerate(images):sheet.paste(crop_strip(im,[center],'bangs tip 2x'),(column*360,top))
            top+=180
        center=next(p for p in checks if p['name']==name)['foreheadCenter'][:2]
        ImageDraw.Draw(sheet).text((8,top+8),name+' | forehead 2x',fill='white',font=FONT)
        for column in [1,2]:sheet.paste(crop_strip(images[column],[center],'forehead 2x'),(column*360,top))
        top+=180
        sheet.crop((0,start,1080,top)).save(OUT/f'panel-{name}.png')
    sheet.save(OUT/'comparison-13poses.png')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    runtime={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'bundle').glob('*.mjs'))}
    stamp=OUT/'runtime-sha256.json'
    if stamp.exists():assert json.loads(stamp.read_text())==runtime,'Cached render belongs to different runtime'
    stamp.write_text(json.dumps(runtime,indent=2)+'\n')
    # Skin, fill, eyes and renderer are unchanged. Preserve the eight existing
    # bare-skin measurements; only visible hair ownership needs a new render.
    for p in (ROOT/'.local/forehead-temple/render').glob('*.png'):
        if p.name.endswith('-bare.png') or p.name.endswith('-owned.png'):
            if not (OUT/p.name).exists():shutil.copy2(p,OUT/p.name)
    features=json.loads((ROOT/'source/viewer/face-feature-fit.json').read_text())['poses']
    poses=[next(p for p in features if p['name']==name) for name in ORDER]
    poses += [dict(name=f'yaw{yaw:+d}',yaw=yaw,pitch=0) for yaw in [-30,-15,15,30]]
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Handler,directory=str(ROOT/'bundle')))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    checks=[];errors=[];performance=[]
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--use-angle=swiftshader'])
            page=browser.new_page(viewport=dict(width=1440,height=1000),device_scale_factor=1)
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{server.server_port}',wait_until='networkidle')
            page.wait_for_function('window.studio?.ready')
            page.evaluate('''async()=>{
              studio.state.auto=false;studio.state.track=false;
              const {Renderer}=await import('./renderer.mjs');
              const c=document.createElement('canvas'),o=document.createElement('canvas');
              Object.assign(c.style,{width:'1100px',height:'1600px'});document.body.append(c);
              window.attachmentCapture=new Renderer(c,o,studio.rig,studio.renderer.images);
            }''')
            for pose in poses:
                cached=[kind for kind in ['full','bare','owned','visibleOwned'] if (OUT/f'{pose["name"]}-{kind}.png').exists()]
                result=page.evaluate('''async({yaw,pitch,cached})=>{
                  studio.setPose(yaw,pitch,0,'Attachment verification');
                  Object.assign(studio.p,{CorrTurn:1,ParamBodyAngleX:0,ParamBreath:0,ParamEyeLOpen:1,ParamEyeROpen:1,ParamEyeBallX:0,ParamEyeBallY:0,ParamHairFront:0,ParamHairSide:0,ParamHairBack:0});
                  const state={...studio.state,auto:false,track:false,hidden:new Set(),mode:'normal',wire:false,grid:false,before:false,surfaceOnly:false,identify:null};
                  let fullMilliseconds=null;
                  const capture=extra=>{
                    attachmentCapture.gl.finish();const start=performance.now();
                    attachmentCapture.render(studio.p,{...state,...extra});
                    attachmentCapture.gl.finish();
                    if(!extra.hidden&&!extra.identify)fullMilliseconds=performance.now()-start;
                    if(attachmentCapture.gl.getError())throw Error('Attachment WebGL error');
                    return attachmentCapture.canvas.toDataURL('image/png').split(',')[1];
                  };
                  const full=cached.includes('full')?null:capture({});
                  const times=[];
                  for(let i=0;i<3;i++){attachmentCapture.gl.finish();const t=performance.now();attachmentCapture.render(studio.p,state);attachmentCapture.gl.finish();times.push(performance.now()-t);}
                  fullMilliseconds=times.sort((a,b)=>a-b)[1];
                  const visibleOwned=cached.includes('visibleOwned')?null:capture({identify:'exposed-skin'});
                  const keep=id=>id==='face-skin'||id.startsWith('eye-')||id==='nose-detail'||id==='mouth-detail'||id.startsWith('ear-');
                  const hidden=new Set(studio.rig.meshes.filter(m=>!keep(m.id)).map(m=>m.id));
                  const bare=cached.includes('bare')?null:capture({hidden}),owned=cached.includes('owned')?null:capture({hidden,identify:'exposed-skin'});
                  const {deform3D}=await import('./rig.mjs'),{SKIN_FILL}=await import('./skin-fill-data.mjs');
                  const face=studio.rig.meshes.find(m=>m.id==='face-skin');
                  return {full,bare,owned,visibleOwned,fullMilliseconds,geometry:{innerRow:SKIN_FILL.innerRow,face:face.positions.map(q=>deform3D(face,q,studio.p)),faceRest:face.positions,foreheadCenter:deform3D(face,[502,310],studio.p)}};
                }''',dict(yaw=pose['yaw'],pitch=pose['pitch'],cached=cached))
                images={}
                for kind in ['full','bare','owned','visibleOwned']:
                    path=OUT/f'{pose["name"]}-{kind}.png'
                    if result[kind]:
                        im=Image.open(io.BytesIO(base64.b64decode(result[kind]))).crop((38,32,1062,1056)).convert('RGBA')
                        im.save(path)
                    images[kind]=np.array(Image.open(path).convert('RGBA'))
                # Exposure means visible ownership in the complete portrait.
                # Dome width and boundary RGB are measured on the bare skin.
                check=measure_pixels(pose['name'],images['bare'],images['visibleOwned'],result['geometry'])
                assert check['exposedPixelsBelowEarLine']==0,(pose['name'],check)
                check['foreheadCenter']=result['geometry']['foreheadCenter']
                yy,xx=np.indices(images['owned'].shape[:2]);line=np.array(result['geometry']['face'])[result['geometry']['innerRow'],:2]
                a,b=line[0],line[-1]
                below=(b[0]-a[0])*(yy+.5-a[1])-(b[1]-a[1])*(xx+.5-a[0])>=0
                bareOwned=(images['owned'][:,:,:3].min(2)>127)&(images['owned'][:,:,3]>127)
                check['hairHiddenPixelsBelowEarLine']=int((bareOwned&below).sum())
                visibleOwned=(images['visibleOwned'][:,:,:3].min(2)>127)&(images['visibleOwned'][:,:,3]>127)
                check['hiddenRibbonPixelsStillVisible']=int((bareOwned&below&visibleOwned).sum())
                check['fullFrameMilliseconds']=result['fullMilliseconds']
                checks.append(check)
                print(pose['name'],'captured',flush=True)
            page.goto(f'http://127.0.0.1:{server.server_port}/r24/index.html',wait_until='networkidle')
            page.wait_for_function('window.studio?.ready')
            baseline=page.evaluate('''async(poses)=>{
              studio.state.auto=false;studio.state.track=false;
              const {Renderer}=await import('./renderer.mjs');
              const c=document.createElement('canvas'),o=document.createElement('canvas');
              Object.assign(c.style,{width:'1100px',height:'1600px'});document.body.append(c);
              const renderer=new Renderer(c,o,studio.rig,studio.renderer.images),times=[],images={};
              for(const pose of poses){
                studio.setPose(pose.yaw,pose.pitch,0,'r24 timing');
                Object.assign(studio.p,{CorrTurn:1,ParamBodyAngleX:0,ParamBreath:0,ParamEyeLOpen:1,ParamEyeROpen:1,ParamEyeBallX:0,ParamEyeBallY:0,ParamHairFront:0,ParamHairSide:0,ParamHairBack:0});
                const state={...studio.state,auto:false,track:false,hidden:new Set(),mode:'normal',wire:false,grid:false,before:false,surfaceOnly:false,identify:null};
                renderer.render(studio.p,state);renderer.gl.finish();const samples=[];
                for(let i=0;i<3;i++){const start=performance.now();renderer.render(studio.p,state);renderer.gl.finish();samples.push(performance.now()-start);}
                times.push({name:pose.name,milliseconds:samples.sort((a,b)=>a-b)[1]});
                images[pose.name]=renderer.canvas.toDataURL('image/png').split(',')[1];
              }
              return {times,images};
            }''',[dict(name=p['name'],yaw=p['yaw'],pitch=p['pitch']) for p in poses])
            for name,data in baseline['images'].items():
                Image.open(io.BytesIO(base64.b64decode(data))).crop((38,32,1062,1056)).save(OUT/f'r24-{name}.png')
            performance=baseline['times']
            browser.close()
    finally:
        server.shutdown();server.server_close();thread.join()
    assert not errors,errors
    front=np.array(Image.open(OUT/'front-full.png'))
    original=np.array(Image.open(OUT/'r24-front.png'))
    changed=int(np.any(front!=original,axis=2).sum())
    result=dict(poses=checks,frontChangedPixels=changed,pageErrors=errors,r24Performance=performance,
                timingMethod='median of three warm full WebGL frames per pose, gl.finish before/after; PNG encoding excluded; software renderer')
    (OUT/'stage4.json').write_text(json.dumps(result,indent=2)+'\n')
    make_sheet(poses,checks)
    print(json.dumps(result),flush=True)
    assert changed==0,('Front changed',changed)

if __name__=='__main__':main()
