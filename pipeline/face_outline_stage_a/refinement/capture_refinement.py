"""Actual Renderer comparisons for 7.17 and the isolated 7.16 far-lock change."""
from pathlib import Path
import base64,functools,http.server,io,json,threading,sys
from PIL import Image,ImageDraw,ImageFont
from playwright.sync_api import sync_playwright
KEYS=[('up-left',-45,30),('up',0,30),('up-right',45,30),('left',-45,0),('front',0,0),('right',45,0),('down-left',-45,-30),('down',0,-30),('down-right',45,-30)]
MIDS=[('mid-left',-22.5,0),('mid-right',22.5,0),('mid-up',0,15),('mid-down',0,-15),('mid-up-left',-22.5,15),('mid-down-left',-22.5,-15),('mid-up-right',22.5,15),('mid-down-right',22.5,-15)]
PROFILES=[('minus67',-67.5,0),('plus67',67.5,0),('minus78',-78.75,0),('plus78',78.75,0)]
class Handler(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*args):pass

def capture(root,out,poses):
 C=out/'captures';C.mkdir(parents=True);errors=[];geometry=[]
 server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Handler,directory=str(root/'bundle')));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 try:
  with sync_playwright() as pw:
   browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--use-angle=swiftshader']);page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1);page.on('pageerror',lambda e:errors.append(str(e)));page.goto(f'http://127.0.0.1:{server.server_port}',wait_until='networkidle');page.wait_for_function('window.studio?.ready')
   page.evaluate('''async()=>{studio.state.auto=false;studio.state.track=false;studio.renderer.render=()=>{};const {Renderer}=await import('./renderer.mjs');const c=document.createElement('canvas'),o=document.createElement('canvas');Object.assign(c.style,{width:'1100px',height:'1600px'});document.body.append(c);window.cap=new Renderer(c,o,studio.rig,studio.renderer.images);}''')
   for name,yaw,pitch in sorted(poses,key=lambda p:p[0]!='front'):
    result=page.evaluate('''async({yaw,pitch})=>{studio.setPose(yaw,pitch,0,'7.17/7.16 refinement');Object.assign(studio.p,{CorrTurn:1,ParamBodyAngleX:0,ParamBreath:0,ParamEyeLOpen:1,ParamEyeROpen:1,ParamEyeBallX:0,ParamEyeBallY:0,ParamHairFront:0,ParamHairSide:0,ParamHairBack:0});const hair=id=>['hair','bangs','cranium','tail-'].some(k=>id.includes(k)),far=yaw>0?'side-hair-right':'side-hair-left';const configs={full:{keep:()=>true},hairless:{keep:id=>!hair(id)},skin:{keep:id=>id==='face-skin'||id.startsWith('ear-')},skinOnly:{keep:id=>id==='face-skin'},hairOnly:{keep:hair},earLeft:{keep:id=>id==='ear-left'},earRight:{keep:id=>id==='ear-right'},fill:{keep:()=>true,identify:'exposed-skin'},farVisible:{keep:()=>true,identify:far},eyeVisible:{keep:()=>true,identify:['eye-left-sclera','eye-right-sclera']},farSurfaceVisible:{keep:()=>true,identify:['hair-top','bangs-'+(yaw>0?'right':'left')]}};const shots={};for(const [kind,c]of Object.entries(configs)){cap.render(studio.p,{...studio.state,hidden:new Set(studio.rig.meshes.filter(m=>!c.keep(m.id)).map(m=>m.id)),auto:false,track:false,mode:'normal',masks:false,wire:false,grid:false,before:false,surfaceOnly:false,identify:c.identify??null});cap.gl.finish();if(cap.gl.getError())throw Error('GL '+kind);shots[kind]=cap.canvas.toDataURL().split(',')[1];}const {deform3D}=await import('./rig.mjs');const {FRAME_CACHE}=await import('./pose-grid.mjs');const p={...studio.p,[FRAME_CACHE]:new Map()},area=(q,t)=>{const[a,b,c]=t.map(i=>q[i]);return(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);};const geometry=[],hairNegative=[];for(const m of studio.rig.meshes){if(m.id==='face-skin'||m.id.startsWith('ear-'))geometry.push({id:m.id,rest:m.positions,triangles:m.triangles,world:m.positions.map(q=>deform3D(m,q,p))});if(hair(m.id)){const world=m.positions.map(q=>deform3D(m,q,p));hairNegative.push({id:m.id,negative:m.triangles.flatMap((t,i)=>area(m.positions,t)*area(world,t)<0?[i]:[])});}}return{shots,geometry,hairNegative,view:cap.view};}''',dict(yaw=yaw,pitch=pitch))
    for kind,bits in result.pop('shots').items():Image.open(io.BytesIO(base64.b64decode(bits))).crop((38,32,1062,1056)).save(C/f'{name}-{kind}.png')
    assert result['view']=={'scale':1,'ox':38,'oy':32};geometry.append(dict(name=name,yaw=yaw,pitch=pitch,**result));print('REFINEMENT CAPTURE',out.name,name,flush=True)
   browser.close()
 finally:server.shutdown();server.server_close();thread.join()
 (C/'geometry-all.json').write_text(json.dumps(geometry)+'\n');(C/'geometry.json').write_text(json.dumps([q for q in geometry if q['name'] in {k[0] for k in KEYS}])+'\n');(C/'audit.json').write_text(json.dumps(dict(pageErrors=errors,poses=len(geometry)),indent=2)+'\n');assert not errors

def rgb(path):
 im=Image.open(path).convert('RGBA');bg=Image.new('RGBA',im.size,(30,36,48,255));bg.alpha_composite(im);return bg.convert('RGB')
def sheet(out,name,rows,columns,box=(230,120,820,710),scale=1):
 w=(box[2]-box[0])*scale;h=(box[3]-box[1])*scale;im=Image.new('RGB',(w*len(columns),(h+30)*len(rows)),(30,36,48));draw=ImageDraw.Draw(im);font=ImageFont.load_default(size=16)
 for row,pose in enumerate(rows):
  for col,(label,path) in enumerate(columns):
   im.paste(rgb(path(pose)).crop(box).resize((w,h)),(col*w,row*(h+30)+30));draw.text((col*w+5,row*(h+30)+5),pose+' | '+label,font=font,fill='white')
 im.save(out/name)
if __name__=='__main__':
 root,before,baseline,out=(Path(p).resolve() for p in sys.argv[1:5]);capture(root,out/'current',KEYS+MIDS+PROFILES);capture(before,out/'face-only',KEYS+MIDS);capture(baseline,out/'baseline',PROFILES)
