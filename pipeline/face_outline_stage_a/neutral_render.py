"""Same current viewer and atlas, before/after only the face/ear calibration modules."""
import base64,functools,http.server,io,threading
from urllib.parse import urlsplit
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

def render_pair(root):
 bundle=root/'bundle';seeds=root/'pipeline/face_outline_stage_a/seed-bundle';errors=[]
 class Handler(http.server.SimpleHTTPRequestHandler):
  def log_message(self,*args):pass
  def do_GET(self):
   if self.path.startswith('/before/'):
    self.path=self.path[len('/before'):];name=urlsplit(self.path).path.lstrip('/')
    if name in ['face-solve-data.mjs','stereo-attachments.mjs']:
     bits=(seeds/name).read_bytes();self.send_response(200);self.send_header('Content-Type','text/javascript');self.send_header('Content-Length',str(len(bits)));self.end_headers();self.wfile.write(bits);return
   return super().do_GET()
 server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Handler,directory=str(bundle)));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();shots=[]
 try:
  with sync_playwright() as pw:
   browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--use-angle=swiftshader'])
   for suffix in ['/before/index.html','/index.html']:
    page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1);page.on('pageerror',lambda e:errors.append(str(e)));page.goto(f'http://127.0.0.1:{server.server_port}'+suffix,wait_until='networkidle');page.wait_for_function('window.studio?.ready',timeout=30000)
    bits=page.evaluate('''async()=>{studio.state.auto=false;studio.state.track=false;studio.renderer.render=()=>{};const {Renderer}=await import('./renderer.mjs');const c=document.createElement('canvas'),o=document.createElement('canvas');Object.assign(c.style,{width:'1100px',height:'1600px'});document.body.append(c);const r=new Renderer(c,o,studio.rig,studio.renderer.images);studio.setPose(0,0,0,'A-stage neutral contract');Object.assign(studio.p,{CorrTurn:1,ParamBodyAngleX:0,ParamBreath:0,ParamEyeLOpen:1,ParamEyeROpen:1,ParamEyeBallX:0,ParamEyeBallY:0,ParamHairFront:0,ParamHairSide:0,ParamHairBack:0});r.render(studio.p,{...studio.state,hidden:new Set(),auto:false,track:false,mode:'normal',masks:false,wire:false,grid:false,before:false,surfaceOnly:false,identify:null});r.gl.finish();if(r.gl.getError())throw Error('neutral GL error');return c.toDataURL().split(',')[1];}''')
    shots.append(np.array(Image.open(io.BytesIO(base64.b64decode(bits))).convert('RGBA')).copy());page.close()
   browser.close()
 finally:server.shutdown();server.server_close();thread.join()
 assert not errors,errors
 return shots
