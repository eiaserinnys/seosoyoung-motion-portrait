"""Focused raster checks for r11, using the actual WebGL entry point."""
import functools, http.server, json, threading
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self,*args):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Handler,directory=str(ROOT/'bundle')))
threading.Thread(target=server.serve_forever,daemon=True).start()
try:
 with sync_playwright() as pw:
  browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--use-angle=swiftshader'])
  page=browser.new_page(viewport={'width':1440,'height':900},device_scale_factor=2)
  page.goto(f'http://127.0.0.1:{server.server_port}',wait_until='networkidle')
  page.wait_for_function('window.studio?.ready',timeout=30000)
  metrics=page.evaluate('''async()=>{
    const {renderer,state,p,rig}=studio,gl=renderer.gl,{deform3D}=await import('./rig.mjs');
    const w=gl.drawingBufferWidth,h=gl.drawingBufferHeight;
    const read=options=>{renderer.render(p,{...state,grid:false,wire:false,...options});const d=new Uint8Array(w*h*4);gl.readPixels(0,0,w,h,gl.RGBA,gl.UNSIGNED_BYTE,d);return d;};
    const solo=ids=>read({mode:'solo',selectedIds:new Set(ids)});
    const diff=(a,b,i)=>Math.abs(a[i]-b[i])+Math.abs(a[i+1]-b[i+1])+Math.abs(a[i+2]-b[i+2]);
    const rows=[];
    for(const gain of [.65,1])for(const yaw of [-50,-25,0,25,50]){
      Object.assign(p,{CorrTurn:gain,ParamAngleX:yaw,ParamAngleY:0,ParamAngleZ:0});
      const full=read({}),near=yaw<0?'right':'left',far=yaw<0?'left':'right';
      const contribution=id=>{const hidden=read({hidden:new Set([id])});let n=0;for(let i=0;i<full.length;i+=4)if(diff(full,hidden,i)>12)n++;return n;};
      const side=solo(['side-hair-'+far,'face-skin']);
      const back=solo(['back-hair-'+far,'back-hair-center']);
      const {scale,ox,oy}=renderer.view;
      let leak=0,earOverlap=0;
      if(yaw)for(let sy=Math.ceil(oy+330*scale);sy<oy+560*scale;sy++){
        let l=w,r=-1;
        for(let x=0;x<w;x++)if(side[((h-1-sy)*w+x)*4+3]>200){l=Math.min(l,x);r=Math.max(r,x);}
        if(r<0)continue;
        for(let x=0;x<w;x++)if((yaw<0?x<l-2:x>r+2)&&back[((h-1-sy)*w+x)*4+3]>200)leak++;
      }
      if(!yaw){
        const cover=solo(['face-skin','side-hair-left','side-hair-right']);
        const hidden=read({hidden:new Set(['ear-left','ear-right'])});
        for(let i=0;i<full.length;i+=4)if(cover[i+3]===255&&cover[i-4+3]===255&&cover[i+4+3]===255&&cover[i-w*4+3]===255&&cover[i+w*4+3]===255&&diff(full,hidden,i)>12)earOverlap++;
      }
      rows.push({gain,yaw,earOverlap,farBackOutsidePixels:leak,farEarPixels:yaw?contribution('ear-'+far):null,
        nearCraniumPixels:contribution('side-cranium-'+near),farCraniumPixels:contribution('side-cranium-'+far)});
    }
    return rows;
  }''')
  (ROOT/'debug/outline_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
  from check_outline_metrics import validate
  validate(metrics)
  print(json.dumps(metrics))
  print("PASS r11 raster occlusion and cranium visibility")
  for side,yaw in [('left',-45),('right',45)]:
   page.evaluate('(yaw)=>{studio.p.CorrTurn=1;studio.setPose(yaw,0,0,"45° 레퍼런스 비교");studio.renderer.render(studio.p,studio.state)}',yaw)
   page.screenshot(path=str(ROOT/'debug'/f'viewer_actual_45_{side}.png'),full_page=True)
  browser.close()
finally:server.shutdown();server.server_close()
