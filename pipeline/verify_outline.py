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
    const {renderer,state,p,rig}=studio,gl=renderer.gl,{deform3D}=await import('./rig.mjs'),{outlineRow}=await import('./face-outline.mjs'),{domeSection}=await import('./hair-surface.mjs');
    const w=gl.drawingBufferWidth,h=gl.drawingBufferHeight;
    const read=options=>{renderer.render(p,{...state,grid:false,wire:false,...options});const d=new Uint8Array(w*h*4);gl.readPixels(0,0,w,h,gl.RGBA,gl.UNSIGNED_BYTE,d);return d;};
    const solo=ids=>read({mode:'solo',selectedIds:new Set(ids)});
    const diff=(a,b,i)=>Math.abs(a[i]-b[i])+Math.abs(a[i+1]-b[i+1])+Math.abs(a[i+2]-b[i+2]);
    const rows=[];
    for(const gain of [.65,1])for(const yaw of [-50,-45,-25,0,25,45,50]){
      Object.assign(p,{CorrTurn:gain,ParamAngleX:yaw,ParamAngleY:0,ParamAngleZ:0});
      const full=read({}),near=yaw<0?'right':'left',far=yaw<0?'left':'right';
      const contribution=id=>{const hidden=read({hidden:new Set([id])});let n=0;for(let i=0;i<full.length;i+=4)if(diff(full,hidden,i)>12)n++;return n;};
      const identification=read({identify:'ear-'+near});
      let foreheadSideHairPixels=0;
      if((gain===.65&&Math.abs(yaw)===50)||(gain===1&&Math.abs(yaw)===45)){
        const hair=read({identify:'side-hair-'+near}),{scale,ox,oy}=renderer.view;
        for(let sy=Math.ceil(oy+280*scale);sy<oy+365*scale;sy++)for(let sx=0;sx<w;sx++){
          const x=(sx-ox)/scale;
          if((yaw<0?x<535:x>469)&&hair[((h-1-sy)*w+sx)*4]>200)foreheadSideHairPixels++;
        }
      }
      let nearEarPixels=0;for(let i=0;i<identification.length;i+=4)if(identification[i]>200)nearEarPixels++;
      const side=solo(['side-hair-'+far,'face-skin']);
      const back=solo(['back-hair-'+far]),lowerBack=solo(['back-hair-'+far,'back-hair-center']);
      const {scale,ox,oy}=renderer.view;
      let leak=0,earOverlap=0;const leakLocations=[];
      if(yaw)for(let sy=Math.ceil(oy+330*scale);sy<oy+560*scale;sy++){
        let l=w,r=-1;
        for(let x=0;x<w;x++)if(side[((h-1-sy)*w+x)*4+3]>200){l=Math.min(l,x);r=Math.max(r,x);}
        if(r<0)continue;
        for(let x=0;x<w;x++)if((yaw<0?x<l-2:x>r+2)&&(sy>=oy+408*scale?lowerBack:back)[((h-1-sy)*w+x)*4+3]>200){leak++;if(leakLocations.length<5)leakLocations.push([(x-ox)/scale,(sy-oy)/scale]);}
      }
      if(!yaw){
        const cover=solo(['face-skin','side-hair-left','side-hair-right']);
        const ears=read({identify:['ear-left','ear-right']});
        for(let i=0;i<full.length;i+=4)if(cover[i+3]===255&&cover[i-4+3]===255&&cover[i+4+3]===255&&cover[i-w*4+3]===255&&cover[i+w*4+3]===255&&ears[i]>200)earOverlap++;
      }
      let headInteriorHolePixels=0;const contour=[],holeLocations=[];
      for(let cy=60;cy<=520;cy+=1){
        const sy=Math.round(oy+cy*scale);let l=w,r=-1;
        for(let x=0;x<w;x++)if(full[((h-1-sy)*w+x)*4+3]>200){l=Math.min(l,x);r=Math.max(r,x);}
        if(r<0)continue;
        if(cy>=80&&cy<=420){
          const {rx,rz,cz,cx}=domeSection(cy),angle=yaw*gain*Math.PI/180;
          const center=502+cx*Math.cos(angle)+cz*Math.sin(angle),radius=Math.hypot(rx*Math.cos(angle),rz*Math.sin(angle));
          // Exclude the outer antialias/loose-strand fringe, not internal gaps.
          for(let x=Math.ceil(ox+(center-radius+15)*scale);x<ox+(center+radius-15)*scale;x++)
            if(full[((h-1-sy)*w+x)*4+3]<10){headInteriorHolePixels++;if(holeLocations.length<20)holeLocations.push([(x-ox)/scale,cy]);}
        }
        if(cy<=420&&cy%10===0)contour.push([cy,(l-ox)/scale,(r-ox)/scale]);
      }
      let cheekEarGapPixels=0;
      if(yaw){
        const m=rig.meshes.find(m=>m.id==='ear-'+near),inner=near==='right'?626:379;
        for(let y=440;y<=490;y++){
          const e=deform3D(m,[inner+(near==='right'?25:-25),y],p);
          const row=outlineRow(e[1]),sourceX=near==='right'?row[1]:row[0];
          const f=deform3D(rig.meshes.find(m=>m.id==='face-skin'),[sourceX,e[1]],p);
          const sy=Math.round(oy+e[1]*scale);
          for(let sx=Math.ceil(ox+Math.min(e[0],f[0])*scale);sx<=ox+Math.max(e[0],f[0])*scale;sx++)
            if(full[((h-1-sy)*w+sx)*4+3]<10)cheekEarGapPixels++;
        }
      }
      let maxConcavity=0;
      for(let i=1;i<contour.length-1;i++){
        const a=contour[i-1],b=contour[i],c=contour[i+1];
        if(c[0]-a[0]!==20)continue;
        maxConcavity=Math.max(maxConcavity,b[1]-(a[1]+c[1])/2,(a[2]+c[2])/2-b[2]);
      }
      rows.push({gain,yaw,foreheadSideHairPixels,headInteriorHolePixels,holeLocations,leakLocations,cheekEarGapPixels,maxConcavity,contour,nearEarPixels:yaw?nearEarPixels:null,nearEarColorContribution:yaw?contribution('ear-'+near):null,earOverlap,farBackOutsidePixels:leak,farEarPixels:yaw?contribution('ear-'+far):null,
        nearCraniumPixels:contribution('side-cranium-'+near),farCraniumPixels:contribution('side-cranium-'+far)});
    }
    return rows;
  }''')
  (ROOT/'debug/outline_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
  from check_outline_metrics import validate
  # Validation follows captures so a failed geometry has inspectable evidence.
  print(json.dumps([{k:v for k,v in r.items() if k!='contour'} for r in metrics]))
  for side,yaw in [('left',-45),('right',45)]:
   page.evaluate('(yaw)=>{studio.p.CorrTurn=1;studio.setPose(yaw,0,0,"45° 레퍼런스 비교");studio.renderer.render(studio.p,studio.state)}',yaw)
   page.screenshot(path=str(ROOT/'debug'/f'viewer_actual_45_{side}.png'),full_page=True)
  browser.close()
  validate(metrics)
  print("PASS r11 raster occlusion and cranium visibility")
finally:server.shutdown();server.server_close()
