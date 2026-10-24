"""Focused raster checks for r11, using the actual WebGL entry point."""
import functools, http.server, json, threading
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
(ROOT/'debug').mkdir(exist_ok=True)
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
  interior=(ROOT/'pipeline/raster_interior.mjs').read_text().replace('../bundle/','./')
  page.add_script_tag(type='module',content=interior+'\nwindow.rasterInterior={mainOpaqueRuns,headInteriorInterval,rowAlphaGaps,foreheadBoundary,foreheadIntrusion};')
  page.wait_for_function('window.rasterInterior')
  metrics=page.evaluate('''async()=>{
    const {renderer,state,p,rig}=studio,gl=renderer.gl,{deform3D}=await import('./rig.mjs'),{outlineRow}=await import('./face-outline.mjs'),{domeSection}=await import('./hair-surface.mjs'),{FRAME_CACHE}=await import('./pose-grid.mjs');
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
        for(let sy=Math.ceil(oy+280*scale);sy<oy+365*scale;sy++){
          const boundary=window.rasterInterior.foreheadBoundary((sy-oy)/scale,p);
          for(let sx=0;sx<w;sx++)if(window.rasterInterior.foreheadIntrusion((sx-ox)/scale,boundary,yaw)&&hair[((h-1-sy)*w+sx)*4]>200)foreheadSideHairPixels++;
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
          const main=window.rasterInterior.mainOpaqueRuns(full,w,h,sy,scale);
          if(!main)throw Error(`No opaque head run at row ${cy}`);
          const [left,right]=window.rasterInterior.headInteriorInterval(cy,p,...main,ox,scale);
          // Exclude the outer antialias/loose-strand fringe, not internal gaps.
          const gaps=window.rasterInterior.rowAlphaGaps(full,w,h,sy,left,right);
          headInteriorHolePixels+=gaps.count;
          for(const x of gaps.locations)if(holeLocations.length<20)holeLocations.push([(x-ox)/scale,cy]);
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
    const pitchedRows=[];
    const poses=[...[-45,45].flatMap(x=>[-30,30].map(y=>[x,y])),...[-33.75,33.75].flatMap(x=>[-22.5,22.5].map(y=>[x,y]))];
    for(const [yaw,pitch] of poses){
      Object.assign(p,{CorrTurn:1,ParamAngleX:yaw,ParamAngleY:pitch,ParamAngleZ:0});
      const pose={...p,[FRAME_CACHE]:new Map()},full=read({}),near=yaw<0?'right':'left',inner=near==='right'?626:379;
      const ear=rig.meshes.find(m=>m.id==='ear-'+near),face=rig.meshes.find(m=>m.id==='face-skin');
      const boundary=[];
      for(let y=347;y<=616;y+=.5)boundary.push(deform3D(face,[outlineRow(y)[near==='right'?1:0],y],pose));
      const {scale,ox,oy}=renderer.view;
      let pitchedCheekEarGapPixels=0;
      for(let y=440;y<=490;y++){
        const e=deform3D(ear,[inner+(near==='right'?25:-25),y],pose),hits=[];
        for(let i=1;i<boundary.length;i++){
          const a=boundary[i-1],b=boundary[i];
          if((e[1]-a[1])*(e[1]-b[1])<=0&&a[1]!==b[1])hits.push(a[0]+(b[0]-a[0])*(e[1]-a[1])/(b[1]-a[1]));
        }
        if(!hits.length)throw Error(`No pitched cheek boundary ${yaw},${pitch},${y}`);
        const faceX=near==='right'?Math.max(...hits):Math.min(...hits),sy=Math.round(oy+e[1]*scale);
        const left=Math.ceil(ox+Math.min(e[0],faceX)*scale),right=Math.floor(ox+Math.max(e[0],faceX)*scale)+1;
        if(left<0||right>w||sy<0||sy>=h)throw Error('Pitched ear/cheek interval outside raster');
        pitchedCheekEarGapPixels+=window.rasterInterior.rowAlphaGaps(full,w,h,sy,left,right).count;
      }
      pitchedRows.push({yaw,pitch,pitchedCheekEarGapPixels});
    }
    return {rows,pitchedRows};
  }''')
  (ROOT/'debug/outline_metrics.json').write_text(json.dumps(metrics['rows'],indent=2)+'\n')
  (ROOT/'debug/pitched_outline_metrics.json').write_text(json.dumps(metrics['pitchedRows'],indent=2)+'\n')
  from check_outline_metrics import validate
  # Validation follows captures so a failed geometry has inspectable evidence.
  print(json.dumps({**metrics,'rows':[{k:v for k,v in r.items() if k!='contour'} for r in metrics['rows']]}))
  for side,yaw in [('left',-45),('right',45)]:
   page.evaluate('(yaw)=>{studio.p.CorrTurn=1;studio.setPose(yaw,0,0,"45° 레퍼런스 비교");studio.renderer.render(studio.p,studio.state)}',yaw)
   page.screenshot(path=str(ROOT/'debug'/f'viewer_actual_45_{side}.png'),full_page=True)
  browser.close()
  validate(metrics['rows'])
  assert all(r['pitchedCheekEarGapPixels']==0 for r in metrics['pitchedRows'])
  print("PASS r11 14-pose raster and 8 pitched cheek/ear gap checks")
finally:server.shutdown();server.server_close()
