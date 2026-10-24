import functools
import http.server
import json
import threading
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
(ROOT/'debug').mkdir(exist_ok=True)
class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self,*args):pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Handler,directory=str(ROOT/'bundle')))
threading.Thread(target=server.serve_forever,daemon=True).start()
errors=[];failures=[];metrics={}
try:
 with sync_playwright() as pw:
  browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--use-angle=swiftshader'])
  page=browser.new_page(viewport={'width':1440,'height':900},device_scale_factor=2)
  page.on('pageerror',lambda e:errors.append(str(e)))
  page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None)
  page.on('response',lambda r:failures.append(r.url) if r.status>=400 else None)
  page.on('requestfailed',lambda r:failures.append(r.url))
  page.goto(f'http://127.0.0.1:{server.server_port}',wait_until='networkidle');page.wait_for_function('window.studio?.ready',timeout=30000)
  def pose(x,y,z=0,grid=False,outfit='sweater',bangs=False):
   page.evaluate('([x,y,z,grid,outfit,bangs])=>{studio.setPose(x,y,z,"3D 구면");Object.assign(studio.state,{grid,outfit,bangs,wire:false,before:false});for(const id of ["grid","bangs","wire"])document.getElementById(id).setAttribute("aria-pressed",String(studio.state[id]));for(const id of ["shirt","sweater"])document.getElementById(id).setAttribute("aria-pressed",String(outfit===id));studio.renderer.render(studio.p,studio.state)}',[x,y,z,grid,outfit,bangs])
  def snap(name):
   page.screenshot(path=str(ROOT/'debug'/name),full_page=True)
  metrics['canopyContinuity']=page.evaluate("""async()=>{
   const {deform3D}=await import('./rig.mjs'),{p,rig}=studio;
   const ids=['hair-top','bangs-left','bangs-right','side-hair-left','side-hair-right','hairpin'];
   let maxError=0,cases=0;
   for(const yaw of [-45,0,45])for(const pitch of [-30,0,30])for(const wind of [0,1]){
    const pose={...p,CorrTurn:1,ParamAngleX:yaw,ParamAngleY:pitch,ParamHairFront:wind,ParamHairSide:-wind};
    for(const xy of [[490,80],[501,120],[512,180],[502,210],[502,245],[330,180],[674,220],[380,300],[625,400]]){
     const points=ids.map(id=>deform3D(rig.meshes.find(m=>m.id===id),xy,pose));
     for(const q of points){maxError=Math.max(maxError,Math.hypot(...q.map((v,i)=>v-points[0][i])));cases++;}
    }
   }return {maxError,cases};
  }""")
  from PIL import Image,ImageDraw
  import io
  tiles=[]
  for pitch in ([] if '--stereo-only' in sys.argv else [30,0,-30]):
   for yaw in [-45,0,45]:
    page.evaluate('studio.p.CorrTurn=1');pose(yaw,pitch)
    raw=Image.open(io.BytesIO(page.locator('#stage').screenshot()))
    v=page.evaluate('studio.renderer.view');s=v['scale'];ox=v['ox'];oy=v['oy']
    tile=raw.crop((ox+152*s,oy+20*s,ox+872*s,oy+740*s)).convert('RGB').resize((480,480))
    ImageDraw.Draw(tile).text((8,8),f'yaw {yaw} / pitch {pitch}',fill='white');tiles.append(tile)
  if tiles:
   sheet=Image.new('RGB',(1440,1440))
   for i,tile in enumerate(tiles):sheet.paste(tile,((i%3)*480,(i//3)*480))
   sheet.save(ROOT/'debug/viewer_9dir_grid.png')
  metrics['stereoRaster']=page.evaluate("""()=>{
   const {p,state,renderer}=studio,gl=renderer.gl,out=[];
   const read=extra=>{renderer.render(p,{...state,grid:false,wire:false,...extra});const a=new Uint8Array(gl.drawingBufferWidth*gl.drawingBufferHeight*4);gl.readPixels(0,0,gl.drawingBufferWidth,gl.drawingBufferHeight,gl.RGBA,gl.UNSIGNED_BYTE,a);return a;};
   for(const gain of [.65,1])for(const yaw of [-1,1].map(s=>s*(gain===1?45:50))){
    Object.assign(p,{CorrTurn:gain,ParamAngleX:yaw,ParamAngleY:0,ParamAngleZ:0});
    const side=yaw<0?'right':'left',ear=read({identify:'ear-'+side}),hair=read({identify:'side-hair-'+side}),full=read({});
    let earPixels=0,foreheadPixels=0;for(let i=0;i<ear.length;i+=4)if(ear[i]>200)earPixels++;
    const {scale,ox,oy}=renderer.view,w=gl.drawingBufferWidth,h=gl.drawingBufferHeight;
    for(let sy=Math.ceil(oy+280*scale);sy<oy+365*scale;sy++)for(let sx=0;sx<w;sx++)if((yaw<0?(sx-ox)/scale<535:(sx-ox)/scale>469)&&hair[((h-1-sy)*w+sx)*4]>200)foreheadPixels++;
    const beads=['tail-left-tie','tail-right-tie'].map(id=>{const hidden=read({hidden:new Set([id])});let n=0;for(let i=0;i<full.length;i+=4)if(Math.abs(full[i]-hidden[i])+Math.abs(full[i+1]-hidden[i+1])+Math.abs(full[i+2]-hidden[i+2])>12)n++;return n;});
    out.push({yaw,gain,earPixels,foreheadPixels,beads});
   }return out;
  }""")
  for side,yaw in [('left',-45),('right',45)]:
   page.evaluate('studio.p.CorrTurn=1');pose(yaw,0);snap('viewer_actual_45_'+side+'.png')
  (ROOT/'debug/pose_metrics.json').write_text(json.dumps({k:metrics[k] for k in ['canopyContinuity','stereoRaster']},indent=2)+'\n')
  print(json.dumps(metrics['stereoRaster']),flush=True)
  assert all(r['earPixels']>=1500 and r['foreheadPixels']==0 and min(r['beads'])>0 for r in metrics['stereoRaster'])
  if '--poses-only' in sys.argv or '--stereo-only' in sys.argv:
   browser.close();raise SystemExit(0)
  page.close()
  page=browser.new_page(viewport={'width':1440,'height':900},device_scale_factor=1)
  page.on('pageerror',lambda e:errors.append(str(e)))
  page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None)
  page.on('response',lambda r:failures.append(r.url) if r.status>=400 else None)
  page.on('requestfailed',lambda r:failures.append(r.url))
  page.goto(f'http://127.0.0.1:{server.server_port}',wait_until='networkidle');page.wait_for_function('window.studio?.ready',timeout=30000)
  page.locator('#correction-recommended').click()
  for name,x,y in [('front',0,0),('turn_left_25',-25,0),('turn_right_25',25,0),('turn_left_50',-50,0),('turn_right_50',50,0),('pitch_up_20',0,20),('pitch_down_20',0,-20)]:
   for grid in [False,True]:
    pose(x,y,grid=grid);snap('viewer_1440_'+name+('_grid' if grid else '')+'.png')
  pose(0,0,15,grid=True);snap('viewer_1440_neck_pivot_roll.png')
  pose(-25,0,grid=True);snap('viewer_1440_sphere_grid.png')
  for name,x,y,z in [('left',-50,-28,-14),('right',50,28,14)]:
   pose(x,y,z);snap('viewer_1440_range_'+name+'.png')
   pose(x,y,z,outfit='shirt');snap('viewer_1440_shirt_'+name+'.png')
  pose(0,0,bangs=True);page.locator('#wire').click();snap('viewer_1440_bangs_hidden_wireframe.png')
  pose(0,0);page.evaluate('studio.p.ParamEyeLOpen=studio.p.ParamEyeROpen=.5;studio.sync();studio.renderer.render(studio.p,studio.state)');snap('viewer_1440_half_blink.png');page.locator('#reset').click()
  metrics['sphereRaster']=page.evaluate("""() => {
    const {renderer,p,state}=studio,gl=renderer.gl,results=[];
    for(const angles of [[0,0,0],[-50,0,0],[50,0,0],[0,40,0],[0,-40,0],[50,28,14]]){
      [p.ParamAngleX,p.ParamAngleY,p.ParamAngleZ]=angles;renderer.render(p,{...state,surfaceOnly:true,grid:false,wire:false});
      const w=gl.drawingBufferWidth,h=gl.drawingBufferHeight,data=new Uint8Array(w*h*4);gl.readPixels(0,0,w,h,gl.RGBA,gl.UNSIGNED_BYTE,data);
      let left=w,right=-1,top=h,bottom=-1,count=0;for(let y=0;y<h;y++)for(let x=0;x<w;x++)if(data[(y*w+x)*4+3]>127){left=Math.min(left,x);right=Math.max(right,x);top=Math.min(top,h-1-y);bottom=Math.max(bottom,h-1-y);count++;}
      results.push({angles,bounds:[left,top,right,bottom],center:[(left+right)/2,(top+bottom)/2],pixels:count});
    }
    return results;
  }""")
  metrics['surfaceComparison']=page.evaluate("""() => {
    const {renderer,p,state}=studio,gl=renderer.gl,frames=[];
    for(const before of [false,true]){
      renderer.render(p,{...state,surfaceOnly:false,before});
      const data=new Uint8Array(gl.drawingBufferWidth*gl.drawingBufferHeight*4);
      gl.readPixels(0,0,gl.drawingBufferWidth,gl.drawingBufferHeight,gl.RGBA,gl.UNSIGNED_BYTE,data);frames.push(data);
    }
    let changedPixels=0;for(let i=0;i<frames[0].length;i+=4)if(frames[0][i]!==frames[1][i]||frames[0][i+1]!==frames[1][i+1]||frames[0][i+2]!==frames[1][i+2])changedPixels++;
    return {changedPixels};
  }""")
  metrics['geometry']=page.evaluate("""async () => {
    const g=await import('./geometry.mjs'),{deform3D,opacity}=await import('./rig.mjs');
    const base=Object.fromEntries(studio.rig.parameters.map(p=>[p.id,p.default])),shell=g.makeSphereMesh(),neck=studio.rig.meshes.find(m=>m.id==='neck');
    let radiusError=0,eyeRadiusError=0,neckYawError=0,eyeLocalHeadLeak=0,bodyHeadLeak=0;
    for(const angles of [[0,0,0],[-50,0,0],[50,0,0],[0,40,0],[0,-40,0],[50,28,14],[-50,-28,-14]]){
      const p={...base,ParamAngleX:angles[0],ParamAngleY:angles[1],ParamAngleZ:angles[2]};
      for(const q of shell.positions){const w=g.headWorld(q,p);radiusError=Math.max(radiusError,Math.abs(Math.hypot(...w.map((v,i)=>v-g.headWorld([0,0,0],p)[i]))-262));}
      for(const side of ['left','right']){
        const c=g.eyeRootWorld(side,p),a=g.eyeballAngles(side,p);eyeLocalHeadLeak=Math.max(eyeLocalHeadLeak,Math.abs(a.lon1-a.lon0),Math.abs(a.lat1-a.lat0));
        for(const q of g.makeSphereMesh(54,24,12).positions){const w=g.eyeWorld(q,side,{...p,ParamEyeBallX:.8,ParamEyeBallY:-.4});eyeRadiusError=Math.max(eyeRadiusError,Math.abs(Math.hypot(...w.map((v,i)=>v-c[i]))-54));}
      }
      if(!angles[1]&&!angles[2])for(const pt of neck.positions){const a=deform3D(neck,pt,p),b=deform3D(neck,pt,base);neckYawError=Math.max(neckYawError,Math.hypot(...a.map((v,i)=>v-b[i])));}
      for(const m of studio.rig.meshes.filter(m=>m.deformerId==='WarpBody'))for(const pt of m.positions){const a=deform3D(m,pt,p),b=deform3D(m,pt,base);bodyHeadLeak=Math.max(bodyHeadLeak,Math.hypot(...a.map((v,i)=>v-b[i])));}
    }
    const shirt=studio.rig.meshes.find(m=>m.id==='upper-body-short-sleeve');
    return {radiusError,eyeRadiusError,neckYawError,eyeLocalHeadLeak,bodyHeadLeak,sphereVertexCount:shell.positions.length,backHemisphere:shell.positions.some(q=>q[2]<-261),frontHemisphere:shell.positions.some(q=>q[2]>261),shirtOpacity:opacity(shirt,base,{...studio.state,outfit:'sweater'})};
  }""")
  metrics['attachment']=page.evaluate("""async()=>{
    const {deform3D}=await import('./rig.mjs'),{CORRECTION_PRESETS}=await import('./geometry.mjs');
    const base={...Object.fromEntries(studio.rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended};
    const area=([a,b,c])=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
    const rows=[];
    for(const gain of [.65,1])for(const [x,y]of [[-50,-28],[-50,28],[50,-28],[50,28],[-25,0],[25,0]]){
      const p={...base,CorrTurn:gain,ParamAngleX:x,ParamAngleY:y};
      for(const id of ['face-skin','hair-top','bangs-left','bangs-right','side-hair-left','side-hair-right']){
        const m=studio.rig.meshes.find(m=>m.id===id),q=m.positions.map(pt=>deform3D(m,pt,p));
        const reversed=m.triangles.filter(t=>area(t.map(i=>q[i]))*area(t.map(i=>m.positions[i]))<0).length;
        const culled=id==='hair-top'||id.startsWith('side-hair')||id.startsWith('bangs');
        rows.push({gain,pose:[x,y],id,folded:culled?0:reversed,backFacing:culled?reversed:0});
      }
    }
    return {maxFolded:Math.max(...rows.map(r=>r.folded)),maxCulledBackFaces:Math.max(...rows.map(r=>r.backFacing)),cases:rows.length,effectiveAutoAngles:[50*base.CorrTurn,28*base.CorrTurn,14]};
  }""")
  metrics['neckOcclusion']=page.evaluate("""async () => {
    const {deform3D}=await import('./rig.mjs'),{renderer,state}=studio,gl=renderer.gl;
    const base=Object.fromEntries(studio.rig.parameters.map(p=>[p.id,p.default])),neck=studio.rig.meshes.find(m=>m.id==='neck');
    return [[-50,0,0],[50,0,0],[0,28,0],[0,-28,0],[-50,-28,-14],[50,28,14]].map(angles=>{
      const p={...base,ParamAngleX:angles[0],ParamAngleY:angles[1],ParamAngleZ:angles[2]};renderer.render(p,{...state,mode:'solo',selectedIds:new Set(['face-skin']),grid:false,wire:false});
      const w=gl.drawingBufferWidth,h=gl.drawingBufferHeight,data=new Uint8Array(w*h*4);gl.readPixels(0,0,w,h,gl.RGBA,gl.UNSIGNED_BYTE,data);const {scale,ox,oy}=renderer.view;
      const points=[436,502,567].map(x=>{const q=deform3D(neck,[x,540],p),xx=Math.floor(ox+q[0]*scale),yy=Math.floor(oy+q[1]*scale);return {point:q,coverAlpha:data[((h-1-yy)*w+xx)*4+3]};});
      return {angles,points};
    });
  }""")
  pose(0,0)
  metrics['neckCapClipping']=page.evaluate("""async()=>{
    const {renderer,state,p}=studio,gl=renderer.gl,{deform3D}=await import('./rig.mjs'),neck=studio.rig.meshes.find(m=>m.id==='neck'),out=[];
    for(const angles of [[-50,0,0],[50,0,0],[0,28,0],[0,-28,0],[-50,-28,-14],[50,28,14]]){
      [p.ParamAngleX,p.ParamAngleY,p.ParamAngleZ]=angles;
      const read=id=>{renderer.render(p,{...state,mode:'solo',selectedIds:new Set([id]),wire:false,grid:false});const a=new Uint8Array(gl.drawingBufferWidth*gl.drawingBufferHeight*4);gl.readPixels(0,0,gl.drawingBufferWidth,gl.drawingBufferHeight,gl.RGBA,gl.UNSIGNED_BYTE,a);return a;};
      const face=read('face-skin'),skin=read('neck'),{scale,ox,oy}=renderer.view,w=gl.drawingBufferWidth,h=gl.drawingBufferHeight;
      let exposed=0;for(let x=436;x<=567;x++){const q=deform3D(neck,[x,540],p),xx=Math.floor(ox+q[0]*scale),yy=h-1-Math.floor(oy+q[1]*scale),i=(yy*w+xx)*4+3;if(face[i]===0&&skin[i]>4)exposed++;}
      out.push({angles,exposed});
    }return out;
  }""")
  metrics['clipping']=page.evaluate("""() => {
    const {renderer,rig,p,state}=studio,gl=renderer.gl;
    const read=id=>{renderer.render(p,{...state,mode:'solo',selectedIds:new Set([id]),wire:false,grid:false});const data=new Uint8Array(gl.drawingBufferWidth*gl.drawingBufferHeight*4);gl.readPixels(0,0,gl.drawingBufferWidth,gl.drawingBufferHeight,gl.RGBA,gl.UNSIGNED_BYTE,data);return data;};
    const results=[];for(const open of [1,.45])for(const [x,y] of [[-25,0],[25,0],[-50,-28],[-50,28],[50,-28],[50,28]]){
      Object.assign(p,{ParamEyeLOpen:open,ParamEyeROpen:open,ParamAngleX:x,ParamAngleY:y,ParamEyeBallX:0,ParamEyeBallY:0});
      for(const side of ['left','right']){const mask=read('eye-'+side+'-sclera');for(const type of ['iris','pupil','highlight']){const draw=read('eye-'+side+'-'+type);let outside=0,visible=0;for(let i=3;i<draw.length;i+=4)if(draw[i]>4){visible++;if(mask[i]===0)outside++;}results.push({open,x,y,side,type,outside,visible});}}
    }
    return results;
  }""")
  metrics['featureVisibility']=page.evaluate("""() => {
    const {renderer,state,p}=studio,gl=renderer.gl,results=[];
    const read=hidden=>{renderer.render(p,{...state,mode:'all',grid:false,wire:false,hidden});const d=new Uint8Array(gl.drawingBufferWidth*gl.drawingBufferHeight*4);gl.readPixels(0,0,gl.drawingBufferWidth,gl.drawingBufferHeight,gl.RGBA,gl.UNSIGNED_BYTE,d);return d;};
    p.ParamEyeLOpen=p.ParamEyeROpen=1;
    for(const [x,y,z,gain=.65]of [[0,0,0],[-25,0,0],[25,0,0],[0,20,0],[0,-20,0],[-50,-28,-14],[50,28,14],[-50,0,0],[50,0,0]]){
      Object.assign(p,{ParamAngleX:x,ParamAngleY:y,ParamAngleZ:z,CorrTurn:gain});const visible=read(new Set());
      for(const id of ['ear-left','ear-right','mouth-detail','nose-detail','tail-left-tie','tail-right-tie',...['left','right'].flatMap(side=>['upper-lashes','lower-line','brow','sclera'].map(part=>'eye-'+side+'-'+part))]){
        const hidden=read(new Set([id]));let pixels=0;for(let i=0;i<hidden.length;i+=4)if(Math.abs(hidden[i]-visible[i])+Math.abs(hidden[i+1]-visible[i+1])+Math.abs(hidden[i+2]-visible[i+2])>12)pixels++;
        results.push({pose:[x,y,z],gain,id,pixels});
      }
    }return results;
  }""")
  metrics['eyeHairOcclusion']=page.evaluate("""() => {
    const {renderer,state,p}=studio,gl=renderer.gl,rows=[];
    const eyes=renderer.rig.meshes.filter(m=>m.id.startsWith('eye-')).map(m=>m.id);
    const read=hidden=>{renderer.render(p,{...state,mode:'all',grid:false,wire:false,hidden,identify:eyes});const d=new Uint8Array(gl.drawingBufferWidth*gl.drawingBufferHeight*4);gl.readPixels(0,0,gl.drawingBufferWidth,gl.drawingBufferHeight,gl.RGBA,gl.UNSIGNED_BYTE,d);return d;};
    for(const yaw of [-45,45]){
      Object.assign(p,{ParamAngleX:yaw,ParamAngleY:0,ParamAngleZ:0,CorrTurn:1,ParamEyeLOpen:1,ParamEyeROpen:1});
      const normal=read(new Set()),uncovered=read(new Set(['side-hair-left','side-hair-right']));let pixels=0;
      for(let i=0;i<normal.length;i+=4)if(uncovered[i]>127&&normal[i]<64)pixels++;
      const hair=renderer.rig.meshes.filter(m=>m.id.includes('hair')||m.id.startsWith('bangs')).map(m=>m.id);
      const brow='eye-'+(yaw<0?'right':'left')+'-brow';
      renderer.render(p,{...state,mode:'all',grid:false,wire:false,hidden:new Set(hair),identify:[brow]});
      const d=new Uint8Array(normal.length);gl.readPixels(0,0,gl.drawingBufferWidth,gl.drawingBufferHeight,gl.RGBA,gl.UNSIGNED_BYTE,d);
      let browWithoutHairPixels=0;for(let i=0;i<d.length;i+=4)if(d[i]>127)browWithoutHairPixels++;
      rows.push({yaw,sideHairCoveredEyePixels:pixels,browWithoutHairPixels});
    }
    return rows;
  }""")
  metrics['correctionPresets']={}
  for preset in ['pure','recommended']:
   page.locator('#correction-'+preset).click()
   metrics['correctionPresets'][preset]=page.evaluate('Object.fromEntries(Object.entries(studio.p).filter(([k])=>k.startsWith("Corr")))')
   for name,x,y,z in [('front',0,0,0),('left',-25,0,0),('right',25,0,0),('up',0,20,0),('down',0,-20,0),('diagonal',-35,-20,-10)]:
    pose(x,y,z);snap('viewer_correction_'+preset+'_'+name+'.png')
  metrics['correctionInput']=page.locator('#CorrDepth').evaluate('(e)=>{e.value=.73;e.dispatchEvent(new Event("input",{bubbles:true}));return studio.p.CorrDepth}')
  metrics['hairControlInput']=page.locator('#CorrHairDepth').evaluate('(e)=>{e.value=.8;e.dispatchEvent(new Event("input",{bubbles:true}));return studio.p.CorrHairDepth}')
  metrics['occlusionControlInput']=page.locator('#CorrHairOcclusion').evaluate('(e)=>{e.value=.5;e.dispatchEvent(new Event("input",{bubbles:true}));return studio.p.CorrHairOcclusion}')
  page.locator('#correction-recommended').click()
  for name,x in [('left',-50),('right',50)]:
   page.evaluate('studio.p.CorrTurn=1;studio.sync()')
   pose(x,-28);snap('viewer_hair_extreme_'+name+'.png')
  page.locator('#correction-recommended').click()
  metrics['hairShell']=page.evaluate("""() => {
    const {renderer,state,p,rig}=studio,gl=renderer.gl,rows=[];
    const read=(s)=>{renderer.render(p,s);const d=new Uint8Array(gl.drawingBufferWidth*gl.drawingBufferHeight*4);gl.readPixels(0,0,gl.drawingBufferWidth,gl.drawingBufferHeight,gl.RGBA,gl.UNSIGNED_BYTE,d);return d;};
    const ids=new Set(rig.meshes.filter(m=>/^(back-hair|hair-top|side-hair|bangs)/.test(m.id)).map(m=>m.id));
    for(const [x,y] of [[0,0],[-25,0],[25,0],[-50,0],[50,0],[0,-20],[0,20]]){
      Object.assign(p,{ParamAngleX:x,ParamAngleY:y,ParamAngleZ:0});
      const s={...state,mode:'all',wire:false,grid:false,hidden:new Set()},full=read(s);
      const contribution=id=>{const hidden=read({...s,hidden:new Set([id])});let n=0;for(let i=0;i<full.length;i+=4)if(Math.abs(full[i]-hidden[i])+Math.abs(full[i+1]-hidden[i+1])+Math.abs(full[i+2]-hidden[i+2])>12)n++;return n;};
      const crown=contribution('hair-top'),ears=['ear-left','ear-right'].map(contribution);
      const hair=read({...s,mode:'solo',selectedIds:ids}),w=gl.drawingBufferWidth,h=gl.drawingBufferHeight;
      let left=w,right=-1,top=h,bottom=-1;
      for(let yy=0;yy<h;yy++)for(let xx=0;xx<w;xx++)if(hair[(yy*w+xx)*4+3]>127){left=Math.min(left,xx);right=Math.max(right,xx);top=Math.min(top,h-1-yy);bottom=Math.max(bottom,h-1-yy);}
      rows.push({pose:[x,y],crownPixels:crown,earPixels:ears,hairBounds:[left,top,right,bottom],hairWidth:right-left+1});
    }return rows;
  }""")
  page.locator('#CorrHairDepth').scroll_into_view_if_needed();pose(0,0);snap('viewer_hair_controls.png')
  # A continuous yaw strip exposes silhouette dents hidden by individual pose checks.
  from PIL import Image,ImageDraw
  frames=[]
  for yaw in [-50,-35,-25,0,25,35,50]:
   pose(yaw,0)
   target=ROOT/'debug'/f'viewer_yaw_{yaw}.png'
   page.screenshot(path=str(target),clip={'x':510,'y':260,'width':360,'height':300})
   frame=Image.open(target).convert('RGB');ImageDraw.Draw(frame).text((8,8),f'input {yaw} / rotation {yaw*.65:.1f}',fill='white');frames.append(frame)
  strip=Image.new('RGB',(360*len(frames),300))
  for i,frame in enumerate(frames):strip.paste(frame,(360*i,0))
  strip.save(ROOT/'debug/viewer_yaw_sweep.png')
  page.locator('#reset').click();pose(0,0)
  page.locator('#after-crop').click();page.wait_for_function('document.getElementById("lightbox-image").complete');metrics['lightbox']=page.locator('#lightbox').evaluate('(e)=>e.open');page.keyboard.press('Escape')
  page.locator('#auto').click();before=page.evaluate('studio.p.ParamAngleX');page.wait_for_function('(v)=>Math.abs(studio.p.ParamAngleX-v)>.1',arg=before,timeout=5000);metrics['autoGaze']=page.evaluate('[studio.p.ParamEyeBallX,studio.p.ParamEyeBallY]');page.locator('#reset').click()
  metrics.update(page.evaluate("""() => ({minFont:Math.min(...[...document.querySelectorAll('button,label,summary,output')].filter(e=>e.getClientRects().length).map(e=>parseFloat(getComputedStyle(e).fontSize))),imageFailures:[...document.images].filter(i=>i.src&&(!i.complete||!i.naturalWidth)).length,webglError:studio.renderer.gl.getError(),pageOverflow:document.documentElement.scrollWidth>innerWidth})"""))
  page.set_viewport_size({'width':1024,'height':768});page.wait_for_timeout(150);snap('viewer_1024_narrow.png');metrics['narrowOverflow']=page.evaluate('document.documentElement.scrollWidth>innerWidth')
  metrics.update(consoleErrors=errors,requestFailures=failures)
  bounds=metrics['sphereRaster'][0]['bounds'];metrics['maxSphereRasterDrift']=max(abs(v-bounds[i]) for r in metrics['sphereRaster'] if r['angles'][2]==0 for i,v in enumerate(r['bounds']))
  metrics['maxSphereRollSizeDrift']=max(abs((r['bounds'][i+2]-r['bounds'][i])-(bounds[i+2]-bounds[i])) for r in metrics['sphereRaster'] for i in [0,1])
  (ROOT/'debug/viewer_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
  print(json.dumps({k:v for k,v in metrics.items() if k not in ['clipping','neckOcclusion','featureVisibility']},ensure_ascii=False))
  assert metrics['surfaceComparison']['changedPixels']>0
  assert metrics['canopyContinuity']['maxError']<1e-4
  assert metrics['correctionInput']==.73
  assert metrics['hairControlInput']==.8 and metrics['occlusionControlInput']==.5
  assert metrics['attachment']['maxFolded']==0
  # Far ears are allowed to disappear behind the cheek in r10.
  assert all(r['pixels']>0 for r in metrics['featureVisibility'] if not r['id'].startswith('ear-'))
  assert all(r['sideHairCoveredEyePixels']>0 and r['browWithoutHairPixels']>0 for r in metrics['eyeHairOcclusion'])
  assert metrics['maxSphereRasterDrift']<=1 and metrics['maxSphereRollSizeDrift']<=1
  g=metrics['geometry'];assert g['radiusError']<1e-9 and g['eyeRadiusError']<1e-9 and g['neckYawError']==0 and g['eyeLocalHeadLeak']==0 and g['bodyHeadLeak']==0 and g['frontHemisphere'] and g['backHemisphere'] and g['shirtOpacity']==0
  # Raw attachment points are diagnostic; neck pixels outside the face are clipped at the cap.
  assert all(r['exposed']==0 for r in metrics['neckCapClipping'])
  assert all(r['outside']==0 for r in metrics['clipping'])
  assert all(r['visible']>0 for r in metrics['clipping'] if r['type']=='iris' and r['open']==1)
  assert metrics['minFont']>=12
  assert not errors and not failures and metrics['imageFailures']==0 and metrics['webglError']==0 and not metrics['pageOverflow'] and not metrics['narrowOverflow'] and metrics['lightbox'] and metrics['autoGaze']==[0,0]
  print('PASS closed sphere raster, child sphere rigidity, yaw-stationary neck, clipping and UI')
  browser.close()
finally:server.shutdown();server.server_close()
