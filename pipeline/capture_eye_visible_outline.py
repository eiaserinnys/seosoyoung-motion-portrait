"""F-B3 R4: visible whole portrait alpha, independent of the eye clip mask."""
import csv,functools,hashlib,http.server,json,sys,threading
from playwright.sync_api import sync_playwright
from capture_sam3_eyes import ROOT,SETUP,Handler,image

OUT=ROOT/'debug/sam3-eye-v295/visible-outline'
SETUP_OUTLINE='''async()=>{
 const {outsideVisibleOutline}=await import('../pipeline/eye_visible_outline.mjs');
 const {FRAME_CACHE}=await import('./pose-grid.mjs');
 const {deform3D}=await import('./rig.mjs');
 const original=eyeCapture.draw;let key,cache;
 eyeCapture.draw=function(mesh,p,...args){
  const next=p.ParamAngleX+','+p.ParamAngleY;
  if(key!==next){key=next;cache=new Map();}p[FRAME_CACHE]=cache;
  return original.call(this,mesh,p,...args);
 };
 const eyeIds=studio.rig.meshes.filter(m=>m.id.startsWith('eye-')||m.id==='closed-eyes-v1').map(m=>m.id);
 window.visibleEyePose=([yaw,pitch,only])=>{
  Object.assign(studio.p,{ParamAngleX:yaw,ParamAngleY:pitch,ParamAngleZ:0,CorrTurn:1,ParamBodyAngleX:0,ParamBreath:0,ParamEyeLOpen:1,ParamEyeROpen:1,ParamEyeBallX:0,ParamEyeBallY:0});
  const r=eyeCapture,gl=r.gl;
  const state={...studio.state,mode:'normal',wire:false,grid:false,masks:false,before:false,surfaceOnly:false};
  r.render(studio.p,{...state,hidden:new Set(eyeIds),identify:null});
  const w=r.canvas.width,h=r.canvas.height;
  if(w!==1100||h!==1600)throw Error('Unexpected visible-outline canvas size');
  const read=()=>{const raw=new Uint8Array(w*h*4);gl.readPixels(0,0,w,h,gl.RGBA,gl.UNSIGNED_BYTE,raw);return raw;};
  const outline=read();
  const result={yaw,pitch,width:w,height:h,meshes:{},outline:r.canvas.toDataURL('image/png').split(',')[1]};
  for(const id of only||eyeIds){
   r.render(studio.p,{...state,hidden:new Set(),identify:id});
   const q=outsideVisibleOutline(read(),outline,w);q.coordinates=q.coordinates.map(([x,y])=>[x,h-1-y]);result.meshes[id]=q;
  }
  r.render(studio.p,{...state,hidden:new Set(),identify:null});result.color=r.canvas.toDataURL('image/png').split(',')[1];
  result.irisBounds={};
  for(const side of ['left','right']){
   const mesh=studio.rig.meshes.find(m=>m.id==='eye-'+side+'-iris');
   const params={...studio.p,[FRAME_CACHE]:cache},points=mesh.positions.map(pt=>deform3D(mesh,pt,params));
   const xs=points.map(p=>p[0]*r.view.scale+r.view.ox),ys=points.map(p=>p[1]*r.view.scale+r.view.oy);result.irisBounds[side]=[Math.min(...xs),Math.min(...ys),Math.max(...xs),Math.max(...ys)];
  }
  if(gl.getError()!==0)throw Error('Visible-outline WebGL error');
  return result;
 };
}'''

def main():
    smoke='--smoke' in sys.argv
    OUT.mkdir(parents=True,exist_ok=True)
    if not smoke:
        digest=hashlib.sha256(b''.join((ROOT/p).read_bytes() for p in ['bundle/renderer.mjs','bundle/eye-sam3-data.mjs','bundle/assets/texture_00.png'])).hexdigest()
        signature=OUT/'inputs.sha256'
        if signature.exists():assert signature.read_text().strip()==digest,'Visible-outline cache belongs to other inputs'
        signature.write_text(digest+'\n')
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Handler,directory=str(ROOT)))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();rows=[]
    poses=[(-54,0)] if smoke else [(sign*yaw,pitch) for pitch in [-30,0,30] for sign in [-1,1] for yaw in range(46,61)]
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--use-angle=swiftshader'])
            page=browser.new_page(viewport=dict(width=1440,height=1000),device_scale_factor=1)
            page.goto(f'http://127.0.0.1:{server.server_port}/bundle/',wait_until='networkidle')
            page.wait_for_function('window.studio?.ready',timeout=30000)
            page.evaluate('window.requestAnimationFrame=()=>0');page.evaluate(SETUP);page.evaluate(SETUP_OUTLINE)
            for yaw,pitch in poses:
                path=OUT/f'p{pitch}-y{yaw}.json'
                if not smoke and path.exists():row=json.loads(path.read_text())
                else:
                    raw=page.evaluate('(a)=>visibleEyePose(a)',[yaw,pitch,['eye-left-brow','eye-left-upper-lashes'] if smoke else None])
                    row={k:v for k,v in raw.items() if k not in ['color','outline']}
                    for kind in ['color','outline']:
                        # Preserve the full 1100x1600 canvas.
                        import base64,io
                        from PIL import Image
                        Image.open(io.BytesIO(base64.b64decode(raw[kind]))).save(OUT/f'p{pitch}-y{yaw}-{kind}.png')
                    if not smoke:path.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
                rows.append(row)
                print('VISIBLE R4',pitch,yaw,{k:v['pixels'] for k,v in row['meshes'].items() if v['pixels']},flush=True)
            browser.close()
    finally:
        server.shutdown();server.server_close();thread.join()
    report=dict(definition='Eye identify >127; full portrait with all eye meshes hidden has alpha exactly 0. No face-mask reference or edge-band exclusion.',poses=rows,complete=len(rows)==90,passed=all(q['pixels']==0 for r in rows for q in r['meshes'].values()))
    path=OUT/('smoke.json' if smoke else 'report.json');path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('VISIBLE OUTLINE',len(rows),'poses',report['passed'],flush=True)
    if not smoke:
        (ROOT/'docs/sam3-eye-visible-outline.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        with (ROOT/'docs/sam3-eye-R4-poses.csv').open('w',newline='') as stream:
            writer=csv.writer(stream,lineterminator='\n');writer.writerow(['pitch','requested_yaw','mesh','eye_role','outside_visible_outline'])
            for r in rows:
                far='left' if r['yaw']<0 else 'right'
                for mesh,q in r['meshes'].items():writer.writerow([r['pitch'],r['yaw'],mesh,'far' if mesh.startswith('eye-'+far+'-') else 'near' if mesh.startswith('eye-') else 'closed',q['pixels']])
        with (ROOT/'docs/sam3-eye-R4-violations.csv').open('w',newline='') as stream:
            writer=csv.writer(stream,lineterminator='\n');writer.writerow(['pitch','requested_yaw','mesh','x','y'])
            for r in rows:
                for mesh,q in r['meshes'].items():
                    for x,y in q['coordinates']:writer.writerow([r['pitch'],r['yaw'],mesh,x,y])

if __name__=='__main__':main()
