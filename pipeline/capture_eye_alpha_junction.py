"""J3: actual isolated eye alpha at requested 0.1-degree steps, one case/run."""
import sys
sys.dont_write_bytecode=True
import functools,http.server,json,threading,hashlib
from pathlib import Path
from playwright.sync_api import sync_playwright
from capture_sam3_eyes import ROOT,SETUP,Handler

OUT=ROOT/'debug/sam3-eye-v293/alpha'
ALPHA_SETUP='''async()=>{
 const {FRAME_CACHE}=await import('./pose-grid.mjs');
 // Geometry is state-independent; reuse the real renderer's frame cache for
 // the two isolated side passes of one pose. All draw and shader calls remain.
 const original=eyeCapture.draw;let key=null,cache;
 eyeCapture.draw=function(mesh,p,...args){
  const next=p.ParamAngleX+','+p.ParamAngleY;
  if(key!==next){key=next;cache=new Map();}p[FRAME_CACHE]=cache;
  return original.call(this,mesh,p,...args);
 };
 window.alphaPose=([yaw,pitch])=>{
  Object.assign(studio.p,{ParamAngleX:yaw,ParamAngleY:pitch,ParamAngleZ:0,CorrTurn:1,
    ParamBodyAngleX:0,ParamBreath:0,ParamEyeLOpen:1,ParamEyeROpen:1,ParamEyeBallX:0,ParamEyeBallY:0});
  const result={yaw,pitch};
  for(const side of ['left','right']){
   const selectedIds=new Set(studio.rig.meshes.filter(m=>m.id.startsWith('eye-'+side+'-')).map(m=>m.id));
   eyeCapture.render(studio.p,{...studio.state,hidden:new Set(),mode:'solo',selectedIds,wire:false,grid:false,before:false,surfaceOnly:false,identify:null});
   const gl=eyeCapture.gl,w=eyeCapture.canvas.width,h=eyeCapture.canvas.height,raw=new Uint8Array(w*h*4);
   gl.readPixels(0,0,w,h,gl.RGBA,gl.UNSIGNED_BYTE,raw);let sum=0;
   for(let i=3;i<raw.length;i+=4)sum+=raw[i];result[side]=sum/255;
   if(gl.getError()!==0)throw Error('J3 WebGL error');
  }
  return result;
 };
}'''


def alpha_report(cases,V_own):
    limit=500*V_own/10;peaks={r:dict(difference=0) for r in ['far','near']}
    for samples in cases:
        case_sign=1 if samples[-1]['yaw']>0 else -1
        for a,b in zip(samples,samples[1:]):
            for side in ['left','right']:
                role='far' if side==('left' if case_sign<0 else 'right') else 'near'
                difference=abs(b[side]-a[side])
                if difference>peaks[role]['difference']:peaks[role]=dict(difference=difference,side=side,pitch=b['pitch'],fromYaw=a['yaw'],toYaw=b['yaw'],fromAlpha=a[side],toAlpha=b[side])
    return dict(limitAlphaPixels=limit,stepRequestedDeg=.1,poses=sum(map(len,cases)),peaks=peaks,passed=all(r['difference']<=limit for r in peaks.values()))


def main():
    pitch,sign=map(int,sys.argv[1].split(','));OUT.mkdir(parents=True,exist_ok=True)
    geometry=json.loads((ROOT/'docs/sam3-eye-junction-v2.json').read_text())
    digest=hashlib.sha256((ROOT/'bundle/renderer.mjs').read_bytes()+(ROOT/'bundle/assets/texture_00.png').read_bytes()+geometry['inputSha'].encode()).hexdigest()
    signature=OUT/'inputs.sha256'
    if signature.exists():assert signature.read_text().strip()==digest,'Alpha cache belongs to other runtime inputs'
    signature.write_text(digest+'\n')
    progress=OUT/f'progress-{pitch}-{sign}.json'
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Handler,directory=str(ROOT)))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();rows=json.loads(progress.read_text()) if progress.exists() else []
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--use-angle=swiftshader'])
            page=browser.new_page(viewport=dict(width=1440,height=1000),device_scale_factor=1)
            page.goto(f'http://127.0.0.1:{server.server_port}/bundle/',wait_until='networkidle')
            page.wait_for_function('window.studio?.ready',timeout=30000)
            page.evaluate('window.requestAnimationFrame=()=>0');page.evaluate(SETUP);page.evaluate(ALPHA_SETUP)
            for tick in range(len(rows),901):
                row=page.evaluate('(a)=>alphaPose(a)',[sign*tick/10,pitch]);rows.append(row)
                if tick%100==0:
                    progress.write_text(json.dumps(rows)+'\n');print('J3',pitch,sign,tick/10,flush=True)
            browser.close()
    finally:
        server.shutdown();server.server_close();thread.join()
    (OUT/f'case-{pitch}-{sign}.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    if len(list(OUT.glob('case-*.json')))==6:
        report=alpha_report([json.loads(path.read_text()) for path in sorted(OUT.glob('case-*.json'))],geometry['J0']['V_own'])
        report['inputSha']=digest
        (ROOT/'docs/sam3-eye-junction-alpha.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
