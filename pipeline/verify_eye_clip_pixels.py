"""Execute the approved R5/R4 checks, stopping at the first exposed eye pose."""
import functools
import http.server
import json
import threading
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
from capture_sam3_eyes import ROOT, ROWS, BOOT, SETUP, Handler, image, binary
from eye_capture_angles import inverse_achieved, achieved

OUT=ROOT/'debug/sam3-eye-v291/far-clip-contract'
DOC=ROOT/'docs/captures'
SIDE_SETUP='''async()=>{
 const {eyeClipControls}=await import('./renderer.mjs');
 window.captureClipPose=([yaw,pitch])=>{
  const result=captureEyes([yaw,pitch,'spill',false]);
  const keep=id=>id==='face-skin'||id.startsWith('eye-')||id==='closed-eyes-v1'||id==='nose-detail'||id.startsWith('mouth-');
  const hidden=new Set(studio.rig.meshes.filter(m=>!keep(m.id)).map(m=>m.id));
  const state={...studio.state,hidden,mode:'normal',wire:false,grid:false,before:false,surfaceOnly:false};
  result.controls={};
  for(const side of ['left','right']){
   const identify=studio.rig.meshes.filter(m=>m.id.startsWith('eye-'+side+'-')).map(m=>m.id);
   eyeCapture.render(studio.p,{...state,identify});
   result[side]=eyeCapture.canvas.toDataURL('image/png').split(',')[1];
   result.controls[side]=eyeClipControls('eye-'+side+'-sclera',yaw);
  }
  if(eyeCapture.gl.getError()!==0)throw Error('Eye clip WebGL error');
  return result;
 };
}'''

def pixel_evidence(captures,side):
    exposed=binary(captures[side])&~binary(captures['faceMask'])
    yy,xx=np.where(exposed)
    return dict(pixels=len(xx),coordinates720=np.column_stack([xx,yy]).tolist())

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    report=dict(rendererContract='profile-only alpha fallback; far-eye smoothstep(0,15) alpha face clip',poses=[])
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Handler,directory=str(ROOT)))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--use-angle=swiftshader'])
            page=browser.new_page(viewport=dict(width=1440,height=1000),device_scale_factor=1)
            page.goto(f'http://127.0.0.1:{server.server_port}/bundle/',wait_until='networkidle')
            page.wait_for_function('window.studio?.ready',timeout=30000);page.evaluate(SETUP);page.evaluate(SIDE_SETUP)
            raw=page.evaluate('(args)=>captureEyes(args)',[0,0,False,False]);neutral=image(raw['color'])
            neutral.save(OUT/'neutral.png')
            old=Image.open(ROOT/'debug/sam3-eye-v291/render/r24-neutral.png').convert('RGBA')
            delta=abs(np.array(neutral).astype(int)-np.array(old).astype(int))[355:460,180:520]
            report['R5']=dict(changedPixels=int(np.any(delta!=0,axis=2).sum()),maxChannelDifferenceLSB=int(delta.max()))
            report['R5']['passed']=report['R5']['changedPixels']==0 and report['R5']['maxChannelDifferenceLSB']<=1
            print('R5',json.dumps(report['R5']),flush=True)
            # Reference inputs are inverted before passing them to either viewer.
            poses=[dict(name=r['name'],targetAchieved=[r['yaw'],r['pitch']],requested=inverse_achieved(BOOT['achieved']['c'],r['yaw'],r['pitch'])) for r in ROWS]
            poses += [dict(name=f'sweep-{yaw}-{pitch}',requested=[yaw,pitch],targetAchieved=achieved(BOOT['achieved']['c'],yaw,pitch)) for yaw in [-45,-40,-35,-30,30,35,40,45] for pitch in [-30,0,30]]
            for pose in poses:
                raw=page.evaluate('(args)=>captureClipPose(args)',pose['requested'])
                captures={k:image(raw[k]) for k in ['color','faceMask','left','right']}
                for k,im in captures.items():im.save(OUT/f"{pose['name']}-{k}.png")
                row={**pose,'actualAchieved':achieved(BOOT['achieved']['c'],*pose['requested']),'controls':raw['controls']}
                row['sides']={side:{**pixel_evidence(captures,side),
                              'role':'far' if pose['requested'][0]*(1 if side=='left' else -1)<0 else 'near'} for side in ['left','right']}
                report['poses'].append(row)
                print('R4',pose['name'],json.dumps({side:(q['role'],q['pixels']) for side,q in row['sides'].items()}),flush=True)
                if any(q['pixels'] for q in row['sides'].values()):
                    report['status']='STOP_R4_PROTRUSION'
                    overlay=np.array(captures['color']).copy()
                    for side in ['left','right']:
                        for x,y in row['sides'][side]['coordinates720']:overlay[y,x]=[255,0,0,255]
                    dest=DOC/f"sam3_eye_clip_failure_{pose['name']}.png"
                    Image.fromarray(overlay).save(dest)
                    report['failureOverlay']=str(dest.relative_to(ROOT))
                    break
            else:
                report['status']='R4_PASS'
            browser.close()
    finally:
        server.shutdown();server.server_close();thread.join()
    (ROOT/'docs/sam3-eye-far-clip-contract.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
