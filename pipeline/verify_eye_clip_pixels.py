"""R5/R4 evidence; only a fully clipped far eye is a protrusion stop gate."""
import functools
import http.server
import json
import threading
import csv
import base64
import io
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
from capture_sam3_eyes import ROOT, ROWS, BOOT, SETUP, Handler, image, binary
from eye_capture_angles import inverse_achieved, achieved

OUT=ROOT/'debug/sam3-eye-v293/far-clip-contract'
BASELINE=ROOT/'debug/sam3-eye-v291/far-clip-contract'
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
   result.controls[side]=eyeClipControls?eyeClipControls('eye-'+side+'-sclera',yaw):{profileFaceClip:Math.abs(yaw)>45,eyeFaceClip:0};
  }
  if(eyeCapture.gl.getError()!==0)throw Error('Eye clip WebGL error');
  return result;
 };
}'''

def pixel_evidence(captures,side):
    exposed=binary(captures[side])&~binary(captures['faceMask'])
    yy,xx=np.where(exposed)
    return dict(pixels=len(xx),coordinates720=np.column_stack([xx,yy]).tolist())

def judge_pose(yaw,counts,baseline):
    criterion='a' if abs(yaw)>=15 else 'b'
    far='left' if yaw<0 else 'right'
    passed={side:counts[side]==0 if criterion=='a' and side==far else True for side in ['left','right']}
    return dict(criterion=criterion,sidePassed=passed,passed=all(passed.values()),
                increasedSides=[s for s in ['left','right'] if counts[s]>baseline[s]])

def save_report(report):
    for row in report['poses']:
        for rev in ['r24','v291']:
            while rev in row['controls'][rev]:
                row['controls'][rev]=row['controls'][rev][rev]
    (ROOT/'docs/sam3-eye-far-clip-contract.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    with (ROOT/'docs/sam3-eye-R4-poses.csv').open('w',newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n');writer.writerow(['pose','criterion','theta_far_achieved','left_role','left_v291','left_r24','right_role','right_v291','right_r24','passed'])
        for r in report['poses']:
            left=r['sides']['left'];right=r['sides']['right']
            writer.writerow([r['name'],r['criterion'],r['thetaFarAchieved'],left['role'],left['pixels'],left['r24Pixels'],right['role'],right['pixels'],right['r24Pixels'],r['passed']])

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    previous=json.loads((ROOT/'docs/sam3-eye-far-clip-contract.json').read_text())
    report=dict(rendererContract='v2: eye soft clip far0..15/near52..67; fixed authored masks and face-skin eye silhouette; nose/mouth retain profile fallback',
                R4Definition='a: achieved theta_far>=15, zero far side only; all other counts report-only against r24',
                R5=previous['R5'],maskBasis='face-skin for both revisions; unchanged face geometry, original r24 eye rendering',poses=[])
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Handler,directory=str(ROOT)))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--use-angle=swiftshader'])
            pages={}
            for rev,path in [('r24','debug/r24/bundle'),('v291','bundle')]:
                page=browser.new_page(viewport=dict(width=1440,height=1000),device_scale_factor=1)
                page.goto(f'http://127.0.0.1:{server.server_port}/{path}/',wait_until='networkidle')
                page.wait_for_function('window.studio?.ready',timeout=30000);page.evaluate('window.requestAnimationFrame=()=>0');page.evaluate(SETUP);page.evaluate(SIDE_SETUP)
                pages[rev]=page
            baseline_page=browser.new_page(viewport=dict(width=1440,height=1000),device_scale_factor=1)
            baseline_page.goto(f'http://127.0.0.1:{server.server_port}/debug/r5-a29/bundle/',wait_until='networkidle')
            baseline_page.wait_for_function('window.studio?.ready',timeout=30000);baseline_page.evaluate('window.requestAnimationFrame=()=>0');baseline_page.evaluate(SETUP)
            neutral={rev:page.evaluate('(a)=>captureEyes(a)',[0,0,False,True]) for rev,page in [('r5-a29',baseline_page),('v291',pages['v291'])]}
            neutral_full={rev:Image.open(io.BytesIO(base64.b64decode(raw['full']))).convert('RGBA') for rev,raw in neutral.items()}
            for rev,raw in neutral.items():
                image(raw['color']).save(OUT/f'{rev}-neutral.png')
                neutral_full[rev].save(OUT/f'{rev}-neutral-full.png')
            diff=np.abs(np.array(neutral_full['v291']).astype(int)-np.array(neutral_full['r5-a29']).astype(int))
            report['R5']=dict(baseline='a29bb358231566d0c90b1b058bc8a153cfc57410',scope='full canvas, hair on, neutral',canvasSize=list(neutral_full['v291'].size),changedPixels=int(np.any(diff,axis=2).sum()),maxChannelDifferenceLSB=int(diff.max()),passed=not np.any(diff))
            baseline_page.close()
            print('R5',json.dumps(report['R5']),flush=True)
            if not report['R5']['passed']:
                report['status']='STOP_R5';save_report(report);browser.close();return
            # Reference inputs are inverted before passing them to either viewer.
            poses=[dict(name=r['name'],targetAchieved=[r['yaw'],r['pitch']],requested=inverse_achieved(BOOT['achieved']['c'],r['yaw'],r['pitch'])) for r in ROWS]
            poses += [dict(name=f'sweep-{yaw}-{pitch}',requested=[yaw,pitch],targetAchieved=achieved(BOOT['achieved']['c'],yaw,pitch)) for yaw in [-45,-40,-35,-30,30,35,40,45] for pitch in [-30,0,30]]
            for pose in poses:
                captures={};evidence={};controls={}
                for rev,page in pages.items():
                    paths={k:OUT/f"{rev}-{pose['name']}-{k}.png" for k in ['color','faceMask','left','right']}
                    baseline_paths={k:BASELINE/f"{rev}-{pose['name']}-{k}.png" for k in paths}
                    if rev=='r24' and all(p.exists() for p in baseline_paths.values()):
                        captures[rev]={k:Image.open(p).convert('RGBA') for k,p in baseline_paths.items()}
                        controls[rev]={s:dict(profileFaceClip=abs(pose['requested'][0])>45,eyeFaceClip=0) for s in ['left','right']}
                        for k,im in captures[rev].items():im.save(paths[k])
                    elif all(p.exists() for p in paths.values()):
                        captures[rev]={k:Image.open(p).convert('RGBA') for k,p in paths.items()}
                        old=next(r for r in previous['poses'] if r['name']==pose['name'])
                        controls[rev]=old['controls'][rev]
                    elif rev=='v291' and pose['name']=='front' and all((OUT/f'front-{k}.png').exists() for k in paths):
                        captures[rev]={k:Image.open(OUT/f'front-{k}.png').convert('RGBA') for k in ['color','faceMask','left','right']}
                        controls[rev]=previous['poses'][0]['controls']
                        for k,im in captures[rev].items():im.save(paths[k])
                    else:
                        raw=page.evaluate('(args)=>captureClipPose(args)',pose['requested'])
                        captures[rev]={k:image(raw[k]) for k in ['color','faceMask','left','right']}
                        controls[rev]=raw['controls']
                        for k,im in captures[rev].items():im.save(OUT/f"{rev}-{pose['name']}-{k}.png")
                # Compare both eye renders against the same face silhouette.
                # Face geometry is unchanged; r24 itself retains its authored clip.
                captures['r24']['faceMask']=captures['v291']['faceMask']
                captures['r24']['faceMask'].save(OUT/f"r24-{pose['name']}-faceMask.png")
                evidence={rev:{side:pixel_evidence(captures[rev],side) for side in ['left','right']} for rev in captures}
                row={**pose,'actualAchieved':achieved(BOOT['achieved']['c'],*pose['requested']),'controls':controls,
                     'thetaFarAchieved':abs(pose['targetAchieved'][0])}
                row['sides']={side:{**evidence['v291'][side],
                              'r24Pixels':evidence['r24'][side]['pixels'],
                              'r24Coordinates720':evidence['r24'][side]['coordinates720'],
                              'role':'far' if pose['requested'][0]*(1 if side=='left' else -1)<0 else 'near'} for side in ['left','right']}
                row.update(judge_pose(pose['targetAchieved'][0],{s:evidence['v291'][s]['pixels'] for s in evidence['v291']},{s:evidence['r24'][s]['pixels'] for s in evidence['r24']}))
                report['poses'].append(row)
                print('R4',pose['name'],row['criterion'],json.dumps({side:(q['pixels'],q['r24Pixels']) for side,q in row['sides'].items()}),flush=True)
                save_report(report)
                if row['criterion']=='a' and not row['passed']:
                    report['status']='STOP_R4_A_PROTRUSION'
                    overlay=np.array(captures['v291']['color']).copy()
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
    save_report(report)
    print(json.dumps(dict(status=report['status'],poses=len(report['poses']),last=report['poses'][-1]),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
