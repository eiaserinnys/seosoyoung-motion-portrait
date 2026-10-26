"""Collect all R4 outside pixels at the unchanged binary threshold, without stops."""
import functools
import http.server
import json
import threading
import csv
import cv2
import numpy as np
from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright
from capture_sam3_eyes import ROOT, ROWS, BOOT, SETUP, Handler, image, binary
from eye_capture_angles import inverse_achieved, achieved

OUT=ROOT/'debug/sam3-eye-v294/far-clip-contract'
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

def outside_samples(captures,side):
    return [dict(x=x,y=y,mask_value=min(captures['faceMask'].getpixel((x,y))[:3]),
                 eye_value=min(captures[side].getpixel((x,y))[:3]),eye_alpha=captures[side].getpixel((x,y))[3])
            for x,y in pixel_evidence(captures,side)['coordinates720']]

def edge_evidence(captures,side):
    mask=binary(captures['faceMask'])
    outside=binary(captures[side])&~mask
    band=cv2.dilate(mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&~mask
    yy,xx=np.where(outside&~band)
    return dict(edgeBandPixels=int((outside&band).sum()),outsideBeyondBandPixels=len(xx),
                outsideBeyondBandCoordinates720=np.column_stack([xx,yy]).tolist())

def export_violations(report):
    directory=DOC/'sam3_eye_R4_violations';directory.mkdir(parents=True,exist_ok=True)
    tiles=[];table=[]
    for row in report['poses']:
        mask=Image.open(OUT/f"v291-{row['name']}-faceMask.png").convert('RGBA')
        color=Image.open(OUT/f"v291-{row['name']}-color.png").convert('RGB')
        for side in ['left','right']:
            eye=Image.open(OUT/f"v291-{row['name']}-{side}.png").convert('RGBA')
            samples=outside_samples({'faceMask':mask,side:eye},side)
            row['sides'][side]['samples']=samples
            if not samples:continue
            yy,xx=np.where(binary(eye));box=(int(xx.min())-8,int(yy.min())-8,int(xx.max())+9,int(yy.max())+9)
            marked=color.copy()
            for point in samples:marked.putpixel((point['x'],point['y']),(255,0,0))
            pair=[im.crop(box).resize(((box[2]-box[0])*4,(box[3]-box[1])*4),Image.Resampling.NEAREST) for im in [color,marked]]
            tile=Image.new('RGB',(pair[0].width*2,pair[0].height+28),'#eeeeee')
            for j,im in enumerate(pair):tile.paste(im,(j*im.width,0))
            ImageDraw.Draw(tile).text((8,pair[0].height+5),f"{row['name']} {side} | raw / marked | {len(samples)} pixels",fill='black')
            dest=directory/f"{row['name']}-{side}.png";tile.save(dest);tiles.append(tile)
            crop=str(dest.relative_to(ROOT));row['sides'][side]['crop4x']=crop
            table.extend([dict(pose=row['name'],eye=side,**point,crop4x=crop) for point in samples])
    with (ROOT/'docs/sam3-eye-R4-violations.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['pose','eye','x','y','mask_value','eye_value','eye_alpha','crop4x'],lineterminator='\n')
        writer.writeheader();writer.writerows(table)
    sheet=Image.new('RGB',(max((im.width for im in tiles),default=400),max(28,sum(im.height for im in tiles))),'#eeeeee');y=0
    for im in tiles:sheet.paste(im,(0,y));y+=im.height
    if not tiles:ImageDraw.Draw(sheet).text((8,8),'No outside pixels at binary127',fill='black')
    sheet.save(DOC/'sam3_eye_R4_violations_4x.png')
    report.update(recordedPixelCount=len(table),affectedEyes=len(tiles),violationTable='docs/sam3-eye-R4-violations.csv',violationSheet='docs/captures/sam3_eye_R4_violations_4x.png')

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
        writer=csv.writer(stream,lineterminator='\n');writer.writerow(['pose','requested_yaw','theta_far_achieved','left_role','left_v291','left_r24','left_edge_band','left_beyond_band','right_role','right_v291','right_r24','right_edge_band','right_beyond_band'])
        for r in report['poses']:
            left=r['sides']['left'];right=r['sides']['right']
            writer.writerow([r['name'],r['requested'][0],r['thetaFarAchieved'],left['role'],left['pixels'],left['r24Pixels'],left['edgeBandPixels'],left['outsideBeyondBandPixels'],right['role'],right['pixels'],right['r24Pixels'],right['edgeBandPixels'],right['outsideBeyondBandPixels']])

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    previous=json.loads((ROOT/'docs/sam3-eye-far-clip-contract.json').read_text())
    report=dict(rendererContract='F-B2: eye soft clip to solid face-skin + skinFillClipMesh union; other clips unchanged',
                R4Definition='Requested |yaw| >=15: far-eye pixels beyond skin-silhouette must be zero. One-pixel edge band is report-only. All raw outside pixels retained at binary127.',
                edgeBandDefinition='Outside binary127 silhouette within its 3x3 dilation (one pixel, 8-neighbour); recorded separately without deleting raw observations.',
                R5=previous['R5'],maskBasis='skin-silhouette for both revisions; original r24 eye rendering',poses=[])
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
            assert report['R5']['passed']
            print('REUSED R5',json.dumps(report['R5']),flush=True)
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
                evidence={rev:{side:{**pixel_evidence(captures[rev],side),**edge_evidence(captures[rev],side)} for side in ['left','right']} for rev in captures}
                row={**pose,'actualAchieved':achieved(BOOT['achieved']['c'],*pose['requested']),'controls':controls,
                     'thetaFarAchieved':abs(pose['targetAchieved'][0])}
                row['sides']={side:{**evidence['v291'][side],
                              'r24Pixels':evidence['r24'][side]['pixels'],
                              'r24Coordinates720':evidence['r24'][side]['coordinates720'],
                              'role':'far' if pose['requested'][0]*(1 if side=='left' else -1)<0 else 'near'} for side in ['left','right']}
                row['increasedSides']=[s for s in ['left','right'] if evidence['v291'][s]['pixels']>evidence['r24'][s]['pixels']]
                report['poses'].append(row)
                print('R4',pose['name'],json.dumps({side:(q['pixels'],q['r24Pixels']) for side,q in row['sides'].items()}),flush=True)
                save_report(report)
            report['status']='R4_RECORDED'
            report['complete']=len(report['poses'])==48
            report['requiredFarEyes']=[dict(pose=r['name'],eye=s,pixels=q['pixels'],edgeBandPixels=q['edgeBandPixels'],outsideBeyondBandPixels=q['outsideBeyondBandPixels']) for r in report['poses'] if abs(r['requested'][0])>=15 for s,q in r['sides'].items() if q['role']=='far']
            report['passed']=all(q['outsideBeyondBandPixels']==0 for q in report['requiredFarEyes'])
            browser.close()
    finally:
        server.shutdown();server.server_close();thread.join()
    export_violations(report)
    save_report(report)
    print(json.dumps(dict(status=report['status'],poses=len(report['poses']),last=report['poses'][-1]),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
