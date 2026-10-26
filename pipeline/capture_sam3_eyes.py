"""Compare actual WebGL eye masks with the approved reference geometry.

The r24 bundle is a read-only git archive in debug/r24/bundle. All captures use
the same renderer size, pose parameters, identify masks and measurement code.
"""
import base64
import functools
import http.server
import io
import json
import threading
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright
from measure_eye_masks import opening26, measure26, ROOT
from segment_eye_references import normalized_image

OUT=ROOT/'debug/sam3-eye-v291/render'
DOC=ROOT/'docs/captures'
M=json.loads((ROOT/'source/viewer/sam3-eye-measurements.json').read_text())
AUDIT=json.loads((ROOT/'debug/sam3-eye-v291/evaluation.json').read_text())
ROWS=json.loads((ROOT/'sam3_masks/references/seedance/selection.json').read_text())['frames']
BOOT=json.loads((ROOT/'source/references/seedance/eye-bootstrap.json').read_text())
ORDER=['up-left','up','up-right','left','front','right','down-left','down','down-right']

class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self,*args):
        pass

SETUP='''async()=>{
 const {Renderer}=await import('./renderer.mjs');
 const canvas=document.createElement('canvas'),overlay=document.createElement('canvas');
 Object.assign(canvas.style,{width:'1100px',height:'1600px'});document.body.append(canvas);
 window.eyeCapture=new Renderer(canvas,overlay,studio.rig,studio.renderer.images);
 window.captureEyes=([yaw,pitch,masks,full])=>{
  studio.setPose(yaw,pitch,0,'SAM eye verification');studio.p.CorrTurn=1;
  Object.assign(studio.p,{ParamBodyAngleX:0,ParamBreath:0,ParamEyeLOpen:1,ParamEyeROpen:1,ParamEyeBallX:0,ParamEyeBallY:0});
  const keep=id=>id==='face-skin'||id.startsWith('eye-')||id==='closed-eyes-v1'||id==='nose-detail'||id.startsWith('mouth-');
  const hidden=new Set(studio.rig.meshes.filter(m=>!keep(m.id)).map(m=>m.id));
  const state={...studio.state,hidden,mode:'normal',wire:false,grid:false,before:false,surfaceOnly:false,identify:null};
  const result={};
  eyeCapture.render(studio.p,state);result.color=canvas.toDataURL('image/png').split(',')[1];
  if(full){eyeCapture.render(studio.p,{...state,hidden:new Set()});result.full=canvas.toDataURL('image/png').split(',')[1];}
  if(masks){
   if(masks===true)for(const side of ['left','right']){
    const ids=suffixes=>suffixes.map(s=>'eye-'+side+'-'+s);
    const groups={env:ids(['sclera','iris','pupil','highlight','upper-lashes','lower-line']),
     inner:ids(['sclera','iris','pupil','highlight']),iris:ids(['iris','pupil','highlight']),lash:ids(['upper-lashes'])};
    for(const [name,identify] of Object.entries(groups)){
     eyeCapture.render(studio.p,{...state,identify});result[side+'-'+name]=canvas.toDataURL('image/png').split(',')[1];
    }
   }
   eyeCapture.render(studio.p,{...state,identify:studio.rig.meshes.filter(m=>m.id.startsWith('eye-')).map(m=>m.id)});
   result.allEyes=canvas.toDataURL('image/png').split(',')[1];
   // Read the mask actually bound for eyes, independent of later nose/mouth draws.
   const eyeIds=studio.rig.meshes.filter(m=>m.id.startsWith('eye-')).map(m=>m.id);
   eyeCapture.render(studio.p,{...state,mode:'solo',hidden:new Set(),selectedIds:new Set(eyeIds),identify:eyeIds});
   const gl=eyeCapture.gl,w=canvas.width,h=canvas.height,raw=new Uint8Array(w*h*4);
   gl.bindFramebuffer(gl.FRAMEBUFFER,eyeCapture.faceFramebuffer);gl.readPixels(0,0,w,h,gl.RGBA,gl.UNSIGNED_BYTE,raw);gl.bindFramebuffer(gl.FRAMEBUFFER,null);
   const mc=document.createElement('canvas');mc.width=w;mc.height=h;const ctx=mc.getContext('2d'),im=ctx.createImageData(w,h);
   for(let y=0;y<h;y++)for(let x=0;x<w;x++){
    const a=raw[((h-1-y)*w+x)*4+3],i=(y*w+x)*4;im.data[i]=im.data[i+1]=im.data[i+2]=a;im.data[i+3]=255;
   }
   ctx.putImageData(im,0,0);result.faceMask=mc.toDataURL('image/png').split(',')[1];
  }
  if(eyeCapture.gl.getError()!==0)throw Error('WebGL eye capture error');
  return result;
 };
}'''

def image(encoded,full=False):
    im=Image.open(io.BytesIO(base64.b64decode(encoded))).convert('RGBA')
    return im.crop((38,32,1062,1056) if full else (190,52,910,772))

def binary(im):
    a=np.array(im)
    return (a[:,:,:3].min(2)>127)&(a[:,:,3]>127)

def measure(captures,row,side):
    env,inner,iris,lash=[binary(captures[side+'-'+kind]) for kind in ['env','inner','iris','lash']]
    # The rendered opening is the visible non-lash aperture. The mask alpha
    # and lash occlusion already supply its semantics; no SAM or RGB cleanup.
    cols=np.flatnonzero(lash.any(0))
    last=np.interp(np.arange(720),cols,np.where(lash,np.arange(720)[:,None],-1).max(0)[cols])
    sy=row['yaw']*(1 if side=='left' else -1)
    q=opening26(env,inner,iris,side,sy,row['pitch'],t_lash=M['K'][side]['tLash'])
    q.update(measure26(inner,inner,iris,iris,lash,last,q,{},side,sy,row['pitch'],rest_ref=M['K'][side]))
    return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in q.items()}

def gates(records,front):
    ref={(r['name'],r['side']):r['q'] for r in M['records']}
    fields=['dTemp','dNas','h','deltaY','a']
    rms={}
    for field in fields:
        rms[field]={}
        for rev in ['r24','v291']:
            errors=[]
            for r in records:
                q=r[rev];o=ref[(r['name'],r['side'])]
                errors.append((q['delta'][1]-o['delta'][1])*M['K'][r['side']]['w'] if field=='deltaY' else q[field]-o[field])
            rms[field][rev]=float(np.sqrt(np.mean(np.square(errors))))
        rms[field]['limit']=2*AUDIT['noise'][field]['sigma']
        rms[field]['passed']=rms[field]['v291']<rms[field]['r24'] and rms[field]['v291']<=rms[field]['limit']
    excluded={o['eye'] for side in ['left','right'] for o in AUDIT['fit'][side]['dNas']['outliers']}
    nasal=[];paired=[]
    for name in [r['name'] for r in ROWS]:
        errors={}
        for side in ['left','right']:
            r=next(r for r in records if r['name']==name and r['side']==side)
            error=r['v291']['dNas']-ref[(name,side)]['dNas']
            if name+':'+side not in excluded:
                errors[side]=error;nasal.append(dict(eye=name+':'+side,errorPx=error))
        if len(errors)==2:paired.append(dict(name=name,errorDifferencePx=abs(errors['left']-errors['right'])))
    limit=3*AUDIT['noise']['dNas']['sigma']
    front_delta=np.abs(np.array(front['v291']).astype(int)-np.array(front['r24']).astype(int))
    # Both visible eye groups and their complete neutral authored bounding box.
    front_eye=front_delta[355:460,180:520]
    return dict(R2=rms,R3=dict(limitPx=limit,excluded=sorted(excluded),
        maxEyeErrorPx=max(abs(r['errorPx']) for r in nasal),maxPairedErrorPx=max(r['errorDifferencePx'] for r in paired),
        errors=nasal,paired=paired),R5=dict(maxChannelDifferenceLSB=int(front_eye.max()),passed=bool(front_eye.max()<=1)))

def sheets(colors):
    names=ORDER+['aux-0213','aux-0255','aux-0074','aux-0102','aux-0276','aux-0277','aux-0043','aux-0241']
    assert {'left','right','up-left','up-right','down-left','aux-0043','aux-0241'}<=set(names)
    rows=[]
    for i,name in enumerate(names):
        row=next(r for r in ROWS if r['name']==name)
        ims=[normalized_image(row),colors['v291'][name],colors['r24'][name]]
        reference=[r['q'] for r in M['records'] if r['name']==name]
        points=np.array([q[k] for q in reference for k in ['Pmin','Pmax','Papex','Pbot']])
        bounds=[(points[:,0].min(),points[:,1].min(),points[:,0].max(),points[:,1].max())]
        for rev in ['v291','r24']:
            mask=np.logical_or(*[binary(Image.open(OUT/f'{rev}-{name}-{side}-env.png').convert('RGBA')) for side in ['left','right']])
            yy,xx=np.where(mask);bounds.append((xx.min(),yy.min(),xx.max(),yy.max()))
        width=int(max(b[2]-b[0] for b in bounds))+32;height=int(max(b[3]-b[1] for b in bounds))+40
        tiles=[]
        for im,b in zip(ims,bounds):
            cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2;x=int(cx-width/2);y=int(cy-height/2)
            tiles.append(im.convert('RGB').crop((x,y,x+width,y+height)).resize((width*2,height*2),Image.Resampling.NEAREST))
        rows.append((name,tiles,width*2,height*2))
    cell_width=max(r[2] for r in rows)+20;sheet=Image.new('RGB',(cell_width*3,sum(r[3]+26 for r in rows)),'#eeeeee');y=0
    for name,tiles,width,height in rows:
        for j,(im,label) in enumerate(zip(tiles,['reference','F-B2','r24'])):
            sheet.paste(im,(j*cell_width,y));ImageDraw.Draw(sheet).text((j*cell_width+4,y+height+3),name+' | '+label,fill='black')
        y+=height+26
    sheet.save(DOC/'sam3_eye_R7_comparison.png')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Handler,directory=str(ROOT)))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    colors={rev:{} for rev in ['r24','v291']};captured={rev:{} for rev in colors};records=[];front={};spill=[]
    # Completed color and identify passes are immutable evidence; resume from
    # them after an interrupted browser command rather than rerendering them.
    cached=all((OUT/f'{rev}-{row["name"]}-color.png').exists() for rev in colors for row in ROWS)
    if cached:
        for rev in colors:
            for row in ROWS:
                captures={kind:Image.open(OUT/f'{rev}-{row["name"]}-{kind}.png').convert('RGBA')
                          for kind in ['color','allEyes','faceMask']+[s+'-'+k for s in ['left','right'] for k in ['env','inner','iris','lash']]}
                colors[rev][row['name']]=captures['color']
                captured[rev][row['name']]={side:measure(captures,row,side) for side in ['left','right']}
                if rev=='v291':spill.append(dict(name=row['name'],pixels=int((binary(captures['allEyes'])&~binary(captures['faceMask'])).sum())))
            front[rev]=Image.open(OUT/f'{rev}-neutral.png').convert('RGBA')
        print('REUSED 48 complete pose captures',flush=True)
        cached_records=[dict(name=row['name'],side=side,**{rev:captured[rev][row['name']][side] for rev in captured}) for row in ROWS for side in ['left','right']]
        (OUT/'observed-metrics.json').write_text(json.dumps(gates(cached_records,front),ensure_ascii=False,indent=2)+'\n')
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--use-angle=swiftshader'])
            for rev,path in [('r24','debug/r24/bundle'),('v291','bundle')]:
                if cached and rev=='r24':continue
                page=browser.new_page(viewport=dict(width=1440,height=1000),device_scale_factor=1)
                page.goto(f'http://127.0.0.1:{server.server_port}/{path}/',wait_until='networkidle')
                page.wait_for_function('window.studio?.ready',timeout=30000);page.evaluate(SETUP)
                for row in ([] if cached else ROWS):
                    raw=page.evaluate('(args)=>captureEyes(args)', [row['yaw'],row['pitch'],True,row['name'] in ORDER])
                    captures={k:image(v,full=k=='full') for k,v in raw.items()}
                    colors[rev][row['name']]=captures['color']
                    captured[rev][row['name']]={side:measure(captures,row,side) for side in ['left','right']}
                    for k,im in captures.items():im.save(OUT/f"{rev}-{row['name']}-{k}.png")
                    if rev=='v291':
                        outside=binary(captures['allEyes'])&~binary(captures['faceMask'])
                        spill.append(dict(name=row['name'],pixels=int(outside.sum())))
                    print('CAPTURE',rev,row['name'],flush=True)
                if not cached:
                    raw=page.evaluate('(args)=>captureEyes(args)',[0,0,False,False]);front[rev]=image(raw['color'])
                    front[rev].save(OUT/f'{rev}-neutral.png')
                if rev=='v291':
                    sweep=[]
                    for yaw in [15,17,19,21,23,25,27,29,30]:
                        requested=yaw*45/BOOT['achieved']['c'][0]
                        dest=OUT/f'sweep-achieved-{yaw}.png'
                        if dest.exists():im=Image.open(dest).convert('RGBA')
                        else:
                            raw=page.evaluate('(args)=>captureEyes(args)',[requested,0,False,False]);im=image(raw['color']);im.save(dest)
                        sweep.append((yaw,im))
                    strip=Image.new('RGB',(len(sweep)*320,300),'#eeeeee')
                    for i,(yaw,im) in enumerate(sweep):
                        strip.paste(im.convert('RGB').crop((405,330,565,470)).resize((320,280)),(i*320,0))
                        ImageDraw.Draw(strip).text((i*320+4,283),f'achieved yaw {yaw}',fill='black')
                    strip.save(DOC/'sam3_eye_R7_temporal_sweep.png')
                    for yaw in [-45,-40,-35,-30,30,35,40,45]:
                        for pitch in [-30,0,30]:
                            dest=OUT/f'spill-{yaw}-{pitch}.json'
                            if dest.exists():result=json.loads(dest.read_text())
                            else:
                                raw=page.evaluate('(args)=>captureEyes(args)',[yaw,pitch,'spill',False])
                                result=dict(yaw=yaw,pitch=pitch,pixels=int((binary(image(raw['allEyes']))&~binary(image(raw['faceMask']))).sum()))
                                dest.write_text(json.dumps(result)+'\n')
                            spill.append(result)
                    page.evaluate('studio.renderer.render(studio.p,studio.state)')
                    page.set_viewport_size(dict(width=1440,height=1000));page.screenshot(path=str(DOC/'sam3_eye_viewer_wide.png'))
                    page.set_viewport_size(dict(width=720,height=1000));page.screenshot(path=str(DOC/'sam3_eye_viewer_narrow.png'))
                page.close()
            browser.close()
    finally:
        server.shutdown();server.server_close();thread.join()
    for row in ROWS:
        for side in ['left','right']:
            records.append(dict(name=row['name'],side=side,**{rev:captured[rev][row['name']][side] for rev in captured}))
    report=gates(records,front);report['R4']=dict(outsidePixels=sum(r['pixels'] for r in spill),poses=spill)
    report['poseInputs']='Same 24 achieved yaw/pitch labels supplied as viewer parameters in both revisions; v291 retains requested-to-achieved A.'
    report['records']=records
    (OUT/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    sheets(colors)
    grid=Image.new('RGB',(1440,1440),'#1e2430')
    for i,name in enumerate(ORDER):
        im=Image.open(OUT/f'v291-{name}-full.png').convert('RGB').crop((152,20,872,740)).resize((480,480))
        ImageDraw.Draw(im).text((8,8),name,fill='white');grid.paste(im,((i%3)*480,(i//3)*480))
    grid.save(DOC/'sam3_eye_9dir_grid.png')
    print(json.dumps({k:v for k,v in report.items() if k not in ['records']},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
