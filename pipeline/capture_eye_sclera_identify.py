"""F-T1 sclera gate using Selaphiel's viewport, identify and iris-centre split."""
import ast,base64,csv,functools,hashlib,http.server,json,sys,threading
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from playwright.sync_api import sync_playwright
from capture_sam3_eyes import ROOT,Handler

OUT=ROOT/'debug/sam3-eye-v295/identify'
HOOK=r'''async ([ww, wh]) => {
  const {deform3D} = await import('./rig.mjs');
  const {viewerPose} = await import('./viewer-pose.mjs');
  const r = studio.renderer, gl = r.gl;
  const m = studio.rig.meshes.find(m => m.id === 'face-skin'); const point = [502, 400]; let k = 0;
  for (let i = 1; i < m.positions.length; i++)
    if (Math.hypot(m.positions[i][0]-point[0], m.positions[i][1]-point[1]) < Math.hypot(m.positions[k][0]-point[0], m.positions[k][1]-point[1])) k = i;
  const read = (x0, y0) => { const buf = new Uint8Array(ww * wh * 4);
    gl.readPixels(x0, gl.drawingBufferHeight - y0 - wh, ww, wh, gl.RGBA, gl.UNSIGNED_BYTE, buf);
    const bits = new Uint8Array(ww * wh); for (let i = 0; i < ww * wh; i++) bits[i] = (buf[4*i] > 127 && buf[4*i+1] > 127 && buf[4*i+2] > 127 && buf[4*i+3] > 127) ? 1 : 0;
    let s = ''; for (let i = 0; i < bits.length; i += 0x8000) s += String.fromCharCode.apply(null, bits.subarray(i, i + 0x8000)); return btoa(s); };
  const eyeIds = studio.rig.meshes.filter(m => m.id.startsWith('eye-') || m.id === 'closed-eyes-v1').map(m => m.id);
  window.__probe = () => {
    const v = r.view, p = viewerPose(studio.p), q = deform3D(m, m.positions[k], p);
    const cx = Math.round(q[0] * v.scale + v.ox), cy = Math.round(q[1] * v.scale + v.oy) + 20, x0 = cx - ww / 2, y0 = cy - wh / 2;
    const far = p.ParamAngleX > 0 ? 'right' : 'left', st = studio.state, base = {...st, mode: 'normal', wire: false, grid: false};
    const g = {cx, cy, far, angles: document.getElementById('angles').textContent};
    r.render(p, {...base, hidden: new Set(), identify: eyeIds}); g.eyes = read(x0, y0);
    r.render(p, {...base, hidden: new Set(eyeIds), identify: 'face-skin'}); g.face = read(x0, y0);
    r.render(p, {...base, hidden: new Set(), identify: ['eye-' + far + '-sclera']}); g.sclera = read(x0, y0);
    r.render(p, {...base, hidden: new Set(), identify: ['iris', 'pupil', 'highlight'].map(s => 'eye-' + far + '-' + s)}); g.iris = read(x0, y0);
    r.render(p, st);
    return g;
  };
}'''
EXTRA='''()=>{
 const r=studio.renderer,gl=r.gl,probe=window.__probe;
 window.scleraPose=()=>{
  const g=probe(),ww=640,wh=260,raw=new Uint8Array(ww*wh*4);
  gl.readPixels(g.cx-ww/2,gl.drawingBufferHeight-(g.cy-wh/2)-wh,ww,wh,gl.RGBA,gl.UNSIGNED_BYTE,raw);
  let s='';for(let i=0;i<raw.length;i+=32768)s+=String.fromCharCode.apply(null,raw.subarray(i,i+32768));
  g.color=btoa(s);if(gl.getError()!==0)throw Error('Sclera identify WebGL error');return g;
 };
}'''

def judge_pair(left,right):
    return dict(rightTemporalVisible=right>0,withinTwoTimes=right<=2*left,
                ratio=right/left if left else None,passed=right>0 and right<=2*left)

def bits(encoded):return np.frombuffer(base64.b64decode(encoded),np.uint8).reshape(260,640)[::-1].astype(bool)

def measure(g):
    sclera,iris=bits(g['sclera']),bits(g['iris'])
    if not iris.any():return dict(scleraTemporal=0,scleraNasal=0,irisVisible=False)
    ic=np.where(iris)[1].mean();xs=np.where(sclera)[1]
    nasal=int((xs>ic).sum() if g['far']=='left' else (xs<ic).sum())
    return dict(scleraTemporal=int(len(xs)-nasal),scleraNasal=nasal,irisVisible=True,irisCenterX=float(ic))

def crop(g):
    a=np.frombuffer(base64.b64decode(g['color']),np.uint8).reshape(260,640,4)[::-1].astype(float)
    rgb=a[:,:,:3]*a[:,:,3:]/255+np.array([35,46,62])*(255-a[:,:,3:])/255
    mask=bits(g['iris'])|bits(g['sclera'])
    box=(0,0,640,260)
    if mask.any():
        yy,xx=np.where(mask);box=(max(0,int(xx.min())-25),max(0,int(yy.min())-30),min(640,int(xx.max())+26),min(260,int(yy.max())+31))
    im=Image.fromarray(rgb.astype(np.uint8)).crop(box)
    return im.resize((im.width*2,im.height*2),Image.Resampling.NEAREST)

def sheet(rows,filename):
    tiles=[]
    for row in rows:
        ims=[Image.open(OUT/f"p{row['pitch']}-y{sign*row['yawAbs']}-2x.png").convert('RGB') for sign in [-1,1]]
        tiles.append((row,ims))
    width=max((im.width for _,ims in tiles for im in ims),default=320)+12
    image=Image.new('RGB',(width*2,max(28,sum(max(im.height for im in ims)+28 for _,ims in tiles))),'#eeeeee');y=0
    for row,ims in tiles:
        for col,(side,im) in enumerate(zip(['left','right'],ims)):
            image.paste(im,(col*width,y));v=row[side]
            ImageDraw.Draw(image).text((col*width+4,y+max(i.height for i in ims)+4),f"p{row['pitch']} y{'-' if col==0 else '+'}{row['yawAbs']} | T {v['scleraTemporal']} N {v['scleraNasal']}",fill='black')
        y+=max(im.height for im in ims)+28
    if not tiles:ImageDraw.Draw(image).text((8,8),'No identify gate failures',fill='black')
    image.save(ROOT/'docs/captures'/filename)

def main():
    smoke='--smoke' in sys.argv
    OUT.mkdir(parents=True,exist_ok=True);rows=[];errors=[]
    digest=hashlib.sha256(b''.join((ROOT/p).read_bytes() for p in ['bundle/renderer.mjs','bundle/eye-sam3-data.mjs','bundle/assets/texture_00.png'])).hexdigest()
    signature=OUT/'inputs.sha256'
    if signature.exists():assert signature.read_text().strip()==digest,'Identify cache belongs to other inputs'
    signature.write_text(digest+'\n')
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Handler,directory=str(ROOT)))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--use-angle=swiftshader'])
            page=browser.new_page(viewport=dict(width=1600,height=1100),device_scale_factor=2)
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{server.server_port}/bundle/',wait_until='networkidle')
            page.wait_for_function('window.studio?.ready',timeout=30000)
            page.locator('summary',has_text='모션 파라미터').click()
            page.locator('input[id="CorrTurn"]').fill('1')
            page.evaluate('window.requestAnimationFrame=()=>0');page.evaluate(HOOK,[640,260]);page.evaluate(EXTRA)
            for pitch in ([0] if smoke else [-30,0,30]):
                page.locator('input[id="ParamAngleY"]').fill(str(pitch))
                for yaw in ([40] if smoke else range(30,59,2)):
                    row=dict(pitch=pitch,yawAbs=yaw)
                    for side,sign in [('left',-1),('right',1)]:
                        path=OUT/f'p{pitch}-y{sign*yaw}.json'
                        if path.exists():g=json.loads(path.read_text())
                        else:
                            page.locator('input[id="ParamAngleX"]').fill(str(sign*yaw))
                            g=page.evaluate('window.scleraPose()');path.write_text(json.dumps(g,ensure_ascii=False)+'\n')
                        row[side]=measure(g);crop(g).save(OUT/f'p{pitch}-y{sign*yaw}-2x.png')
                    row.update(judge_pair(row['left']['scleraTemporal'],row['right']['scleraTemporal']));rows.append(row)
                    print('SCLERA',json.dumps(row),flush=True)
            browser.close()
    finally:
        server.shutdown();server.server_close();thread.join()
    assert not errors,errors
    report=dict(method='Selaphiel 1600x1100 DPR2 viewer, 640x260 nasal-bridge window, RGB/alpha>127 identify, temporal/nasal split by mean visible iris x',rows=rows,complete=len(rows)==45,failed=[r for r in rows if not r['passed']],passed=all(r['passed'] for r in rows))
    if smoke:
        (OUT/'smoke.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print('IDENTIFY SMOKE COMPLETE',len(rows),'pair',flush=True);return
    (ROOT/'docs/sam3-eye-sclera-identify.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    with (ROOT/'docs/sam3-eye-sclera-identify.csv').open('w',newline='') as stream:
        w=csv.writer(stream,lineterminator='\n');w.writerow(['pitch','yaw_abs','left_temporal','left_nasal','right_temporal','right_nasal','right_over_left','right_visible','within_2x','passed'])
        for r in rows:w.writerow([r['pitch'],r['yawAbs'],r['left']['scleraTemporal'],r['left']['scleraNasal'],r['right']['scleraTemporal'],r['right']['scleraNasal'],r['ratio'],r['rightTemporalVisible'],r['withinTwoTimes'],r['passed']])
    sheet(rows,'sam3_eye_sclera_identify_pairs_2x.png');sheet(report['failed'],'sam3_eye_sclera_identify_failures_2x.png')
    print('SCLERA COMPLETE',len(rows),'pairs; failures',len(report['failed']),flush=True)

if __name__=='__main__':main()
