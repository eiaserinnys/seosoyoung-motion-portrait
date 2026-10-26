"""Current F-B3/F-T1 R7 evidence; diagnostic measurements are separate gates."""
import functools,http.server,json,shutil,threading
import numpy as np
from PIL import Image,ImageDraw
from playwright.sync_api import sync_playwright
import capture_sam3_eyes as c
from eye_capture_angles import inverse_achieved

ROOT=c.ROOT;OUT=ROOT/'debug/sam3-eye-v295/final-render';DOC=ROOT/'docs/captures'
FOCUS=[('right','right'),('aux-0241','right'),('up-right','right'),('aux-0213','right')]
BOUNDS='''async()=>{
 const {FRAME_CACHE}=await import('./pose-grid.mjs');const {deform3D}=await import('./rig.mjs');
 const original=eyeCapture.draw;let key,cache;
 eyeCapture.draw=function(mesh,p,...args){
  const next=p.ParamAngleX+','+p.ParamAngleY;if(next!==key){key=next;cache=new Map();}p[FRAME_CACHE]=cache;
  return original.call(this,mesh,p,...args);
 };
 window.currentIrisBounds=()=>{
  const p={...studio.p,[FRAME_CACHE]:new Map()},result={};
  for(const side of ['left','right']){
   const mesh=studio.rig.meshes.find(m=>m.id==='eye-'+side+'-iris'),pts=mesh.positions.map(pt=>deform3D(mesh,pt,p));
   const xs=pts.map(p=>p[0]*eyeCapture.view.scale+eyeCapture.view.ox-190),ys=pts.map(p=>p[1]*eyeCapture.view.scale+eyeCapture.view.oy-52);
   result[side]=[Math.min(...xs),Math.min(...ys),Math.max(...xs),Math.max(...ys)];
  }return result;
 };
}'''

def eye_crop(im,bounds):
    x0,y0,x1,y1=bounds;box=(int(x0)-25,int(y0)-35,int(x1)+26,int(y1)+26)
    cropped=im.convert('RGB').crop(box)
    return cropped.resize((cropped.width*2,cropped.height*2),Image.Resampling.NEAREST)

def strip(tiles):
    width=max(im.width for _,im in tiles)+16;height=max(im.height for _,im in tiles)+28
    result=Image.new('RGB',(width*len(tiles),height),'#eeeeee')
    for i,(label,im) in enumerate(tiles):result.paste(im,(i*width,0));ImageDraw.Draw(result).text((i*width+4,height-22),label,fill='black')
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True);c.OUT=OUT
    names=c.ORDER+['aux-0213','aux-0255','aux-0074','aux-0102','aux-0276','aux-0277','aux-0043','aux-0241']
    colors={rev:{} for rev in ['r24','v291']};bounds={}
    for name in names:
        for path in (ROOT/'debug/sam3-eye-v293/final-render').glob(f'r24-{name}-*.png'):shutil.copyfile(path,OUT/path.name)
        colors['r24'][name]=Image.open(OUT/f'r24-{name}-color.png').convert('RGBA')
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(c.Handler,directory=str(ROOT)))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--use-angle=swiftshader'])
            page=browser.new_page(viewport=dict(width=1440,height=1000),device_scale_factor=1)
            page.goto(f'http://127.0.0.1:{server.server_port}/bundle/',wait_until='networkidle');page.wait_for_function('window.studio?.ready',timeout=30000)
            page.evaluate('window.requestAnimationFrame=()=>0');page.evaluate(c.SETUP);page.evaluate(BOUNDS)
            for name in names:
                row=next(r for r in c.ROWS if r['name']==name);requested=inverse_achieved(c.BOOT['achieved']['c'],row['yaw'],row['pitch'])
                path=OUT/f'v291-{name}-bounds.json'
                if path.exists():bounds[name]=json.loads(path.read_text())
                else:
                    raw=page.evaluate('(a)=>captureEyes(a)',[*requested,True,name in c.ORDER])
                    for kind,value in raw.items():c.image(value,full=kind=='full').save(OUT/f'v291-{name}-{kind}.png')
                    bounds[name]=page.evaluate('window.currentIrisBounds()');path.write_text(json.dumps(bounds[name])+'\n')
                colors['v291'][name]=Image.open(OUT/f'v291-{name}-color.png').convert('RGBA');print('R7',name,flush=True)
            browser.close()
    finally:
        server.shutdown();server.server_close();thread.join()
    c.sheets(colors)
    focus=strip([(name+':'+side+' | far sclera 2x',eye_crop(colors['v291'][name],bounds[name][side])) for name,side in FOCUS]);focus.save(DOC/'sam3_eye_far_sclera_2x.png')
    r4=json.loads((ROOT/'docs/sam3-eye-visible-outline.json').read_text());high=[]
    outside_tiles=[];outside_evidence=[]
    for row in r4['poses']:
        points=[point for q in row['meshes'].values() for point in q['coordinates']]
        if not points:continue
        im=Image.open(ROOT/f"debug/sam3-eye-v295/visible-outline/p{row['pitch']}-y{row['yaw']}-color.png").convert('RGBA')
        outline=Image.open(ROOT/f"debug/sam3-eye-v295/visible-outline/p{row['pitch']}-y{row['yaw']}-outline.png").convert('RGBA')
        for mesh,q in row['meshes'].items():
            for x,y in q['coordinates']:outside_evidence.append(dict(pitch=row['pitch'],requestedYaw=row['yaw'],mesh=mesh,x=x,y=y,renderColorRGBA=list(im.getpixel((x,y))),eyesHiddenRGBA=list(outline.getpixel((x,y)))))
        original=Image.new('RGBA',im.size,(35,46,62,255));original.alpha_composite(im);original=original.convert('RGB');marked=original.copy()
        for x,y in points:marked.putpixel((x,y),(255,0,0))
        xs=[p[0] for p in points];ys=[p[1] for p in points];box=(min(xs)-40,min(ys)-40,max(xs)+41,max(ys)+41)
        pair=[]
        for label,picture in [('original',original),('outside marked',marked)]:
            cut=picture.crop(box);pair.append((f"p{row['pitch']} y{row['yaw']} | {label} | {len(points)}px",cut.resize((cut.width*4,cut.height*4),Image.Resampling.NEAREST)))
        outside_tiles.append(strip(pair))
    if outside_tiles:
        sheet=Image.new('RGB',(max(im.width for im in outside_tiles),sum(im.height for im in outside_tiles)),'#eeeeee');y=0
        for im in outside_tiles:sheet.paste(im,(0,y));y+=im.height
        sheet.save(DOC/'sam3_eye_R4_visible_outside_4x.png')
    (ROOT/'docs/sam3-eye-R4-visible-pixels.json').write_text(json.dumps(dict(coordinateSpace='full 1100x1600 portrait canvas',pixels=outside_evidence),ensure_ascii=False,indent=2)+'\n')
    for pitch in [-30,0,30]:
        lines=[]
        for sign in [-1,1]:
            tiles=[]
            for yaw in range(50,59,2):
                row=next(r for r in r4['poses'] if r['yaw']==sign*yaw and r['pitch']==pitch);side='left' if sign<0 else 'right'
                im=Image.open(ROOT/f'debug/sam3-eye-v295/visible-outline/p{pitch}-y{sign*yaw}-color.png')
                tiles.append((f'p{pitch} y{sign*yaw} {side} 2x',eye_crop(im,row['irisBounds'][side])))
            lines.append(strip(tiles))
        sheet=Image.new('RGB',(max(im.width for im in lines),sum(im.height for im in lines)),'#eeeeee');y=0
        for im in lines:sheet.paste(im,(0,y));y+=im.height
        sheet.save(DOC/f'sam3_eye_high_turn_p{pitch}_2x.png')
        if pitch==0:high.append(sheet)
    original=Image.open(DOC/'sam3_eye_R7_comparison.png').convert('RGB');parts=[original,focus,*high]
    result=Image.new('RGB',(max(im.width for im in parts),sum(im.height for im in parts)),'#eeeeee');y=0
    for im in parts:result.paste(im,(0,y));y+=im.height
    result.save(DOC/'sam3_eye_R7_comparison.png')
    nose=strip([(f'p30 y{yaw} | nose overlap',Image.open(ROOT/f'debug/sam3-eye-v295/identify/p30-y{yaw}-2x.png').convert('RGB')) for yaw in range(38,55,2)])
    nose.save(DOC/'sam3_eye_nose_overlap_p30_2x.png')
    grid=Image.new('RGB',(1440,1440),'#1e2430')
    for i,name in enumerate(c.ORDER):
        im=Image.open(OUT/f'v291-{name}-full.png').convert('RGB').crop((152,20,872,740)).resize((480,480));ImageDraw.Draw(im).text((8,8),name,fill='white');grid.paste(im,((i%3)*480,(i//3)*480))
    grid.save(DOC/'sam3_eye_9dir_grid.png')
    report=dict(poses=names,focusEyes=FOCUS,highYawRequested=list(range(50,59,2)),highPitch=[-30,0,30],referenceAndR24Reused=True)
    (ROOT/'docs/sam3-eye-r7-captures.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('R7 COMPLETE',len(names),'reference poses + focus + high turns',flush=True)

if __name__=='__main__':main()
