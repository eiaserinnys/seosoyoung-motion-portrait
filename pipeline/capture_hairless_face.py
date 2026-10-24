"""Actual WebGL face-only renders, registered references and geometry sheets."""
import base64,functools,http.server,io,json,threading
import cv2
import numpy as np
from PIL import Image,ImageDraw
from playwright.sync_api import sync_playwright
from hairless_face_reference import ROOT,FILES,geometry,raw_mask,matrix,registered_point
BG=(30,36,48);GREEN=(80,255,120);PINK=(255,90,200)
ORDER=['up-left','up','up-right','left','front','right','down-left','down','down-right']
CROP=(260,220,740,700)

class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self,*args):pass

def webgl_renders(poses):
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Handler,directory=str(ROOT/'bundle')))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    errors=[];renders={}
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--use-angle=swiftshader'])
            page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
            page.on('pageerror',lambda e:errors.append(str(e)))
            page.on('console',lambda m:errors.append(m.text) if m.type=='error' else None)
            page.on('response',lambda r:errors.append(str(r.status)+' '+r.url) if r.status>=400 else None)
            page.goto(f'http://127.0.0.1:{server.server_port}',wait_until='networkidle')
            page.wait_for_function('window.studio?.ready',timeout=30000)
            page.evaluate('''async()=>{
              studio.setPose(0,0,0,'Face-only verification');
              const {Renderer}=await import('./renderer.mjs');
              const canvas=document.createElement('canvas'),overlay=document.createElement('canvas');
              Object.assign(canvas.style,{width:'1100px',height:'1600px'});document.body.append(canvas);
              window.faceCapture=new Renderer(canvas,overlay,studio.rig,studio.renderer.images);
            }''')
            tiles=[]
            for name in ORDER:
                p=next(p for p in poses if p['name']==name)
                data=page.evaluate('''([yaw,pitch])=>{
                  studio.setPose(yaw,pitch,0,'Face-only verification');studio.p.CorrTurn=1;
                  Object.assign(studio.p,{ParamBodyAngleX:0,ParamBreath:0,ParamEyeLOpen:1,ParamEyeROpen:1,ParamEyeBallX:0,ParamEyeBallY:0});
                  const keep=id=>id==='face-skin'||id.startsWith('eye-')||id==='nose-detail'||id==='mouth-detail';
                  const hidden=new Set(studio.rig.meshes.filter(m=>!keep(m.id)).map(m=>m.id));
                  const state={...studio.state,hidden,mode:'normal',wire:false,grid:false,before:false,surfaceOnly:false,identify:null};
                  faceCapture.render(studio.p,state);
                  if(faceCapture.gl.getError()!==0)throw Error('WebGL error');
                  const hairless=faceCapture.canvas.toDataURL('image/png').split(',')[1];
                  faceCapture.render(studio.p,{...state,hidden:new Set()});
                  if(faceCapture.gl.getError()!==0)throw Error('Full portrait WebGL error');
                  return {hairless,full:faceCapture.canvas.toDataURL('image/png').split(',')[1]};
                }''',[p['yaw'],p['pitch']])
                # Renderer at scale=1: canvas origin = (38,32).
                im=Image.open(io.BytesIO(base64.b64decode(data['hairless']))).crop((38,32,1062,1056)).convert('RGBA')
                bg=Image.new('RGBA',(1024,1024),BG+(255,));bg.alpha_composite(im);renders[name]=bg.convert('RGB')
                full=Image.open(io.BytesIO(base64.b64decode(data['full']))).crop((38,32,1062,1056)).convert('RGBA')
                full_bg=Image.new('RGBA',(1024,1024),BG+(255,));full_bg.alpha_composite(full)
                tile=full_bg.convert('RGB').crop((152,20,872,740)).resize((480,480))
                ImageDraw.Draw(tile).text((8,8),name+' | full portrait',fill='white');tiles.append(tile)
            # Verify the existing viewer entry point responds to its actual sliders.
            page.get_by_role('button',name='상 +20°',exact=True).click()
            assert page.evaluate('studio.p.ParamAngleY')==20
            browser.close()
            assert not errors,errors
    finally:
        server.shutdown();server.server_close();thread.join()
    grid=Image.new('RGB',(1440,1440),BG)
    for i,tile in enumerate(tiles):grid.paste(tile,((i%3)*480,(i//3)*480))
    grid.save(ROOT/'docs/captures/viewer_9dir_grid.png')
    return renders

def reference(p):
    rgb=np.array(Image.open(ROOT/'source/references'/FILES[p['name']]).convert('RGB'))
    mask=raw_mask(p['name']);show=mask.copy()
    raw=p['feature'].get('rawCenters',p['feature']['centers'])
    for k,pt in raw.items():
        x,y=pt[0]-152,pt[1]-20
        rx,ry=(52,48) if k in ['left','right'] else (42,18) if k.startswith('brow') else (25,25)
        show[max(0,int(y-ry)):min(720,int(y+ry)),max(0,int(x-rx)):min(720,int(x+rx))]=255
    a=matrix(p)
    warped=cv2.warpAffine(rgb,a,(1024,1024),flags=cv2.INTER_LINEAR,borderValue=BG)
    visible=cv2.warpAffine(show,a,(1024,1024),flags=cv2.INTER_NEAREST)>127
    canvas=np.zeros((1024,1024,3),np.uint8);canvas[:]=BG;canvas[visible]=warped[visible]
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        pts=cv2.transform(c.astype(float),a).round().astype(np.int32)
        cv2.polylines(canvas,[pts],True,GREEN,2,cv2.LINE_AA)
    return Image.fromarray(canvas)

def geometry_overlay(p,background,grid=True):
    out=background.copy();draw=ImageDraw.Draw(out)
    skin=next(m for m in p['meshes'] if m['id']=='face-skin')
    if grid:
        for tri in skin['triangles']:
            q=[tuple(skin['positions'][i][:2]) for i in tri];draw.line(q+[q[0]],fill=(142,69,115),width=1)
    # Visible alpha>127 contour: front silhouette samples, not transparent mesh padding.
    for side in [0,1]:draw.line([tuple(row['points'][side][:2]) for row in p['boundary']],fill=PINK,width=2)
    for m in p['meshes']:
        if m['id']=='face-skin':continue
        pos=np.array(m['positions']);box=(*pos[:,:2].min(0),*pos[:,:2].max(0))
        draw.rectangle(box,outline=(255,190,90),width=1)
    raw=p['feature'].get('rawCenters',p['feature']['centers'])
    for k,pt in raw.items():
        x,y=registered_point(p,pt);draw.line((x-5,y,x+5,y),fill=GREEN,width=2);draw.line((x,y-5,x,y+5),fill=GREEN,width=2)
    return out

def comparison_region(p,im):
    # The source skin above restY347 is cut into bang-shaped artwork. Its shape
    # supplies no face-silhouette evidence; compare the shared lower region.
    points=[tuple(pt) for pt in p['comparisonTop']]
    mask=Image.new('L',im.size,0)
    ImageDraw.Draw(mask).polygon([(0,points[0][1]),*points,(1024,points[-1][1]),(1024,1024),(0,1024)],fill=255)
    return Image.composite(im,Image.new('RGB',im.size,BG),mask)

def main():
    poses=geometry();renders=webgl_renders(poses)
    comparison=Image.new('RGB',(2040,1020),BG);overlay=Image.new('RGB',(1440,1440),BG)
    for i,name in enumerate(ORDER):
        p=next(p for p in poses if p['name']==name);ref=reference(p);render=renders[name]
        # Same crop/scale for reference and render; green SAM, pink rig visible edge.
        for side in [0,1]:
            pts=[tuple(row['points'][side][:2]) for row in p['boundary']]
            ImageDraw.Draw(ref).line(pts,fill=PINK,width=2);ImageDraw.Draw(render).line(pts,fill=PINK,width=2)
        pair=[comparison_region(p,ref),comparison_region(p,render)]
        for j,im in enumerate(pair):
            tile=im.crop(CROP).resize((340,340));d=ImageDraw.Draw(tile)
            d.text((6,6),name+(' | REF green / rig pink' if j==0 else ' | WebGL face only'),fill='white')
            comparison.paste(tile,(((i%3)*2+j)*340,(i//3)*340))
        geom=comparison_region(p,geometry_overlay(p,reference(p))).crop(CROP).resize((480,480))
        d=ImageDraw.Draw(geom);d.text((8,8),name+' | SAM green / mesh pink / boxes amber',fill='white')
        overlay.paste(geom,((i%3)*480,(i//3)*480))
    comparison.save(ROOT/'docs/captures/hairless_face_9dir_comparison.png')
    overlay.save(ROOT/'docs/captures/face_geom_9dir_overlay.png')
    print('PASS WebGL face-only + full portrait 9 poses, pose button, HTTP/console/WebGL errors 0; 3 sheets saved')
if __name__=='__main__':main()
