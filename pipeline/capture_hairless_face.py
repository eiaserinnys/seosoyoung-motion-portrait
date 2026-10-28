"""Actual WebGL face-only renders, registered references and geometry sheets."""
import base64,functools,http.server,io,json,threading
import cv2
import numpy as np
from PIL import Image,ImageDraw
from playwright.sync_api import sync_playwright
from hairless_face_reference import ROOT,FILES,geometry,raw_mask,matrix,registered_point
BG=(30,36,48);GREEN=(80,255,120);PINK=(255,90,200)
RED=(255,64,64);CYAN=(64,224,255)
ORDER=['up-left','up','up-right','left','front','right','down-left','down','down-right']
CROP=(260,220,740,700)

class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self,*args):pass

def webgl_renders(poses,order=ORDER,save_grid=True,inspect=None,include_ears=False,inspect_pose=None):
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
              studio.setPose(0,0,0,'Face-only verification');studio.p.CorrHeadBase=1;
              const {Renderer}=await import('./renderer.mjs');
              const canvas=document.createElement('canvas'),overlay=document.createElement('canvas');
              Object.assign(canvas.style,{width:'1100px',height:'1600px'});document.body.append(canvas);
              window.faceCapture=new Renderer(canvas,overlay,studio.rig,studio.renderer.images);
            }''')
            tiles=[]
            for name in order:
                p=next(p for p in poses if p['name']==name)
                data=page.evaluate('''([yaw,pitch,includeEars])=>{
                  studio.setPose(yaw,pitch,0,'Face-only verification');studio.p.CorrTurn=1;
                  Object.assign(studio.p,{ParamBodyAngleX:0,ParamBreath:0,ParamEyeLOpen:1,ParamEyeROpen:1,ParamEyeBallX:0,ParamEyeBallY:0});
                  const keep=id=>id==='face-skin'||id.startsWith('eye-')||id==='nose-detail'||id==='mouth-detail'||(includeEars&&id.startsWith('ear-'));
                  const hidden=new Set(studio.rig.meshes.filter(m=>!keep(m.id)).map(m=>m.id));
                  const state={...studio.state,hidden,mode:'normal',wire:false,grid:false,before:false,surfaceOnly:false,identify:null};
                  faceCapture.render(studio.p,state);
                  if(faceCapture.gl.getError()!==0)throw Error('WebGL error');
                  const hairless=faceCapture.canvas.toDataURL('image/png').split(',')[1];
                  faceCapture.render(studio.p,{...state,hidden:new Set()});
                  if(faceCapture.gl.getError()!==0)throw Error('Full portrait WebGL error');
                  return {hairless,full:faceCapture.canvas.toDataURL('image/png').split(',')[1]};
                }''',[p['yaw'],p['pitch'],include_ears])
                if inspect_pose is not None:inspect_pose(page,p,data)
                # Renderer at scale=1: canvas origin = (38,32).
                im=Image.open(io.BytesIO(base64.b64decode(data['hairless']))).crop((38,32,1062,1056)).convert('RGBA')
                renders[name]=im
                full=Image.open(io.BytesIO(base64.b64decode(data['full']))).crop((38,32,1062,1056)).convert('RGBA')
                full_bg=Image.new('RGBA',(1024,1024),BG+(255,));full_bg.alpha_composite(full)
                tile=full_bg.convert('RGB').crop((152,20,872,740)).resize((480,480))
                ImageDraw.Draw(tile).text((8,8),name+' | full portrait',fill='white');tiles.append(tile)
            # Verify the existing viewer entry point responds to its actual sliders.
            page.get_by_role('button',name='상 +20°',exact=True).click()
            assert page.evaluate('studio.p.ParamAngleY')==20
            for yaw in [-90,-67.5,0,67.5,90]:
                page.locator('#ParamAngleX').evaluate('''(el,value)=>{
                    studio.p.CorrTurn=1;el.value=value;el.dispatchEvent(new Event('input',{bubbles:true}));
                }''',yaw)
                assert page.evaluate('studio.p.ParamAngleX*studio.p.CorrTurn')==yaw
            if inspect is not None:inspect(page)
            browser.close()
            assert not errors,errors
    finally:
        server.shutdown();server.server_close();thread.join()
    if save_grid:
        grid=Image.new('RGB',(1440,1440),BG)
        for i,tile in enumerate(tiles):grid.paste(tile,((i%3)*480,(i//3)*480))
        grid.save(ROOT/'docs/captures/viewer_9dir_grid.png')
    return renders

def warped_reference(p):
    rgb=np.array(Image.open(ROOT/'source/references'/FILES[p['name']]).convert('RGB'))
    return cv2.warpAffine(rgb,matrix(p),(1024,1024),flags=cv2.INTER_LINEAR,borderValue=BG)

def reference_boundary(p,canvas):
    contours,_=cv2.findContours(raw_mask(p['name']),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        pts=cv2.transform(c.astype(float),matrix(p)).round().astype(np.int32)
        cv2.polylines(canvas,[pts],True,GREEN,2,cv2.LINE_AA)

def reference(p):
    mask=raw_mask(p['name']);show=mask.copy()
    raw=p['feature'].get('rawCenters',p['feature']['centers'])
    for k,pt in raw.items():
        x,y=pt[0]-152,pt[1]-20
        rx,ry=(52,48) if k in ['left','right'] else (42,18) if k.startswith('brow') else (25,25)
        show[max(0,int(y-ry)):min(720,int(y+ry)),max(0,int(x-rx)):min(720,int(x+rx))]=255
    a=matrix(p)
    warped=warped_reference(p)
    visible=cv2.warpAffine(show,a,(1024,1024),flags=cv2.INTER_NEAREST)>127
    canvas=np.zeros((1024,1024,3),np.uint8);canvas[:]=BG;canvas[visible]=warped[visible]
    reference_boundary(p,canvas)
    return Image.fromarray(canvas)

def geometry_overlay(p,background,difference,grid=True):
    canvas=np.array(background).astype(float)
    for mask,color,alpha in zip(difference[:3],[RED,RED,CYAN],[1.,.5,.45]):
        canvas[mask]=(1-alpha)*canvas[mask]+alpha*np.array(color)
    out=Image.fromarray(canvas.astype(np.uint8));draw=ImageDraw.Draw(out)
    skin=next(m for m in p['meshes'] if m['id']=='face-skin')
    if grid:
        for tri in skin['triangles']:
            q=[tuple(skin['positions'][i][:2]) for i in tri];draw.line(q+[q[0]],fill=(142,69,115),width=1)
    canvas=np.array(out);reference_boundary(p,canvas);out=Image.fromarray(canvas);draw=ImageDraw.Draw(out)
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

def comparison_mask(p,size=(1024,1024)):
    # The source skin above restY347 is cut into bang-shaped artwork. Its shape
    # supplies no face-silhouette evidence; compare the shared lower region.
    points=[tuple(pt) for pt in p['comparisonTop']]
    mask=Image.new('L',size,0)
    ImageDraw.Draw(mask).polygon([(0,points[0][1]),*points,(1024,points[-1][1]),(1024,1024),(0,1024)],fill=255)
    return mask

def comparison_region(p,im):
    return Image.composite(im,Image.new('RGB',im.size,BG),comparison_mask(p,im.size))

def silhouette_difference(p,im):
    region=np.array(comparison_mask(p))>127
    a=(np.array(im)[:,:,3]>127)&region
    b=(cv2.warpAffine(raw_mask(p['name']),matrix(p),(1024,1024),flags=cv2.INTER_NEAREST)>127)&region
    red=a&~b;cyan=b&~a;gray=a&b
    bg_mask=warped_reference(p).max(axis=2)<40
    red_bg=red&bg_mask;red_hair=red&~bg_mask
    values={}
    for name,mask in [('red',red),('cyan',cyan),('redBg',red_bg),('redHair',red_hair)]:
        count=int(np.count_nonzero(mask));values[name+'Pixels']=count
        values[name+'MaxDepth']=round(float(cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5).max()),3) if count else 0.0
    return red_bg,red_hair,cyan,gray,values

def difference_image(p,warped,difference):
    gray=np.array(warped.convert('L'))//2
    canvas=np.repeat(gray[:,:,None],3,axis=2)
    canvas[np.array(comparison_mask(p))==0]=BG
    for mask,color,alpha in zip(difference,[RED,RED,CYAN,(128,128,128)],[1.,.5,1.,1.]):
        canvas[mask]=((1-alpha)*canvas[mask]+alpha*np.array(color)).astype(np.uint8)
    return Image.fromarray(canvas)

def difference_label(name,values):
    # Keep the existing font and tile size; the added reference metric wraps.
    return (f"{name} | R_bg:{values['redBgPixels']}({values['redBgMaxDepth']:.1f}px)\n"
            f"R_hair:{values['redHairPixels']} C:{values['cyanPixels']}({values['cyanMaxDepth']:.1f}px)")

def main():
    poses=geometry();renders=webgl_renders(poses)
    comparison=Image.new('RGB',(3060,1020),BG);overlay=Image.new('RGB',(1440,1440),BG)
    differences={}
    for i,name in enumerate(ORDER):
        p=next(p for p in poses if p['name']==name);ref=reference(p)
        red_bg,red_hair,cyan,gray,values=silhouette_difference(p,renders[name]);differences[name]=values
        difference=[red_bg,red_hair,cyan,gray]
        bg=Image.new('RGBA',(1024,1024),BG+(255,));bg.alpha_composite(renders[name]);render=bg.convert('RGB')
        # Same crop/scale for reference and render; green SAM, pink rig visible edge.
        for side in [0,1]:
            pts=[tuple(row['points'][side][:2]) for row in p['boundary']]
            ImageDraw.Draw(ref).line(pts,fill=PINK,width=2);ImageDraw.Draw(render).line(pts,fill=PINK,width=2)
        pair=[comparison_region(p,ref),comparison_region(p,render),
              difference_image(p,Image.fromarray(warped_reference(p)),difference)]
        for j,im in enumerate(pair):
            tile=im.crop(CROP).resize((340,340));d=ImageDraw.Draw(tile)
            label=name+(' | REF green / rig pink' if j==0 else ' | WebGL face only')
            if j==2:label=difference_label(name,values)
            d.multiline_text((6,6),label,fill='white')
            comparison.paste(tile,(((i%3)*3+j)*340,(i//3)*340))
        geom=comparison_region(p,geometry_overlay(p,reference(p),difference)).crop(CROP).resize((480,480))
        d=ImageDraw.Draw(geom);d.text((8,8),name+' | SAM green / mesh pink / boxes amber',fill='white')
        overlay.paste(geom,((i%3)*480,(i//3)*480))
    comparison.save(ROOT/'docs/captures/hairless_face_9dir_comparison.png')
    overlay.save(ROOT/'docs/captures/face_geom_9dir_overlay.png')
    from verify_hairless_face import write_metrics
    failed=write_metrics(poses,differences)
    print('PASS WebGL face-only + full portrait 9 poses, pose button, HTTP/console/WebGL errors 0; 3 sheets saved')
    if failed:raise SystemExit(1)
if __name__=='__main__':main()
