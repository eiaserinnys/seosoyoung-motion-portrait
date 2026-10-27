"""7.11 rotated base + face t; 10 residual fits, same grid/Powell scheme."""
import base64,functools,http.server,itertools,json,threading,time
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.optimize import minimize
from playwright.sync_api import sync_playwright
from config import V,A,T,NAMES
from earmath import warp,measure,feasible_translation
from optimizer_coords import encode,decode,Incumbent
class Handler(http.server.SimpleHTTPRequestHandler):
 def log_message(self,*args):pass

def plot_failure(out,n,side,frame,target,pred,grid,best,theta):
 rgb=np.array(frame.convert('RGB'))
 for mask,color in [(pred,(255,70,190)),(target,(65,255,100))]:
  c,_=cv2.findContours(np.uint8(mask),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);cv2.drawContours(rgb,c,-1,color,2)
 yy,xx=np.where(pred|target);box=[max(0,int(xx.min())-25),max(0,int(yy.min())-25),min(1024,int(xx.max()+1)+25),min(1024,int(yy.max()+1)+25)]
 Image.fromarray(rgb).crop(box).resize(((box[2]-box[0])*3,(box[3]-box[1])*3)).save(out/'mask-overlay-3x.png')
 cells=41;size=20;im=Image.new('RGB',(cells*size+100,cells*size+90),(30,36,48));d=ImageDraw.Draw(im);font=ImageFont.load_default(size=14)
 for tx in range(-80,81,4):
  for ty in range(-80,81,4):
   scores=[q[3] for q in grid if q[0]==tx and q[1]==ty];val=max(scores);feasible=np.hypot(tx,ty)<=80;color=tuple(int(v) for v in ([40+210*val,40+170*val,160-100*val] if feasible else [35,35,40]));x=50+(tx+80)//4*size;y=45+(ty+80)//4*size;d.rectangle((x,y,x+size-1,y+size-1),fill=color)
 d.text((8,5),n+' '+side+' | max IoU over sx; infeasible t gray',fill='white',font=font)
 for tx,ty,c in [(best['theta'][0],best['theta'][1],'white'),(theta[0],theta[1],'red')]:
  x=50+(tx+80)/4*size+size/2;y=45+(ty+80)/4*size+size/2;d.ellipse((x-5,y-5,x+5,y+5),outline=c,width=2)
 d.text((50,875),'tx: -80 .. +80; ty: -80(top) .. +80',fill='white',font=font);im.save(out/'grid-IoU-map.png')

def main():
 cv2.setNumThreads(1);start=time.time();E=V/'ears';E.mkdir();bases=json.loads((V/'ear-bases.json').read_text())['poses'];keys=json.loads((T/'bundle/ear-transform-data.mjs').read_text().split('=')[1].strip().rstrip(';'));old=json.loads((A/'v711-filled/ear-fit.json').read_text());results=[];errors=[]
 server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Handler,directory=str(T/'bundle')));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 try:
  with sync_playwright() as pw:
   browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--use-angle=swiftshader']);page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1);page.on('pageerror',lambda e:errors.append(str(e)));page.goto(f'http://127.0.0.1:{server.server_port}',wait_until='networkidle');page.wait_for_function('window.studio?.ready');page.evaluate((A/'ear-fit/renderer.js').read_text())
   def shot(theta,side,yaw,pitch,box,full=False):
    q=page.evaluate('q=>renderEar(q)',dict(theta=np.asarray(theta).tolist(),side=side,yaw=yaw,pitch=pitch,targetBox=box,frontSource=full));a,b,c,d=q['box'];return np.frombuffer(base64.b64decode(q['bits']),np.uint8).reshape(d-b,c-a),q['box']
   for n in NAMES:
    meta=json.loads((A/'inputs'/n/'manifest.json').read_text());pose=bases[n];key=next(k for k in keys if k['yaw']==pose['yaw'] and k['pitch']==pose['pitch'])
    for side,e in meta['ears'].items():
     if not e['visible'] or [n,side] not in json.loads((V.parent/'selection.json').read_text())['selected']:continue
     out=E/(n+'-'+side);out.mkdir();target=np.array(Image.open(A/'inputs'/n/f'ear-{side}-mask.png'))>127;yy,xx=np.where(target);box=[max(0,int(xx.min())-10),max(0,int(yy.min())-10),min(1024,int(xx.max()+1)+10),min(1024,int(yy.max()+1)+10)]
     if (n,side)==('left','right'):box=[625,408,714,537]
     source,_=shot([0,0,1,1,0],side,pose['yaw'],pose['pitch'],box,True);Image.fromarray(source).save(out/'B-base-alpha.png');center=pose['sides'][side]['Bcenter'];grid=[];best=None
     for tx,ty,sx in itertools.product(range(-80,81,4),range(-80,81,4),[.6,.7,.8,.9,1.,1.1]):
      theta=[tx,ty,sx,1.,0.];pred,truth,origin=warp(source,target,theta,center);m=measure(pred,truth,origin,box,side);ok=feasible_translation(theta);grid.append([tx,ty,sx,m['wholeScreenIoU'],ok])
      if ok and (best is None or m['wholeScreenIoU']>best['warpIoU']):best={'theta':theta,'warpIoU':m['wholeScreenIoU']}
     (out/'grid.json').write_text(json.dumps({'bestFeasible':best,'candidates':grid})+'\n');print('711 EAR GRID',n,side,best,flush=True);history=[];incumbent=Incumbent()
     def evaluate(theta,stage,full=False):
      alpha,window=shot(theta,side,pose['yaw'],pose['pitch'],box);a,b,c,d=window;truth=target[b:d,a:c];assert truth.sum()==target.sum();pred=alpha>127;m=measure(pred,truth,[a,b],box,side,full);history.append({'stage':stage,'theta':np.asarray(theta).tolist(),**m});incumbent.observe(theta,m['wholeScreenIoU']);return m,pred,window
     def objective(x,stage):
      theta=decode(x)
      return 1-evaluate(theta,stage)[0]['wholeScreenIoU']
     initial,_,_=evaluate(best['theta'],'grid-renderer',True)
     first=minimize(lambda x:objective(np.r_[x,1.,0.],'3DOF'),encode(best['theta'])[:3],method='Powell',bounds=[(-80,80),(-1,1),(.6,1.1)],options={'direc':np.diag([4.,.1,.1]),'maxfev':600,'maxiter':80,'xtol':1e-3,'ftol':1e-5})
     theta=incumbent.theta.copy();after3,_,_=evaluate(theta,'after-3DOF-best-evaluated',True);second=None
     if after3['helixSymmetricRMS']>4:
      second=minimize(lambda x:objective(x,'5DOF'),encode(theta),method='Powell',bounds=[(-80,80),(-1,1),(.6,1.1),(.85,1.1),(-15,15)],options={'direc':np.diag([4.,.1,.1,.05,3.]),'maxfev':800,'maxiter':100,'xtol':1e-3,'ftol':1e-5});theta=incumbent.theta.copy()
     final,pred,window=evaluate(theta,'final-renderer',True);a,b,c,d=window;full=np.zeros((1024,1024),np.uint8);full[b:d,a:c]=np.uint8(pred)*255;Image.fromarray(full).save(out/'final-renderer-alpha.png');predw,truthw,origin=warp(source,target,theta,center);fw=measure(predw,truthw,origin,box,side,True)
     key[side]={'t':theta[:2].tolist(),'sx':float(theta[2]),'s':float(theta[3]),'theta':float(theta[4])};before=next(q for q in old if q['pose']==n and q['side']==side)
     failed=not feasible_translation(theta) or final['wholeScreenIoU']<.70
     report={'pose':n,'side':side,'yaw':pose['yaw'],'pitch':pose['pitch'],'B':pose['sides'][side],'targetBox':box,'grid':best,'gridRenderer':initial,'after3DOF':after3,'finalRenderer':final,'finalWarp':fw,'parameters':key[side],'translationNorm':float(np.linalg.norm(theta[:2])),'translationWithin80':feasible_translation(theta),'previousFilledParameters':before['parameters'],'previousFilledTranslationNorm':before['translationNorm'],'conditionalScaleRotationUsed':second is not None,'firstNfev':int(first.nfev),'secondNfev':int(second.nfev) if second is not None else 0,'earFailed':failed,'optimizerTranslationCoordinates':'tx in [-80,80], ty=u*sqrt(6400-tx^2), u in [-1,1]','bestEvaluatedRendererIoU':incumbent.score}
     (out/'fit.json').write_text(json.dumps(report,indent=2)+'\n');(out/'history.json').write_text(json.dumps(history)+'\n');results.append(report);(V/'ear-fit.json').write_text(json.dumps(results,indent=2)+'\n')
     if failed:plot_failure(out,n,side,Image.open(A/'inputs'/n/'reference-frame.png'),target,full>127,grid,best,theta)
     print('711 EAR DONE',n,side,'IoU',final['wholeScreenIoU'],'residualT',key[side],'norm',report['translationNorm'],'failed',failed,flush=True)
   browser.close()
 finally:server.shutdown();server.server_close();thread.join()
 assert not errors and len(results)==len(json.loads((V.parent/'selection.json').read_text())['selected'])
 (T/'bundle/ear-transform-data.mjs').write_text('export const EAR_TRANSFORMS='+json.dumps(keys,separators=(',',':'))+';\n');(V/'ear-audit.json').write_text(json.dumps({'elapsedSeconds':round(time.time()-start,2),'pageErrors':errors,'fits':results,'failedEars':[r['pose']+'-'+r['side'] for r in results if r['earFailed']]},indent=2)+'\n')
 print('711 ALL10 EARS FINISHED',round(time.time()-start,2),'seconds',flush=True)
if __name__=='__main__':main()
