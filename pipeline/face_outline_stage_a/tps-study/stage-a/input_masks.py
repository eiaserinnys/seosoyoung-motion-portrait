from pathlib import Path as _ProjectPath
PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[4]
"""7.10.2 registered input masks. No runtime or solver mutations."""
import inspect,json,sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation,binary_fill_holes
V=Path(__file__).resolve().parent;O=V.parent;P=O.parent;R=PROJECT_ROOT
sys.path.insert(0,str(R/'pipeline'))
from ponytail_mesh import registration_matrix
from ponytail_observations import skin_contour
NAMES=['left','right','up','down','up-left','up-right','down-left','down-right']
THRESHOLD=3916*.5

def skin_boundary(rgb,landmarks):
    # Same pre-convex-hull path as existing reference_contour.py.
    source=inspect.getsource(skin_contour)
    source=source[:source.index('    skin[:]=0')]+"    return face[:,0,:].astype(float)\n"
    space={'cv2':cv2,'np':np};exec(source,space)
    return space['skin_contour'](rgb,landmarks,chin_y=landmarks[6,1])

def registered_ear_box(contour,matrix):
    q=np.asarray(contour,float)@matrix[:,:2].T+matrix[:,2]
    lo=np.floor(q.min(0)-10).astype(int);hi=np.ceil(q.max(0)+10).astype(int)
    return [max(0,int(lo[0])),max(0,int(lo[1])),min(1024,int(hi[0])),min(1024,int(hi[1]))],q

def segment_ear(rgb,box):
    x0,y0,x1,y1=box;pixels=rgb[y0:y1,x0:x1].astype(float);r,g,b=pixels.transpose(2,0,1)
    seed=(r>150)&(g>85)&(b>65)&(r-g>15)&(g-b>4)&(pixels.mean(2)>110)
    clean=cv2.morphologyEx(np.uint8(seed),cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    n,labels,stats,_=cv2.connectedComponentsWithStats(clean,connectivity=8)
    mask=np.zeros((1024,1024),bool)
    if n>1:
        chosen=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));mask[y0:y1,x0:x1]=binary_fill_holes(labels==chosen)
    return mask

def hair_band(rgb):return binary_dilation(binary_dilation(rgb.mean(2)<90,iterations=2),iterations=4)

def prepare_pose(name,key,feature):
    dest=V/'inputs'/name;dest.mkdir(parents=True)
    rgb=np.array(Image.open(R/key['image']).convert('RGB'));landmarks=np.array(key['landmarks'],float)
    matrix=registration_matrix(key,feature)
    registered=cv2.warpAffine(rgb,matrix,(1024,1024),borderValue=(255,255,255))
    boundary=skin_boundary(rgb,landmarks)@matrix[:,:2].T+matrix[:,2]
    (V/'reference-contours').mkdir(exist_ok=True)
    (V/'reference-contours'/f'{name}.json').write_text(json.dumps(boundary.tolist())+'\n')
    skin=np.zeros((1024,1024),np.uint8);cv2.fillPoly(skin,[np.rint(boundary).astype(np.int32)],1);skin=skin>0
    band=hair_band(registered);target=skin.copy();ears={}
    for side,contour in zip(['left','right'],key['earContours']):
        box,points=registered_ear_box(contour,matrix)
        if name=='left' and side=='right':
            mask=np.array(Image.open(O/'v2/reference-ear-mask.png'))>127;source='approved v2 left/right ear; preserved pixels'
        else:mask=segment_ear(registered,box);source='registered earContours bbox +10, specified color/closing/component/fill'
        area=int(mask.sum());visible=area>=THRESHOLD
        Image.fromarray(np.uint8(mask)*255).save(dest/f'ear-{side}-mask.png')
        # Invisible ears are not reference observations or TPS weight exceptions.
        if visible:target|=mask
        yy,xx=np.where(mask)
        ears[side]={'box':box,'maskPixels':area,'visible':visible,'minimumVisiblePixels':THRESHOLD,
            'maskPixelsOutsideBox':int(mask.sum()-mask[box[1]:box[3],box[0]:box[2]].sum()),
            'bounds':[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)] if len(xx) else None,
            'source':source,'registeredInputContour':points.tolist()}
    Image.fromarray(registered).save(dest/'reference-frame.png')
    for title,mask in [('skin-mask',skin),('reference-mask',target),('hairband',band)]:Image.fromarray(np.uint8(mask)*255).save(dest/f'{title}.png')
    registered_landmarks=landmarks@matrix[:,:2].T+matrix[:,2]
    report={'name':name,'sourceImage':key['image'],'registrationMatrix':matrix.tolist(),'registeredLandmarks':registered_landmarks.tolist(),
        'anchorCenters':feature['centers'],'ears':ears,'skinPixels':int(skin.sum()),'unionPixels':int(target.sum()),'hairBandPixels':int(band.sum()),
        'colorRule':'R>150,G>85,B>65,R-G>15,G-B>4,mean>110;1px closing;largest component;fill holes',
        'skinBoundaryMethod':'existing reference_contour.py pre-hull GrabCut boundary','referenceFrameActualYawPitch':[key['yaw'],key['pitch']]}
    (dest/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('C1 MASKS',name,json.dumps({s:{k:v for k,v in e.items() if k not in ['registeredInputContour']} for s,e in ears.items()}),flush=True)
    return report
