"""Clean neutral skin backing and the unchanged Seedance polar surface for WebGL."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from ponytail_model import predict_center,predict_radii,assert_registration
from ponytail_observations import ROOT,digest


def build():
    path=ROOT/'source/viewer/ponytail-seedance-registration.json'
    data=json.loads(path.read_text());assert_registration(data)
    image=np.array(Image.open(ROOT/'source/references/ponytail/ponytail_front.png').convert('RGB'))
    points=np.array(data['frontConstraint']['landmarks'])
    feature_mask=np.zeros(image.shape[:2],np.uint8)
    for i,(x,y) in enumerate(points[:6]):
        axes=(55,46) if i<2 else (60,21) if i<4 else (25,28) if i==4 else (35,17)
        cv2.ellipse(feature_mask,(round(x),round(y)),axes,0,0,360,255,-1)
    for x,y in points[7:9]:cv2.ellipse(feature_mask,(round(x),round(y)),(55,75),0,0,360,255,-1)
    # Extend measured skin colors across removed features and beyond the source
    # hairline; the learned geometry, rather than painted hair, owns the edge.
    skin=np.zeros(image.shape[:2],np.uint8)
    cv2.fillPoly(skin,[np.rint(data['frontConstraint']['contour']).astype(np.int32)],1)
    dark=image.mean(axis=2)<160
    cleaned=cv2.inpaint(image,np.uint8((feature_mask>0)|dark)*255,9,cv2.INPAINT_TELEA)
    warm=image[:,:,0].astype(int)-image[:,:,2].astype(int)>25
    valid=(skin>0)&(~dark)&warm&(feature_mask==0)
    nearest=distance_transform_edt(~valid,return_distances=False,return_indices=True)
    extended=cv2.GaussianBlur(cleaned[nearest[0],nearest[1]],(0,0),6)
    # Original face RGB provides the exact overlap color. Its immutable atlas
    # remains untouched; only this separate backing texture is generated.
    rig=json.loads((ROOT/'bundle/assets/seosoyoung-rig.json').read_text())
    mesh=next(m for m in rig['meshes'] if m['id']=='face-skin')
    ax,ay,w,h=mesh['atlas'];bx,by=mesh['bbox'][:2]
    atlas=np.array(Image.open(ROOT/'bundle/assets/texture_00.png').convert('RGBA'))
    original=atlas[ay:ay+h,ax:ax+w]
    rgb=np.full((1024,1024,3),0,np.uint8);rgb[20:740,152:872]=extended
    # Match the untouched face's neutral forehead tone without copying its
    # cropped edge pixels or inpainted fringe into the larger backing surface.
    source=rgb[290:330,460:540].astype(float)
    target=original[290-by:330-by,460-bx:540-bx,:3].astype(float)
    delta=np.median(target-source,axis=(0,1))
    rgb=np.clip(rgb.astype(float)+delta,0,255).astype(np.uint8)
    # RGB outside the neutral image is never sampled by this polar mesh.
    y=np.arange(1024)[:,None];fade=np.clip((500-y)/90,0,1);fade=fade*fade*(3-2*fade)
    mask=np.broadcast_to(np.rint(255*fade).astype(np.uint8),(1024,1024)).copy()
    mask[:20]=0;mask[740:]=0;mask[:,:152]=0;mask[:,872:]=0
    old_alpha=np.zeros((1024,1024),np.uint8);old_alpha[by:by+h,bx:bx+w]=original[:,:,3]
    transition=np.clip((y-390)/40,0,1)
    mask=np.rint(mask*(1-transition+transition*old_alpha/255)).astype(np.uint8)
    Image.fromarray(np.dstack((rgb,mask))).save(ROOT/'bundle/assets/exposed-skin.png')
    # Feather the original upper crop into the backing. The measured lower
    # face keeps its authored alpha, and the original atlas remains immutable.
    cap=np.zeros((1024,1024),np.uint8)
    cv2.fillPoly(cap,[np.rint(np.array(data['frontConstraint']['contour'])+[152,20]).astype(np.int32)],255)
    lower=np.clip((y-390)/20,0,1);cap=cap*(1-lower)+255*lower
    distance=cv2.distanceTransform(np.uint8(old_alpha>127),cv2.DIST_L2,3)
    upper=np.clip((410-y)/20,0,1)
    cap=np.rint(cap*(1-upper+upper*np.clip(distance/12,0,1))).astype(np.uint8)
    Image.fromarray(np.dstack((np.full((1024,1024,3),255,np.uint8),cap))).save(ROOT/'bundle/assets/ponytail-skin-alpha.png')
    shell=data['silhouette'];theta=np.array(data['theta'])
    center=predict_center(shell,[0,0,0,0,0,0]);radius=predict_radii(shell,[0,0,0,0,0,0])
    rays=np.column_stack((np.cos(theta),np.sin(theta)));n=len(theta)
    origin=points[:2].mean(0)
    # The fitted shell is in the eye-centered model gauge; UVs sample the
    # canonical image gauge. Camera alignment is applied only at runtime.
    neutral=np.vstack((center,*[center+radius[:,None]*rays*f for f in np.linspace(1/12,1,12)]))+origin
    triangles=[[0,1+i,1+(i+1)%n] for i in range(n)]
    for ring in range(11):
        for i in range(n):
            j=(i+1)%n;a=1+ring*n+i;b=1+ring*n+j;c=a+n;d=b+n
            triangles.extend([[a,c,d],[a,d,b]])
    model=data['model']
    output=dict(sourceSha256=digest(path),shell=shell,theta=theta.tolist(),neutral=neutral.tolist(),triangles=triangles,
                eye3D=model['points3D'][:2],eyeResidual=np.array(model['residual'])[:,:2].tolist(),
                poses=[dict(yaw=k['requestedAngles'][0],pitch=k['requestedAngles'][1],
                            sourceYaw=k['pose'][0] if k['name']!='front' else 0.,
                            sourcePitch=k['pose'][1] if k['name']!='front' else 0.) for k in data['keys']])
    (ROOT/'bundle/skin-fill-data.mjs').write_text('// Generated by pipeline/build_skin_fill.py.\nexport const SKIN_FILL='+json.dumps(output,separators=(',',':'))+';\n')
    print('PASS clean exposed skin backing; canonical/registration/atlas/rig unchanged')

if __name__=='__main__':build()
