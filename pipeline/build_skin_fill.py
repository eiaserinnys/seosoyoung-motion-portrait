"""Clean neutral skin backing and the unchanged Seedance polar surface for WebGL."""
import json
import collections
import copy
import subprocess
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
    skin=cv2.erode(skin,np.ones((35,35),np.uint8))
    bright_skin=(image.mean(axis=2)>200)&(image[:,:,0]>240)&(image[:,:,1]>200)&(image[:,:,2]>175)
    valid=(skin>0)&(~dark)&warm&bright_skin&(feature_mask==0)
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
    source=rgb[400:440,470:530].astype(float)
    target=original[400-by:440-by,470-bx:530-bx,:3].astype(float)
    delta=np.median(target-source,axis=(0,1))
    rgb=np.clip(rgb.astype(float)+delta,0,255).astype(np.uint8)
    # Both surfaces are opaque: the ribbon shares the actual boundary and a
    # 16px interior skirt. The auxiliary RGBA stores RGB repair, not alpha loss.
    y=np.arange(1024)[:,None]
    ear_height=float(points[7:9,1].min()+20-10)
    mask=np.zeros((1024,1024),np.uint8);mask[20:548,152:872]=255
    Image.fromarray(np.dstack((rgb,mask))).save(ROOT/'bundle/assets/exposed-skin.png')
    old_alpha=np.zeros((1024,1024),np.uint8);old_alpha[by:by+h,bx:bx+w]=original[:,:,3]
    distance=cv2.distanceTransform(np.uint8(old_alpha>127),cv2.DIST_L2,3)
    weight=np.clip(1-distance/24.,0,1)*(y<ear_height)*(old_alpha>127)
    weight=weight*weight*(3-2*weight)
    Image.fromarray(np.dstack((rgb,np.rint(weight*255).astype(np.uint8)))).save(ROOT/'bundle/assets/ponytail-skin-alpha.png')
    shell=copy.deepcopy(data['silhouette']);theta=np.array(data['theta'])
    mirror=np.array([np.argmin(np.abs(np.angle(np.exp(1j*(np.pi-th-theta))))) for th in theta])
    sign=np.array([1,-1,1,1,-1,1,-1,1,-1,1],float)
    for key in ['front','depth']:
        value=np.array(shell[key]);shell[key]=((value+value[mirror])/2).tolist()
    value=np.array(shell['residual']);shell['residual']=((value+sign[:,None]*value[:,mirror])/2).tolist()
    shell['center']['point3D'][0]=0.
    value=np.array(shell['center']['residual'])
    value[:,0]=(value[:,0]-sign*value[:,0])/2
    value[:,1]=(value[:,1]+sign*value[:,1])/2
    shell['center']['residual']=value.tolist()
    # The largest boundary loop is the outside; the mesh also has small holes.
    edges=collections.Counter(tuple(sorted((a,b))) for tri in mesh['triangles'] for a,b in zip(tri,tri[1:]+tri[:1]))
    adj=collections.defaultdict(list)
    for (a,b),count in edges.items():
        if count==1:adj[a].append(b);adj[b].append(a)
    loops=[];remaining=set(adj)
    while remaining:
        start=next(iter(remaining));loop=[start];previous=-1;current=start
        while True:
            nxt=next(i for i in adj[current] if i!=previous)
            if nxt==start:break
            loop.append(nxt);previous,current=current,nxt
        remaining.difference_update(loop);loops.append(loop)
    loop=max(loops,key=len);rest=np.array(mesh['positions'])
    # Rotate the loop at a lower vertex so the selected upper run is continuous.
    cut=next(i for i,v in enumerate(loop) if rest[v,1]>=ear_height)
    loop=loop[cut:]+loop[:cut]
    inner=[i for i in loop if rest[i,1]<ear_height]
    if rest[inner[0],0]>rest[inner[-1],0]:inner.reverse()
    m=97;triangles=[]
    for k in range(5):
        for j in range(m-1):
            a=k*m+j;b=a+1;c=a+m;d=c+1
            triangles.extend([[a,c,d],[a,d,b]])
    poses=[dict(yaw=k['requestedAngles'][0],pitch=k['requestedAngles'][1],
                sourceYaw=k['pose'][0],sourcePitch=k['pose'][1]) for k in data['keys']]
    for row in poses:
        if row['yaw']==0:row['sourceYaw']=0.
        if row['pitch']==0 and row['yaw']==0:row['sourcePitch']=0.
    for pitch in [-30,0,30]:
        pair=[k for k in poses if k['pitch']==pitch and k['yaw']!=0]
        yaw=np.mean([abs(k['sourceYaw']) for k in pair]);p=np.mean([k['sourcePitch'] for k in pair])
        for k in pair:k.update(sourceYaw=float(np.sign(k['yaw'])*yaw),sourcePitch=float(p))
    model=data['model']
    def paired_controls(indices):
        pt=np.array(model['points3D'])[indices].copy()
        half=(pt[1,0]-pt[0,0])/2
        pt[:,0]=[-half,half];pt[:,1:]=pt[:,1:].mean(0)
        res=np.array(model['residual'])[:,indices].copy()
        res[:,0,0]=(res[:,0,0]-sign*res[:,1,0])/2
        res[:,1,0]=-sign*res[:,0,0]
        res[:,0,1]=(res[:,0,1]+sign*res[:,1,1])/2
        res[:,1,1]=sign*res[:,0,1]
        return pt.tolist(),res.tolist()
    eye,eye_res=paired_controls([0,1]);ear,ear_res=paired_controls([7,8])
    output=dict(sourceSha256=digest(path),shell=shell,theta=theta.tolist(),mirror=mirror.tolist(),earHeight=ear_height,
                innerRow=inner,columns=m,neutral=[],triangles=triangles,
                eye3D=eye,eyeResidual=eye_res,ear3D=ear,earResidual=ear_res,poses=poses)
    target=ROOT/'bundle/skin-fill-data.mjs'
    def save():target.write_text('// Generated by pipeline/build_skin_fill.py.\nexport const SKIN_FILL='+json.dumps(output,separators=(',',':'))+';\n')
    save()
    # Reuse runtime geometry for exact rest UVs, rather than a second geometry path.
    output['neutral']=json.loads(subprocess.check_output(['node',str(ROOT/'pipeline/skin_fill_geometry.mjs'),'--neutral'],text=True))
    save()
    print(f'PASS clean upper ribbon {6*m} vertices; frozen registration/atlas/rig unchanged')

if __name__=='__main__':build()
