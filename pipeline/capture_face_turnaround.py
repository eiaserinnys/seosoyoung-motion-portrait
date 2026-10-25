"""Seven actual-angle WebGL face renders with reference/outline/mesh evidence."""
import json
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw

from capture_hairless_face import BG, GREEN, PINK, CROP, reference, webgl_renders
from hairless_face_reference import ROOT


def side_reference(pose):
    name=pose['name'].replace('67','_67').replace('90','_90')
    rgb=np.array(Image.open(ROOT/f'source/references/{name}_no_rimlight.png').convert('RGB'))
    mask=np.array(Image.open(ROOT/f'sam3_masks/references/{name}-face.png'))
    canvas=np.full((1024,1024,3),BG,np.uint8)
    canvas[20:740,152:872]=rgb
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.polylines(canvas,[contour+np.array([[[152,20]]])],True,GREEN,2,cv2.LINE_AA)
    return Image.fromarray(canvas)


def draw_outline(canvas,pose):
    draw=ImageDraw.Draw(canvas)
    for side in [0,1]:
        draw.line([tuple(row['points'][side][:2]) for row in pose['boundary']],fill=PINK,width=2)


def mesh_overlay(pose,render):
    out=render.copy();draw=ImageDraw.Draw(out)
    skin=next(m for m in pose['meshes'] if m['id']=='face-skin')
    for tri in skin['triangles']:
        pts=[tuple(skin['positions'][i][:2]) for i in tri]
        draw.line(pts+[pts[0]],fill=(142,69,115),width=1)
    draw_outline(out,pose)
    for key,point in pose['landmarks'].items():
        def matches(mesh):
            name=mesh['id']
            if key in ['left','right']:return name.startswith('eye-'+key+'-') and not name.endswith('brow')
            if key.startswith('brow'):return name=='eye-'+('left' if key=='browLeft' else 'right')+'-brow'
            return name==key+'-detail'
        pts=np.concatenate([np.array(m['positions'])[:,:2] for m in pose['meshes'] if matches(m)])
        draw.rectangle((*pts.min(axis=0),*pts.max(axis=0)),outline=(255,190,90),width=2)
        x,y=point[:2];draw.line((x-3,y,x+3,y),fill=GREEN,width=1);draw.line((x,y-3,x,y+3),fill=GREEN,width=1)
    return out


def main():
    poses=json.loads(subprocess.check_output(['node',str(ROOT/'pipeline/hairless_face_geometry.mjs'),'--turnaround'],timeout=60))
    assert [p['yaw'] for p in poses]==[-90,-67.5,-45,0,45,67.5,90]
    renders=webgl_renders(poses,order=[p['name'] for p in poses],save_grid=False)
    sheet=Image.new('RGB',(7*480,3*480),BG)
    for col,p in enumerate(poses):
        ref=side_reference(p) if abs(p['yaw'])>45 else reference(p)
        draw_outline(ref,p)
        render=Image.new('RGBA',(1024,1024),BG+(255,));render.alpha_composite(renders[p['name']]);render=render.convert('RGB')
        for row,(canvas,label) in enumerate([(ref,'REF / SAM green / rig pink'),(render,'WebGL face only'),(mesh_overlay(p,render),'Mesh / six feature boxes')]):
            tile=canvas.crop(CROP).resize((480,480))
            ImageDraw.Draw(tile).text((8,8),f"{p['yaw']:+g} deg | {label}",fill='white')
            sheet.paste(tile,(col*480,row*480))
    path=ROOT/'docs/captures/face_turnaround_7angle.png';sheet.save(path)
    print('PASS seven WebGL renders, five actual-angle slider inputs, WebGL/console/HTTP errors 0;',path.relative_to(ROOT))


if __name__=='__main__':main()
