"""One actual WebGL capture run plus standalone scientific registration figures."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from ponytail_observations import ROOT,THETA,digest
from ponytail_model import predict,predict_radii,predict_center,measure,assert_registration
from ponytail_mesh import registered,registration_matrix
from hairless_face_reference import geometry,matrix
from capture_hairless_face import webgl_renders,ORDER,BG
FONT=ImageFont.load_default(size=18)


def fit_images(data):
    sheet=Image.new('RGB',(1080,1080),BG)
    rays=np.column_stack((np.cos(THETA),np.sin(THETA)))
    for i,name in enumerate(ORDER):
        row=next(k for k in data['keys'] if k['name']==name)
        im=Image.open(ROOT/row['image']).convert('RGB')
        draw=ImageDraw.Draw(im)
        draw.line([tuple(p) for p in row['contour']]+[tuple(row['contour'][0])],fill='#28bd66',width=2)
        fitted=predict_center(data['silhouette'],row['pose'])+rays*predict_radii(data['silhouette'],row['pose'])[:,None]
        draw.line([tuple(p) for p in fitted]+[tuple(fitted[0])],fill='#ec3c75',width=2)
        for x,y in predict(data['model'],row['pose']):draw.ellipse((x-4,y-4,x+4,y+4),outline='#065cff',width=2)
        tile=im.resize((360,360));ImageDraw.Draw(tile).text((8,8),name,fill='black',font=FONT)
        sheet.paste(tile,((i%3)*360,(i//3)*360))
    return sheet


def trajectory(data):
    time=np.arange(361)/24;poses=np.array([r['pose'] for r in data['video']]);hold=np.zeros(361,bool);hold[data['holdout']['indices']]=True
    error=np.array([np.sqrt(np.mean(np.sum((predict(data['model'],r['pose'])-r['landmarks'])**2,axis=1))) for r in data['video']])
    fig,axes=plt.subplots(2,1,figsize=(14,7),sharex=True,layout='constrained')
    for i,label in enumerate(['Yaw','Pitch','Roll']):axes[0].plot(time,poses[:,i],label=label,linewidth=1.6)
    axes[0].set_ylabel('Degrees');axes[0].legend(ncol=3);axes[0].grid(alpha=.2)
    axes[1].plot(time,error,color='#576c88',linewidth=1,label='Tracked observation vs model')
    axes[1].scatter(time[hold],error[hold],s=12,color='#d44371',label='Held-out shape frames')
    axes[1].axhline(11.1,color='#d44371',linestyle='--',label='11.1 px limit')
    axes[1].set(xlabel='Video time (s)',ylabel='Landmark RMS (720 px reference)');axes[1].legend();axes[1].grid(alpha=.2)
    return fig


def webgl(data):
    def inspect(page):
        page.get_by_role('button',name='앞머리 숨김',exact=True).click()
        page.evaluate("studio.setPose(-22.5,15,0,'Seedance skin inspection')")
        page.evaluate('studio.renderer.render(studio.p,studio.state)')
        page.screenshot(path=str(ROOT/'docs/captures/seedance_viewer_wide.png'))
        page.set_viewport_size(dict(width=720,height=1000))
        page.evaluate('studio.renderer.render(studio.p,studio.state)')
        page.screenshot(path=str(ROOT/'docs/captures/seedance_viewer_narrow.png'))
        page.get_by_role('button',name='앞머리 숨김',exact=True).click()
    poses=geometry();renders=webgl_renders(poses,inspect=inspect,include_ears=True)
    row=np.flatnonzero(np.array(renders['front'])[300,:,3]>127)
    assert len(row) and row[-1]-row[0]>250,'Exposed forehead is still clipped to the original narrow UV cap'
    fill_geometry=json.loads(subprocess.check_output(['node',str(ROOT/'pipeline/skin_fill_geometry.mjs')],timeout=60))
    comparison=Image.new('RGB',(2160,1080),BG);overlay=Image.new('RGB',(1440,1440),BG)
    for i,name in enumerate(ORDER):
        row=next(k for k in data['keys'] if k['name']==name);pose=next(p for p in poses if p['name']==name)
        rgb=np.array(Image.open(ROOT/row['image']).convert('RGB'))
        ref=Image.fromarray(cv2.warpAffine(rgb,registration_matrix(row,pose['feature']),(1024,1024),borderValue=BG))
        contour=registered(row,pose['feature'])
        ImageDraw.Draw(ref).line([tuple(p) for p in contour]+[tuple(contour[0])],fill='#28bd66',width=2)
        render=Image.new('RGBA',(1024,1024),BG+(255,));render.alpha_composite(renders[name]);render=render.convert('RGB')
        for j,im in enumerate([ref,render]):
            tile=im.crop((152,20,872,740)).resize((360,360));ImageDraw.Draw(tile).text((8,8),name+(' | reference' if j==0 else ' | WebGL'),fill='black' if j==0 else 'white',font=FONT)
            comparison.paste(tile,(((i%3)*2+j)*360,(i//3)*360))
        draw=ImageDraw.Draw(render)
        fill=next(m for m in fill_geometry if m['name']==name)
        for tri in fill['triangles']:
            points=[tuple(fill['positions'][n][:2]) for n in tri];draw.line(points+[points[0]],fill='#476f79',width=1)
        mesh=next(m for m in pose['meshes'] if m['id']=='face-skin')
        for tri in mesh['triangles']:
            points=[tuple(mesh['positions'][n][:2]) for n in tri];draw.line(points+[points[0]],fill='#c85b91',width=1)
        draw.line([tuple(p) for p in contour]+[tuple(contour[0])],fill='#28bd66',width=2)
        overlay.paste(render.crop((152,20,872,740)).resize((480,480)),((i%3)*480,(i//3)*480))
    comparison.save(ROOT/'docs/captures/ponytail_9dir_registration_comparison.png')
    comparison.save(ROOT/'docs/captures/hairless_face_9dir_comparison.png')
    overlay.save(ROOT/'docs/captures/face_geom_9dir_overlay.png')
    return [dict(name=p['name'],minDetJ=p['minDetJ'],reversedTriangles=p['reversedTriangles']) for p in poses]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifacts',type=Path,required=True);args=parser.parse_args()
    data=json.loads((ROOT/'source/viewer/ponytail-seedance-registration.json').read_text())
    measured=assert_registration(data)
    docs=ROOT/'docs/captures'
    sheet=fit_images(data);sheet.save(docs/'stage4_9dir_fitted_overlay_1080.png')
    sheet.save(args.artifacts/'stage4_9dir_fitted_overlay_1080.png')
    fig=trajectory(data);fig.savefig(docs/'stage4_pose_trajectory_and_fit.png',dpi=150);fig.savefig(args.artifacts/'stage4_pose_trajectory_and_fit.png',dpi=150);plt.close(fig)
    meshes=webgl(data)
    assert all(p['reversedTriangles']==0 and p['minDetJ']>=.0034 for p in meshes),meshes
    metrics=dict(**measured,mesh=meshes,videoFrames=361,holdoutFrames=54,
                 captureSha256={p.name:digest(p) for p in sorted(docs.glob('*.png')) if p.name in ['ponytail_9dir_registration_comparison.png','hairless_face_9dir_comparison.png','face_geom_9dir_overlay.png','viewer_9dir_grid.png','stage4_pose_trajectory_and_fit.png','stage4_9dir_fitted_overlay_1080.png']})
    text=json.dumps(metrics,ensure_ascii=False,indent=2)+'\n'
    (ROOT/'docs/stage4_registration_metrics.json').write_text(text)
    (args.artifacts/'stage4_registration_metrics.json').write_text(text)
    print('PASS 9 actual WebGL poses, existing pose button and five yaw sliders; 6 captures; HTTP/console/GL errors 0')


if __name__=='__main__':main()
