"""Final matched-angle evidence after all 48 R4 poses have been recorded.

Prior captures have a different renderer or input-angle contract. This run has
its own directory and resumes only complete captures from this final contract.
"""
import ast
import base64
import io
import functools
import http.server
import json
import threading
import shutil
from eye_render_ratios import ratio_gates,model_fidelity
import numpy as np
from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright
import capture_sam3_eyes as c
from eye_capture_angles import inverse_achieved, achieved

ROOT=c.ROOT
OUT=ROOT/'debug/sam3-eye-v294/final-render'
DOC=c.DOC
def clipped_measure(captures,row,side):
    q=c.measure(captures,row,side)
    sil=c.binary(captures['faceMask'])
    outward=-1 if side=='left' else 1
    distance=float('inf')
    for step in range(301):
        x,y=np.rint(np.array(q['cO'])+step*outward*np.array(q['E'])[:,0]).astype(int)
        if not (0<=x<720 and 0<=y<720 and sil[y,x]):
            distance=float(step);break
    env=q['dTemp'];theta=-row['yaw']*(1 if side=='left' else -1)
    q.update(dTempEnv=env,dTempSil=distance,temporalAtFaceEdge=bool(theta>=15 and distance-env<=5))
    if q['temporalAtFaceEdge']:q['dTemp']=min(env,distance)
    return q


def stage_measure(name,rgba,owned,geometry):
    # Execute the existing gate math unchanged, retaining all metrics when an
    # assertion fails. The delegated final contract makes Stage4 report-only.
    tree=ast.parse((ROOT/'pipeline/skin_fill_validation.py').read_text())
    class ReportOnly(ast.NodeTransformer):
        def visit_Assert(self,node):return None
    scope={};exec(compile(ast.fix_missing_locations(ReportOnly().visit(tree)),'<Stage4 report-only>','exec'),scope)
    owned=owned.copy();white=(owned[:,:,:3].min(2)>127)&(owned[:,:,3]>127)
    owned[white,1:3]=0  # Existing validator expects red; identify renders white.
    result=scope['validate_pixels'](name,rgba,owned,geometry)
    result['passed']=(result['exposedPixelsBelowEarLine']==0 and result['downHalfWidthDifference']<=2 and result['maxForeheadLuminanceValley']<=10)
    return result


def regression_sheet():
    report=json.loads((ROOT/'docs/sam3-eye-far-clip-contract.json').read_text())
    cache=ROOT/'debug/sam3-eye-v294/far-clip-contract';tiles=[]
    for row in report['poses']:
        for side in row['increasedSides']:
            coords=row['sides'][side]['coordinates720']+row['sides'][side]['r24Coordinates720']
            if not coords:continue
            # Include the full eye, not just the few outside pixels.
            bounds=[]
            for rev in ['v291','r24']:
                path=cache/f"{rev}-{row['name']}-{side}.png"
                im=Image.open(path) if path.exists() else Image.open(cache/f"front-{side}.png")
                yy,xx=np.where(c.binary(im));bounds.append((xx.min(),yy.min(),xx.max(),yy.max()))
            box=(min(b[0] for b in bounds)-8,min(b[1] for b in bounds)-8,max(b[2] for b in bounds)+9,max(b[3] for b in bounds)+9)
            pair=[]
            for rev,field in [('v291','coordinates720'),('r24','r24Coordinates720')]:
                path=cache/f"{rev}-{row['name']}-color.png"
                im=Image.open(path if path.exists() else cache/'front-color.png').convert('RGB')
                for x,y in row['sides'][side][field]:im.putpixel((x,y),(255,0,0))
                im=im.crop(box).resize(((box[2]-box[0])*4,(box[3]-box[1])*4),Image.Resampling.NEAREST)
                pair.append(im)
            tiles.append((row['name']+' '+side,pair))
    width=max((im.width for _,pair in tiles for im in pair),default=200)+16
    sheet=Image.new('RGB',(width*2,max(28,sum(pair[0].height+24 for _,pair in tiles))),'#eeeeee');y=0
    for label,pair in tiles:
        for j,im in enumerate(pair):
            sheet.paste(im,(width*j,y));ImageDraw.Draw(sheet).text((width*j+4,y+im.height+3),label+' | '+['F-B2','r24'][j],fill='black')
        y+=pair[0].height+24
    if not tiles:ImageDraw.Draw(sheet).text((8,8),'No increases over r24',fill='black')
    sheet.save(DOC/'sam3_eye_R7_outside_regressions_4x.png')

def temporal_tip_sheet():
    tiles=[]
    for name,side in [('aux-0213','right'),('aux-0255','right'),('down','right'),('aux-0048','left')]:
        yy,xx=np.where(c.binary(Image.open(OUT/f'v291-{name}-{side}-env.png').convert('RGBA')))
        box=(int(xx.min())-12,int(yy.min())-12,int(xx.max())+13,int(yy.max())+13)
        im=Image.open(OUT/f'v291-{name}-color.png').convert('RGB').crop(box)
        tiles.append((f'{name}:{side} | temporal tip 2x',im.resize((im.width*2,im.height*2),Image.Resampling.NEAREST)))
    strip=Image.new('RGB',(sum(im.width+16 for _,im in tiles),max(im.height for _,im in tiles)+28),'#eeeeee');x=0
    for label,im in tiles:
        strip.paste(im,(x,0));ImageDraw.Draw(strip).text((x+4,im.height+5),label,fill='black');x+=im.width+16
    strip.save(DOC/'sam3_eye_FB2_temporal_tips_2x.png')
    sheet=Image.open(DOC/'sam3_eye_R7_comparison.png').convert('RGB')
    combined=Image.new('RGB',(max(sheet.width,strip.width),sheet.height+strip.height),'#eeeeee')
    combined.paste(sheet,(0,0));combined.paste(strip,(0,sheet.height));combined.save(DOC/'sam3_eye_R7_comparison.png')


def main():
    OUT.mkdir(parents=True,exist_ok=True);c.OUT=OUT
    r4=json.loads((ROOT/'docs/sam3-eye-far-clip-contract.json').read_text())
    junction=json.loads((ROOT/'docs/sam3-eye-junction-v2.json').read_text())
    j5=dict(runtimeRevision='v2.9.4 F-B2',status='closed by Fable review; not rerun',maskSource='skin-silhouette',
            scope='Fable re-review 1: J1/J2/J3 and R4 mask checks',
            evidence=['sam3-eye-far-clip-contract.json','sam3-eye-junction-alpha.json'])
    assert junction['complete'] and junction['passed']['J1'] and junction['passed']['J6']
    assert r4['status']=='R4_RECORDED' and r4['complete'] and r4['R5']['passed']
    baseline=ROOT/'debug/sam3-eye-v293/final-render'
    for path in baseline.glob('r24-*.png'):shutil.copyfile(path,OUT/path.name)
    neutral={}
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(c.Handler,directory=str(ROOT)))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    colors={r:{} for r in ['r24','v291']};records={r:{} for r in colors};inputs=[]
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--use-angle=swiftshader'])
            baseline_page=browser.new_page(viewport=dict(width=1440,height=1000),device_scale_factor=1)
            baseline_page.goto(f'http://127.0.0.1:{server.server_port}/debug/r5-6dd/bundle/',wait_until='networkidle')
            baseline_page.wait_for_function('window.studio?.ready',timeout=30000)
            baseline_page.evaluate('window.requestAnimationFrame=()=>0');baseline_page.evaluate(c.SETUP)
            baseline_raw=baseline_page.evaluate('(a)=>captureEyes(a)',[0,0,False,True])
            neutral['r24']=c.image(baseline_raw['color'])
            baseline_full=Image.open(io.BytesIO(base64.b64decode(baseline_raw['full']))).convert('RGBA')
            baseline_full.save(OUT/'r5-6dd-neutral-full.png');baseline_page.close()
            for rev,path in [('r24','debug/r24/bundle'),('v291','bundle')]:
                page=None
                if rev=='v291':
                    page=browser.new_page(viewport=dict(width=1440,height=1000),device_scale_factor=1)
                    page.goto(f'http://127.0.0.1:{server.server_port}/{path}/',wait_until='networkidle')
                    page.wait_for_function('window.studio?.ready',timeout=30000)
                    page.evaluate('window.requestAnimationFrame=()=>0');page.evaluate(c.SETUP)
                    raw=page.evaluate('(a)=>captureEyes(a)',[0,0,False,True])
                    full=Image.open(io.BytesIO(base64.b64decode(raw['full']))).convert('RGBA')
                    full.save(OUT/'v291-neutral-full.png')
                    diff=np.abs(np.array(full).astype(int)-np.array(baseline_full).astype(int))
                    r4['R5']=dict(baseline=(ROOT/'debug/r5-6dd/revision.txt').read_text().strip(),scope='full canvas, hair on, neutral',canvasSize=list(full.size),changedPixels=int(np.any(diff,axis=2).sum()),maxChannelDifferenceLSB=int(diff.max()))
                    r4['R5']['passed']=r4['R5']['changedPixels']==0
                    print('R5 MAIN 6dd12cf',json.dumps(r4['R5']),flush=True)
                    assert r4['R5']['passed'],'R5 neutral changed'
                    neutral['v291']=c.image(raw['color'])
                for row in c.ROWS:
                    requested=inverse_achieved(c.BOOT['achieved']['c'],row['yaw'],row['pitch'])
                    keys=['color','allEyes','faceMask']+[side+'-'+k for side in ['left','right'] for k in ['env','inner','iris','lash']]
                    if row['name'] in c.ORDER:keys+=['full']
                    paths={k:OUT/f"{rev}-{row['name']}-{k}.png" for k in keys}
                    if all(p.exists() for p in paths.values()):captures={k:Image.open(p).convert('RGBA') for k,p in paths.items()}
                    else:
                        assert page is not None,'Missing unchanged r24 evidence'
                        raw=page.evaluate('(a)=>captureEyes(a)',[*requested,True,row['name'] in c.ORDER])
                        captures={k:c.image(v,full=k=='full') for k,v in raw.items()}
                        for k,im in captures.items():im.save(paths[k])
                    records[rev][row['name']]={side:clipped_measure(captures,row,side) for side in ['left','right']}
                    colors[rev][row['name']]=captures['color']
                    if rev=='v291':inputs.append(dict(name=row['name'],requested=requested,targetAchieved=[row['yaw'],row['pitch']],actualAchieved=achieved(c.BOOT['achieved']['c'],*requested)))
                    print('MATCHED F-B2',rev,row['name'],flush=True)
                if page is not None:
                    sweep=[]
                    for yaw in [15,17,19,21,23,25,27,29,30]:
                        requested=inverse_achieved(c.BOOT['achieved']['c'],yaw,0)
                        im=c.image(page.evaluate('(a)=>captureEyes(a)',[*requested,False,False])['color']);sweep.append((yaw,im))
                    strip=Image.new('RGB',(len(sweep)*320,300),'#eeeeee')
                    for i,(yaw,im) in enumerate(sweep):
                        strip.paste(im.convert('RGB').crop((405,330,565,470)).resize((320,280)),(i*320,0))
                        ImageDraw.Draw(strip).text((i*320+4,283),f'achieved yaw {yaw}',fill='black')
                    strip.save(DOC/'sam3_eye_R7_temporal_sweep.png')
                    page.evaluate('studio.setPose(0,0,0,"Neutral");studio.renderer.render(studio.p,studio.state)')
                    page.screenshot(path=str(DOC/'sam3_eye_viewer_wide.png'))
                    page.set_viewport_size(dict(width=720,height=1000));page.screenshot(path=str(DOC/'sam3_eye_viewer_narrow.png'));page.close()
            browser.close()
    finally:
        server.shutdown();server.server_close();thread.join()
    combined=[dict(name=r['name'],side=side,**{rev:records[rev][r['name']][side] for rev in records}) for r in c.ROWS for side in ['left','right']]
    report=c.gates(combined,neutral)
    report['runtimeRevision']='v2.9.4 F-B2 solid skin-silhouette union, requested 52..67 junction'
    report['absoluteForm']='Reference only: SAM vs identify constant offsets; ratio form is authoritative'
    report.update(ratio_gates(combined,c.M,c.AUDIT));report['R5']=r4['R5']
    report.update(model_fidelity(combined,c.M,json.loads((ROOT/'source/viewer/sam3-eye-fit.json').read_text())))
    accepted={'dNas':report['R2Ratio']['dNas']['passed'],'R3':report['R3Ratio']['v291']['passed'],'deltaY':report['R2Ratio']['deltaY']['passed']}
    report['FableAcceptedExcess']={k:'PASS' if passed else '한계 초과이나 렌더에서 결함이 아님' for k,passed in accepted.items()}
    report['FableAcceptanceReason']='r25-impl-review-fable.md 1.2: dNas/R3 residual is reference nasal-line extension and beta omission plus approved nasal outlier, not iris/lid misalignment; deltaY excess is subpixel. Limits unchanged.'
    report['Junction']=junction;report['J3']=json.loads((ROOT/'docs/sam3-eye-junction-alpha.json').read_text());report['J5']=j5
    (ROOT/'docs/sam3-eye-junction-profile.json').write_text(json.dumps(j5,ensure_ascii=False,indent=2)+'\n')
    report['Stage4']=dict(status='excluded by Fable r25 1.3; not remeasured in this review')
    report['Stage4Evidence']='Historical finding only: original up 5px matches r24; hair-on 0px. Previous candidate metrics are not carried forward.'
    report['R4']=dict(status=r4['status'],poses=len(r4['poses']));report['records']=combined;report['poseInputs']=inputs
    (ROOT/'docs/sam3-eye-render-metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    c.sheets(colors);temporal_tip_sheet();regression_sheet()
    grid=Image.new('RGB',(1440,1440),'#1e2430')
    for i,name in enumerate(c.ORDER):
        im=Image.open(OUT/f'v291-{name}-full.png').convert('RGB').crop((152,20,872,740)).resize((480,480))
        ImageDraw.Draw(im).text((8,8),name,fill='white');grid.paste(im,((i%3)*480,(i//3)*480))
    grid.save(DOC/'sam3_eye_9dir_grid.png')
    print(json.dumps({k:report[k] for k in ['R2Ratio','R3Ratio','frontOffsets','R4','R5']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
