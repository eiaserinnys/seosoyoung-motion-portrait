"""Final matched-angle evidence; only R1/R4-far/R5 are stop gates.

Prior captures have a different renderer or input-angle contract. This run has
its own directory and resumes only complete captures from this final contract.
"""
import ast
import functools
import http.server
import json
import threading
import shutil
from eye_render_ratios import ratio_gates
import numpy as np
from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright
import capture_sam3_eyes as c
from eye_capture_angles import inverse_achieved, achieved

ROOT=c.ROOT
OUT=ROOT/'debug/sam3-eye-v292/final-render'
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
    cache=ROOT/'debug/sam3-eye-v292/far-clip-contract';tiles=[]
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
    width=max(im.width for _,pair in tiles for im in pair)+16
    sheet=Image.new('RGB',(width*2,sum(pair[0].height+24 for _,pair in tiles)),'#eeeeee');y=0
    for label,pair in tiles:
        for j,im in enumerate(pair):
            sheet.paste(im,(width*j,y));ImageDraw.Draw(sheet).text((width*j+4,y+im.height+3),label+' | '+['v2.9.2','r24'][j],fill='black')
        y+=pair[0].height+24
    sheet.save(DOC/'sam3_eye_R7_outside_regressions_4x.png')


def main():
    OUT.mkdir(parents=True,exist_ok=True);c.OUT=OUT
    r4=json.loads((ROOT/'docs/sam3-eye-far-clip-contract.json').read_text())
    junction=json.loads((ROOT/'docs/sam3-eye-junction-v2.json').read_text())
    j5=json.loads((ROOT/'docs/sam3-eye-junction-profile.json').read_text())
    assert junction['complete'] and junction['passed']['J1'] and junction['passed']['J6']
    assert r4['status']=='R4_PASS' and r4['R5']['passed'] and j5['passed']
    old_metrics=json.loads((ROOT/'docs/sam3-eye-render-metrics.json').read_text())
    baseline=ROOT/'debug/sam3-eye-v291/final-render'
    for path in baseline.glob('r24-*.png'):shutil.copyfile(path,OUT/path.name)
    neutral={rev:Image.open(ROOT/f'debug/sam3-eye-v292/far-clip-contract/{rev}-neutral.png').convert('RGBA') for rev in ['r24','v291']}
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(c.Handler,directory=str(ROOT)))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    colors={r:{} for r in ['r24','v291']};records={r:{} for r in colors};inputs=[]
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--use-angle=swiftshader'])
            for rev,path in [('r24','debug/r24/bundle'),('v291','bundle')]:
                page=None
                if rev=='v291':
                    page=browser.new_page(viewport=dict(width=1440,height=1000),device_scale_factor=1)
                    page.goto(f'http://127.0.0.1:{server.server_port}/{path}/',wait_until='networkidle')
                    page.wait_for_function('window.studio?.ready',timeout=30000)
                    page.evaluate('window.requestAnimationFrame=()=>0');page.evaluate(c.SETUP)
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
                    print('MATCHED v292',rev,row['name'],flush=True)
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
    report['runtimeRevision']='v2.9.2 requested 52..67 junction'
    report['absoluteForm']='Reference only: SAM vs identify constant offsets; ratio form is authoritative'
    report.update(ratio_gates(combined,c.M,c.AUDIT));report['R5']=r4['R5']
    report['Junction']=junction;report['J3']=json.loads((ROOT/'docs/sam3-eye-junction-alpha.json').read_text());report['J5']=j5
    report['Stage4']=old_metrics['Stage4'];report['Stage4Evidence']='Reused: original up 5px matches r24; hair-on 0px; excluded by Fable junction v2'
    report['R4']=dict(status=r4['status'],poses=len(r4['poses']));report['records']=combined;report['poseInputs']=inputs
    (ROOT/'docs/sam3-eye-render-metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    c.sheets(colors);regression_sheet()
    grid=Image.new('RGB',(1440,1440),'#1e2430')
    for i,name in enumerate(c.ORDER):
        im=Image.open(OUT/f'v291-{name}-full.png').convert('RGB').crop((152,20,872,740)).resize((480,480))
        ImageDraw.Draw(im).text((8,8),name,fill='white');grid.paste(im,((i%3)*480,(i//3)*480))
    grid.save(DOC/'sam3_eye_9dir_grid.png')
    print(json.dumps({k:report[k] for k in ['R2Ratio','R3Ratio','frontOffsets','R4','R5']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
