"""J5 actual far-eye mask checks at the six authored/profile requests."""
import sys
sys.dont_write_bytecode=True
import functools,http.server,json,threading
from playwright.sync_api import sync_playwright
from capture_sam3_eyes import ROOT,SETUP,Handler,image
from verify_eye_clip_pixels import SIDE_SETUP,pixel_evidence


def main():
    out=ROOT/'debug/sam3-eye-v292/profile-checks';out.mkdir(parents=True,exist_ok=True);rows=[]
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Handler,directory=str(ROOT)))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--use-angle=swiftshader'])
            page=browser.new_page(viewport=dict(width=1440,height=1000),device_scale_factor=1)
            page.goto(f'http://127.0.0.1:{server.server_port}/bundle/',wait_until='networkidle')
            page.wait_for_function('window.studio?.ready',timeout=30000);page.evaluate('window.requestAnimationFrame=()=>0');page.evaluate(SETUP);page.evaluate(SIDE_SETUP)
            for pitch in [-30,0,30]:
                for sign in [-1,1]:
                    for magnitude in [50,55,60,67,80,90]:
                        yaw=sign*magnitude;raw=page.evaluate('(a)=>captureClipPose(a)',[yaw,pitch])
                        captures={k:image(raw[k]) for k in ['color','faceMask','left','right']}
                        side='left' if sign<0 else 'right';evidence=pixel_evidence(captures,side)
                        row=dict(yaw=yaw,pitch=pitch,farSide=side,controls=raw['controls'],**evidence);rows.append(row)
                        for k,im in captures.items():im.save(out/f'{yaw}-{pitch}-{k}.png')
                        report=dict(passed=all(r['pixels']==0 for r in rows),poses=rows)
                        (ROOT/'docs/sam3-eye-junction-profile.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
                        print('J5',yaw,pitch,evidence['pixels'],flush=True)
                        if evidence['pixels']:
                            browser.close();return
            browser.close()
    finally:
        server.shutdown();server.server_close();thread.join()


if __name__=='__main__':main()
