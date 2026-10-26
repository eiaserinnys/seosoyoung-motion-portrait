"""R5 full neutral portrait vs the immutable 6dd12cf main capture."""
import base64,functools,hashlib,http.server,io,json,threading
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
from capture_sam3_eyes import ROOT,SETUP,Handler

def main():
    out=ROOT/'debug/sam3-eye-v295/neutral';out.mkdir(parents=True,exist_ok=True)
    baseline=ROOT/'debug/sam3-eye-v294/final-render/r5-6dd-neutral-full.png'
    assert (ROOT/'debug/r5-6dd/revision.txt').read_text().strip()=='6dd12cfe12b89a3b67b23938996a62c70e281173'
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Handler,directory=str(ROOT)))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--use-angle=swiftshader'])
            page=browser.new_page(viewport=dict(width=1440,height=1000),device_scale_factor=1)
            page.goto(f'http://127.0.0.1:{server.server_port}/bundle/',wait_until='networkidle')
            page.wait_for_function('window.studio?.ready',timeout=30000);page.evaluate('window.requestAnimationFrame=()=>0');page.evaluate(SETUP)
            raw=page.evaluate('(a)=>captureEyes(a)',[0,0,False,True])
            current=Image.open(io.BytesIO(base64.b64decode(raw['full']))).convert('RGBA');current.save(out/'neutral-full.png')
            browser.close()
    finally:
        server.shutdown();server.server_close();thread.join()
    reference=Image.open(baseline).convert('RGBA');assert current.size==reference.size==(1100,1600)
    diff=np.abs(np.array(current).astype(int)-np.array(reference).astype(int));changed=int(np.any(diff,axis=2).sum())
    report=dict(baseline='6dd12cfe12b89a3b67b23938996a62c70e281173',baselineImageSha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),scope='full canvas, hair on, neutral',canvasSize=list(current.size),changedPixels=changed,maxChannelDifferenceLSB=int(diff.max()),passed=changed==0)
    (ROOT/'docs/sam3-eye-neutral-quality.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
