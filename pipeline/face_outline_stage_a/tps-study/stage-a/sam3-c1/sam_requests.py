from pathlib import Path as _ProjectPath
PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[5]
"""Bounded fal SAM3 calls on small registered ROIs; no local fit or rendering.
Remote waits run outside the shared heavy slot. Keys remain in process memory.
"""
import concurrent.futures,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
V=Path(__file__).resolve().parent;A=V.parent;O=A.parent;P=O.parent;R=PROJECT_ROOT
sys.path.insert(0,str(R/'pipeline'))
from stage1_sam3_segment import call_sam3_on_crop
NAMES=['left','right','up','down','up-left','up-right','down-left','down-right']
FACE_PROMPT='face (skin) of anime girl excluding ears neck and hair'

def specs():
    jobs=[]
    source_check=json.loads((A/'sam3-source-comparison.json').read_text())
    assert len(source_check)==8 and not any(q['pixelsEqual'] for q in source_check)
    for name in NAMES:
        frame=A/'inputs'/name/'reference-frame.png';sha=hashlib.sha256(frame.read_bytes()).hexdigest()
        jobs.append({'id':name+'-face','pose':name,'part':'face','frame':str(frame),'frameSHA256':sha,
            'prompt':FACE_PROMPT,'crop':[150,120,850,740],'boxes':None,'upscale':1})
        meta=json.loads((A/'inputs'/name/'manifest.json').read_text())
        for side in ['left','right']:
            # Existing bbox is earContours +10. Another30 gives requested +40.
            x0,y0,x1,y1=meta['ears'][side]['box']
            box=[max(0,x0-30),max(0,y0-30),min(1024,x1+30),min(1024,y1+30)]
            crop=[max(0,box[0]-40),max(0,box[1]-40),min(1024,box[2]+40),min(1024,box[3]+40)]
            jobs.append({'id':name+'-ear-'+side,'pose':name,'part':'ear-'+side,'frame':str(frame),'frameSHA256':sha,
                'prompt':'ear','crop':crop,'boxes':[box],'upscale':1})
    assert len(jobs)==24
    return jobs

def safe_call(job,key,call=call_sam3_on_crop):
    im=Image.open(job['frame']).convert('RGBA')
    try:mask,meta=call(key,im,tuple(job['crop']),job['prompt'],boxes=job['boxes'],upscale=job['upscale'])
    except ValueError as e:
        if str(e)!='SAM 3 returned an empty mask list':raise
        mask=np.zeros((im.height,im.width),np.uint8)
        meta={'prompt':job['prompt'],'crop_box':job['crop'],'upscale':job['upscale'],'num_masks_returned':0,'score':0.,'area_px':0,'emptyMaskResponse':True}
    return mask,meta

def execute(job,key):
    out=V/'raw'/job['pose'];out.mkdir(parents=True,exist_ok=True)
    meta_path=out/(job['part']+'.json');mask_path=out/(job['part']+'.png')
    fingerprint=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()
    if meta_path.exists() and mask_path.exists():
        old=json.loads(meta_path.read_text());assert old['requestFingerprint']==fingerprint
        print('SAM CACHE',job['id'],flush=True);return old
    started=time.time();mask,meta=safe_call(job,key)
    Image.fromarray(mask).save(mask_path)
    result={**meta,'id':job['id'],'pose':job['pose'],'part':job['part'],'frameSHA256':job['frameSHA256'],
        'boxPrompts':job['boxes'],'requestFingerprint':fingerprint,'endpoint':'fal-ai/sam-3/image',
        'elapsedSeconds':round(time.time()-started,2),'logicalRequestCount':1,'maskSHA256':hashlib.sha256(mask_path.read_bytes()).hexdigest()}
    meta_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('SAM DONE',job['id'],'score',result['score'],'area',result['area_px'],'seconds',result['elapsedSeconds'],flush=True)
    return result

def main():
    sys.path.insert(0,str(Path(os.environ['AGENT_COMMON_FILES_DIR'])/'lib'))
    from secret_loader import get_secret
    from scratch_workspace import resolve_scratch_dir
    resolve_scratch_dir();key=get_secret('shared/fal','api_key');assert bool(key)
    jobs=specs();(V/'request-plan.json').write_text(json.dumps(jobs,ensure_ascii=False,indent=2)+'\n')
    # One actual image first, then bounded independent calls; no alternate prompts.
    done=[execute(jobs[0],key)]
    print('SAM ACTUAL ONE-IMAGE SMOKE PASS',jobs[0]['id'],flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(execute,j,key) for j in jobs[1:]]
        for future in concurrent.futures.as_completed(futures):
            done.append(future.result());(V/'call-manifest.json').write_text(json.dumps(sorted(done,key=lambda x:x['id']),ensure_ascii=False,indent=2)+'\n')
    print('SAM REMOTE COMPLETE',len(done),'logical image calls; key omitted; local fits not started',flush=True)
if __name__=='__main__':main()
