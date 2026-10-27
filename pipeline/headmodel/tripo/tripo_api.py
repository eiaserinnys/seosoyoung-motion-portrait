"""Bounded official Tripo v3 calls; credentials stay in process memory."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import requests
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=HERE/'results/tripo'
ENDPOINT='https://openapi.tripo3d.ai/v3'
MODEL='v3.1-20260211'


def save(name,value):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


class TripoError(RuntimeError):
    def __init__(self,code,message,fatal_auth=False):
        super().__init__(f'Tripo error code {code}: {message}')
        self.code=code
        self.fatal_auth=fatal_auth


class Client:
    def __init__(self):
        sys.path.insert(0,str(Path(os.environ['AGENT_COMMON_FILES_DIR'])/'lib'))
        from secret_loader import get_secret
        self.key=get_secret('shared/tripo','api_key')
        self.session=requests.Session()
        self.session.headers['Authorization']='Bearer '+self.key
        history=OUT/'generation.json'
        self.calls=json.loads(history.read_text(encoding='utf-8')) if history.exists() else []

    def request(self,method,path,**kwargs):
        response=self.session.request(method,ENDPOINT+path,timeout=120,**kwargs)
        try: payload=response.json()
        except ValueError: raise TripoError(response.status_code,'Non-JSON API response') from None
        code=payload.get('code',response.status_code)
        if response.status_code in (401,403) or code:
            message=str(payload.get('message','API rejected request')).replace(self.key,'[redacted]')
            auth=(response.status_code in (401,403) or code in (1000,1001) or
                  'invalid api key' in message.lower() or 'unauthorized' in message.lower())
            raise TripoError(code,message,fatal_auth=auth)
        return payload['data']

    def generate(self,name,image,attempt):
        if len(self.calls)>=4: raise RuntimeError('Four-generation bound reached')
        tick=time.perf_counter()
        with open(image,'rb') as file:
            token=self.request('POST','/files',files={'file':(Path(image).name,file,'image/png')})['file_token']
        parameters={'input':token,'model':MODEL}
        record={'name':name,'attempt':attempt,'model':MODEL,'endpoint':ENDPOINT+'/generation/image-to-model',
                'parameters':dict(parameters,input='uploaded file_token (asset reference omitted)'),
                'input_path':str(Path(image).relative_to(ROOT)),'status':'submitting'}
        self.calls.append(record); save('generation.json',self.calls)
        try:
            task=self.request('POST','/generation/image-to-model',json=parameters)
            task_id=task['task_id']; record.update(task_id=task_id,status='queued')
            save('generation.json',self.calls)
            while True:
                result=self.request('GET','/tasks/'+task_id)
                status=result['status']
                print('TRIPO',name,task_id,status,result.get('progress'),flush=True)
                if status in ('success','failed','cancelled'): break
                if time.perf_counter()-tick>1500: raise RuntimeError('Generation poll time bound exceeded')
                time.sleep(10)
            record.update(status=status,elapsed_seconds=time.perf_counter()-tick,
                          credits_consumed=result.get('credits_consumed'),
                          created_at=result.get('created_at'),completed_at=result.get('completed_at'))
            if status!='success':
                raise TripoError(result.get('error_code',-1),str(result.get('error_message',status)).replace(self.key,'[redacted]'))
            url=result['output']['model_url']
            # Download without forwarding the API Authorization header to CDN.
            response=requests.get(url,timeout=180); response.raise_for_status()
            path=OUT/(name+'.glb'); path.write_bytes(response.content)
            record.update(glb_path=str(path.relative_to(ROOT)),glb_bytes=path.stat().st_size,
                          sha256=hashlib.sha256(response.content).hexdigest())
            save('generation.json',self.calls)
            return path
        except TripoError as error:
            record.update(status='failed',error_code=error.code,message=str(error),elapsed_seconds=time.perf_counter()-tick)
            save('generation.json',self.calls); raise


def inputs():
    """Deterministic head/neck crop; no generative image edits."""
    source=HERE/'inputs/still-hairoff.png'
    original=Image.open(source).convert('RGBA')
    keep=np.zeros((original.height,original.width),np.uint8)
    for name in ['head','face','ears','neck']:
        layer=np.array(Image.open(HERE/f'inputs/{name}.png').convert('RGBA'))
        keep=np.maximum(keep,layer[:,:,3])
    # The neck crop ends before the shoulders and the scattered hair remnants.
    box=(410,115,860,565)
    rgba=np.array(original); rgba[:,:,3]=np.minimum(rgba[:,:,3],keep)
    cut=Image.fromarray(rgba).crop(box)
    white=Image.new('RGBA',cut.size,'white'); white.alpha_composite(cut)
    path=HERE/'inputs/T1-input.png'; white.convert('RGB').save(path)
    t2=ROOT/'source/character/still.png'
    save('inputs.json',{'T1':{'source_commit':'ded2be6e17efd911243222fa8d1f492dfcad7ff3',
         'source':'experiments/see-through/still/still-hairoff.png','crop_box':box,'background':'white',
         'selection':'union of original head, face, ears and neck layer alpha; no painting'},
         'T2':{'source':str(t2.relative_to(ROOT)),'input_size':Image.open(t2).size,
               'preprocessing':'none; supplied original full front art'},
         'documentation':['https://developers.tripo3d.ai/en/docs/generation-image-to-model/standard',
                          'https://developers.tripo3d.ai/en/docs/models-and-versions']})
    return path,t2


def generate_all():
    t1,t2=inputs(); client=Client(); paths={}
    for name,path in [('T1',t1),('T2',t2)]:
        previous=[r for r in client.calls if r['name']==name and r['status']=='success']
        if previous:
            paths[name]=ROOT/previous[-1]['glb_path']
            continue
        try: paths[name]=client.generate(name,path,1)
        except TripoError as error:
            # Authorization/credit failures stop part 2 immediately.
            if error.fatal_auth or error.code in (2010,401,403,1000,1001): raise
            paths[name]=client.generate(name+'-retry',path,2)
    return paths
