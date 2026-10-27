import json,sys
import numpy as np
from config import V,T
import stage_a
inputs=json.loads((V/'a-inputs.json').read_text());old=json.loads((V/'runtime-current.json').read_text())['prior']
solved=stage_a.solve(inputs)
if 'ponytail' in old:solved['ponytail']=old['ponytail']
solved['family45']=old.get('family45')
(V/'face-observed.json').write_text(json.dumps(solved,separators=(',',':'))+'\n')
(T/'bundle/face-solve-data.mjs').write_text('export const FACE_SOLVE='+json.dumps(solved,separators=(',',':'))+';\n')
print('OBSERVED FACE SOLVE',json.dumps([{'name':p['name'],'t':p['t'],**p['stats']} for p in solved['poses']]),flush=True)
