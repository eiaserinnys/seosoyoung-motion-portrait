import json
from config import V,A
old=json.loads((A/'v711-filled/ear-fit.json').read_text());new=json.loads((V/'refit/ear-fit.json').read_text());merged=[next((q for q in new if q['pose']==e['pose'] and q['side']==e['side']),e) for e in old];changed=sorted(set(q['pose'] for q in new if q['parameters']!=next(e for e in old if e['pose']==q['pose'] and e['side']==q['side'])['parameters']));(V/'ear-fit.json').write_text(json.dumps(merged,indent=2)+'\n');(V/'changed-poses.json').write_text(json.dumps(changed)+'\n')
