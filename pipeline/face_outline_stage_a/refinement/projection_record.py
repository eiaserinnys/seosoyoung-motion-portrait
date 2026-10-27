"""Persist only approved offline family45 moves, independently of runtime queries."""
import hashlib,json

def save(before,after,projection,path,source_paths):
 poses=[]
 for entry in projection:
  name=entry['name'];a=next(p for p in before['family45']['poses'] if p['name']==name);b=next(p for p in after['family45']['poses'] if p['name']==name)
  moves=[]
  for i in entry.get('movedIds',[]):
   moves.append(dict(index=i,rest=before['points'][i],duBefore=a['du'][i],duAfter=b['du'][i],delta=[b['du'][i][d]-a['du'][i][d] for d in [0,1]]))
  poses.append(dict(name=name,moves=moves,maxMove=entry.get('maxMove',0)))
 record=dict(definition='expected B observation = raw authored observation + stored offline free-X projection displacement',fixedPolicy='anchors plus samples left/right/center',sources={name:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for name,p in source_paths.items()},poses=poses)
 path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');return record
