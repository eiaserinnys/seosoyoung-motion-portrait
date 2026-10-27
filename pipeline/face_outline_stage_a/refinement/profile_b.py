"""7.18 B: uniform mesh rows, family45 solve and runtime-equivalent free-X projection."""
from pathlib import Path
import copy,json,shutil,subprocess,sys
import numpy as np
from close_anchors import read_face,write_face,close
from triangle_projection import project,ProjectionConflict
from profile_a import restore_b
from projection_record import save

def main(root,out):
 out.mkdir(parents=True,exist_ok=True);stage=root/'pipeline/face_outline_stage_a/tps-study/stage-a/v711-filled'
 restore_b(root)
 sys.path.insert(0,str(stage));import stage_a
 raw=subprocess.check_output(['node',str(root/'pipeline/face_solve_inputs.mjs'),'--base'],cwd=root,timeout=600)
 inputs=json.loads(raw);(out/'profile-inputs.json').write_bytes(raw)
 face=read_face(root);face['family45']=stage_a.solve_family45(inputs,face['Z']);write_face(root,face)
 report={'rowAudit':inputs['family45']['rowAudit'],'anchorBefore':close(root,inputs,out/'anchors-before.json'),'projection':[]}
 # A diagnostic module in a data-only copy exposes the prior without invoking
 # the runtime correction that this offline operation is meant to bake in.
 face=read_face(root);before_projection=copy.deepcopy(face)
 diag=out/'raw-prior';shutil.copytree(root/'bundle',diag/'bundle')
 f=diag/'bundle/face-solve.mjs';(diag/'bundle/face-solve-diag.mjs').write_text(f.read_text()+'\nexport {face45Prior as diagnosticPrior};\n')
 values=json.loads(subprocess.check_output(['node',str(Path(__file__).with_name('family_nodes.mjs')),str(diag)],timeout=120))
 fixed=set(face['anchors'].values())|{i for s in face['samples'] for i in [s['left'],s['right'],s['center']]}
 for pose in face['family45']['poses']:
  world=np.array(next(p for p in values if p['name']==pose['name'])['world'])
  try:
   target,stats=project(face['points'],world,face['triangles'],fixed)
   for i,d in enumerate(target[:,0]-world[:,0]):pose['du'][i][0]+=float(d)
   report['projection'].append({'name':pose['name'],**stats})
  except ProjectionConflict as e:report['projection'].append({'name':pose['name'],'conflict':str(e)})
 write_face(root,face);report['anchorAfter']=close(root,inputs,out/'anchors-after.json')
 save(before_projection,read_face(root),report['projection'],root/'pipeline/face_outline_stage_a/accepted/family45-projection.json',{'before':diag/'bundle/face-solve-data.mjs','after':root/'bundle/face-solve-data.mjs'})
 (out/'profile-b.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print('B family regenerated',json.dumps(report['projection']),flush=True)
if __name__=='__main__':main(*(Path(a).resolve() for a in sys.argv[1:3]))
