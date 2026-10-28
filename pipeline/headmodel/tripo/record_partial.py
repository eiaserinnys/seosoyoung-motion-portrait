"""Freeze measured partial results while awaiting the execution-budget decision.

Only JSON/header/hash reads and metadata writes; no API, rasterization or fit.
"""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import struct

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
FIT=HERE.parent/'fit'
OUT=HERE/'results/tripo'
FOUT=FIT/'results/fit2f'


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def glb_header(path):
    with path.open('rb') as stream:
        magic,version,total=struct.unpack('<III',stream.read(12))
        if magic!=0x46546C67 or version!=2: raise ValueError('Expected GLB2')
        length,kind=struct.unpack('<II',stream.read(8))
        if kind!=0x4E4F534A: raise ValueError('Expected GLB JSON chunk')
        data=json.loads(stream.read(length))
    accessors=data['accessors']; positions=set(); count=0
    for mesh in data['meshes']:
        for primitive in mesh['primitives']:
            positions.add(primitive['attributes']['POSITION'])
            if primitive.get('mode',4)!=4: raise ValueError('Expected triangle primitives')
            count+=accessors[primitive['indices']]['count']//3 if 'indices' in primitive else accessors[primitive['attributes']['POSITION']]['count']//3
    return {'declared_position_vertices':sum(accessors[i]['count'] for i in positions),
            'declared_triangles':count,'bytes':total,'note':'GLB accessor declarations, before position welding'}


def main():
    now=datetime.now(timezone.utc).isoformat()
    generation=read(OUT/'generation.json')
    successful=[r for r in generation if r['status']=='success']
    for item in successful:
        path=ROOT/item['glb_path']
        with path.open('rb') as file: actual=hashlib.file_digest(file,'sha256').hexdigest()
        if actual!=item['sha256']: raise ValueError('Generated GLB changed')
    t1=read(OUT/'T1-alignment.json')
    t2={'status':'hole_fill_failed','raw_glb':glb_header(OUT/'T2.glb'),
        'failure':'MeshLab meshing_close_holes requires edge manifoldness; largest position-welded component has non-manifold edges.',
        'prepared_mesh':None,'alignment':None,'repair_applied':False}
    save(OUT/'T2-mesh-failure.json',t2)
    archive=FOUT/'pre-edge-clamp/compare.json'
    if not archive.exists(): shutil.copyfile(OUT/'compare.json',archive)
    compare=read(OUT/'compare.json')
    compare['status']='partial; awaiting additional runner approval and T2 topology decision'
    compare['models']['B3-step1']['curvature']=read(FOUT/'b3-step1-residuals.json')['curvature']
    compare['models']['B3-w3'].pop('curvature',None)
    compare['models']['B3-w3']['curvature_status']='Pending corrected cotangent remeasurement; geometric projection residuals remain valid.'
    compare['models']['B3-step2']={'available':False,'reason':'Corrective solve with sum-edge-then-clamp cotangent weights pending.',
                                  'superseded_results':'pipeline/headmodel/fit/results/fit2f/pre-edge-clamp/'}
    compare['models']['T1-A']={'aligned':True,'measured':False,'parameters':t1['pass2'],
                              'reason':'Nine-key residuals and final sheets await approved runner.'}
    compare['models']['T1-B3']={'available':False,'reason':'Not executed; additional runner approval pending.'}
    compare['models']['T2-A']={'available':False,'reason':t2['failure']}
    status={'status':'partial','checked_at':now,'endpoint':'https://openapi.tripo3d.ai/v3','authenticated':True,
            'created_models':len(successful),'credits_consumed':sum(r['credits_consumed'] for r in successful),
            'T1':'global alignment complete','T2':t2['failure'],'additional_runners_pending':4}
    compare.update(tripo_status=status,generation=generation)
    save(OUT/'status.json',status); save(OUT/'compare.json',compare)
    save(FOUT/'residuals.json',{'status':'partial','models':{k:v for k,v in compare['models'].items() if k.startswith(('A','B3'))}})
    step2=read(FOUT/'b3-step2-residuals.json')
    step2.update(valid_for_spec=False,superseded_by='Correct edge-weight clamp, corrective solve pending; original copy preserved in pre-edge-clamp.')
    save(FOUT/'b3-step2-residuals.json',step2)
    endpoint=read(OUT/'endpoint-check.json')
    endpoint.update(checked_at=datetime.fromtimestamp((OUT/'endpoint-check.json').stat().st_mtime,timezone.utc).isoformat(),
                    historical_phase='before user purchased credits; not the current balance')
    save(OUT/'endpoint-check.json',endpoint)
    sheets=read(FOUT/'sheet-sha256.json')
    for item in sheets['sheets']:
        path=ROOT/item['path']
        item['validity']='superseded cotangent implementation' if ('step2' in path.name or 'shape-step1' in path.name) else 'valid: unchanged uniform-L geometry and mask contours'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']: raise ValueError('Original sheet changed')
    existing={item['path'] for item in sheets['sheets']}
    for path in sorted((FIT/'sheets').glob('fit2f-*-pre-edge-clamp.png')):
        relative=str(path.relative_to(ROOT))
        if relative not in existing:
            sheets['sheets'].append({'path':relative,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                     'validity':'superseded operator archive'})
    save(FOUT/'sheet-sha256.json',sheets)
    tsheets=[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
              'role':'initial generated mesh axis review, before global fitting' if p.name=='T1-axes.png' else 'superseded UV seam diagnostic'}
             for p in sorted((HERE/'sheets').glob('*.png'))]
    save(OUT/'sheet-sha256.json',{'status':'partial; final three Tripo sheets not generated','sheets':tsheets})
    validation={
        'status':'incomplete; awaiting additional execution approval',
        'runner_calls':9,'original_runner_limit':6,'overrun_acknowledged':True,
        'runner_history':['prepare','step1','step2 with incorrect cot clamp','Tripo import missing requests',
            'Tripo wrong .com endpoint upload rejected','ICT sheets','funded Tripo generation; UV-island preparation stopped',
            'position-weld preparation; dense raster processing stopped','prefiltered Tripo alignment: T1 complete, T2 hole-fill failed'],
        'additional_runners_requested':4,
        'pending_runs':['ICT step2 with corrected cot weights','T2 topology repair and global alignment','T1-B3','final corrected ICT and Tripo comparison/sheets'],
        'pending_design_decision':'T2 coordinate-preserving non-manifold splitting before hole fill is proposed, not implemented.',
        'contracts_result':'5 passed, 1 failed (remesh maximum distance 3.0878357047105056px >1.5px)',
        'latest_targeted_checks':{'open_boundary_and_raster_equivalence':'1 passed .226s',
            'edge_cot_clamp_gaussian_constant':'RED old edge1.2 instead of.325; GREEN1 passed .009s',
            'API_auth_resume_fixture':'1 passed .001s','reused_2d_contracts':'3 passed with boundary test,4 total .357s',
            'legacy_zbuffer_mask':'2 passed .027s'},
        'test2_collision_median_before_after_px':read(FOUT/'remesh.json')['target_collision'],
        'step1_curvature':read(FOUT/'b3-step1-residuals.json')['curvature'],
        'step2_status':'Original result superseded: triangle-wise cot clamp differed from the specified edge-wise clamp. Correction implemented and unit-tested; solve not rerun.',
        'T1_visibility_equivalence':read(OUT/'T1-visibility-equivalence.json'),
        'T1_A_final_parameters':{k:t1['pass2'][k] for k in ['rotation_offset_deg','scale_xyz','translation_ict','thickness_px','elapsed_seconds']},
        'generated_models':len(successful),'credits_consumed':sum(r['credits_consumed'] for r in successful),
        'generation_create_attempts':len(generation),'prior_2010_attempts':len([r for r in generation if r.get('error_code')==2010]),
        'api_key_changes':0,'T2':t2,
        'independent_review':'Current code/edge-clamp correction passed static review. Initial ICT images reviewed; corrected step2 and Tripo final visual review pending.',
        'visual_findings':'Initial step2 reduced fine faceting, but was produced by the superseded operator and is not the specified experiment. Generated T1 is a complete head after position welding; fitted T1 sheets not generated.',
        'not_executed':['corrected step2 solve','T1-B3','T2-A','nine-key Tripo residuals','final three Tripo sheets','B130','2e','A refit','lambda/mu/weight/taper/gate search','PR','merge','deployment'],
        'unrelated_warning':'pip reported existing pdfminer-six requires missing cryptography through host PYTHONPATH; no PDF functionality or unrelated package was modified.',
        'protected_paths':'No existing 2d source, headmodel data/model, bundle or runtime modified.'}
    save(FOUT/'validation.json',validation)
    print(json.dumps({'status':status,'T1_A':validation['T1_A_final_parameters'],'T2':t2,
                      'step1_curvature':validation['step1_curvature'],'generated_glb_hashes_verified':len(successful)},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
