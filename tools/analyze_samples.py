"""Compare private Core samples and extracted raw tables. No model data is bundled.
Usage: python analyze_samples.py TABLES.json.gz SAMPLES.json.gz REGIONS.json OUT.json.gz
REGIONS: {face: mesh ID, regions: {name: [mesh IDs]}, channels: [X, Y, Z]}
"""
import gzip,json,sys
from pathlib import Path
import numpy as np
from deformation_metrics import metrics,ordinary_points,rms,grid_curvature_change,sphere_projection


def analyze(t,s,config):
    ids=s['meshIds']; frames={k:[np.asarray(v).reshape(-1,2)*[1,-1] for v in p['vertices']] for k,p in s['poses'].items()}
    rest=frames['neutral'];face=rest[ids.index(config['face'])];low=face.min(axis=0);high=face.max(axis=0)
    width=float(high[0]-low[0]);center=(low+high)/2;radius=float(np.max(high-low)*.6)
    X,Y,Z=config['channels'];report={'width':width,'head_assumption':{'center':center.tolist(),'radius':radius,'rule':'0.6 times max neutral face bounding span'},'mesh_inventory':[], 'regions':{},'warps':[], 'roll_blend_targets':[]}
    for i,id in enumerate(ids):
        categories=[k for k,v in config['regions'].items() if id in v]
        report['mesh_inventory'].append({'id':id,'regions':categories,'neutral_opacity':s['poses']['neutral']['opacities'][i]})
    for region,mids in config['regions'].items():
        depth_sign=-1 if region=='hair_back' else 1
        ii=[ids.index(id) for id in mids];a=np.concatenate([rest[i] for i in ii]);results={}
        for pose in frames:
            b=np.concatenate([frames[pose][i] for i in ii]);r=metrics(a/width,b/width)
            if pose.startswith('xy_'):
                x,y=map(float,pose.split('_')[1:]);base=sphere_projection(a,center,radius,x,y,depth_sign)
                r['sphere_residual_rms']=rms(b-base)/width
                r['sphere_radius_sensitivity']=[rms(b-sphere_projection(a,center,float(np.max(high-low)*f),x,y,depth_sign))/width for f in [.5,.6,.8]]
                if x and y:
                    key=lambda v: str(int(v))
                    bx=np.concatenate([frames[f'xy_{key(x)}_0'][i] for i in ii]);by=np.concatenate([frames[f'xy_0_{key(y)}'][i] for i in ii])
                    joint=b-bx-by+a;r['joint_rms']=rms(joint)/width
                    bj=base-sphere_projection(a,center,radius,x,0,depth_sign)-sphere_projection(a,center,radius,0,y,depth_sign)+a
                    r['joint_residual_vs_sphere_rms']=rms(joint-bj)/width
            results[pose]=r
        report['regions'][region]={'depth_sign':depth_sign,'mesh_count':len(ii),'vertex_count':len(a),'poses':results}
    # Ordinary local control grids: exactly distinguish local deformation from inherited world motion.
    local_frames={};defaults={p['id']:p['default'] for p in t['parameters']}
    for d in t['deformers']:
        if d['kind']!=0:continue
        obj=t['warps'][d['specific_index']];band=obj['binding_band_index']
        axes=[t['normal_parameter_bindings'][i]['parameter_id'] for i in t['normal_binding_bands'][band]['binding_indices']] if band>=0 else []
        if not set(axes)&set(config['channels']+config.get('extra_channels',[])):continue
        a=ordinary_points(t,'warp',obj,{})
        span=float(np.max(np.ptp(a,axis=0))); span=span if span>0 else 1
        rows={};points={}
        for pose,p in s['poses'].items():
            b=ordinary_points(t,'warp',obj,p['parameters']);r=metrics(a/span,b/span)
            r['second_difference_change']=grid_curvature_change(a,b,obj['rows'],obj['columns'])/span
            rows[pose]=r;points[pose]=b.tolist()
        report['warps'].append({'id':d['id'],'parent_index':d['parent_deformer_index'],'rows':obj['rows'],'columns':obj['columns'],'axes':axes,'local_neutral_span':span,'poses':rows})
        local_frames[d['id']]=points
    # Store roll blend delta rows verbatim only in the private output; no uncertain constraint evaluator.
    for b in t['blend_keyform_bindings']:
        if b['parameter_id']!=Z:continue
        binding=t['blend_parameter_bindings'][b['parameter_binding_index']];kind=b['target_kind'];rows=[]
        for offset,key in enumerate(binding['keys']):
            k=t[kind+'_keyforms'][b['keyform_start']+offset]; row={'key':key,'fields':k}
            if kind in ('warp','mesh'):
                obj=t['warps' if kind=='warp' else 'meshes'][b['target_index']];n=obj['vertex_count'];start=k['position_start'];p=np.array(t['position_floats'][start:start+2*n]).reshape(-1,2)
                row['delta_rms_local']=rms(p);row['positions']=p.tolist()
            rows.append(row)
        report['roll_blend_targets'].append({'id':b['target_id'],'kind':kind,'constraints':b['constraint_indices'],'rows':rows})
    return report,local_frames

if __name__=='__main__':
    tf,sf,cf,output=sys.argv[1:]; t=json.load(gzip.open(tf,'rt'));s=json.load(gzip.open(sf,'rt'));config=json.loads(Path(cf).read_text())
    report,local=analyze(t,s,config)
    report['local_control_points']=local
    with gzip.open(output,'wt') as f:json.dump(report,f,ensure_ascii=False)
    print(json.dumps({'meshes':len(report['mesh_inventory']),'regions':len(report['regions']),'head_warps':len(report['warps'])}))
