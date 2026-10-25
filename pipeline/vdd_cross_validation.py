"""Pin and LOPO errors on fixed prior-visible normal observations, in pixels.

No held-out observations enter fitting or budget selection. This evaluates
screen-space correspondence prediction, separately from the raster gate.
"""
import json
import subprocess
import types
import numpy as np
from fit_hair_solve import ROOT,observations,fit_joint
from hair_solve_quality import compare_captures


def evaluate(base):
    old=types.ModuleType('baseline_fit');old.__file__=str(ROOT/'pipeline/fit_hair_solve.py')
    exec(subprocess.check_output(['git','show',base+':pipeline/fit_hair_solve.py']),old.__dict__)
    inputs=json.loads(subprocess.check_output(['node',str(ROOT/'pipeline/hair_solve_inputs.mjs')],timeout=60))
    features={p['name']:p for p in json.loads((ROOT/'source/viewer/face-feature-fit.json').read_text())['poses']}
    before=json.loads(subprocess.check_output(['git','show',base+':source/viewer/hair-solve.json']))
    after=json.loads((ROOT/'source/viewer/hair-solve.json').read_text())
    systems=[observations(p,features[p['name']],ROOT/'debug/hair_solve_prior') for p in inputs]
    domains={p['name']:old.observations(p,features[p['name']],ROOT/'debug/hair_solve_prior') for p in inputs}
    rows=[]
    def rms(system,du):
        error=np.sum((system['c']@np.array(du)-system['d'])*system['normal'],axis=1)
        return float(np.sqrt(np.average(error**2,weights=system['w'])))
    for held in systems:
        name=held['name'];group=[s for s in systems if s['pitch']==held['pitch']]
        settings=next(p for p in after['poses'] if p['name']==name)['regularization']
        fitted=fit_joint(group,settings['conformal'],settings['symmetry'],settings['view'],held_out=name)
        prediction=next(p for p in fitted if p['name']==name)
        domain=domains[name]
        rows.append(dict(heldOut=name,sampleCount=len(domain['d']),
                         before=dict(pinRms=rms(domain,next(p for p in before['poses'] if p['name']==name)['du']),
                                     lopoRms=rms(domain,np.zeros((192,2)))),
                         after=dict(pinRms=rms(domain,next(p for p in after['poses'] if p['name']==name)['du']),
                                    lopoRms=rms(domain,prediction['du'])),
                         predictionDuRms=prediction['stats']['duRms']))
    face_before=json.loads(subprocess.check_output(['git','show',base+':source/viewer/face-solve.json']))
    face=json.loads((ROOT/'source/viewer/face-solve.json').read_text())
    def face_summary(data):
        return dict(pin=[dict(name=p['name'],observationRms=p['stats']['rms']) for p in data['poses'] if p['yaw'] or p['pitch']],
                    lopo=data['crossValidation'])
    means={variant:{key:float(np.mean([p[variant][key] for p in rows])) for key in ['pinRms','lopoRms']} for variant in ['before','after']}
    poses=list(features.values())
    raster=dict(main=compare_captures(poses,ROOT/'debug/hair_solve_prior',ROOT/'debug/vdd_main'),
                after=compare_captures(poses,ROOT/'debug/hair_solve_prior',ROOT/'debug/hair_h0/current'))
    return dict(rasterCaptures=raster,hair=dict(definition='weighted normal-distance RMS on identical baseline prior-visible observations; not raster RMS; independent baseline LOPO predicts zero residual',
                          folds=rows,mean=means,heldOutPolicy='remove all data terms for the held-out chart; fit other charts and predict via mirror/adjacent-yaw priors; budget selection uses predicted displacements only'),
                face=dict(definition='stored face solve: pin observation RMS and eight leave-one-pose-out silhouette/feature RMS; face files unchanged',
                          before=face_summary(face_before),after=face_summary(face)))


if __name__=='__main__':
    import sys
    print(json.dumps(evaluate(sys.argv[1]),ensure_ascii=False))
