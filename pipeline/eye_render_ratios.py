"""Fable's front-normalized log ratios, converted back to reference pixels."""
import numpy as np


def model_fidelity(records,measurement,fit):
    """Fable per_eye2's front-normalized render/model split, in still pixels."""
    reference={(r['name'],r['side']):r for r in measurement['records']}
    front={r['side']:r['v291'] for r in records if r['name']=='front'}
    unclipped={'aux-0017','aux-0342','aux-0048','aux-0213','aux-0255','down','down-right','aux-0074'}
    iris_poses={'up-left':'left','down-left':'left','right':'right','up-right':'right'}
    temporal=[];iris=[]
    for row in records:
        name,side=row['name'],row['side'];ref=reference[name,side];b=fit[side]
        theta=np.clip(ref['yaw']*(-1 if side=='left' else 1),-45,45)
        rT=float(np.exp(np.interp(theta,b['knotsDeg'],b['r_T'])))
        rN=float(np.exp(np.interp(theta,b['knotsDeg'],b['r_N'])))
        render=row['v291']
        if name in unclipped and theta>0:
            error=float(np.log((render['dTemp']/front[side]['dTemp'])/rT)*measurement['rest'][side]['dTemp'])
            temporal.append(dict(name=name,side=side,theta=float(theta),renderDTemp=render['dTemp'],modelDTemp=b['rest']['dTemp']*rT,
                rawErrorPx=render['dTemp']-b['rest']['dTemp']*rT,normalizedErrorPx=error,passed=abs(error)<=2,
                scope='eye clip' if abs(error)<=2 else 'remaining face silhouette limitation; outside this eye PR per Fable 1.3'))
        if iris_poses.get(name)==side:
            iris.append(dict(name=name,side=side,theta=float(theta),modelIrisDiameter=2*b['rest']['a']*float(np.cos(np.deg2rad(max(theta,0)))),
                modelOpeningWidth=b['rest']['dTemp']*rT+b['rest']['dNas']*rN,renderIrisChord=2*render['a'],renderOpeningWidth=render['dTemp']+render['dNas']))
    return dict(FBTemporal=temporal,FAIris=iris,modelComparison='per_eye2: log((render/render-front)/model-ratio) × measured still dTemp; raw width difference also retained')


def ratio_error(render,render_front,reference,rest,field,w0):
    if field=='deltaY':
        d=(np.array(render['cI'])-render['cO'])@np.array(render['E'])
        d0=(np.array(render_front['cI'])-render_front['cO'])@np.array(render_front['E'])
        return float(d[1]-d0[1]-(reference['delta'][1]-rest['delta'][1])*w0)
    return float((np.log(render[field]/render_front[field])-np.log(reference[field]/rest[field]))*reference[field])


def ratio_gates(records,measurement,audit):
    ref={(r['name'],r['side']):r['q'] for r in measurement['records']}
    front={r['side']:r for r in records if r['name']=='front'}
    fields=['dTemp','dNas','h','deltaY','a'];rms={};errors={}
    for field in fields:
        per={}
        for rev in ['r24','v291']:
            rows=[dict(name=r['name'],side=r['side'],errorPx=ratio_error(r[rev],front[r['side']][rev],ref[(r['name'],r['side'])],measurement['rest'][r['side']],field,measurement['K'][r['side']]['w'])) for r in records]
            per[rev]=rows
        rms[field]={rev:float(np.sqrt(np.mean([r['errorPx']**2 for r in per[rev]]))) for rev in per}
        rms[field].update(limit=2*audit['noise'][field]['sigma'])
        rms[field]['passed']=rms[field]['v291']<=rms[field]['limit']
        rms[field]['betterThanR24']=rms[field]['v291']<rms[field]['r24'];errors[field]=per
    excluded={o['eye'] for s in ['left','right'] for o in audit['fit'][s]['dNas']['outliers']}
    R3={};limit=3*audit['noise']['dNas']['sigma']
    for rev,rows in errors['dNas'].items():
        included=[r for r in rows if r['name']+':'+r['side'] not in excluded]
        values={(r['name'],r['side']):r['errorPx'] for r in included};paired=[]
        for name in sorted({r['name'] for r in included}):
            if (name,'left') in values and (name,'right') in values:
                paired.append(dict(name=name,errorDifferencePx=abs(values[name,'left']-values[name,'right'])))
        R3[rev]=dict(maxEyeErrorPx=max(abs(r['errorPx']) for r in included),maxPairedErrorPx=max(r['errorDifferencePx'] for r in paired),errors=included,paired=paired,
                     allEyesMaxErrorPx=max(abs(r['errorPx']) for r in rows))
        R3[rev]['passed']=max(R3[rev]['maxEyeErrorPx'],R3[rev]['maxPairedErrorPx'])<=limit
    offsets={s:{rev:{f:front[s][rev][f]-measurement['rest'][s][f] for f in ['dTemp','dNas','h','a']} for rev in ['v291','r24']} for s in ['left','right']}
    return dict(R2Ratio=rms,R3Ratio=dict(limitPx=limit,excluded=sorted(excluded),**R3),frontOffsets=offsets,
                ratioErrors=errors,normalization='Fable ratio_r2.py: rendered front reference pose / measured still rest; 48 eyes including front; deltaY uses projected center change instead of log')
