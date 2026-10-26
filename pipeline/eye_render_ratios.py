"""Fable's front-normalized log ratios, converted back to reference pixels."""
import numpy as np


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
