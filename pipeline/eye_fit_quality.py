"""Independent v2.9 evaluator; v28 input/report contract, replaced fitting rules.
No designer prototype imported; no threshold or constant search.
"""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
from functools import partial
from fit_eye_sam3 import temporal_huber_weights
json.dumps=partial(json.dumps,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())

P=Path(__file__).resolve().parents[1];OUT=P/'debug/sam3-eye-v295'
OUT.mkdir(parents=True,exist_ok=True)
D=json.loads((P/'source/viewer/sam3-eye-measurements.json').read_text());rest=D['rest'];K=D['K']
rows=[r for r in D['records'] if 'q' in r];N=len(rows)
censored_temporal=np.array([r['q']['temporalAtFaceEdge'] for r in rows])
groups=json.loads((P/'source/references/seedance/eye-bootstrap.json').read_text())['groups']
fields=['dTemp','dNas','h','deltaY','a','bI'];core=fields[:4]
theta=np.array([-r['yaw']*(1 if r['side']=='left' else -1) for r in rows])
t=np.deg2rad(theta);p=np.array([r['pitch']/30 for r in rows]);knots=np.arange(-45,46,15)
ids=[r['name']+':'+r['side'] for r in rows]
side_masks={s:np.array([r['side']==s for r in rows]) for s in ['left','right']}
Y={};W={};BASE={};VALUES={};noise={}
for f in fields:
    vals=[];ys=[];ws=[];bs=[]
    for r in rows:
        q=r['q'];s=r['side'];b=rest[s]
        value=q['delta'][1]*K[s]['w'] if f=='deltaY' else q[f]
        base=K[s]['w'] if f=='deltaY' else b[f]
        weight=q['weightDistance'] if f in ['dTemp','dNas'] else q['weightX'] if f=='a' else q['weightY'] if f in ['deltaY','bI'] else 1.
        if q['measurementFailed'] or (f=='bI' and not q['bObserved']):weight=0.
        vals.append(np.nan if value is None else value);bs.append(base);ws.append(weight)
        ys.append(q['delta'][1]-b['delta'][1] if f=='deltaY' else np.log(value/base) if value is not None else 0.)
    VALUES[f]=np.array(vals);Y[f]=np.array(ys);W[f]=np.array(ws);BASE[f]=np.array(bs)
    ss=0.;df=0;clusters=[]
    for group in groups:
        for side,sm in side_masks.items():
            valid=sm&np.array([r['name'] in group and not r['q']['measurementFailed'] for r in rows])&np.isfinite(VALUES[f])
            if f=='dTemp':valid&=~censored_temporal
            if f=='a':valid&=W[f]>0
            if f=='bI':valid&=np.array([r['q']['bObserved'] for r in rows])
            x=VALUES[f][valid]
            if len(x)>1:
                within=float(np.sum((x-x.mean())**2));ss+=within;df+=len(x)-1
                clusters.append(dict(group=group,side=side,n=len(x),ss=within))
    sd=float(np.sqrt(ss/df)) if df else None
    noise[f]=dict(pooledSD=sd,df=df,ss=ss,sigma=max(sd,1.) if sd is not None else None,floorApplied=sd is not None and sd<1,clusters=clusters)

def table_design(degrees):
    # Six nonnegative increments and a signed free video intercept.
    far=np.column_stack([-np.clip(degrees/15-j,0,1) for j in range(3)])
    near=np.column_stack([np.clip(-degrees/15-j,0,1) for j in range(3)])
    return np.column_stack([far,near,np.ones(len(degrees)),-np.ones(len(degrees))])

def design(f):
    if f in ['dTemp','dNas']:return table_design(theta),np.zeros(N)
    if f in ['h','deltaY']:return np.column_stack([p,p*p,t*t,t*p,np.ones(N)]),np.zeros(N)
    if f=='a':return np.column_stack([np.log(np.cos(t)),np.ones(N)]),np.zeros(N)
    return np.ones((N,1)),np.log(np.cos(p*np.pi/6))

def solve(f,mask):
    X,offset=design(f);is_table=f in ['dTemp','dNas'];c=np.zeros(X.shape[1]);weights=W[f]*mask
    sigma=noise[f]['sigma'];k=1.5*sigma
    for step in range(200):
        pred=offset+X@c;beta=c[-2]-c[-1] if is_table else c[-1];run=pred-beta
        scale=BASE[f] if f=='deltaY' else BASE[f]*np.exp(np.clip(run,-20,20))
        err=(pred-Y[f])*scale
        huber=temporal_huber_weights(err,censored_temporal,k) if f=='dTemp' else np.minimum(1.,k/np.maximum(abs(err),1e-12))
        ww=weights*huber*scale**2
        good=ww>0;A=X[good]*np.sqrt(ww[good,None]);b=(Y[f]-offset)[good]*np.sqrt(ww[good])
        if is_table:new=nnls(A,b,maxiter=1000)[0]
        else:
            # Retain v28's .001 quadratic ridge on shape terms, not video bias.
            ridge=np.eye(X.shape[1])*.001;ridge[-1,-1]=1e-12
            new=np.linalg.solve(A.T@A+ridge,A.T@b)
        change=float(np.max(abs(new-c)));c=new
        if change<1e-9:break
    pred=offset+X@c;beta=c[-2]-c[-1] if is_table else c[-1];run=pred-beta
    scale=BASE[f] if f=='deltaY' else BASE[f]*np.exp(np.clip(run,-20,20))
    err=(pred-Y[f])*scale
    return dict(coeff=c,beta=beta,errors=err,pred=pred,run=run,iterations=step+1,converged=change<1e-9,coefficientChange=change)

models={};folds=[];fits={};checks=[];warnings=[]
def gate(name,value,limit,kind='max',sigma=None,warnable=False):
    passed=value<=limit if kind=='max' else value>=limit
    excess=(value-limit) if kind=='max' else (limit-value)
    verdict='PASS' if passed else 'WARN' if warnable and sigma is not None and excess<sigma else 'FAIL'
    item=dict(name=name,value=value,limit=limit,status=verdict)
    checks.append(item)
    if verdict=='WARN':warnings.append(item)

for side,sm in side_masks.items():
    fits[side]={};models[side]={}
    for f in fields:
        m=solve(f,sm);models[side][f]=m;fit_ok=sm&(W[f]>0);ok=fit_ok.copy()
        if f=='dTemp':ok&=~censored_temporal
        drop=ok&(abs(m['errors'])>4*noise[f]['sigma']);keep=ok&~drop
        cv=np.full(N,np.nan);fold_results=[]
        for group in groups:
            held=sm&np.array([r['name'] in group for r in rows]);train=sm&~held
            fm=solve(f,train);cv[held]=fm['errors'][held]
            fold_results.append(dict(group=group,iterations=fm['iterations'],converged=fm['converged'],coefficientChange=fm['coefficientChange']))
        weight=W[f][keep];ee=m['errors'][keep];cc=cv[keep]
        rms=float(np.sqrt(np.average(ee**2,weights=weight)));median=float(np.median(abs(ee)))
        loco=float(np.sqrt(np.average(cc**2,weights=weight)))
        X,_=design(f);rank=int(np.linalg.matrix_rank(X[fit_ok]));wsum=float(W[f][fit_ok].sum())
        beta_px=float(m['beta']*BASE[f][sm][0]);sig=noise[f]['sigma']
        item=dict(coefficients=m['coeff'].tolist(),beta=float(m['beta']),betaPx=beta_px,
            iterations=m['iterations'],converged=m['converged'],coefficientChange=m['coefficientChange'],folds=fold_results,
            eligible=int(fit_ok.sum()),weightSum=wsum,rank=rank,rms=rms,median=median,loco=loco,
            outliers=[dict(eye=ids[i],residualPx=float(m['errors'][i])) for i in np.flatnonzero(drop)],
            outlierFraction=float(drop.sum()/ok.sum()),reportOnly=f not in core,
            residuals=[dict(eye=ids[i],residualPx=float(m['errors'][i]),locoPx=float(cv[i]),weight=float(W[f][i]),outlier=bool(drop[i])) for i in np.flatnonzero(sm&(~censored_temporal if f=='dTemp' else True))])
        name=side+':'+f
        if f in ['dTemp','dNas']:
            logs=table_design(knots)@m['coeff']-m['beta'];ratios=np.exp(logs)
            # Half-open intervals; last interval includes +45. No sample is exactly a knot.
            # Coverage includes lower-bound fitting observations; residual
            # statistics and outlier rejection above use uncensored eyes only.
            counts=[int(np.sum((fit_ok&~drop)&(theta>=lo)&(theta<(hi if hi<45 else hi+1e-9)))) for lo,hi in zip(knots[:-1],knots[1:])]
            item.update(knotsDeg=knots.tolist(),knotLogs=logs.tolist(),knotRatios=ratios.tolist(),intervalCounts=counts)
            gate(name+':identification',min(counts),2,kind='min')
            gate(name+':monotone',float(np.max(np.diff(logs))),1e-12)
            gate(name+':ratioMin',float(ratios.min()),.25,kind='min');gate(name+':ratioMax',float(ratios.max()),1.5)
        else:
            gate(name+':weightSum',wsum,3*X.shape[1],kind='min');gate(name+':rank',rank,X.shape[1],kind='min')
        gate(name+':convergence',int(m['converged'] and all(z['converged'] for z in fold_results)),1,kind='min')
        if f in core:
            for label,value,limit in [('RMS',rms,1.5*sig),('median',median,sig),('LOCO',loco,2*sig),('G2',abs(beta_px),3.)]:
                gate(name+':'+label,value,limit,sigma=sig,warnable=True)
            gate(name+':outlierFraction',item['outlierFraction'],.1)
        fits[side][f]=item

area=[]
for r in rows:
    threshold=.5*K[r['side']]['irisArea']*np.cos(np.deg2rad(abs(r['yaw'])))
    area.append(dict(eye=r['name']+':'+r['side'],area=r['clean']['irisArea'],threshold=float(threshold),passGate=r['clean']['irisArea']>=threshold))
pixels=all((not x['skin'] and x['opening']) if (x['x'],x['y'])!=(251,371) else (not x['opening'] and not x['iris']) for x in D['pixelChecks'])
hashchanges=[path for path,h in D['hashes'].items() if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=h]
protected={}  # Audit-only external files are not repository inputs.
protected_changes=[path for path,h in protected.items() if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=h]
censored=[];suspects=[]
for i,r in enumerate(rows):
    q=r['q'];rec=dict(eye=ids[i],theta=float(theta[i]),dTempEnv=q['dTempEnv'],dTempSil=q['dTempSil'],difference=q['dTempSil']-q['dTempEnv'],dTemp=q['dTemp'])
    if q['temporalAtFaceEdge']:censored.append(rec)
    if q['silhouetteSuspect']:suspects.append(rec)
gate('measurementCount',sum(not r['q']['measurementFailed'] for r in rows),48,kind='min')
gate('coverageRefined',sum(r['clean']['lashCoverageRefined']>=.9 for r in rows),48,kind='min')
gate('irisArea',sum(x['passGate'] for x in area),48,kind='min')
gate('pixelPreservation',int(pixels and len(D['pixelChecks'])==7),1,kind='min')
gate('inputHashes',len(hashchanges)+len(protected_changes),0)
gate('bObserved',D['diagnostics']['bObservedCount'],24,kind='min')
gate('WARNcount',len(warnings),6)
failures=[x for x in checks if x['status']=='FAIL']
status='CONSENSUS HOLD' if failures else 'CONSENSUS PASS-WARN' if warnings else 'CONSENSUS PASS'
asymmetry={};touch={}
for side,sm in side_masks.items():
    near=sm&(theta<0);ww=np.array([r['q']['w'] for r in rows])[near]
    depths=np.array([r['q']['nasalCutDepth'] for r in rows])[sm]
    asymmetry[side]=dict(stillGapTemporal=rest[side]['gapTemporal'],stillGapNasal=rest[side]['gapNasal'],nearCount=int(near.sum()),nearWidths=ww.tolist(),nearMin=float(ww.min()),nearMedian=float(np.median(ww)),nearMax=float(ww.max()),nearRatioMin=float(ww.min()/rest[side]['w']),nearRatioMax=float(ww.max()/rest[side]['w']))
    touch[side]=dict(nasalTouch=sum(r['q']['nasalTouch'] for r in rows if r['side']==side),bothTouch=sum(r['q']['touchL'] and r['q']['touchR'] for r in rows if r['side']==side),depthMin=float(depths.min()),depthMedian=float(np.median(depths)),depthMax=float(depths.max()),depths=[dict(eye=ids[i],touch=rows[i]['q']['nasalTouch'],depth=rows[i]['q']['nasalCutDepth']) for i in np.flatnonzero(sm)])
result=dict(status=status,gates=checks,failures=failures,warnings=warnings,noise=noise,fit=fits,censored=censored,silhouetteSuspect=suspects,irisArea=area,inputHashChanges=hashchanges,protectedChanges=protected_changes,K=K,rest=rest,restMinusK=D['restMinusK'],asymmetry=asymmetry,nasalContact=touch,coverage=dict(refinedPass=sum(r['clean']['lashCoverageRefined']>=.9 for r in rows),rawPass=sum(r['clean']['lashCoverageRaw']>=.9 for r in rows)))
(OUT/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status=status,noise={f:{k:v for k,v in n.items() if k!='clusters'} for f,n in noise.items()},warnings=warnings,failures=failures,censored=censored,suspects=suspects),ensure_ascii=False,indent=2))
