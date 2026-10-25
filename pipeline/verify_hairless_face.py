"""Face-only quantitative gates. Exits nonzero and preserves every failed value."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from hairless_face_reference import ROOT,geometry,raw_mask,matrix,mask_interval,registered_point
R11_CHIN_MIDPOINT={'up-left':430.1,'up-right':573.9}
UP_DIAGONAL_LIMITS=dict(redBgPixels=120,redBgMaxDepth=2.,cyanPixels=1500,cyanMaxDepth=8.)
DIAGONAL_JAW_LIMITS=dict(chinMidpointError=10.,leftMaxOutwardStep=0.,rightMaxOutwardStep=0.,
                         maxSecondDifference=15.,maxMidpointReversal=5.)

def rms(values):return float(np.sqrt(np.mean(np.square(values))))
def expected_opening(poses,p,side):
    def raw(q,k):
        f=q['feature'];scale=f.get('registration',{}).get('scale',1)
        return f['measuredScales'][k][1]*scale
    if not p['pitch']:return 1.0
    if not p['yaw']:return (raw(p,'left')+raw(p,'right'))/2
    if p['pitch']==-30:
        canonical=next(q for q in poses if q['yaw']==45 and q['pitch']==-30)
        return raw(canonical,side if p['yaw']>0 else 'right' if side=='left' else 'left')
    mirror=next(q for q in poses if q['yaw']==-p['yaw'] and q['pitch']==p['pitch'])
    return float(np.sqrt(raw(p,side)*raw(mirror,'right' if side=='left' else 'left')))

def outline_shape(p):
    rows=p['outline']['rows']
    widths=[r[4]-r[2] for r in rows]
    peak=max(widths);peak_rows=[r[0] for r,w in zip(rows,widths) if abs(w-peak)<1e-7]
    start=465 if p['name']=='up' else max(480,peak_rows[0])
    lower=[w for r,w in zip(rows,widths) if r[0]>=start]
    increase=max([0.,*np.diff(lower).tolist()])
    prior=p['outline'].get('priorRows',[])
    upper=max([0.,*[abs(w-(q[2]-q[1])) for r,w,q in zip(rows,widths,prior) if r[0]<=465]])
    at={r[0]:w for r,w in zip(rows,widths)}
    return dict(peakRestY=peak_rows[0],peakCount=len(peak_rows),lowerStart=start,
                maxLowerWidthIncrease=increase,maxLowerHalfWidthIncrease=increase/2,
                maxUpperPriorError=upper,pearMargin=at[450]-at[490],widths=[[r[0],w] for r,w in zip(rows,widths)])

def diagonal_jaw(p):
    rows=np.asarray([r for r in p['outline']['rows'] if r[0]>=570],float)
    edges=rows[:,[2,4]];steps=np.diff(edges,axis=0);mid=edges.mean(axis=1)
    r11=R11_CHIN_MIDPOINT[p['name']]
    reverse=np.diff(mid)*(1 if p['name']=='up-left' else -1)
    return dict(rows=rows.tolist(),chinMidpoint=float(mid[-1]),r11ChinMidpoint=r11,
                chinMidpointError=float(abs(mid[-1]-r11)),
                leftMaxOutwardStep=float(max(0.,np.max(-steps[:,0]))),
                rightMaxOutwardStep=float(max(0.,np.max(steps[:,1]))),
                maxSecondDifference=float(np.abs(np.diff(edges,n=2,axis=0)).max()),
                maxMidpointReversal=float(max(0.,np.max(reverse))))

def metrics(poses):
    result=[]
    for p in poses:
        target_errors=[]
        for row in p['outline']['rows']:
            # Neutral yaw above465 is hair-occluded SAM, not observed skin.
            if row[0]<(440 if p['pitch'] else 465):continue
            actual=next(b['points'] for b in p['boundary'] if b['y']==row[0])
            target_errors.extend([actual[0][0]-row[2],actual[1][0]-row[4]])
        # Raw mask distances remain reference-only: even the far cheek can be
        # cut by hair. Only classified BG samples are silhouette evidence.
        sides=[0,1] if not p['yaw'] else [0 if p['yaw']<0 else 1]
        raw_errors=[];sample_count=0
        for row in p['boundary']:
            for side in sides:
                q=row['points'][side];xs=mask_interval(p,q[1])
                if xs is not None:raw_errors.append(q[0]-xs[side]);sample_count+=1
        a=matrix(p);raw_chin=float(np.where(raw_mask(p['name'])>127)[0].max()*a[1,1]+a[1,2])
        chin=float(np.mean([q[1] for q in p['boundary'][-1]['points']]))
        lm=p['landmarks'];eye=(lm['left'][1]+lm['right'][1])/2
        length_errors={};raw_length_errors={}
        if p['pitch']:
            raw={k:registered_point(p,v) for k,v in p['feature']['rawCenters'].items()}
            reye=(raw['left'][1]+raw['right'][1])/2
            raw_length_errors={'eyeChin':(chin-eye)-(raw_chin-reye),
                           'eyeNose':(lm['nose'][1]-eye)-(raw['nose'][1]-reye),
                           'noseMouth':(lm['mouth'][1]-lm['nose'][1])-(raw['mouth'][1]-raw['nose'][1])}
            target=p['feature']['centers'];teye=(target['left'][1]+target['right'][1])/2
            length_errors={'eyeChin':(chin-eye)-(p['outline']['yChin']-teye),
                           'eyeNose':(lm['nose'][1]-eye)-(target['nose'][1]-teye),
                           'noseMouth':(lm['mouth'][1]-lm['nose'][1])-(target['mouth'][1]-target['nose'][1])}
        eyes={k:{'shear':v['shear'],'heightRatio':v['heightRatio'],
                 'expectedOpeningRatio':expected_opening(poses,p,k),
                 'openingRatioError':abs(v['heightRatio']-expected_opening(poses,p,k)),
                 'ballRatioError':max(abs(r-1) for r in v['ballRatios'].values())} for k,v in p['eyes'].items()}
        cheeks=p['cheeks'];cheek_error=abs(cheeks['0.25']-cheeks['0.5'])
        observations=p['outline'].get('boundarySamples',[])
        bg_errors=[p['outline']['rows'][i][2 if side=='left' else 4]-s[side]['xCanvas']
                   for i,s in enumerate(observations) for side in ['left','right'] if s[side]['validBG']]
        counts={side:{kind:sum(s[side]['class']==kind for s in observations) for kind in ['BG','HAIR','SKIN']} for side in ['left','right']}
        result.append(dict(name=p['name'],yaw=p['yaw'],pitch=p['pitch'],targetRms=rms(target_errors),
                           targetChinError=abs(chin-p['outline']['yChin']),rawVisibleRms=rms(raw_errors),
                           rawSampleCount=sample_count,rawChinError=abs(chin-raw_chin),
                           lengthErrors=length_errors,rawLengthErrors=raw_length_errors,eyes=eyes,cheekError=cheek_error,
                           cheekDisplacements=cheeks,minDetJ=p['minDetJ'],reversedTriangles=p['reversedTriangles'],outlineShape=outline_shape(p),
                           boundaryClassCounts=counts,validBgSamples=len(bg_errors),bgFitRmsReference=rms(bg_errors) if bg_errors else None))
        if p['name'] in R11_CHIN_MIDPOINT:result[-1]['diagonalJaw']=diagonal_jaw(p)
    return result

def failures(rows,protected):
    failed=[]
    for p in rows:
        def check(label,value,limit):
            if abs(value)>limit:failed.append(dict(pose=p['name'],gate=label,value=float(value),limit=limit))
        check('symmetric outline RMS',p['targetRms'],2.5)
        check('symmetric chin Y',p['targetChinError'],2)
        for k,v in p['lengthErrors'].items():check(k,v,5)
        for k,v in p['eyes'].items():
            check(k+' shear',v['shear'],1e-7)
            check(k+' opening ratio',v['openingRatioError'],.03)
            check(k+' ball height',v['ballRatioError'],1e-7)
        check('reversed triangles',p['reversedTriangles'],0)
        if p['minDetJ'] <= 0:failed.append(dict(pose=p['name'],gate='positive Jacobian',value=p['minDetJ']))
        if p['pitch']:
            s=p['outlineShape']
            check('lower width increase',s['maxLowerWidthIncrease'],.5)
            check('lower half-width increase',s['maxLowerHalfWidthIncrease'],.5)
            if s['peakRestY']<440 or s['peakRestY']>490 or s['peakCount']!=1:
                failed.append(dict(pose=p['name'],gate='single cheek peak',value=s['peakRestY'],count=s['peakCount']))
            if p['pitch']>0:check('upper prior width',s['maxUpperPriorError'],8)
            if p['name']=='up' and s['pearMargin']<=8:failed.append(dict(pose=p['name'],gate='pear inversion',value=s['pearMargin'],minimumExclusive=8))
        if p['name'] in R11_CHIN_MIDPOINT:
            for key,limit in UP_DIAGONAL_LIMITS.items():check(key,p[key],limit)
            for key,limit in DIAGONAL_JAW_LIMITS.items():check(key,p['diagonalJaw'][key],limit)
    if protected:failed.append(dict(gate='protected paths',paths=protected,limit=0))
    return failed

def protected_changes():
    paths=subprocess.check_output(['git','-C',str(ROOT),'diff','--name-only','origin/main'],text=True).splitlines()
    # All non-face runtime/artwork/mesh/depth inputs stay byte identical.
    face_modules={'face-features.mjs','face-outline.mjs','face-prior.mjs','face-feature-prior.mjs','face-contour.mjs','face-solve.mjs','face-solve-data.mjs','feature-data.mjs','outline-data.mjs','index.html'}
    forbidden=['hair','dome','cranium','stereo-attachments','side-strands','bead','tail','neck','ear-','fit_depth']
    # This task explicitly combines face calibration with a residual wrapper;
    # the three 3D prior modules remain protected and have byte-level tests.
    joint_paths={'bundle/rig.mjs','bundle/hair-solve.mjs','bundle/hair-solve-data.mjs','source/viewer/hair-solve.json',
                 'pipeline/fit_hair_solve.py','pipeline/hair_solve_inputs.mjs','pipeline/hair_solve_metrics.mjs',
                 'pipeline/hair_solve_quality.py','pipeline/accept_hair_solve.py','pipeline/capture_hair_solve_prior.py',
                 'pipeline/hair_report.py','pipeline/probe_hair_only.py','pipeline/hair_revision_metrics.py','pipeline/test_hair_solve.py','pipeline/test_hair_solve.mjs'}
    return [p for p in paths if p not in joint_paths and ((p.startswith('bundle/') and Path(p).name not in face_modules and not approved_angle_domain(p))
            or (p.startswith(('source/','parts/','sam3_masks/','pipeline/'))
                # Face-scoped validation may inspect protected hair; its runtime
                # geometry and renderer still cannot change in this face task.
                and p not in {'pipeline/test_hair_pixels.py','pipeline/test_hair_r15.mjs','pipeline/verify_face_solve_hair.py','pipeline/test_hair_texture_budget.mjs'} and 'hairless' not in p and any(word in p for word in forbidden)))]

def angle_domain_only(previous,current):
    """D2 permits AngleX bounds and pass-through keys in the protected rig."""
    expected=json.loads(json.dumps(previous))
    angle=next(p for p in expected['parameters'] if p['id']=='ParamAngleX')
    angle.update(min=-90,max=90)
    for deformer in expected['deformers']:
        if deformer['id'] in ['WarpNeck','WarpHead']:
            keys=deformer['keyforms']['ParamAngleX']
            deformer['keyforms']['ParamAngleX']=[dict(value=-90,output=-90),*keys,dict(value=90,output=90)]
    return expected==current

def approved_angle_domain(path):
    if path!='bundle/assets/seosoyoung-rig.json':return False
    previous=json.loads(subprocess.check_output(['git','-C',str(ROOT),'show','origin/main:'+path],text=True,timeout=60))
    return angle_domain_only(previous,json.loads((ROOT/path).read_text()))

def self_test():
    """Every gate must reject its own injected violation, before trusting it."""
    valid=dict(name='injected',yaw=0,pitch=30,targetRms=0,targetChinError=0,rawVisibleRms=0,rawChinError=0,
               lengthErrors={'eyeChin':0,'eyeNose':0,'noseMouth':0},cheekError=0,minDetJ=1,reversedTriangles=0,
               eyes={'left':dict(shear=0,openingRatioError=0,ballRatioError=0)},
               outlineShape=dict(peakRestY=450,peakCount=1,maxLowerWidthIncrease=0,maxLowerHalfWidthIncrease=0,maxUpperPriorError=0,pearMargin=26))
    assert not failures([valid],[])
    cases=[('targetRms',2.501),('targetChinError',2.01),('minDetJ',-.1),('reversedTriangles',1)]
    for key,value in cases:
        mutated=json.loads(json.dumps(valid));mutated[key]=value;assert failures([mutated],[]),key
    for key in valid['lengthErrors']:
        mutated=json.loads(json.dumps(valid));mutated['lengthErrors'][key]=5.01;assert failures([mutated],[]),key
    for key,value in [('shear',.1),('openingRatioError',.031),('ballRatioError',.1)]:
        mutated=json.loads(json.dumps(valid));mutated['eyes']['left'][key]=value;assert failures([mutated],[]),key
    assert failures([valid],['bundle/hair-shell.mjs'])
    down={**valid,'pitch':-30,'minDetJ':0};assert failures([down],[])
    for key,value in [('peakRestY',510),('peakCount',2),('maxLowerWidthIncrease',.501),('maxLowerHalfWidthIncrease',.501),('maxUpperPriorError',8.01)]:
        mutated=json.loads(json.dumps(valid));mutated['outlineShape'][key]=value;assert failures([mutated],[]),key
    up={**valid,'name':'up','outlineShape':{**valid['outlineShape'],'pearMargin':8}};assert failures([up],[])
    diagonal={**valid,'name':'up-left','yaw':-45,**{key:limit for key,limit in UP_DIAGONAL_LIMITS.items()},
              'diagonalJaw':DIAGONAL_JAW_LIMITS.copy(),'redHairPixels':1000000}
    assert not failures([diagonal],[])
    for key,limit in UP_DIAGONAL_LIMITS.items():
        mutated=json.loads(json.dumps(diagonal));mutated[key]=limit+.001;assert failures([mutated],[]),key
    for key,limit in DIAGONAL_JAW_LIMITS.items():
        mutated=json.loads(json.dumps(diagonal));mutated['diagonalJaw'][key]=limit+.001;assert failures([mutated],[]),key
    print('PASS gate self-test: 27 injected violations rejected')

def write_metrics(poses,differences):
    self_test()
    rows=metrics(poses)
    for p in rows:p.update(differences[p['name']])
    protected=protected_changes();failed=failures(rows,protected)
    out=dict(definition='visible target registration: pitch restY440..616, neutral yaw restY465..616 (upper SAM is hair occlusion); pitch-only single peak440..490 and monotone lower width; up upper prior±8 and width450>width490+8; r11 XY retained; positive mesh Jacobians replace the old S6 formula; comparison excludes restY<347; raw contours/lengths reference-only; redBg is render-only over registered RGB max<40; redHair is remaining render-only occlusion',
             visibility='face skin + eyes + brows + nose + closed mouth only; all hair/ears/neck/body OFF',
             approvedUpDiagonalLimits={**UP_DIAGONAL_LIMITS,**DIAGONAL_JAW_LIMITS},
             poses=rows,protectedChanges=protected,failures=failed,passed=not failed)
    (ROOT/'docs/face_hairless_metrics.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print('pose         targetRMS rawMaskRefRMS chinY lengthMax shear cheek')
    for p in rows:
        print(f"{p['name']:12} {p['targetRms']:7.3f} {p['rawVisibleRms']:12.3f} {p['rawChinError']:5.3f} {max(map(abs,p['lengthErrors'].values()),default=0):9.3f} {max(v['shear'] for v in p['eyes'].values()):5.1f} {p['cheekError']:5.3f}")
    print(json.dumps(failed,ensure_ascii=False,indent=2))
    return failed

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:self_test();return
    from capture_hairless_face import webgl_renders,silhouette_difference
    poses=geometry();renders=webgl_renders(poses)
    failed=write_metrics(poses,{p['name']:silhouette_difference(p,renders[p['name']])[-1] for p in poses})
    raise SystemExit(1 if failed else 0)
if __name__=='__main__':main()
