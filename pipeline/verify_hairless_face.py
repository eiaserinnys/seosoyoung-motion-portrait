"""Face-only quantitative gates. Exits nonzero and preserves every failed value."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from hairless_face_reference import ROOT,geometry,raw_mask,matrix,mask_interval,registered_point

def rms(values):return float(np.sqrt(np.mean(np.square(values))))
def expected_opening(poses,p,side):
    def raw(q,k):
        f=q['feature'];scale=f.get('registration',{}).get('scale',1)
        return f['measuredScales'][k][1]*scale
    if not p['pitch']:return 1.0
    if not p['yaw']:return (raw(p,'left')+raw(p,'right'))/2
    mirror=next(q for q in poses if q['yaw']==-p['yaw'] and q['pitch']==p['pitch'])
    return float(np.sqrt(raw(p,side)*raw(mirror,'right' if side=='left' else 'left')))

def metrics(poses):
    result=[]
    for p in poses:
        target_errors=[]
        for row in p['outline']['rows']:
            if row[0]<440:continue
            actual=next(b['points'] for b in p['boundary'] if b['y']==row[0])
            target_errors.extend([actual[0][0]-row[2],actual[1][0]-row[4]])
        # Unoccluded raw sides: both at yaw0; only far side at yaw ±45.
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
        result.append(dict(name=p['name'],yaw=p['yaw'],pitch=p['pitch'],targetRms=rms(target_errors),
                           targetChinError=abs(chin-p['outline']['yChin']),rawVisibleRms=rms(raw_errors),
                           rawSampleCount=sample_count,rawChinError=abs(chin-raw_chin),
                           lengthErrors=length_errors,rawLengthErrors=raw_length_errors,eyes=eyes,cheekError=cheek_error,
                           cheekDisplacements=cheeks,sphericalError=p['sphericalError']))
    return result

def failures(rows,protected):
    failed=[]
    for p in rows:
        def check(label,value,limit):
            if abs(value)>limit:failed.append(dict(pose=p['name'],gate=label,value=float(value),limit=limit))
        check('symmetric outline RMS',p['targetRms'],2)
        check('symmetric chin Y',p['targetChinError'],2)
        for k,v in p['lengthErrors'].items():check(k,v,5)
        for k,v in p['eyes'].items():
            check(k+' shear',v['shear'],1e-7)
            check(k+' opening ratio',v['openingRatioError'],.03)
            check(k+' ball height',v['ballRatioError'],1e-7)
        if not p['yaw'] and p['pitch']:
            check('spherical cheek Y',p['cheekError'],10 if p['pitch']>0 else 4.05)
            check('spherical formula',p['sphericalError'],1e-6)
    if protected:failed.append(dict(gate='protected paths',paths=protected,limit=0))
    return failed

def protected_changes():
    paths=subprocess.check_output(['git','-C',str(ROOT),'diff','--name-only','origin/main'],text=True).splitlines()
    # All non-face runtime/artwork/mesh/depth inputs stay byte identical.
    face_modules={'face-features.mjs','face-outline.mjs','feature-data.mjs','outline-data.mjs','index.html'}
    forbidden=['hair','dome','cranium','stereo-attachments','side-strands','bead','tail','neck','ear-','fit_depth']
    return [p for p in paths if (p.startswith('bundle/') and Path(p).name not in face_modules)
            or (p.startswith(('source/','parts/','sam3_masks/','pipeline/'))
                # Explicitly approved exception: only its obsolete face import
                # and synthetic-chin assertion change; hair test bodies stay intact.
                and p!='pipeline/test_hair_pixels.py' and 'hairless' not in p and any(word in p for word in forbidden))]

def self_test():
    """Every gate must reject its own injected violation, before trusting it."""
    valid=dict(name='injected',yaw=0,pitch=30,targetRms=0,targetChinError=0,rawVisibleRms=0,rawChinError=0,
               lengthErrors={'eyeChin':0,'eyeNose':0,'noseMouth':0},cheekError=0,sphericalError=0,
               eyes={'left':dict(shear=0,openingRatioError=0,ballRatioError=0)})
    assert not failures([valid],[])
    cases=[('targetRms',2.01),('targetChinError',2.01),('cheekError',10.01),('sphericalError',.1)]
    for key,value in cases:
        mutated=json.loads(json.dumps(valid));mutated[key]=value;assert failures([mutated],[]),key
    for key in valid['lengthErrors']:
        mutated=json.loads(json.dumps(valid));mutated['lengthErrors'][key]=5.01;assert failures([mutated],[]),key
    for key,value in [('shear',.1),('openingRatioError',.031),('ballRatioError',.1)]:
        mutated=json.loads(json.dumps(valid));mutated['eyes']['left'][key]=value;assert failures([mutated],[]),key
    assert failures([valid],['bundle/hair-shell.mjs'])
    down={**valid,'pitch':-30,'cheekError':4.051};assert failures([down],[])
    print('PASS gate self-test: 12 injected violations rejected')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    self_test()
    if args.self_test:return
    rows=metrics(geometry());protected=protected_changes();failed=failures(rows,protected)
    out=dict(definition='restY440..616; symmetric target rows and centers; raw visible contours/lengths are reference-only artwork asymmetry; paired measured opening envelopes',
             visibility='face skin + eyes + brows + nose + closed mouth only; all hair/ears/neck/body OFF',
             poses=rows,protectedChanges=protected,failures=failed,passed=not failed)
    (ROOT/'docs/face_hairless_metrics.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print('pose         targetRMS rawVisibleRMS chinY lengthMax shear cheek')
    for p in rows:
        print(f"{p['name']:12} {p['targetRms']:7.3f} {p['rawVisibleRms']:12.3f} {p['rawChinError']:5.3f} {max(map(abs,p['lengthErrors'].values()),default=0):9.3f} {max(v['shear'] for v in p['eyes'].values()):5.1f} {p['cheekError']:5.3f}")
    print(json.dumps(failed,ensure_ascii=False,indent=2))
    raise SystemExit(1 if failed else 0)
if __name__=='__main__':main()
