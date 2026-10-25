"""Select only residual poses that improve the actual rendered prior comparison."""
import json
from pathlib import Path
from fit_hair_solve import write
from hair_solve_quality import compare_captures
import probe_hair_only as probe

ROOT=Path(__file__).resolve().parents[1]


def main():
    poses=sorted(json.loads((ROOT/'source/viewer/face-feature-fit.json').read_text())['poses'],key=lambda p:(-p['pitch'],p['yaw']))
    current=probe.capture(poses,False)
    quality=compare_captures(poses,ROOT/'debug/hair_solve_prior',current)
    data=json.loads((ROOT/'source/viewer/hair-solve.json').read_text())
    rejected={p['name'] for p in quality['poses'] if not p['pass'] and p['name'] in {r['name'] for r in data['poses']}}
    for pose in data['poses']:
        if pose['name'] in rejected:
            pose['candidateStats']=pose['stats'];pose['du']=[[0.,0.]]*192
            pose['stats']={**pose['stats'],'duRms':0.,'duMax':0.,'predictedRms':pose['stats']['priorRms']}
            pose['selection']='prior retained: candidate raster RMS increased'
        else:pose['selection']='candidate accepted'
    data['candidateComparison']=quality
    write(data)
    if rejected:
        # Render only changed poses; preserve unchanged neutral capture hashes.
        names=['geometry-sha256.json','reference-mix.json']
        saved={n:(current/n).read_bytes() for n in names}
        probe.capture([p for p in poses if p['name'] in rejected],False)
        for n,value in saved.items():(current/n).write_bytes(value)
    final=compare_captures(poses,ROOT/'debug/hair_solve_prior',current)
    data['comparison']=final;write(data)
    (current/'hair-solve-comparison.json').write_text(json.dumps(final,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(final,ensure_ascii=False))
    assert final['pass'],'Actual raster monotonicity failed'


if __name__=='__main__':main()
