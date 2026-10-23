"""Assertions shared by raster capture and mutation checks."""
import copy,json
from pathlib import Path

def validate(rows):
    assert rows and {r['gain'] for r in rows}=={.65,1}
    assert all(r['earOverlap']==0 for r in rows)
    assert all(r['farBackOutsidePixels']==0 for r in rows)
    assert all(r['farCraniumPixels']==0 for r in rows)
    assert all(r['nearCraniumPixels']>0 for r in rows if r['yaw'])
    assert all(r['nearCraniumPixels']==0 for r in rows if not r['yaw'])
    assert all(r['farEarPixels']==0 for r in rows if abs(r['yaw']*r['gain'])>=25)

if __name__=='__main__':
    rows=json.loads((Path(__file__).resolve().parents[1]/'debug/outline_metrics.json').read_text())
    validate(rows)
    for key in ['earOverlap','farBackOutsidePixels','farCraniumPixels','farEarPixels']:
        broken=copy.deepcopy(rows);broken[0][key]=1
        try:validate(broken)
        except AssertionError:continue
        raise AssertionError('Checker accepted injected '+key)
    print('PASS raster contract and 4 injected violations rejected')
