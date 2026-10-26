import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--frames", required=True)
args=parser.parse_args()
import ast,json,hashlib
from pathlib import Path
import numpy as np,cv2
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'source/references/seedance/silhouettes';OUT.mkdir(parents=True,exist_ok=True)
path=ROOT/'pipeline/ponytail_observations.py';tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='skin_contour');src=ast.get_source_segment(path.read_text(),node)
src=src[:src.index('    moments=cv2.moments(skin)')]+'    return skin\n';exec(compile(src,'<Stage4 skin mask before ray sampling>','exec'))
reg=json.loads((ROOT/'source/viewer/ponytail-seedance-registration.json').read_text());rows=json.loads((ROOT/'sam3_masks/references/seedance/selection.json').read_text())['frames'];video={r['frame']:r for r in reg['video']};cv2.setNumThreads(1);report=[]
for row in rows:
 path=Path(args.frames)/f"frame_{row['frame']:04d}.png"
 rgb=cv2.cvtColor(cv2.resize(cv2.imread(str(path)),(720,720),interpolation=cv2.INTER_CUBIC),cv2.COLOR_BGR2RGB);points=np.array(video[row['frame']]['landmarks']);cv2.setRNGSeed(42)
 raw=skin_contour(rgb,points,points[6,1]);norm=cv2.warpAffine(raw,np.array(row['normalization']),(720,720),flags=cv2.INTER_NEAREST,borderValue=0)
 dest=OUT/f"{row['name']}.png";Image.fromarray(norm.astype('uint8')*255).save(dest)
 report.append(dict(name=row['name'],frame=row['frame'],source=str(path),sourceSha=hashlib.sha256(path.read_bytes()).hexdigest(),mask=str(dest),maskSha=hashlib.sha256(dest.read_bytes()).hexdigest(),area=int(norm.sum())))
(OUT/'manifest.json').write_text(json.dumps(dict(source='Stage4 skin_contour pixel mask reconstructed before polar sampling; per-frame RNG42; largest convex hull; original saved masks unavailable',codeSha=hashlib.sha256((ROOT/'pipeline/ponytail_observations.py').read_bytes()).hexdigest(),frames=report),ensure_ascii=False,indent=2)+'\n');print('MASKS',len(report),'MIN_AREA',min(r['area'] for r in report))
