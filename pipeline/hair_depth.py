"""Calibrate a cached Depth Anything V2 prior with measured hair attachments.

Inference uses the canonical front-image loader introduced by the face-depth
PR. Normal dome rebuilds need only this checked-in 128-square depth cache.
"""
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'source/viewer/hair-depth-prior.npz'
META = ROOT / 'source/viewer/hair-depth-prior.json'


def build_depth_prior(infer=False):
    source = ROOT / 'source/references/head_front_ref.png'
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if infer:
        from fit_depth import load_depth_map, MODEL_ID, MODEL_REVISION
        depth = load_depth_map()[::8, ::8].astype(np.float32)
        np.savez_compressed(CACHE, depth=depth)
        META.write_text(json.dumps({
            'model': MODEL_ID, 'revision': MODEL_REVISION,
            'license': 'Apache-2.0', 'source': str(source.relative_to(ROOT)),
            'sourceSha256': source_sha, 'canvas': [1024, 1024],
            'sourceOffset': [152, 20], 'sampleStep': 8,
            'preprocessing': 'canonical fit_depth.load_depth_map; sigma4; face ROI symmetry',
        }, ensure_ascii=False, indent=2) + '\n')
    metadata = json.loads(META.read_text())
    if metadata['sourceSha256'] != source_sha:
        raise ValueError('Hair depth cache does not match head_front_ref.png; rebuild with --infer-depth')
    depth = np.load(CACHE)['depth']
    # The independent yaw fit anchors the absolute surface. The learned prior
    # contributes internal bends, with its broad background/portrait bias removed.
    script = """
    import {hairLift} from './bundle/hair-shell.mjs';
    import {domeSection} from './bundle/hair-surface.mjs';
    import {STRANDS,strandDepth} from './bundle/side-strands.mjs';
    import {BEADS,beadWorld,earWorld} from './bundle/stereo-attachments.mjs';
    import {CORRECTION_PRESETS} from './bundle/geometry.mjs';
    const p={...CORRECTION_PRESETS.recommended,ParamAngleX:0,ParamAngleY:0,ParamAngleZ:0};
    const anchors=[];
    for(const y of [100,150,200,250,300])for(const u of [-.8,-.4,0,.4,.8]){
      const {cx,rx}=domeSection(y),x=502+cx+u*rx;
      anchors.push({kind:'crown-45-dome',point:[x,y],z:hairLift([x,y],p)[2]});
    }
    for(const side of ['left','right']){
      for(const [y,l,r] of STRANDS[side])for(const u of [0,.5,1]){
        const point=[l+(r-l)*u,y];anchors.push({kind:'strand-'+side,point,z:Math.max(strandDepth(point,side),earWorld(point,side,p)[2]+4)});
      }
      anchors.push({kind:'bead-'+side,point:BEADS[side].front,z:beadWorld(side,p)[2]});
    }
    console.log(JSON.stringify(anchors));
    """
    anchors = json.loads(subprocess.check_output(
        ['node', '--input-type=module', '-e', script], text=True, cwd=ROOT))
    def sample(x, y):
        return float(cv2.remap(depth, np.array([[x / 8]], np.float32),
                              np.array([[y / 8]], np.float32), cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)[0, 0])
    raw = np.array([sample(*a['point']) for a in anchors])
    target = np.array([a['z'] for a in anchors])
    design = np.column_stack([raw, np.ones(len(raw))])
    alpha, beta = np.linalg.lstsq(design, target, rcond=None)[0]
    error = design @ [alpha, beta] - target
    detail = (depth - cv2.GaussianBlur(depth, (0, 0), sigmaX=3, sigmaY=3)) * alpha
    x_values = list(range(192, 817, 8))
    y_values = list(range(64, 553, 8))
    grid = [[round(float(detail[y // 8, x // 8]), 5) for x in x_values] for y in y_values]
    result = {'metadata': {**metadata,
              'cacheSha256': hashlib.sha256(CACHE.read_bytes()).hexdigest(),
              'alpha': float(alpha), 'beta': float(beta), 'anchorCount': len(anchors),
              'anchorRms': float(np.sqrt(np.mean(error ** 2))),
              'detailSigmaPixels': 24, 'detailRange': [float(detail.min()), float(detail.max())],
              'method': '45-degree dome/strand/bead absolute Z; calibrated DAV2 internal depth detail'},
              'x': x_values, 'y': y_values, 'relief': grid, 'anchors': anchors}
    print('PASS hair depth prior', json.dumps(result['metadata']))
    return result
