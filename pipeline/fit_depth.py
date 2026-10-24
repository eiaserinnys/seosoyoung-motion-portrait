"""Fit a symmetric 3D face-depth surface to the approved outline edge depths."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = 'depth-anything/Depth-Anything-V2-Small-hf'
MODEL_REVISION = '5426e4f0f36572d16453bbda7a8389317b1bef99'
CANVAS_SIZE = 1024
X_VALUES = sorted(set(list(range(336, 665, 8)) + [502, 668]))
Y_VALUES = sorted(set(list(range(400, 623, 6)) + [465, 624]))
EDGE_SCALE = 4 * np.sin(np.deg2rad(45))
FACE_RADIUS = 262.0
FACE_CENTER_Y = 326.0
REFERENCE_CORR_VERTICAL = 1.45
REFERENCE_CORR_DEPTH = 0.55


def interpolate_rows(rows, y, columns):
    if y <= rows[0][0]:
        return [float(rows[0][column]) for column in columns]
    if y >= rows[-1][0]:
        return [float(rows[-1][column]) for column in columns]
    for a, b in zip(rows, rows[1:]):
        if y <= b[0]:
            t = (y - a[0]) / (b[0] - a[0])
            return [float(a[column] + (b[column] - a[column]) * t) for column in columns]
    raise AssertionError('outline row interpolation fell through')


def contour_at(rows, y):
    left, right = interpolate_rows(rows, y, [1, 2])
    left45, right45, other_left45, other_right45 = interpolate_rows(rows, y, [3, 4, 5, 6])
    z_edge = ((other_left45 - left45) + (other_right45 - right45)) / EDGE_SCALE
    return left, right, z_edge


def depth_at(depth, x, y):
    return float(cv2.remap(
        depth,
        np.array([[x]], dtype=np.float32),
        np.array([[y]], dtype=np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )[0, 0])


def smoothstep(value):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


def corrected_center_depth(y):
    # Matches correctedLift([502, y], 0, recommended) through the unextended
    # upper ellipsoid and reaches 134.1px at the measured cheek row y=465.
    ry = FACE_RADIUS * REFERENCE_CORR_VERTICAL
    rz = FACE_RADIUS * REFERENCE_CORR_DEPTH
    normalized_y = (y - FACE_CENTER_Y) / ry
    return rz * np.sqrt(max(0.0, 1.0 - normalized_y * normalized_y))


def load_depth_map():
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    source = Image.open(ROOT / 'source/references/head_front_ref.png').convert('RGBA')
    canvas = Image.new('RGBA', (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 255))
    canvas.alpha_composite(source, (152, 20))
    image = canvas.convert('RGB')
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model.eval()
    inputs = processor(images=image, return_tensors='pt')
    with torch.inference_mode():
        outputs = model(**inputs)
        result = processor.post_process_depth_estimation(
            outputs, target_sizes=[(CANVAS_SIZE, CANVAS_SIZE)]
        )[0]['predicted_depth']
    depth = result.detach().cpu().numpy().astype(np.float32)
    if depth.ndim == 3:
        depth = depth[0]
    if depth.shape != (CANVAS_SIZE, CANVAS_SIZE):
        raise ValueError(f'Unexpected depth map size: {depth.shape}')

    # The authored portrait is mirrored around canvas x=502, i.e. x -> 1004-x.
    roi = depth[:, 336:669]
    depth[:, 336:669] = (roi + roi[:, ::-1]) * 0.5
    return cv2.GaussianBlur(depth, (0, 0), sigmaX=4, sigmaY=4)


def fit_grid(depth, rows):
    boundary = []
    for row in rows:
        y = float(row[0])
        left, right, z_edge = contour_at(rows, y)
        boundary.extend([(depth_at(depth, left, y), z_edge),
                         (depth_at(depth, right, y), z_edge)])
    model_values = np.array([item[0] for item in boundary], dtype=np.float64)
    edge_values = np.array([item[1] for item in boundary], dtype=np.float64)
    design = np.column_stack([model_values, np.ones_like(model_values)])
    alpha, beta = np.linalg.lstsq(design, edge_values, rcond=None)[0]
    predicted_edges = design @ np.array([alpha, beta])
    residual = edge_values - predicted_edges
    total = edge_values - edge_values.mean()
    r_squared = 1 - float(residual @ residual) / max(float(total @ total), 1e-12)

    da_grid = np.asarray([[alpha * depth_at(depth, x, y) + beta for x in X_VALUES]
                          for y in Y_VALUES], dtype=np.float64)
    center_bulges = []
    for y in range(465, 591, 6):
        left, right, _ = contour_at(rows, y)
        mid = (left + right) * 0.5
        edge_da = (depth_at(depth, left, y) + depth_at(depth, right, y)) * 0.5
        center_bulges.append(max(0.0, alpha * depth_at(depth, mid, y) + beta - (alpha * edge_da + beta)))
    model_bulge = float(np.median(center_bulges)) if center_bulges else 0.0
    chin_edge_depth = contour_at(rows, 616)[2]

    z_grid = []
    centerline_rows = []
    for yi, y in enumerate(Y_VALUES):
        left, right, z_edge = contour_at(rows, y)
        mid = 502.0
        half = max((right - left) * 0.5, 1e-6)
        surface_left, surface_right = mid - half, mid + half
        left_da = alpha * depth_at(depth, surface_left, y) + beta
        right_da = alpha * depth_at(depth, surface_right, y) + beta
        da_edge = (left_da + right_da) * 0.5
        da_mid = alpha * depth_at(depth, mid, y) + beta
        da_denominator = max(da_mid - da_edge, 1e-4)
        if y <= 465:
            z_mid_target = corrected_center_depth(y)
        else:
            t = np.clip((y - 465) / 151.0, 0, 1)
            z_mid_target = corrected_center_depth(465) + (chin_edge_depth - corrected_center_depth(465)) * smoothstep(t)
        center_arch = max(0.0, z_mid_target - z_edge)
        centerline_rows.append([float(y), float(z_edge + center_arch)])
        row = []
        for xi, x in enumerate(X_VALUES):
            normalized_x = (x - mid) / half
            sphere = max(0.0, 1.0 - normalized_x * normalized_x)
            da_norm = np.clip((da_grid[yi, xi] - da_edge) / da_denominator, 0.0, 1.0)
            shape = (0.35 * da_norm + 0.65 * sphere) * sphere
            row.append(float(z_edge + center_arch * shape))
        z_grid.append(row)

    metadata = {
        'model': MODEL_ID,
        'revision': MODEL_REVISION,
        'canvas': [CANVAS_SIZE, CANVAS_SIZE],
        'sourceOffset': [152, 20],
        'symmetryAxis': 502,
        'gaussianSigma': 4,
        'edgeFit': {'alpha': float(alpha), 'beta': float(beta), 'rSquared': r_squared,
                    'samples': len(boundary)},
        'medianCenterBulge': model_bulge,
        'centerlineProfile': {'topY': 465, 'topDepth': float(corrected_center_depth(465)),
                              'chinY': 616, 'chinDepth': float(chin_edge_depth),
                              'depthModelWeight': 0.35, 'sphereModelWeight': 0.65},
        'edgeDepthFormula': '((right45Left-left45Left)+(right45Right-left45Right))/(4*sin(45deg))',
    }
    return z_grid, metadata, centerline_rows


def write_outputs(grid, metadata, rows, centerline_rows):
    result = {
        'metadata': metadata,
        'x': X_VALUES,
        'y': Y_VALUES,
        'z': grid,
        'centerRows': centerline_rows,
        'edgeRows': [[float(row[0]), *contour_at(rows, float(row[0]))] for row in rows],
    }
    json_text = json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    (ROOT / 'source/viewer/face-depth.json').write_text(json_text + '\n')
    module = '// Generated by pipeline/fit_depth.py from Depth Anything V2 and OUTLINE_ROWS.\n'
    module += 'export const FACE_DEPTH=' + json_text + ';\n'
    module += '''
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
function interval(values,value){
 if(value<=values[0])return [0,0];
 for(let i=1;i<values.length;i++)if(value<=values[i])return [i-1,(value-values[i-1])/(values[i]-values[i-1])];
 return [values.length-2,1];
}
function sampleGridRaw(x,y){
 const [iy,ty]=interval(FACE_DEPTH.y,y),[ix,tx]=interval(FACE_DEPTH.x,x),z=FACE_DEPTH.z;
 const a=z[iy][ix]+(z[iy][ix+1]-z[iy][ix])*tx;
 const b=z[iy+1][ix]+(z[iy+1][ix+1]-z[iy+1][ix])*tx;
 return a+(b-a)*ty;
}
function sampleGrid(x,y){return (sampleGridRaw(x,y)+sampleGridRaw(1004-x,y))/2;}
function centerAt(y){
 const rows=FACE_DEPTH.centerRows;
 if(y<=rows[0][0])return rows[0][1];
 for(let i=1;i<rows.length;i++)if(y<=rows[i][0]){
   const a=rows[i-1],b=rows[i],t=(y-a[0])/(b[0]-a[0]);
   return a[1]+(b[1]-a[1])*t;
 }
 return rows.at(-1)[1];
}
function contourAt(y){
 const rows=FACE_DEPTH.edgeRows;
 let edge;
 if(y<=rows[0][0])edge=rows[0].slice(1);
 else for(let i=1;i<rows.length;i++)if(y<=rows[i][0]){
   const a=rows[i-1],b=rows[i],t=(y-a[0])/(b[0]-a[0]);
   edge=a.slice(1).map((v,j)=>v+(b[j+1]-v)*t);break;
 }
 if(!edge)edge=rows.at(-1).slice(1);
 const [left,right,z]=edge,half=(right-left)/2;
 return [502-half,502+half,z];
}
export function faceDepthEdgeAt(y){return contourAt(y)[2];}
export function faceDepthAt(x,y){
 const raw=sampleGrid(x,y),[l,r,target]=contourAt(y);
 if(x<l||x>r||r<=l)return raw;
 const u=clamp((x-l)/(r-l),0,1),leftError=target-sampleGrid(l,y),rightError=target-sampleGrid(r,y);
 const edgeAdjusted=raw+leftError*(1-u)+rightError*u;
 const centerError=centerAt(y)-(sampleGrid(502,y)+(leftError+rightError)*.5);
 return edgeAdjusted+centerError*4*u*(1-u);
}
'''
    (ROOT / 'bundle/face-depth.mjs').write_text(module)


def build():
    trace = json.loads((ROOT / 'source/viewer/outline-traces.json').read_text())
    rows = trace['rows']
    depth = load_depth_map()
    grid, metadata, centerline_rows = fit_grid(depth, rows)
    write_outputs(grid, metadata, rows, centerline_rows)
    print('PASS depth fit', json.dumps({
        'model': metadata['model'], 'revision': metadata['revision'],
        'edgeFitR2': round(metadata['edgeFit']['rSquared'], 4),
        'centerlineTopDepth': round(metadata['centerlineProfile']['topDepth'], 3),
        'grid': [len(Y_VALUES), len(X_VALUES)],
    }))


if __name__ == '__main__':
    build()
