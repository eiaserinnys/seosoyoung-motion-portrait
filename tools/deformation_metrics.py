"""Numerical diagnostics only. Coordinates and model-derived reports stay private."""
import numpy as np


def rms(a):
    return float(np.sqrt(np.mean(np.sum(np.asarray(a)**2, axis=-1))))


def metrics(rest, target):
    """Affine fit and nonaffine residual; dimensions are those of the supplied frame."""
    a, b = np.asarray(rest, dtype=float), np.asarray(target, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2:
        raise ValueError('matching N by 2 coordinates required')
    design = np.column_stack([a, np.ones(len(a))])
    fit, _, rank, _ = np.linalg.lstsq(design, b, rcond=None)
    if rank < 3:
        raise ValueError('non-collinear rest coordinates required')
    F = fit[:2].T
    U, scales, Vt = np.linalg.svd(F)
    rotation = U @ Vt
    stretch = Vt.T @ np.diag(scales) @ Vt
    return dict(displacement_rms=rms(b-a), displacement_max=float(np.linalg.norm(b-a, axis=1).max()),
                mean_delta=(b-a).mean(axis=0).tolist(), affine=F.tolist(),
                principal_stretches=scales.tolist(), shear=float(stretch[0, 1]),
                rotation_degrees=float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))),
                nonaffine_rms=rms(b-design@fit))


def grid_curvature_change(rest, target, rows, columns):
    """RMS change of second differences, not intrinsic 3D curvature."""
    a=np.asarray(rest).reshape(rows+1, columns+1, 2)
    b=np.asarray(target).reshape(a.shape)
    values=[np.diff(b-a,n=2,axis=k).reshape(-1,2) for k in (0,1) if a.shape[k]>2]
    return rms(np.concatenate(values)) if values else 0.0


def ordinary_points(tables, kind, obj, overrides):
    """Evaluate only ordinary stored keyforms in the object's parent-local frame.

    Blend shapes, parent transformations, quad interpolation and glue are deliberately
    excluded here. Final world geometry must be sampled with Cubism Core.
    """
    params={p['id']:p['default'] for p in tables['parameters']} | overrides
    band=obj['binding_band_index']; corners=[(0,1.0)]; stride=1
    if band>=0:
        for index in tables['normal_binding_bands'][band]['binding_indices']:
            binding=tables['normal_parameter_bindings'][index]
            keys=binding['keys'];value=params[binding['parameter_id']]
            if value<=keys[0]: choices=[(0,1)]
            elif value>=keys[-1]: choices=[(len(keys)-1,1)]
            else:
                j=int(np.searchsorted(keys,value)-1); w=(value-keys[j])/(keys[j+1]-keys[j]);choices=[(j,1-w),(j+1,w)]
            corners=[(i+j*stride,w*v) for i,w in corners for j,v in choices]
            stride*=len(keys)
    result=np.zeros((obj['vertex_count'],2))
    for index,weight in corners:
        row=tables[kind+'_keyforms'][obj['keyform_start']+index]
        start=row['position_start']
        result+=weight*np.asarray(tables['position_floats'][start:start+2*len(result)]).reshape(-1,2)
    return result


def sphere_projection(points, center, radius, yaw, pitch, depth_sign=1):
    """Orthographic, image y-down; returns the unchanged image at zero rotation."""
    p=np.asarray(points)-center;x,y=p.T;z=depth_sign*np.sqrt(np.maximum(0,radius**2-x*x-y*y))
    a,b=np.radians([yaw,pitch]);X=x*np.cos(a)+z*np.sin(a);Z=-x*np.sin(a)+z*np.cos(a)
    return np.column_stack([X,y*np.cos(b)-Z*np.sin(b)])+center
