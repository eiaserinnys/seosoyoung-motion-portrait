"""Raster contracts for neutral silhouettes, temple RGB and posterior volume."""
import math
import hashlib
import numpy as np


def approved_top_gate(yaw, pitch, rendered):
    """Historical completion bounds; direct reference residuals stay diagnostic."""
    if pitch > 0 and not yaw:
        return {'target': 92, 'tolerance': 8, 'pass': abs(rendered-92) <= 8}
    if yaw and pitch > 0:
        target = 55 if yaw < 0 else 48
        return {'target': target, 'tolerance': 6, 'pass': abs(rendered-target) <= 6}
    if yaw and pitch < 0:
        return {'maximum': 13, 'pass': rendered <= 13}
    return None


def protected_files_match(root, entries):
    return all(hashlib.sha256((root / path).read_bytes()).hexdigest() == expected
               for expected, path in entries)


def posterior_error_gate(yaw, error):
    limit = 16 if yaw < 0 else 20
    return {'maximum': limit, 'value': error,
            'pass': error is not None and math.isfinite(error) and error <= limit}


def posterior_shift_gate(shift):
    return {'gate': 'H2 raster exterior shift from zero-supplement stage a <=60px',
            'pass': math.isfinite(shift) and shift <= 60, 'value': shift}


def row_span(mask, y):
    xs = np.flatnonzero(mask[y])
    return [int(xs[0]), int(xs[-1])] if len(xs) else None


def neutral_silhouette(before, after):
    differences, missing = [], 0
    for y in range(80, 521):
        a, b = row_span(before, y), row_span(after, y)
        if a is None and b is None:
            continue
        if a is None or b is None:
            missing += 1
        else:
            differences.extend(abs(a[i]-b[i]) for i in [0, 1])
    median = float(np.median(differences)) if differences else 0
    p95 = float(np.percentile(differences, 95)) if differences else 0
    return {'median': median, 'p95': p95, 'missingRows': missing,
            'pass': not missing and median <= 1 and p95 <= 2}


def rimlight_metrics(before, after, before_mask, after_mask, yaw):
    x0, x1 = (560, 860) if yaw < 0 else (164, 464)
    roi = np.s_[120:420, x0:x1]
    # Skin and transparent background cannot be counted as hair rimlight.
    old_luma = before[roi][:, :, :3] @ np.array([.2126, .7152, .0722])
    new_luma = after[roi][:, :, :3] @ np.array([.2126, .7152, .0722])
    old_mask, new_mask = before_mask[roi], after_mask[roi]
    if not old_mask.any():
        return {'pass': False, 'reduction': None, 'baselinePixels': 0, 'pixels': 0}
    threshold = float(np.percentile(old_luma[old_mask], 98))
    old_count = int(((old_luma >= threshold) & old_mask).sum())
    count = int(((new_luma >= threshold) & new_mask).sum())
    reduction = 1-count/old_count if old_count else 0
    return {'window': [x0, 120, x1, 420], 'threshold': threshold,
            'baselinePixels': old_count, 'pixels': count, 'reduction': reduction,
            'minimumReduction': .50, 'pass': reduction >= .50}


def posterior_metrics(reference, current, baseline, yaw):
    side, outward = (1, 1) if yaw < 0 else (0, -1)
    old_errors, errors, upper, upper_increase = [], [], [], []
    upper_samples = []
    missing = 0
    for y in range(80, 521):
        r, c, b = [row_span(mask, y) for mask in [reference, current, baseline]]
        if r is None or c is None or b is None:
            missing += 1
            continue
        if 340 <= y <= 440:
            old_errors.append(abs(b[side]-r[side]))
            errors.append(abs(c[side]-r[side]))
        if 120 <= y <= 200:
            cur_overhang = (c[side]-r[side])*outward
            base_overhang = (b[side]-r[side])*outward
            upper.append(cur_overhang)
            upper_increase.append(max(0, cur_overhang) - max(0, base_overhang))
            upper_samples.append({'y': y, 'reference': r[side], 'baseline': b[side],
                                  'current': c[side], 'increase': upper_increase[-1]})
    old_mean = float(np.mean(old_errors)) if old_errors else None
    mean = float(np.mean(errors)) if errors else None
    improvement = old_mean-mean if mean is not None else None
    maximum = max(upper, default=None)
    increase = max(upper_increase, default=None)
    return {'baselineMeanAbsoluteDelta': old_mean, 'meanAbsoluteDelta': mean,
            'improvement': improvement, 'upperMaximumOverhang': maximum,
            'upperMaximumIncrease': increase, 'lowerRows': [340, 440],
            'upperRows': [120, 200], 'missingRows': missing,
            'upperWorstRow': max(upper_samples, key=lambda row: row['increase'], default=None),
            'pass': not missing and improvement is not None and improvement >= 10
                    and increase is not None and increase <= 13}
