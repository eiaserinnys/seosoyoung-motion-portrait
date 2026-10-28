"""Build tentative cheek and chin observations in the fixed video-to-art frame."""
import argparse
import hashlib
import json
import resource
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'pipeline/headmodel/data'
REGISTRATION = ROOT / 'source/viewer/ponytail-seedance-registration.json'
REASONS = ['볼·턱 후보', '윗부분 관측 안 함', '귀 또는 인접부', '가려짐 또는 불명', '짧은 조각']


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_frames():
    fit = [f for f in range(1, 362) if f % 12 not in (5, 6, 7)]
    validation = [f for f in range(1, 362) if f % 12 in (5, 6, 7)]
    return fit, validation


def mask_to_video(to720):
    return np.linalg.inv(np.asarray(to720, float) @ np.diag([640 / 720, 640 / 720, 1.]))


def lookup(image, points):
    xy = np.rint(points).astype(int)
    xy[:, 0] = np.clip(xy[:, 0], 0, image.shape[1] - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, image.shape[0] - 1)
    return image[xy[:, 1], xy[:, 0]]


def classify_samples(points, normals, head, ear, neck, eye_y):
    outside = points + 5 * normals
    upper = points[:, 1] <= eye_y + 12
    near_ear = (lookup(ear, points) != 0) | (lookup(ear, outside) != 0)
    hidden = (lookup(head, outside) != 0) & (lookup(neck, outside) == 0)
    reasons = np.zeros(len(points), np.uint8)
    reasons[hidden] = 3
    reasons[near_ear] = 2
    reasons[upper] = 1
    return reasons == 0, reasons


def contour(mask):
    outlines, _ = cv2.findContours(np.uint8(mask > 0), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(outlines, key=cv2.contourArea)[:, 0, :].astype(float)


def normals(points):
    tangent = np.roll(points, -3, axis=0) - np.roll(points, 3, axis=0)
    result = np.column_stack((tangent[:, 1], -tangent[:, 0]))
    if cv2.contourArea(points.astype(np.float32), oriented=True) < 0:
        result = -result
    return result / np.maximum(1e-8, np.linalg.norm(result, axis=1)[:, None])


def refine(points, normal, image):
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32), (3, 3), .7)
    shifts = np.arange(-5, 5.01, .5)
    query = points[:, None, :] + normal[:, None, :] * shifts[None, :, None]
    values = cv2.remap(gray, query[:, :, 0].astype(np.float32), query[:, :, 1].astype(np.float32), cv2.INTER_LINEAR)
    offsets = shifts[np.argmin(values + 2 * np.abs(shifts)[None, :], axis=1)]
    offsets = np.median(np.array([np.roll(offsets, k) for k in (-2, -1, 0, 1, 2)]), axis=0)
    return points + normal * offsets[:, None]


def remove_short_sections(keep, reasons):
    for _ in range(2):
        counts = sum(np.roll(keep, k).astype(int) for k in range(-4, 5))
        removed = keep & (counts < 5)
        keep[removed] = False
        reasons[removed] = 4
    return keep, reasons


def process(frames_dir, body_path, frames, output):
    registration = json.loads(REGISTRATION.read_text())['video']
    bodies = {r['frame']: r for r in json.loads(body_path.read_text())}
    available = sorted(int(p.name.split('_')[1]) for p in DATA.glob('frame_*') if (p / 'skin.png').exists())
    fit, validation = split_frames()

    @lru_cache(maxsize=8)
    def image(frame):
        path = frames_dir / f'frame_{frame:04d}.png'
        bgr = cv2.imread(str(path))
        return cv2.resize(bgr, (720, 720), interpolation=cv2.INTER_CUBIC), sha256(path)

    @lru_cache(maxsize=6)
    def source(frame):
        directory = DATA / f'frame_{frame:04d}'
        manifest_path = directory / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        assert image(frame)[1] == manifest['source_image_sha256']
        transform = mask_to_video(manifest['to720'])[:2]
        paths = {'skin': 'skin.png', 'head': 'head.png', 'neck': 'raw/neck.png', 'ear': 'raw/ear.png'}
        masks = {name: cv2.warpAffine(cv2.imread(str(directory / path), cv2.IMREAD_GRAYSCALE), transform,
                                      (720, 720), flags=cv2.INTER_NEAREST)
                 for name, path in paths.items()}
        meta = {'frame': frame, 'frame_sha256': image(frame)[1], 'manifest_sha256': sha256(manifest_path),
                'mask_sha256': {name: sha256(directory / path) for name, path in paths.items()},
                'normalization_applied': manifest['normalization_applied'], 'to720': manifest['to720']}
        return masks, meta

    def warped(target, source_frame):
        masks, meta = source(source_frame)
        target_gray = cv2.cvtColor(image(target)[0], cv2.COLOR_BGR2GRAY)
        source_gray = cv2.cvtColor(image(source_frame)[0], cv2.COLOR_BGR2GRAY)
        forward = cv2.calcOpticalFlowFarneback(target_gray, source_gray, None, .5, 4, 25, 5, 7, 1.5, 0)
        backward = cv2.calcOpticalFlowFarneback(source_gray, target_gray, None, .5, 4, 25, 5, 7, 1.5, 0)
        y, x = np.indices((720, 720), dtype=np.float32)
        mx, my = x + forward[:, :, 0], y + forward[:, :, 1]
        moved = {name: cv2.remap(mask, mx, my, cv2.INTER_NEAREST) for name, mask in masks.items()}
        reverse = cv2.remap(backward, mx, my, cv2.INTER_LINEAR)
        consistency = np.linalg.norm(forward + reverse, axis=2)
        return moved, meta, consistency

    rows = []
    for f in frames:
        if f in available:
            masks, meta = source(f)
            sources = [meta]
            transfer = None
            other = None
        else:
            previous = max(v for v in available if v < f)
            following = min(v for v in available if v > f)
            candidates = [warped(f, previous), warped(f, following)]
            selected = 0 if f - previous <= following - f else 1
            masks, _, transfer = candidates[selected]
            other = candidates[1 - selected][0]
            sources = [dict(value[1], side=side) for value, side in zip(candidates, ('앞', '뒤'))]

        p = contour(masks['skin'])
        n = normals(p)
        ear = cv2.dilate(masks['ear'], np.ones((13, 13), np.uint8))
        tracked = registration[f - 1]
        keep, reasons = classify_samples(p, n, masks['head'], ear, masks['neck'],
                                         np.mean(np.asarray(tracked['landmarks'])[:2, 1]))
        keep, reasons = remove_short_sections(keep, reasons)
        p = refine(p, n, image(f)[0])
        matrix = np.asarray(bodies[f]['G_after_body'])
        art = p @ matrix[:, :2].T + matrix[:, 2]
        transfer_stats = None
        if transfer is not None and np.any(keep):
            consistency = lookup(transfer, p[keep])
            comparison = cKDTree(contour(other['skin'])).query(p[keep])[0]
            transfer_stats = {'왕복_중앙값': float(np.median(consistency)),
                              '반대쪽_선_차이_중앙값': float(np.median(comparison)),
                              '반대쪽_선_차이_95': float(np.percentile(comparison, 95))}
        rows.append({'frame': f, 'frame_sha256': image(f)[1], 'pose': tracked['pose'][:3],
                     'body_to_front': bodies[f]['body_to_front'], 'G_fixed': bodies[f]['G_fixed'],
                     'G_after_body': bodies[f]['G_after_body'], 'body_status': bodies[f]['status'],
                     'source_kind': '같은 영상' if f in available else '옮긴 선',
                     'sources': sources, 'transfer_diagnostic': transfer_stats,
                     'confidence': '자동선 · 확인 전', 'points720': np.round(p, 4).tolist(),
                     'pointsArt': np.round(art, 4).tolist(), 'keep': keep.tolist(), 'reason': reasons.tolist()})
        print(f'관측 {f}: 사용 {int(keep.sum())}/{len(keep)}', flush=True)

    result = {'coordinate': 'G × 몸통 고정', 'registration_sha256': sha256(REGISTRATION),
              'body_sha256': sha256(body_path), 'reason_codes': REASONS,
              'fit_frames': fit, 'validation_frames': validation, 'observations': rows}
    output.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':')) + '\n')
    print(f'SAVED {output}: {len(rows)} observations', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames-dir', type=Path, required=True)
    parser.add_argument('--body-matrices', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, nargs='+')
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
    cv2.setNumThreads(1)
    process(args.frames_dir, args.body_matrices, args.frames or list(range(1, 362)), args.output)


if __name__ == '__main__':
    main()
