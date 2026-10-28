import argparse
import json
import resource
import sys
from pathlib import Path

resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))

import cv2
import numpy as np
import shapely
from PIL import Image, ImageDraw, ImageFont

root = Path('/home/eias/workspace/.projects/seosoyoung-motion-portrait--feat-face-video-refit-f0ed41f9')
sys.path.insert(0, str(root / 'pipeline'))
from face_video_contour import centered_basis, contour, project

parser = argparse.ArgumentParser()
parser.add_argument('--result', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--solution', type=int)
args = parser.parse_args()
cv2.setNumThreads(1)
result = json.loads(args.result.read_text())
solution = result['solutions'][args.solution] if args.solution is not None else None
stats = {row['frame']: row for row in (solution['stats'] if solution is not None else result['stats'])}
data = json.loads((root / 'source/viewer/face-solve.json').read_text())
obs = json.loads((root / 'source/viewer/face-video-observations.json').read_text())['observations']
rows = {r['frame']: r for r in obs}
n = data['meshCount']
rest = np.asarray(data['points'][:n])
triangles = np.asarray(data['triangles'])
depth = np.asarray(result['depth'])
source_texture = cv2.imread(str(root / 'parts/face-skin.png'), cv2.IMREAD_UNCHANGED)
frames_dir = Path('/home/eias/workspace/.local/artifacts/20261025-ponytail-seedance-reg/frames')
font = ImageFont.truetype('/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf', 22)
sheet = Image.new('RGB', (1920, 5 * 590), 'white')


def raster_face(q):
    canvas = np.zeros((1024, 1024, 4), np.uint8)
    for triangle in triangles:
        src = (rest[triangle] - [306, 218]).astype(np.float32)
        dest = q[triangle].astype(np.float32)
        a, b, c = dest
        if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) <= 0:
            continue
        x0, y0 = np.floor(dest.min(axis=0)).astype(int)
        x1, y1 = np.ceil(dest.max(axis=0)).astype(int) + 1
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(1024, x1), min(1024, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        local = dest - [x0, y0]
        matrix = cv2.getAffineTransform(src, local.astype(np.float32))
        warped = cv2.warpAffine(source_texture, matrix, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(mask, np.rint(local).astype(np.int32), 255)
        block = canvas[y0:y1, x0:x1]
        valid = (mask > 0) & (warped[:, :, 3] > 0)
        block[valid] = warped[valid]
    return canvas


for row_index, frame in enumerate(result['frames']):
    row = rows[frame]
    local = np.column_stack((rest - [502, 326], depth))
    translation = np.zeros(2)
    if solution is not None:
        basis = centered_basis(*row['pose'][:2])
        local += (basis @ np.asarray(solution['C_coefficients']).reshape(35, -1)).reshape(n, 3)
        shift = basis @ np.asarray(solution['T_coefficients'])
        angle = np.deg2rad(row['pose'][2])
        translation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) @ shift[:2]
    q = project(local, row['pose']) + translation
    face = raster_face(q)
    video = cv2.imread(str(frames_dir / f'frame_{frame:04d}.png'))
    video = cv2.resize(video, (720, 720), interpolation=cv2.INTER_CUBIC)
    video = cv2.warpAffine(video, np.asarray(row['G_after_body']), (1024, 1024), flags=cv2.INTER_CUBIC,
                           borderValue=(238, 241, 245))
    face_rgb = np.full((1024, 1024, 3), (238, 241, 245), np.uint8)
    alpha = face[:, :, 3:4].astype(float) / 255
    face_rgb = np.uint8(face[:, :, :3] * alpha + face_rgb * (1 - alpha))
    overlay = np.uint8(video * (1 - .5 * alpha) + face[:, :, :3] * (.5 * alpha))
    points, keep = np.asarray(row['pointsArt']), np.asarray(row['keep'])
    for i in range(len(points)):
        j = (i + 1) % len(points)
        if keep[i] and keep[j]:
            cv2.line(overlay, tuple(map(int, np.rint(points[i]))), tuple(map(int, np.rint(points[j]))), (30, 30, 230), 2)
    surface = contour(q, triangles)
    for first, second in surface['overlap_triangles']:
        area = shapely.Polygon(q[triangles[first]]).intersection(shapely.Polygon(q[triangles[second]]))
        center = tuple(map(int, np.rint([area.centroid.x, area.centroid.y])))
        cv2.circle(overlay, center, 8, (180, 20, 150), 2)
    images = [video, face_rgb, overlay]
    for col, image in enumerate(images):
        tile = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).crop((185, 150, 825, 700))
        sheet.paste(tile, (col * 640, row_index * 590 + 40))
    draw = ImageDraw.Draw(sheet)
    for col, title in enumerate(['영상', '맞춘 얼굴판', '겹침']):
        draw.text((col * 640 + 12, row_index * 590 + 8), title, font=font, fill=(30, 30, 30))
    if solution is not None:
        label = f'보정 {args.solution + 1}' + (' · 정체' if solution['diagnostics']['line_search_stalled'] else '')
        draw.text((850, row_index * 590 + 8), label, font=font, fill=(30, 30, 30))
    draw.text((3 * 640 - 500, row_index * 590 + 8), '보라: 접힌 곳', font=font, fill=(30, 30, 30))
    draw.text((130, row_index * 590 + 8), f'{frame}번', font=font, fill=(30, 30, 30))
    draw.text((225, row_index * 590 + 8), f'뒷면 {stats[frame]["back_faces"]} · 겹침 {stats[frame]["overlap_pairs"]}',
              font=font, fill=(30, 30, 30))
sheet.save(args.output)
print(f'SAVED {args.output}: 5장', flush=True)
