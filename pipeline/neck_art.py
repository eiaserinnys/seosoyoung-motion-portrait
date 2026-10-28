"""Restore visible neck texels from the original artwork, preserving covered fill."""

import cv2
import numpy as np
from PIL import Image


def apply_reference(neck, still, sweater, face, necklace, hair, origin=(0, 0)):
    neck = np.asarray(neck, dtype=np.uint8)
    still = np.asarray(still, dtype=np.uint8)
    sweater = np.asarray(sweater) > 0
    has_sweater = sweater.any(axis=0)
    collar_y = np.argmax(sweater, axis=0) - 2
    remaining = has_sweater[None, :] & (np.arange(neck.shape[0])[:, None] + origin[1] < collar_y[None, :])
    zones = {}
    for name, mask in [('face', face), ('necklace', necklace), ('hair', hair)]:
        zones[name] = remaining & np.asarray(mask, dtype=bool)
        remaining &= ~zones[name]
    zones['background'] = remaining & (still[:, :, 3] < 128)
    zones['visible'] = remaining & ~zones['background']
    result = neck.copy()
    result[:, :, 3][zones['background']] = 0
    result[zones['visible']] = still[zones['visible']]
    keep = np.where(zones['background'], 0, 255).astype(np.uint8)
    counts = {name: int(mask.sum()) for name, mask in zones.items()}
    counts['outside_collar_rule'] = int(neck.shape[0] * neck.shape[1] - sum(counts.values()))
    # Only repair isolated transparent neck texels surrounded by opaque neck art.
    # Preserved background cutouts and non-neck mesh seams remain untouched.
    opaque = result[:, :, 3] == 255
    isolated = (result[1:-1, 1:-1, 3] == 0) & (keep[1:-1, 1:-1] == 255)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx or dy:
                isolated &= opaque[1+dy:opaque.shape[0]-1+dy, 1+dx:opaque.shape[1]-1+dx]
    ys, xs = np.where(isolated)
    for y, x in zip(ys + 1, xs + 1):
        neighborhood = result[y-1:y+2, x-1:x+2, :3].reshape(9, 3)
        result[y, x, :3] = np.median(np.delete(neighborhood, 4, axis=0), axis=0).astype(np.uint8)
        result[y, x, 3] = 255
    counts['filled_isolated_holes'] = [[int(x + 1 + origin[0]), int(y + 1 + origin[1])] for y, x in zip(ys, xs)]
    counts['changed_pixels'] = int(np.count_nonzero(np.any(result != neck, axis=2)))
    return result, keep, counts


def crop_inputs(root, bbox):
    x, y, width, height = bbox
    masks = root / 'sam3_masks'
    load = lambda path: np.asarray(Image.open(path).convert('L')) > 0
    still = np.asarray(Image.open(root / 'source/character/still.png').convert('RGBA'))[y:y+height, x:x+width]
    sweater = load(masks / 'sweater_mask.png')[:, x:x+width]
    face = load(masks / 'face-skin_mask.png')
    necklace = load(masks / 'necklace_mask.png')
    hair = np.zeros_like(face)
    for path in sorted(masks.glob('*_mask.png')):
        if 'hair' in path.name or path.name == 'bangs-front_mask.png':
            hair |= load(path)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    face = cv2.dilate(face.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    necklace = cv2.dilate(necklace.astype(np.uint8), kernel) > 0
    hair = cv2.dilate(hair.astype(np.uint8), kernel) > 0
    crop = np.s_[y:y+height, x:x+width]
    return still, sweater, face[crop], necklace[crop], hair[crop]
