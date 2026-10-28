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
    necklace = cv2.dilate(necklace.astype(np.uint8), kernel) > 0
    hair = cv2.dilate(hair.astype(np.uint8), kernel) > 0
    crop = np.s_[y:y+height, x:x+width]
    return still, sweater, face[crop], necklace[crop], hair[crop]
