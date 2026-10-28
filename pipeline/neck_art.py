"""Remove externally connected shoulder fill where the original artwork is dark."""

import cv2
import numpy as np
from PIL import Image


SKIN_FRACTION = 0.7
STILL_FRACTION = 0.6


def removal_mask(neck, still, sweater, origin=(0, 0)):
    neck = np.asarray(neck, dtype=np.uint8)
    still = np.asarray(still, dtype=np.uint8)
    sweater = np.asarray(sweater) > 0
    has_sweater = sweater.any(axis=0)
    collar_y = np.argmax(sweater, axis=0) - 2
    allowed = has_sweater[None, :] & (np.arange(neck.shape[0])[:, None] + origin[1] < collar_y[None, :])
    neck_luma = cv2.cvtColor(neck[:, :, :3], cv2.COLOR_RGB2GRAY)
    still_luma = cv2.cvtColor(still[:, :, :3], cv2.COLOR_RGB2GRAY)
    opaque = neck[:, :, 3] > 0
    median = np.zeros(neck.shape[1], dtype=np.float32)
    for x in range(neck.shape[1]):
        skin = neck_luma[:, x][(neck[:, x, 3] >= 128) & allowed[:, x]]
        if skin.size:
            median[x] = np.median(skin)
    candidate = (allowed & opaque & (neck_luma >= median[None, :] * SKIN_FRACTION)
                 & ((still_luma <= neck_luma * STILL_FRACTION) | (still[:, :, 3] < 128)))

    _, outside_labels, _, _ = cv2.connectedComponentsWithStats((~opaque).astype(np.uint8), connectivity=8)
    edge_labels = set(np.unique(np.r_[outside_labels[0], outside_labels[-1],
                                      outside_labels[:, 0], outside_labels[:, -1]])) - {0}
    outside = np.isin(outside_labels, list(edge_labels))
    touches_outside = cv2.dilate(outside.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    touches_outside[0, :] = touches_outside[-1, :] = True
    touches_outside[:, 0] = touches_outside[:, -1] = True
    count, labels, _, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=8)
    remove_labels = set(np.unique(labels[touches_outside])) - {0}
    remove = np.isin(labels, list(remove_labels))
    center_left, center_right = max(0, 495-origin[0]), min(neck.shape[1], 506-origin[0])
    if center_left < center_right and np.any(remove[:, center_left:center_right]):
        raise ValueError('지울 목 성분이 가운데 열에 닿습니다')
    return remove, {'candidate_pixels': int(candidate.sum()), 'candidate_components': count-1,
                    'removed_components': len(remove_labels), 'removed_pixels': int(remove.sum()),
                    'retained_pixels': int((candidate & ~remove).sum())}


def crop_inputs(root, bbox):
    x, y, width, height = bbox
    still = np.asarray(Image.open(root / 'source/character/still.png').convert('RGBA'))[y:y+height, x:x+width]
    sweater = np.asarray(Image.open(root / 'sam3_masks/sweater_mask.png').convert('L'))[:, x:x+width]
    return still, sweater


def clear_background(neck, remove):
    result = np.array(neck, dtype=np.uint8, copy=True)
    result[:, :, 3][remove] = 0
    return result
