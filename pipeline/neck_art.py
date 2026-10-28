"""Trim shoulder fill above the original collar while retaining covered skin."""

import cv2
import numpy as np
from PIL import Image


def removal_mask(neck, face, sweater):
    neck = np.asarray(neck) > 0
    face = np.asarray(face) > 0
    sweater = np.asarray(sweater) > 0
    expanded = cv2.dilate(neck.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    has_sweater = sweater.any(axis=0)
    top = np.argmax(sweater, axis=0)
    rows = np.arange(neck.shape[0])[:, None]
    return ~expanded & ~face & has_sweater[None, :] & (rows < top[None, :] - 2)


def load_removal_mask(root):
    masks = [np.asarray(Image.open(root / 'sam3_masks' / name).convert('L'))
             for name in ('neck_mask.png', 'face-skin_mask.png', 'sweater_mask.png')]
    return removal_mask(*masks)


def clear_background(neck, remove):
    result = np.array(neck, dtype=np.uint8, copy=True)
    result[:, :, 3][remove] = 0
    return result
