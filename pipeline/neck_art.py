"""Keep neck artwork within the transparent boundary of the original canvas."""

import numpy as np


def clear_background(neck, reference):
    result = np.array(neck, dtype=np.uint8, copy=True)
    result[:, :, 3][np.asarray(reference)[:, :, 3] == 0] = 0
    return result
