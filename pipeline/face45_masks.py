"""Sample side-family SAM masks in the approved r11 canvas registration."""
import json
import sys

import numpy as np
from PIL import Image

from fit_outline import ROOT, interval, main_component


def sample_rows(rows):
    result = {}
    for name in ['left67', 'left90', 'right67', 'right90']:
        file = name.replace('67', '_67').replace('90', '_90')+'-face.png'
        mask = main_component(np.array(Image.open(ROOT/'sam3_masks/references'/file)))
        result[name] = []
        for y in rows:
            ty = round(420+(y-420)*195/196) if y >= 420 else y
            result[name].append([y, *[x+152 for x in interval(mask, ty-20)]])
    return result


if __name__ == '__main__':
    print(json.dumps(sample_rows(json.load(sys.stdin))))
