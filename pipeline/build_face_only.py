"""Refit facial data and refresh viewer HTML without rebuilding any hair asset."""
from pathlib import Path
import shutil
from fit_features import build as fit_features
from fit_outline import build as fit_outline
def build():
    root=Path(__file__).resolve().parents[1]
    fit_features();fit_outline()
    shutil.copyfile(root/'source/viewer/index.html',root/'bundle/index.html')
    print('PASS face-only fits and viewer HTML; hair/atlas/rig untouched')
if __name__=='__main__':build()
