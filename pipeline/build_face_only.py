"""Refit facial data and refresh viewer HTML without rebuilding any hair asset."""
from pathlib import Path
import shutil
from fit_features import build as fit_features
from fit_outline import build as fit_outline
def build():
    root=Path(__file__).resolve().parents[1]
    fit_features();fit_outline()
    if (root/'source/viewer/ponytail-seedance-registration.json').exists():
        from optimize_3d_2d import main as solve
        from solve_to_bundle import main as compile_solve
        solve();compile_solve()
        from build_skin_fill import build as skin_fill
        skin_fill()
    shutil.copyfile(root/'source/viewer/index.html',root/'bundle/index.html')
    print('PASS face-only fits and viewer HTML; hair/atlas/rig untouched')
if __name__=='__main__':build()
