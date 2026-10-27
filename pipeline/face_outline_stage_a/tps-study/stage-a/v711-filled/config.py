from pathlib import Path as _ProjectPath
PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[5]
from pathlib import Path
V=Path(__file__).resolve().parent;A=V.parent;O=A.parent;P=O.parent;T=PROJECT_ROOT;R=PROJECT_ROOT
NAMES=['left','right','up','down','up-left','up-right','down-left','down-right']
def near_sides(yaw):return ['right'] if yaw<0 else ['left'] if yaw>0 else ['left','right']
