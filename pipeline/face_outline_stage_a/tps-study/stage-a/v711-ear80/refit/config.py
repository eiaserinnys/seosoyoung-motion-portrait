from pathlib import Path as _ProjectPath
PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[6]
from pathlib import Path
V=Path(__file__).resolve().parent;A=V.parent.parent;O=A.parent;P=O.parent;T=PROJECT_ROOT
NAMES=['left','right','up','down','up-left','up-right','down-left','down-right']
