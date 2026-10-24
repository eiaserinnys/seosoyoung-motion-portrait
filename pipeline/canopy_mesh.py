"""Shared canvas lattice for overlapping hair textures.

Identical world samples alone do not guarantee identical raster interpolation:
overlapping parts must also share cell vertices and the diagonal direction.
Texture alpha keeps the original silhouette and every isolated hair strand.
"""
import json,math
from pathlib import Path
from PIL import Image,ImageChops
IDS={'hair-top','bangs-left','bangs-right','side-hair-left','side-hair-right','hairpin'}
def canopy_mesh(bbox,atlas,size,alpha):
 x,y,w,h=bbox;ax,ay,_,_=atlas;step=8
 xs=sorted({x,x+w,*range(math.ceil(x/step)*step,x+w,step)})
 ys=sorted({y,y+h,*range(math.ceil(y/step)*step,y+h,step)})
 positions=[[px,py] for py in ys for px in xs]
 uvs=[[(ax+px-x)/size,(ay+py-y)/size] for px,py in positions]
 triangles=[];cols=len(xs)
 for j in range(len(ys)-1):
  for i in range(cols-1):
   if not alpha.crop((xs[i]-x,ys[j]-y,xs[i+1]-x,ys[j+1]-y)).getbbox():continue
   a=j*cols+i;b=a+1;c=a+cols;d=c+1
   triangles.extend([[a,b,c],[b,d,c]])
 used=sorted({i for t in triangles for i in t});indices={old:new for new,old in enumerate(used)}
 return dict(positions=[positions[i] for i in used],uvs=[uvs[i] for i in used],triangles=[[indices[i] for i in t] for t in triangles],topology='shared-canopy-lattice')
def update():
 path=Path(__file__).resolve().parents[1]/'bundle/assets/seosoyoung-rig.json'
 rig=json.loads(path.read_text())
 alpha=ImageChops.lighter(Image.open(path.parent/'texture_00.png').getchannel('A'),Image.open(path.parent/'texture_before_00.png').getchannel('A'))
 for m in rig['meshes']:
  if m['id'] in IDS:
   ax,ay,w,h=m['atlas'];m.update(canopy_mesh(m['bbox'],m['atlas'],rig['atlasSize'],alpha.crop((ax,ay,ax+w,ay+h))))
 path.write_text(json.dumps(rig,ensure_ascii=False,separators=(',',':')))
 print('PASS shared lattice for six canopy layers')
if __name__=='__main__':update()
