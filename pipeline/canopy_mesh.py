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
 x,y,w,h=bbox;ax,ay,_,_=atlas;step=12
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
 from stage3_build_atlas_and_rig import cranium_mesh
 from silhouette_mesh import silhouette_mesh
 path=Path(__file__).resolve().parents[1]/'bundle/assets/seosoyoung-rig.json'
 rig=json.loads(path.read_text())
 after=Image.open(path.parent/'texture_00.png').convert('RGBA')
 before=Image.open(path.parent/'texture_before_00.png').convert('RGBA')
 alpha=ImageChops.lighter(after.getchannel('A'),before.getchannel('A'))
 dome=json.loads((path.parents[2]/'source/viewer/dome-fit.json').read_text())
 seam_rows=dome['frontSeam'][::6]
 if seam_rows[-1]!=dome['frontSeam'][-1]:seam_rows.append(dome['frontSeam'][-1])
 seams=[[(row[side],row[0]) for row in seam_rows] for side in [1,2]]
 counts=lambda:[sum(len(m[key]) for m in rig['meshes']) for key in ['positions','triangles']]
 old=counts()
 for m in rig['meshes']:
  ax,ay,w,h=m['atlas'];box=(ax,ay,ax+w,ay+h)
  if m['id'] in IDS:
   m.update(canopy_mesh(m['bbox'],m['atlas'],rig['atlasSize'],alpha.crop(box)))
  elif m['id'].startswith('side-cranium-'):
   m.update(cranium_mesh(m['id'].split('-')[-1],m['bbox'],m['atlas']))
  elif m['id'].startswith(('back-hair-','tail-')):
   union=after.crop(box);union.putalpha(alpha.crop(box))
   m.update(silhouette_mesh(union,m['bbox'],m['atlas'],rig['atlasSize'],
            [(306.5,544.2),(691.5,542.6),(379,465),(626,465)],
            seams if m['id'].startswith('back-hair-') else (),max_area=800,simplify_tol=3.0))
 path.write_text(json.dumps(rig,ensure_ascii=False,separators=(',',':')))
 print(json.dumps({'before':old,'after':counts(),'changedHairMeshes':15}))
if __name__=='__main__':update()
