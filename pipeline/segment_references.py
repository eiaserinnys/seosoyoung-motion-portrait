"""SAM 3 reference masks. Cached by input SHA; no credentials in artifacts."""
import hashlib,json,os,sys
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from stage1_sam3_segment import call_sam3_on_crop
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'sam3_masks/references'
SPECS={
 'left-crown':('left_45_ref_720.png',(30,0,680,550),'hair',None,[(50,5,660,540)]),
 'right-crown':('right_45_ref_720.png',(30,0,680,550),'hair',[(180,200,1),(350,80,1),(100,330,1),(370,470,0)],None),
 'left-face':('left_45_ref_720.png',(130,200,555,620),'face of anime girl excluding ears neck and hair',[(280,490,1),(370,500,1),(270,570,1),(500,430,0),(360,650,0)],[(150,210,480,605)]),
 'right-face':('right_45_ref_720.png',(240,200,585,620),'face of anime girl excluding ears neck and hair',[(450,490,1),(350,500,1),(430,570,1),(220,430,0),(360,650,0)],[(265,210,565,605)]),
 'right-cranium':('left_45_ref_720.png',(430,120,675,505),'brown hair on side and back of head behind ear',[(580,310,1),(570,380,1),(570,450,1),(500,430,0),(650,480,0)],[(460,130,650,500)]),
 'left-cranium':('right_45_ref_720.png',(40,120,290,550),'dark brown hair on side and back of head behind ear',[(120,300,1),(130,380,1),(150,480,1),(210,430,0),(50,530,0)],[(60,130,270,540)]),
}
FACE_ROIS={
 'up':((185,150,540,515),[(360,400,1),(300,450,1),(420,450,1),(360,505,0)]),
 'up-left':((125,180,455,525),[(260,415,1),(240,475,1),(350,455,1),(300,515,0)]),
 'up-right':((275,180,610,530),[(490,415,1),(515,475,1),(410,465,1),(400,520,0)]),
 'down':((190,270,520,635),[(360,540,1),(300,560,1),(420,560,1),(360,630,0)]),
 'down-left':((135,250,445,630),[(250,520,1),(260,580,1),(340,560,1),(360,615,0)]),
 'down-right':((280,250,610,650),[(470,540,1),(450,600,1),(360,550,1),(370,640,0)]),
}
# Visible jaw-edge samples in the source references. SAM treats neck skin as
# part of the same object in pitched views; this explicit contour clips that
# connected neck while retaining the original SAM mask separately.
JAWS={
 'up':[(210,404),(232,428),(270,449),(315,466),(358,474),(399,466),(444,450),(480,432),(503,407)],
 'up-left':[(144,412),(151,438),(176,465),(213,499),(250,503),(296,493),(345,475),(384,452),(410,418)],
 'up-right':[(339,420),(360,455),(403,482),(453,501),(500,508),(541,503),(566,471),(586,446),(593,412)],
 'down':[(211,535),(233,567),(270,593),(312,611),(353,621),(388,610),(431,589),(470,559),(490,529)],
 'down-left':[(153,486),(165,526),(190,565),(224,598),(259,614),(294,606),(338,584),(381,559),(409,529)],
 'down-right':[(321,534),(348,574),(386,606),(423,629),(456,639),(486,619),(521,594),(550,563),(572,529)],
}
for direction,(crop,points) in FACE_ROIS.items():
 SPECS[direction+'-face']=(direction.replace('-','_')+'_ref_720.png',crop,
  'face skin of anime girl excluding ears neck and hair',points,None)
def main():
 OUT.mkdir(exist_ok=True,parents=True)
 manifest_path=OUT/'manifest.json'
 manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
 sys.path.insert(0,str(Path(os.environ['AGENT_COMMON_FILES_DIR'])/'lib'))
 from secret_loader import get_secret
 key=None
 for name,(file,crop,prompt,points,boxes) in SPECS.items():
  src=ROOT/'source/references'/file;sha=hashlib.sha256(src.read_bytes()).hexdigest()
  if name not in sys.argv[1:] and manifest.get(name,{}).get('sha256')==sha and (OUT/(name+'.png')).exists():continue
  if key is None:key=get_secret('shared/fal','api_key')
  mask,meta=call_sam3_on_crop(key,Image.open(src).convert('RGBA'),crop,prompt,points,boxes)
  Image.fromarray(mask).save(OUT/(name+'.png'))
  if name.removesuffix('-face') in JAWS:Image.fromarray(mask).save(OUT/(name+'-raw.png'))
  manifest[name]={**meta,'source':file,'sha256':sha,'endpoint':'fal-ai/sam-3/image'}
  manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
  print(name,meta['score'],meta['area_px'],flush=True)
 for direction,jaw in JAWS.items():
  name=direction+'-face';target=OUT/(name+'.png');raw=OUT/(name+'-raw.png')
  if not raw.exists():raw.write_bytes(target.read_bytes())
  mask=np.array(Image.open(raw));clip=np.zeros_like(mask)
  cv2.fillPoly(clip,[np.array([(0,0),(0,jaw[0][1]),*jaw,(719,jaw[-1][1]),(719,0)],np.int32)],255)
  Image.fromarray(np.minimum(mask,clip)).save(target)
  manifest[name]['jawConstraint']=jaw
  manifest[name]['maskSha256']=hashlib.sha256(target.read_bytes()).hexdigest()
 manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
