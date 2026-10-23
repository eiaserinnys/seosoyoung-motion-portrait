"""SAM 3 reference masks. Cached by input SHA; no credentials in artifacts."""
import hashlib,json,os,sys
from pathlib import Path
from PIL import Image
from stage1_sam3_segment import call_sam3_on_crop
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'sam3_masks/references'
SPECS={
 'left-crown':('left_45_ref_720.png',(30,0,680,550),'hair',None,[(50,5,660,540)]),
 'right-crown':('right_45_ref_720.png',(30,0,680,550),'hair',None,[(50,5,660,540)]),
 'left-face':('left_45_ref_720.png',(130,200,555,620),'face of anime girl excluding ears neck and hair',[(280,490,1),(370,500,1),(270,570,1),(500,430,0),(360,650,0)],[(150,210,480,605)]),
 'right-face':('right_45_ref_720.png',(240,200,585,620),'face of anime girl excluding ears neck and hair',[(450,490,1),(350,500,1),(430,570,1),(220,430,0),(360,650,0)],[(265,210,565,605)]),
 'right-cranium':('left_45_ref_720.png',(430,120,675,505),'brown hair on side and back of head behind ear',[(580,310,1),(570,380,1),(570,450,1),(500,430,0),(650,480,0)],[(460,130,650,500)]),
 'left-cranium':('right_45_ref_720.png',(40,120,290,550),'dark brown hair on side and back of head behind ear',[(120,300,1),(130,380,1),(150,480,1),(210,430,0),(50,530,0)],[(60,130,270,540)]),
}
def main():
 OUT.mkdir(exist_ok=True,parents=True)
 manifest_path=OUT/'manifest.json'
 manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
 sys.path.insert(0,str(Path(os.environ['AGENT_COMMON_FILES_DIR'])/'lib'))
 from secret_loader import get_secret
 key=get_secret('shared/fal','api_key')
 for name,(file,crop,prompt,points,boxes) in SPECS.items():
  src=ROOT/'source/references'/file;sha=hashlib.sha256(src.read_bytes()).hexdigest()
  if name not in sys.argv[1:] and manifest.get(name,{}).get('sha256')==sha and (OUT/(name+'.png')).exists():continue
  mask,meta=call_sam3_on_crop(key,Image.open(src).convert('RGBA'),crop,prompt,points,boxes)
  Image.fromarray(mask).save(OUT/(name+'.png'))
  manifest[name]={**meta,'source':file,'sha256':sha,'endpoint':'fal-ai/sam-3/image'}
  manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
  print(name,meta['score'],meta['area_px'],flush=True)
if __name__=='__main__':main()
