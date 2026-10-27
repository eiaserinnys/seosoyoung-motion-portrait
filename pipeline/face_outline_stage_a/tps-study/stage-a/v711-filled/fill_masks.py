"""Fill only enclosed SAM omissions, preserve external silhouette and approved left."""
import hashlib,json,shutil,sys
import numpy as np,cv2
from PIL import Image
from scipy.ndimage import binary_fill_holes,label
from config import V,A,NAMES
sys.path.insert(0,str(A));from fit_inputs import fit_pose
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def external(mask):
 c,_=cv2.findContours(np.uint8(mask),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
 out=np.zeros(mask.shape,np.uint8);cv2.drawContours(out,c,-1,1,1);return out
def filled(mask):
 out=binary_fill_holes(mask);assert not (mask&~out).any();assert np.array_equal(external(mask),external(out));assert not (binary_fill_holes(out)&~out).any();return out
def main():
 cv2.setNumThreads(1);snap=V/'before-inputs';snap.mkdir()
 for n in ['inputs','tps']:shutil.copytree(A/n,snap/n,symlinks=True)
 for n in ['design.md','C1-input-audit.json']:shutil.copyfile(A/n,snap/n)
 left={str(p.relative_to(A)):sha(p) for p in (A/'inputs/left').iterdir() if p.is_file()};left['tps/left.json']=sha(A/'tps/left.json');rows=[]
 for name in NAMES:
  if name=='left':continue
  d=A/'inputs'/name;row={'name':name,'masks':{}}
  for key in ['reference-mask','skin-mask']:
   path=d/f'{key}.png';before=np.array(Image.open(path))>127;after=filled(before);added=after&~before
   row['masks'][key]={'beforePixels':int(before.sum()),'afterPixels':int(after.sum()),'filledPixels':int(added.sum()),'holeComponents':int(label(added)[1]),'externalBoundaryChangedPixels':int((external(before)!=external(after)).sum()),'remainingHolePixels':int((binary_fill_holes(after)&~after).sum())}
   Image.fromarray(np.uint8(after)*255).save(path)
  meta=json.loads((d/'manifest.json').read_text());meta.update(skinPixels=row['masks']['skin-mask']['afterPixels'],unionPixels=row['masks']['reference-mask']['afterPixels'],maskRevision='enclosed SAM eye/brow/mouth holes filled; exterior unchanged',holeFill=row['masks'])
  (d/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
  print('HOLE FILL',json.dumps(row),flush=True);rows.append(row)
 (V/'mask-fill-audit.json').write_text(json.dumps({'poses':rows,'leftSHA256':left,'leftUntouched':True},indent=2)+'\n')
 # The existing approved objective, weights, Jacobian and initialization are reused exactly.
 for row in rows:
  n=row['name'];(A/'inputs'/n/'tps.json').unlink();row['TPS']=fit_pose(n,max_nfev=400)
  assert row['TPS']['maxNfev']==400 and row['TPS']['secondDifferenceLambda']==.25 and row['TPS']['controlGrid']==[8,8]
  (V/'mask-fill-audit.json').write_text(json.dumps({'poses':rows,'leftSHA256':left,'leftUntouched':True},indent=2)+'\n')
 for path,h in left.items():assert sha(A/path)==h
 for n in NAMES:
  for key in ['hairband','reference-frame','ear-left-mask','ear-right-mask']:
   assert sha(A/'inputs'/n/f'{key}.png')==sha(snap/'inputs'/n/f'{key}.png')
 lines=['# 내부 구멍 교정 입력','', 'left는 그대로 보존하고 나머지7포즈의 reference-mask와 skin-mask 내부 구멍을 채웠다. 외부 윤곽 변경은 모두0px다. 승인된 목적식·가중·초기화·정칙화로 TPS를 포즈마다 한 번씩 max400으로 다시 맞췄다. TPS 단독 IoU나 제어점 변위를 새 기준으로 쓰지 않았다.','', '|포즈|reference 전/후 면적|채운 면적/구멍 수|skin 전/후 면적|외부 윤곽 변경|TPS 종료 상태|nfev|','|:--|:--|:--|:--|--:|:--|--:|']
 for row in rows:
  ref=row['masks']['reference-mask'];skin=row['masks']['skin-mask'];f=row['TPS'];lines.append(f"|{row['name']}|{ref['beforePixels']}/{ref['afterPixels']}|{ref['filledPixels']}/{ref['holeComponents']}|{skin['beforePixels']}/{skin['afterPixels']}|0|{f['message']} (success={f['success']})|{f['nfev']}|")
 lines+=['', '이전 입력과 TPS는before-inputs/에 보존했다. left 입력 전체와TPS SHA,귀 마스크·프레임·기존 띠8포즈의 바이트 불변을 확인했다. 설정 변경이나 새 TPS 후보는 없다. 다음 단계는 얼굴 솔브→귀 적합→활성 집합→C2→조건부 C3다.']
 (V/'inputs.md').write_text('\n'.join(lines)+'\n');print('ALL7 HOLES FILLED AND TPS REFITTED; LEFT AND EXTERIOR UNCHANGED',flush=True)
if __name__=='__main__':main()
