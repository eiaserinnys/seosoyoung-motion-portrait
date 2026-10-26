"""Cache SAM 3 eye observations; camera normalization never changes facial shape.

Input frames are resized to the registration's 720px image coordinates, then
camera similarity is inverted once. Masks retain that normalized 720px frame;
the measurement stage adds the artwork offset (152, 20).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
from PIL import Image, ImageDraw
from stage1_sam3_segment import call_sam3_on_crop

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'sam3_masks/references/seedance'
REG = ROOT/'source/viewer/ponytail-seedance-registration.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def camera_inverse(pose, origin):
    r = np.radians(pose[2])
    linear = np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]])/np.exp(pose[3])
    return np.column_stack((linear, origin-linear@pose[4:6]))


def select_frames(inputs):
    data = json.loads(REG.read_text())
    origin = np.array(data['frontConstraint']['landmarks'])[:2].mean(0)
    rows = []
    for k in data['keys']:
        rows.append(dict(name=k['name'], frame=k['sourceFrame'], yaw=k['yaw'], pitch=k['pitch'],
                         normalization=k['normalization'], image=k['image'],
                         eyeCenters=k['landmarks'][:2], role='representative',
                         requested=k['requestedAngles']))
    chosen = {r['frame'] for r in rows}
    # Three independent near-down frames, intermediate diagonals, then plateaus.
    targets = [(0,-28),(2,-28),(-3,-28),(-20,15),(20,15),(-20,-15),(20,-15),
               (-18,0),(18,0),(0,15),(0,-18),(-36,0),(36,0),(0,28),(0,-19)]
    video = data['video']
    for i, (y,p) in enumerate(targets):
        candidates = [v for v in video if v['frame'] not in chosen and
                      (i>=3 or (abs(v['pose'][0])<5 and v['pose'][1]<-24))]
        def score(v):
            d = ((v['pose'][0]-y)/45)**2+((v['pose'][1]-p)/30)**2
            return (d, v['frame'])
        v = min(candidates, key=score)
        chosen.add(v['frame'])
        a = camera_inverse(v['pose'], origin)
        target = ROOT/'source/references/seedance/frames'/f"frame_{v['frame']:04d}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(inputs/'frames'/target.name, target)
        eyes = np.array(v['landmarks'])[:2]@a[:,:2].T+a[:,2]
        rows.append(dict(name=f"aux-{v['frame']:04d}", frame=v['frame'], yaw=v['pose'][0],
                         pitch=v['pose'][1], normalization=a.tolist(), image=str(target.relative_to(ROOT)),
                         eyeCenters=eyes.tolist(), role='heldout' if i>=12 else 'training'))
    return rows


def normalized_image(row):
    image = Image.open(ROOT/row['image']).convert('RGBA')
    if row['role']=='representative':
        return image
    rgb = np.array(image.resize((720,720), Image.Resampling.BICUBIC))
    return Image.fromarray(cv2.warpAffine(rgb, np.array(row['normalization']), (720,720),
                                        flags=cv2.INTER_CUBIC, borderValue=(255,255,255,255)))


def prompts(center, side):
    x,y = center
    shift = -13 if side=='left' else 13
    x += shift
    crop = tuple(int(v) for v in (x-85,y-95,x+85,y+65))
    eye = (int(x-59),int(y-49),int(x+59),int(y+45))
    iris = (int(center[0]-36),int(y-47),int(center[0]+36),int(y+44))
    return crop, {
        'opening': ('anime eye eyeball including white sclera and iris',
                    [(int(x),int(y),1),(int(x-33),int(y),1),(int(x+33),int(y),1)], [eye]),
        'iris': ('colorful iris of eye', [(int(center[0]),int(y+5),1)], [iris]),
        'upper-lashes': ('black upper eyelash line', [(int(x),int(y-47),1)],
                         [(int(x-67),int(y-70),int(x+67),int(y-23))]),
        'brow': ('eyebrow', [(int(x),int(y-80),1)],
                 [(int(x-66),int(y-95),int(x+66),int(y-58))]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--only', nargs='*')
    parser.add_argument('--select-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = select_frames(args.inputs)
    selection = dict(schema=1, registrationSha256=digest(REG), coordinates='normalized 720px; artwork offset [152,20]', frames=rows)
    (OUT/'selection.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2)+'\n')
    sheet = Image.new('RGB', (960, len(rows)//4*200), '#eeeeee')
    for i,row in enumerate(rows):
        image = normalized_image(row)
        x,y = np.array(row['eyeCenters']).mean(0)
        tile = image.crop((int(x-125),int(y-105),int(x+125),int(y+65))).convert('RGB').resize((240,170))
        sheet.paste(tile, ((i%4)*240,(i//4)*200))
        ImageDraw.Draw(sheet).text(((i%4)*240+4,(i//4)*200+172), f"{row['name']} {row['yaw']:.1f}/{row['pitch']:.1f}", fill='black')
    (ROOT/'debug').mkdir(exist_ok=True)
    sheet.save(ROOT/'debug/sam3-eye-selection.png')
    if args.select_only:
        print(f'Selected {len(rows)} frames; pure down: '+str(sum(abs(r['yaw'])<5 and r['pitch']<-24 for r in rows)))
        return
    path = OUT/'manifest.json'
    manifest = json.loads(path.read_text()) if path.exists() else {}
    sys.path.insert(0,str(Path(os.environ['AGENT_COMMON_FILES_DIR'])/'lib'))
    from secret_loader import get_secret
    key = get_secret('shared/fal','api_key')
    jobs = []
    for row in rows:
        if args.only and row['name'] not in args.only: continue
        image = normalized_image(row)
        for side,center in zip(['left','right'],row['eyeCenters']):
            crop,specs = prompts(center,side)
            for kind,(prompt,points,boxes) in specs.items():
                name = f"{row['name']}-eye-{side}-{kind}"
                signature = dict(sourceSha256=digest(ROOT/row['image']), normalization=row['normalization'],
                                 crop=list(crop),prompt=prompt,points=points,boxes=boxes,upscale=2)
                fingerprint = hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
                if manifest.get(name,{}).get('inputSha256')==fingerprint and (OUT/(name+'.png')).exists():continue
                jobs.append((name,image,crop,prompt,points,boxes,signature,fingerprint))
    def run(job):
        name,image,crop,prompt,points,boxes,signature,fingerprint = job
        mask,meta = call_sam3_on_crop(key,image,crop,prompt,points,boxes,upscale=2)
        Image.fromarray(mask).save(OUT/(name+'.png'))
        return name, {**meta,**signature,'inputSha256':fingerprint,'maskSha256':digest(OUT/(name+'.png')),
                      'endpoint':'fal-ai/sam-3/image'}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run,j) for j in jobs]):
            name,meta = future.result()
            manifest[name] = meta
            path.write_text(json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
            print(name, meta['score'], meta['area_px'], flush=True)


if __name__=='__main__':
    main()
