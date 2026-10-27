import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from input_masks import V,NAMES

def boundary(mask):return cv2.findContours(np.uint8(mask),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]

def draw_pose(name):
    dest=V/'inputs'/name;rgb=np.array(Image.open(dest/'reference-frame.png').convert('RGB'));m=json.loads((dest/'manifest.json').read_text())
    skin=np.array(Image.open(dest/'skin-mask.png'))>127;band=np.array(Image.open(dest/'hairband.png'))>127
    out=rgb.astype(float);out[band]=.83*out[band]+.17*np.array([60,120,255]);out=np.uint8(np.clip(out,0,255))
    cv2.drawContours(out,boundary(band),-1,(45,125,255),1,cv2.LINE_AA)
    for side in ['left','right']:
        ear=np.array(Image.open(dest/f'ear-{side}-mask.png'))>127
        out[ear]=np.uint8(.76*out[ear]+.24*np.array([255,65,65]));cv2.drawContours(out,boundary(ear),-1,(255,65,65),2,cv2.LINE_AA)
    cv2.drawContours(out,boundary(skin),-1,(65,255,110),2,cv2.LINE_AA)
    Image.fromarray(out).save(dest/'C1-overlay.png')
    return Image.fromarray(out),m

def main():
    S=V/'sheets';S.mkdir(exist_ok=True);font=ImageFont.load_default(size=17)
    w,h=700,620;sheet=Image.new('RGB',(w*4,h*2+40),(30,36,48));draw=ImageDraw.Draw(sheet)
    draw.text((8,8),'C1: GREEN skin contour | RED ear mask | BLUE hair band | ear <1958px: invisible (recorded only)',fill='white',font=font)
    for i,name in enumerate(NAMES):
        im,m=draw_pose(name);x=i%4*w;y=i//4*h+40
        sheet.paste(im.crop((150,150,850,720)),(x,y+50))
        ears='  '.join(f"{side[0].upper()}:{e['maskPixels']}px {'visible' if e['visible'] else 'invisible'}" for side,e in m['ears'].items())
        draw.text((x+7,y+5),name,fill='white',font=font);draw.text((x+7,y+27),ears,fill='white',font=font)
    path=S/'C1-eight-pose-mask-overlays.png';sheet.save(path)
    print('C1 SHEET',str(path),sheet.size,flush=True)
if __name__=='__main__':main()
