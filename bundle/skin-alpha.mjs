// Compose only the face tile; the authored atlas and every hair tile stay intact.
export function applySkinAlpha(atlas,mask,rig){
 const skin=rig.meshes.find(m=>m.id==='face-skin'),[x,y,w,h]=skin.atlas,[bx,by]=skin.bbox;
 const tile=document.createElement('canvas');tile.width=w;tile.height=h;
 const t=tile.getContext('2d');t.drawImage(atlas,x,y,w,h,0,0,w,h);
 t.globalCompositeOperation='destination-in';t.drawImage(mask,-bx,-by);
 const result=document.createElement('canvas');result.width=atlas.width;result.height=atlas.height;
 const ctx=result.getContext('2d');ctx.drawImage(atlas,0,0);ctx.clearRect(x,y,w,h);ctx.drawImage(tile,x,y);
 return result;
}
