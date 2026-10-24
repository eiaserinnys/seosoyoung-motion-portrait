import {strandWorld} from './side-strands.mjs';
import {hairSurface,domeSection,craniumPosition} from './hair-surface.mjs';
import {earWorld} from './stereo-attachments.mjs';
import {HEAD_CENTER,clamp,correction,correctedLift,artWorld} from './geometry.mjs';

const smooth=t=>{t=clamp(t,0,1);return t*t*(3-2*t);};

const smax=(a,b,k=8)=>(a+b+Math.sqrt((a-b)**2+k*k))/2;
// The front dome meets the rear shell at longitude ±90°, where Z is cz.
export function hairLift(point,p,relief=0) {
 const [x,y]=point,{cx,rx,cz}=domeSection(y);
 const s=clamp((x-HEAD_CENTER[0]-cx)/Math.max(1,rx),-1,1);
 const u=clamp((250-y)/192,0,1),cap=Math.sqrt(Math.max(.08,1-u*u));
 const lowerMix=smooth((y-230)/120),skinY=clamp(y,220,350);
 const zSkinMid=correctedLift([502,skinY],0,p)[2];
 const zCenter=(195*cap*(1-lowerMix)+(zSkinMid+12)*lowerMix)*correction(p,'CorrHairDepth');
 const zDome=cz+(zCenter-cz)*Math.sqrt(Math.max(0,1-s*s));
 const zFace=correctedLift([x,skinY],0,p)[2];
 const foreheadFade=(1-smooth((Math.abs(s)-.72)/.28))*smooth((y-210)/35);
 return [x-HEAD_CENTER[0],y-HEAD_CENTER[1],zDome+(smax(zDome,zFace+6)-zDome)*foreheadFade+relief];
}
const shellWorld=(point,p,relief=0)=>hairSurface(point,p,0,relief);
const fringeSurface=(point,p)=>artWorld(hairLift(point,p,0),p);
// One coordinate field owns every overlapping front-hair pixel. Layer order
// belongs to the renderer; it must never alter the rotated geometry.
export function canopyWorld(point,p) {
 const [x,y]=point,world=fringeSurface(point,p);
 const side=x<HEAD_CENTER[0]?'left':'right';
 const lockWeight=smooth((y-340)/45);
 if(lockWeight){
  const lock=templeWorld(point,side,p);
  for(let i=0;i<3;i++)world[i]+=(lock[i]-world[i])*lockWeight;
 }
 const free=smooth((y-245)/180),outer=smooth((Math.abs(x-HEAD_CENTER[0])-75)/70);
 const sway=9*free*free*(p.ParamHairFront*(1-outer)+p.ParamHairSide*outer);
 const delta=artWorld([sway,0,0],p),origin=artWorld([0,0,0],p);
 return world.map((v,i)=>v+delta[i]-origin[i]);
}
export const hairWorld=canopyWorld;
export const fringeWorld=canopyWorld;
export const craniumWeight=()=>1;
export function craniumWorld([x,y,lambda],side,p) {
 const world=craniumPosition([x,y,lambda],p);
 const {cx,rx}=domeSection(y),sign=Math.sign(lambda)||(side==='left'?-1:1);
 const crownBlend=1-smooth((y-345)/35);
 if(!crownBlend)return world;
 const seam=canopyWorld([HEAD_CENTER[0]+cx+sign*rx,y],p);
 if(Math.abs(Math.abs(lambda)-Math.PI/2)<1e-4)return world.map((v,i)=>v+(seam[i]-v)*crownBlend);
 const w=clamp((Math.abs(lambda)-Math.PI/2)/(Math.PI*.18),0,1);
 return [world[0],world[1]+(seam[1]-world[1])*crownBlend*(1-w),world[2]];
}
export function farSideWeight(side,p) {
  const angle=p.ParamAngleX*(side==='left'?-1:1);
  return smooth(angle/50)*correction(p,'CorrHairOcclusion');
}
function templeWorld(point,side,p) {
 // Hanging front locks retain the original forehead-depth attachment rather
 // than being flattened onto the narrower skull at ear height. Rotation moves
 // them in front of the cheek, uncovering the posterior ear naturally.
 const lock=strandWorld(point,side,p),fringe=fringeSurface(point,p),lower=smooth((point[1]-260)/147);
 const shell=fringe.map((v,i)=>v+(lock[i]-v)*lower);
 const hanging=smooth((point[1]-350)/57);
 for(let i=0;i<3;i++)shell[i]+=(lock[i]-shell[i])*hanging;
 return shell;
}
export function sideWorld(point,side,p,ear=false){return ear?earWorld(point,side,p):canopyWorld(point,p);}
export function backWorld(point,p) {
 const back=shellWorld(point,p,-4),upper=1-smooth((point[1]-380)/85);
 if(!upper)return back;
 // Background is painted first, so the upper backing can share the dome
 // exactly. A depth-only inset opens a projected crack at the far temple.
 const canopy=artWorld(hairLift(point,p,0),p);
 return back.map((v,i)=>v+(canopy[i]-v)*upper);
}
