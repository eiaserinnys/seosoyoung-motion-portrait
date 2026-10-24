import {strandWorld} from './side-strands.mjs';
import {hairLift,pitchHairDepth,fringeRotation,hairPitchShift,domeSection,hairSurface,craniumPosition} from './hair-surface.mjs';
import {earWorld} from './stereo-attachments.mjs';
import {outlinePitchShift} from './face-outline.mjs';
import {HEAD_CENTER,clamp,correction,artWorld} from './geometry.mjs';
export {hairLift,pitchHairDepth,fringeRotation} from './hair-surface.mjs';

const smooth=t=>{t=clamp(t,0,1);return t*t*(3-2*t);};

const shellWorld=(point,p,relief=0)=>hairSurface(point,p,0,relief);
export function referenceFringe(point,p,relief=0) {
 const local=hairLift(point,p,relief);
 if(!p.ParamAngleY)return artWorld(local,p);
 const unrolled={...p,ParamAngleZ:0};
 const a=artWorld(local,unrolled),b=artWorld([local[0],local[1],pitchHairDepth(point,p,relief)],unrolled);
 const q=[a[0],b[1],a[2]];
 if(!p.ParamAngleZ)return q;
 return artWorld(q.map((v,i)=>v-(i<2?HEAD_CENTER[i]:0)),{...p,ParamAngleX:0,ParamAngleY:0});
}
// Approved perimeter and lower locks stay on the r15 surface. Only the
// crown interior consumes the inferred depth and measured 2D residual.
export function fringeSurface(point,p,relief=0){
 const approved=referenceFringe(point,p,relief);
 if(!p.ParamAngleY)return approved;
 const {cx,rx}=domeSection(point[1]),u=clamp((point[0]-502-cx)/Math.max(1,rx),-1,1);
 const t=clamp((point[1]-70)/(310-70),0,1),envelope=t*(1-t)*(1-u*u)**2;
 if(!envelope)return approved;
 const rotated=fringeRotation(point,p,relief),shift=hairPitchShift(point,p,rotated);
 return approved.map((v,i)=>v+(rotated[i]-v)*4*envelope+shift[i]*envelope);
}
// One coordinate field owns every overlapping front-hair pixel. Layer order
// belongs to the renderer; it must never alter the rotated geometry.
function canopy(point,p,surface) {
 const [x,y]=point,world=surface(point,p);
 const side=x<HEAD_CENTER[0]?'left':'right';
 const lower=smooth((y-405)/55),xStart=126-42*lower,width=14;
 const lockWeight=smooth((Math.abs(x-HEAD_CENTER[0])-xStart)/width)*smooth((y-340)/67);
 if(lockWeight){
  const lock=templeWorld(point,side,p,surface);
  for(let i=0;i<3;i++)world[i]+=(lock[i]-world[i])*lockWeight;
 }
 const free=smooth((y-245)/180),outer=smooth((Math.abs(x-HEAD_CENTER[0])-75)/70);
 const sway=9*free*free*(p.ParamHairFront*(1-outer)+p.ParamHairSide*outer);
 const delta=artWorld([sway,0,0],p),origin=artWorld([0,0,0],p);
 return world.map((v,i)=>v+delta[i]-origin[i]);
}
export const canopyWorld=(point,p)=>canopy(point,p,fringeSurface);
const referenceCanopy=(point,p)=>canopy(point,p,referenceFringe);
export const hairWorld=canopyWorld;
export const fringeWorld=canopyWorld;
export const craniumWeight=()=>1;
export function craniumWorld([x,y,lambda],side,p) {
 const world=craniumPosition([x,y,lambda],p);
 const {cx,rx}=domeSection(y),sign=Math.sign(lambda)||(side==='left'?-1:1);
 const edge=[HEAD_CENTER[0]+cx+sign*rx,y],horizonY=referenceFringe(edge,p)[1];
 const crownBlend=1-smooth((y-345)/35);
 if(!crownBlend)return [world[0],horizonY,world[2]];
 const seam=referenceCanopy(edge,p),yOut=horizonY+(seam[1]-horizonY)*crownBlend;
 if(Math.abs(Math.abs(lambda)-Math.PI/2)<1e-4)return [world[0]+(seam[0]-world[0])*crownBlend,yOut,world[2]+(seam[2]-world[2])*crownBlend];
 return [world[0],yOut,world[2]];
}
export const craniumRotation=craniumWorld;
export function farSideWeight(side,p) {
  const angle=p.ParamAngleX*(side==='left'?-1:1);
  return smooth(angle/50)*correction(p,'CorrHairOcclusion');
}
function templeWorld(point,side,p,surface) {
 const inner=side==='left'?379:626;
 const outward=Math.max(0,(point[0]-inner)*(side==='left'?-1:1));
 const join=(1-smooth(outward/30))*smooth((point[1]-380)/70);
 // Hanging front locks retain the original forehead-depth attachment rather
 // than being flattened onto the narrower skull at ear height. Rotation moves
 // them in front of the cheek, uncovering the posterior ear naturally.
 const lock=strandWorld(point,side,p),fringe=surface(point,p),lower=smooth((point[1]-260)/147);
 const shell=fringe.map((v,i)=>v+(lock[i]-v)*lower);
 const hanging=smooth((point[1]-350)/57);
 for(let i=0;i<3;i++)shell[i]+=(lock[i]-shell[i])*hanging;
 const shift=outlinePitchShift(point,p);
 shell[0]+=shift[0]*join;
 return shell;
}
export function sideWorld(point,side,p,ear=false){return ear?earWorld(point,side,p):canopyWorld(point,p);}
export function backWorld(point,p) {
 const back=shellWorld(point,p,-4),upper=1-smooth((point[1]-380)/85);
 if(!upper)return back;
 // Background is painted first, so the upper backing can share the dome
 // exactly. A depth-only inset opens a projected crack at the far temple.
 const canopy=referenceFringe(point,p,0);
 return back.map((v,i)=>v+(canopy[i]-v)*upper);
}
