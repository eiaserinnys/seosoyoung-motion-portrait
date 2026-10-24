import {strandWorld} from './side-strands.mjs';
import {hairLift,pitchHairDepth,domeSection,hairSurface,craniumPosition,sphereEase,pitchEase,hairPoseValue} from './hair-surface.mjs';
import {earWorld} from './stereo-attachments.mjs';
import {outlinePitchShift} from './face-outline.mjs';
import {HEAD_CENTER,clamp,correction,artWorld} from './geometry.mjs';
import {DOME} from './dome-data.mjs';
export {hairLift,pitchHairDepth} from './hair-surface.mjs';

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
// H1 keeps the authored yaw X/Z. H5 changes only the free fringe's Y.
export function fringeSurface(point,p,relief=0){
 const local=hairLift(point,p,relief);
 if(!p.ParamAngleY)return artWorld(local,p);
 const unrolled={...p,ParamAngleZ:0};
 const a=artWorld(local,unrolled),b=artWorld([local[0],local[1],pitchHairDepth(point,p,relief)],unrolled);
 let yWorld=b[1];
 if(point[1]>230){
  const neutralY=artWorld(local,{...unrolled,ParamAngleY:0})[1];
  const kappa=DOME.tipLag.kappa+hairPoseValue(DOME.tipLag.poses,p,pose=>pose.kappa-DOME.tipLag.kappa);
  yWorld=neutralY+(yWorld-neutralY)*(1-(1-kappa)*sphereEase((point[1]-230)/190));
 }
 const q=[a[0],yWorld,a[2]];
 if(!p.ParamAngleZ)return q;
 return artWorld(q.map((v,i)=>v-(i<2?HEAD_CENTER[i]:0)),{...p,ParamAngleX:0,ParamAngleY:0});
}
// One coordinate field owns every overlapping front-hair pixel. Layer order
// belongs to the renderer; it must never alter the rotated geometry.
function canopy(point,p,surface) {
 const [x,y]=point,world=surface(point,p);
 const smooth=t=>pitchEase(t,p);
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
 const point=[x,y,lambda];
 if(!p.ParamAngleY)return legacyCranium(point,side,p);
 const world=craniumPosition(point,p),weight=sphereEase(Math.abs(p.ParamAngleY*correction(p,'CorrTurn'))/30);
 const noPosterior=craniumPosition(point,p,false),posteriorY=world[1]-noPosterior[1];
 const {cx,rx}=domeSection(y),sign=lambda<0?-1:1,edge=[HEAD_CENTER[0]+cx+sign*rx,y];
 const fringe=fringeSurface(edge,p),seam=canopyWorld(edge,p),crownBlend=1-pitchEase((y-345)/35,p);
 // Pitch the rear horizon with the front shell, then add only the bounded
 // rear residual. Raw deep-skull pitch Y would form a hanging flap.
 const yOut=fringe[1]+(seam[1]-fringe[1])*crownBlend+posteriorY;
 const collar=(1-sphereEase(Math.abs(Math.abs(lambda)-Math.PI/2)/(Math.PI/12)))*weight;
 if(collar){
  for(const i of [0,2])world[i]+=(seam[i]-world[i])*collar;
 }
 world[1]=yOut;
 if(weight===1)return world;
 // The neutral seam's authored X/Z correction must fade out continuously;
 // a binary branch drops several pixels as automatic pitch crosses zero.
 const neutral={...p,ParamAngleY:0},rest=craniumPosition(point,neutral),legacy=legacyCranium(point,side,neutral);
 return world.map((v,i)=>i===1?v:v+(legacy[i]-rest[i])*(1-weight));
}
function legacyCranium([x,y,lambda],side,p){
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
 const smooth=t=>pitchEase(t,p);
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
 if(p.ParamAngleY&&point[1]>405){
  const neutral=templeWorld(point,side,{...p,ParamAngleY:0},surface);
  const hang=sphereEase((point[1]-405)/139)*sphereEase(Math.abs(p.ParamAngleY*correction(p,'CorrTurn'))/30);
  shell[1]+=(neutral[1]-shell[1])*hang;
 }
 return shell;
}
export function sideWorld(point,side,p,ear=false){return ear?earWorld(point,side,p):canopyWorld(point,p);}
export function backWorld(point,p) {
 const back=shellWorld(point,p,-4),upper=1-pitchEase((point[1]-380)/85,p);
 if(!upper)return back;
 // Background is painted first, so the upper backing can share the dome
 // exactly. A depth-only inset opens a projected crack at the far temple.
 const canopy=p.ParamAngleY?fringeSurface(point,p,0):referenceFringe(point,p,0);
 return back.map((v,i)=>v+(canopy[i]-v)*upper);
}
