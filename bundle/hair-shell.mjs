import {strandWorld} from './side-strands.mjs';
import {hairSurface,domeSection,craniumPosition} from './hair-surface.mjs';
import {earWorld} from './stereo-attachments.mjs';
import {DOME} from './dome-data.mjs';
import {outlinePitchShift} from './face-outline.mjs';
import {HEAD_CENTER,HEAD_RADIUS,clamp,radians,correction,correctedLift,artWorld} from './geometry.mjs';

const smooth=t=>{t=clamp(t,0,1);return t*t*(3-2*t);};

// Preserve the established forehead/fringe correction; the complete outer
// skull is owned by hairSurface, not this front illustration relief.
export function hairLift(point,p,relief=0) {
  const q=correctedLift(point,0,p),y=point[1];
  if(y<310){
    const t=clamp((y-(HEAD_CENTER[1]-HEAD_RADIUS))/246,0,1);
    const dx=(point[0]-HEAD_CENTER[0])/DOME.rx;
    const v=1-dx*dx-((y-DOME.cy)/DOME.ry)**2;
    // Tangent continuation avoids a square-root derivative cliff at the limb.
    const upper=v>=.16?Math.sqrt(v):Math.max(0,.4+(v-.16)/.8);
    // Ease the pole into the sampled artwork ribbon; its finite slope avoids
    // a square-root cliff at the first row while keeping the pole at zero.
    q[2]=q[2]*smooth(t)+30*correction(p,'CorrHairDepth')*upper*Math.sin(Math.PI*t)**2;
  }
  // Measured outer forehead clearance. The center fringe remains unchanged;
  // the temple ribbon joins its shallower attachment before the free locks.
  const temple=smooth((Math.abs(point[0]-HEAD_CENTER[0])-105)/45)*smooth((y-240)/65)*(1-smooth((y-350)/60));
  q[2]-=28*temple;
  q[2]+=relief;
  return q;
}
const shellWorld=(point,p,relief=0)=>hairSurface(point,p,0,relief);
const fringeSurface=(point,p)=>artWorld(hairLift(point,p,0),p);
// One coordinate field owns every overlapping front-hair pixel. Layer order
// belongs to the renderer; it must never alter the rotated geometry.
export function canopyWorld(point,p) {
 const [x,y]=point,world=fringeSurface(point,p);
 const side=x<HEAD_CENTER[0]?'left':'right';
 const lower=smooth((y-405)/55),xStart=126-42*lower,width=14;
 const lockWeight=smooth((Math.abs(x-HEAD_CENTER[0])-xStart)/width)*smooth((y-340)/67);
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
 const dome=craniumPosition([x,y,lambda],p);
 const w=1-smooth((Math.abs(lambda)-Math.PI/2)/(Math.PI/4));
 if(!w)return dome;
 const {cx,rx,cz}=domeSection(y),sign=Math.sign(lambda)||(side==='left'?-1:1);
 const domeSeam=artWorld([cx+sign*rx,y-HEAD_CENTER[1],cz],p);
 const seam=canopyWorld([HEAD_CENTER[0]+cx+sign*rx,y],p);
 // Reuse the endpoint exactly instead of retaining cos(pi/2) roundoff.
 if(Math.abs(lambda)===Math.PI/2)return seam;
 return dome.map((v,i)=>v+(seam[i]-domeSeam[i])*w);
}
export function farSideWeight(side,p) {
  const angle=p.ParamAngleX*(side==='left'?-1:1);
  return smooth(angle/50)*correction(p,'CorrHairOcclusion');
}
function templeWorld(point,side,p) {
 const inner=side==='left'?379:626;
 const outward=Math.max(0,(point[0]-inner)*(side==='left'?-1:1));
 const join=(1-smooth(outward/30))*smooth((point[1]-380)/70);
 // Hanging front locks retain the original forehead-depth attachment rather
 // than being flattened onto the narrower skull at ear height. Rotation moves
 // them in front of the cheek, uncovering the posterior ear naturally.
 const lock=strandWorld(point,side,p),fringe=fringeSurface(point,p),lower=smooth((point[1]-260)/147);
 const shell=fringe.map((v,i)=>v+(lock[i]-v)*lower);
 const hanging=smooth((point[1]-350)/57);
 for(let i=0;i<3;i++)shell[i]+=(lock[i]-shell[i])*hanging;
 const world=shell,shift=outlinePitchShift(point,p);
 for(const i of [0,1])world[i]+=shift[i]*join;
 return world;
}
export function sideWorld(point,side,p,ear=false){return ear?earWorld(point,side,p):canopyWorld(point,p);}
export function backWorld(point,p) {
 const back=shellWorld(point,p,-4),upper=1-smooth((point[1]-380)/85);
 if(!upper)return back;
 const canopy=artWorld(hairLift(point,p,-4),p);
 return back.map((v,i)=>v+(canopy[i]-v)*upper);
}
