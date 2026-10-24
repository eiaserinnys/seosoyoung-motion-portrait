import {strandWorld} from './side-strands.mjs';
import {hairSurface,domeSection,craniumPosition} from './hair-surface.mjs';
import {earWorld} from './stereo-attachments.mjs';
import {outlinePitchShift} from './face-outline.mjs';
import {HEAD_CENTER,HEAD_RADIUS,clamp,correction,correctedLift,artWorld} from './geometry.mjs';
import {DOME} from './dome-data.mjs';

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
// Keep the yaw dome depth; pitch uses a pole-convergent upper arc so its
// center does not sink below the rim when the head looks downward.
export function pitchHairDepth(point,p,relief=0) {
 const q=correctedLift(point,0,p),y=point[1];
 if(y<310){
  const t=clamp((y-(HEAD_CENTER[1]-HEAD_RADIUS))/246,0,1),dx=(point[0]-HEAD_CENTER[0])/DOME.rx;
  const v=1-dx*dx-((y-DOME.cy)/DOME.ry)**2,upper=v>=.16?Math.sqrt(v):Math.max(0,.4+(v-.16)/.8);
  q[2]=q[2]*smooth(t)+30*correction(p,'CorrHairDepth')*upper*Math.sin(Math.PI*t)**2;
 }
 return q[2]+relief;
}
export function fringeSurface(point,p,relief=0) {
 const local=hairLift(point,p,relief);
 if(!p.ParamAngleY)return artWorld(local,p);
 const unrolled={...p,ParamAngleZ:0};
 const a=artWorld(local,unrolled),b=artWorld([local[0],local[1],pitchHairDepth(point,p,relief)],unrolled);
 const q=[a[0],b[1],a[2]];
 if(!p.ParamAngleZ)return q;
 return artWorld(q.map((v,i)=>v-(i<2?HEAD_CENTER[i]:0)),{...p,ParamAngleX:0,ParamAngleY:0});
}
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
 const world=craniumPosition([x,y,lambda],p);
 const {cx,rx}=domeSection(y),sign=Math.sign(lambda)||(side==='left'?-1:1);
 const edge=[HEAD_CENTER[0]+cx+sign*rx,y],horizonY=fringeSurface(edge,p)[1];
 const crownBlend=1-smooth((y-345)/35);
 if(!crownBlend)return [world[0],horizonY,world[2]];
 const seam=canopyWorld(edge,p),yOut=horizonY+(seam[1]-horizonY)*crownBlend;
 if(Math.abs(Math.abs(lambda)-Math.PI/2)<1e-4)return [world[0]+(seam[0]-world[0])*crownBlend,yOut,world[2]+(seam[2]-world[2])*crownBlend];
 return [world[0],yOut,world[2]];
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
 const canopy=fringeSurface(point,p,0);
 return back.map((v,i)=>v+(canopy[i]-v)*upper);
}
