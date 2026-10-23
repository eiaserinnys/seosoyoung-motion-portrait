import {strandWorld} from './side-strands.mjs';
import {hairSurface} from './hair-surface.mjs';
import {earWorld} from './stereo-attachments.mjs';
import {DOME} from './dome-data.mjs';
export {craniumWeight,craniumWorld} from './cranium-surface.mjs';
import {outlineWorld} from './face-outline.mjs';
import {HEAD_CENTER,HEAD_RADIUS,clamp,radians,correction,correctedLift,artWorld} from './geometry.mjs';

const smooth=t=>{t=clamp(t,0,1);return t*t*(3-2*t);};
const lowerWeight=y=>smooth((y-210)/100);

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
  q[2]+=relief;
  return q;
}
export function hairWorld(point,p,relief=0) {return hairSurface(point,p,0,relief);}
export function fringeWorld(point,p,relief) {return artWorld(hairLift(point,p,relief),p);}
export function farSideWeight(side,p) {
  const angle=p.ParamAngleX*(side==='left'?-1:1);
  return smooth(angle/50)*correction(p,'CorrHairOcclusion');
}
export function sideWorld(point,side,p,ear=false) {
 if(ear)return earWorld(point,side,p);
 const inner=side==='left'?379:626;
 const outward=Math.max(0,(point[0]-inner)*(side==='left'?-1:1));
 const join=(1-smooth(outward/30))*lowerWeight(point[1]);
 // Hanging front locks retain the original forehead-depth attachment rather
 // than being flattened onto the narrower skull at ear height. Rotation moves
 // them in front of the cheek, uncovering the posterior ear naturally.
 const shell=hairWorld(point,p,4),lock=strandWorld(point,side,p),face=outlineWorld(point,p,0);
 const hanging=smooth((point[1]-350)/57);
 for(let i=0;i<3;i++)shell[i]+=(lock[i]-shell[i])*hanging;
 const world=shell.map((v,i)=>v+(face[i]-v)*join);
 return world;
}
export function backWorld(point,p) {return hairWorld(point,p,-4);}
