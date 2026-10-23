import {DOME} from './dome-data.mjs';
export {craniumWeight,craniumWorld} from './cranium-surface.mjs';
import {outlineWorld} from './face-outline.mjs';
import {HEAD_CENTER,HEAD_RADIUS,clamp,radians,correction,correctedLift,artWorld} from './geometry.mjs';

const smooth=t=>{t=clamp(t,0,1);return t*t*(3-2*t);};
const lowerWeight=y=>smooth((y-210)/100);
const outerWeight=x=>smooth((Math.abs(x-HEAD_CENTER[0])-124)/180);

// One continuous art surface: the upper cap reaches z=0 at the north pole,
// then joins the unchanged forehead/cheek surface by y=310. The spherical
// upper section uses radius 262 without extending its pole along a tangent.
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
export function hairWorld(point,p,relief=0,fringe=false) {
  const q=hairLift(point,p,relief),world=artWorld(q,p),dx=point[0]-HEAD_CENTER[0];
  if(dx*p.ParamAngleX<0){
    const lower=lowerWeight(point[1]);
    const boundary=fringe?1-lower:1-lower*(1-outerWeight(point[0]));
    const weight=smooth((Math.abs(dx)-90)/180)*boundary*correction(p,'CorrHairDepth')*
      Math.min(1,Math.abs(p.ParamAngleX*correction(p,'CorrTurn'))/50);
    const unrolled=artWorld(q,{...p,ParamAngleZ:0});
    const shift=(point[0]-unrolled[0])*weight,roll=radians(p.ParamAngleZ);
    world[0]+=shift*Math.cos(roll);world[1]+=shift*Math.sin(roll);
  }
  return world;
}
export function fringeWorld(point,p,relief) {return hairWorld(point,p,relief,true);}
export function farSideWeight(side,p) {
  const angle=p.ParamAngleX*(side==='left'?-1:1);
  return smooth(angle/50)*correction(p,'CorrHairOcclusion');
}
export function sideWorld(point,side,p,ear=false) {
  const far=farSideWeight(side,p),inner=side==='left'?379:626;
  const outward=Math.max(0,(point[0]-inner)*(side==='left'?-1:1));
  const lower=ear?1:lowerWeight(point[1]);
  const amount=.55*far*lower;
  const x=ear?inner+(point[0]-inner)*(1-amount):point[0]+(side==='left'?1:-1)*outward*amount;
  const xy=[x,point[1]];
  const world=ear?artWorld(correctedLift(xy,-12,p),p):hairWorld(xy,p,4*smooth(outward/80));
  // Visibility changes camera depth only. Rotating this bias into XY folds
  // the far ribbon during combined yaw and downward pitch.
  if(!ear){
    const face=outlineWorld(xy,p),base=artWorld(correctedLift(xy,0,p),p);
    const join=1;
    world[0]+=(face[0]-base[0])*join;world[1]+=(face[1]-base[1])*join;
  }
  world[2]-=far*80*(ear?1:lower*smooth(outward/80));
  return world;
}

// The back ribbon uses the same attachment and a slightly stronger inward fold.
// Relief is camera-depth only, so it cannot push the far edge outwards in yaw.
export function backWorld(point,p) {
  const side=point[0]<HEAD_CENTER[0]?'left':'right';
  const far=farSideWeight(side,p),inner=side==='left'?379:626;
  const outward=Math.max(0,(point[0]-inner)*(side==='left'?-1:1));
  const x=point[0]+(side==='left'?1:-1)*outward*.95*far*lowerWeight(point[1]);
  const xy=[x,point[1]],q=hairWorld(xy,p,0);
  const face=outlineWorld(xy,p),base=artWorld(correctedLift(xy,0,p),p);
  q[0]+=face[0]-base[0];q[1]+=face[1]-base[1];q[2]-=4;
  return q;
}
