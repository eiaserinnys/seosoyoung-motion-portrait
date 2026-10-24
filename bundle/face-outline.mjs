import {HEAD_CENTER,artWorld,correctedLift,correction,clamp,radians} from './geometry.mjs';
import {OUTLINE_ROWS} from './outline-data.mjs';
export {OUTLINE_ROWS} from './outline-data.mjs';
// Cubic Hermite interpolation keeps the sampled mask residual smooth in height.
export function outlineRow(y){
 if(y<=OUTLINE_ROWS[0][0])return OUTLINE_ROWS[0].slice(1);
 for(let i=1;i<OUTLINE_ROWS.length;i++)if(y<=OUTLINE_ROWS[i][0]){
  const a=OUTLINE_ROWS[i-1],b=OUTLINE_ROWS[i],prev=OUTLINE_ROWS[Math.max(0,i-2)],next=OUTLINE_ROWS[Math.min(OUTLINE_ROWS.length-1,i+1)];
  const h=b[0]-a[0],t=(y-a[0])/h;
  return a.slice(1).map((v,j)=>{
   const k=j+1,m0=(b[k]-prev[k])/(b[0]-prev[0]),m1=(next[k]-a[k])/(next[0]-a[0]);
   return (2*t**3-3*t*t+1)*v+(t**3-2*t*t+t)*h*m0+(-2*t**3+3*t*t)*b[k]+(t**3-t*t)*h*m1;
  });
 }
 return OUTLINE_ROWS.at(-1).slice(1);
}
function legacyOutline(point,world,p){
 const angle=p.ParamAngleX*correction(p,'CorrTurn'),theta=radians(clamp(angle,-45,45));
 const weight=correction(p,'CorrOutline')*clamp((point[1]-250)/80,0,1);
 if(!theta||!weight)return world;
 const sampleY=Math.max(465,point[1]);
 const [l,r,ll,lr,rl,rr,ty]=outlineRow(sampleY);
 const [frontL,frontR]=outlineRow(point[1]),u=clamp((point[0]-frontL)/(frontR-frontL),0,1);
 const half=(r-l)/2,offset=(l+r)/2-HEAD_CENTER[0];
 const residual=(yaw,targets)=>{
  const pose={...p,ParamAngleX:yaw,CorrTurn:1,ParamAngleY:0,ParamAngleZ:0};
  return [targets[0]-offset-artWorld(correctedLift([HEAD_CENTER[0]-half,sampleY],0,p),pose)[0],
          targets[1]-offset-artWorld(correctedLift([HEAD_CENTER[0]+half,sampleY],0,p),pose)[0]];
 };
 const neg=residual(-45,[ll,lr]),pos=residual(45,[rl,rr]);
 const near=(pos[1]-neg[0])/2,far=(pos[0]-neg[1])/2;
 const minus=-near*(1-u)-far*u,plus=far*(1-u)+near*u;
 const odd=Math.sin(theta)/Math.sin(Math.PI/4),even=(1-Math.cos(theta))/(1-Math.cos(Math.PI/4));
 const shift=((plus-minus)*odd+(plus+minus)*even)*.5*weight;
 const dy=(ty-Math.min(sampleY,616))*even*weight,roll=radians(p.ParamAngleZ);
 return [world[0]+shift*Math.cos(roll)-dy*Math.sin(roll),world[1]+shift*Math.sin(roll)+dy*Math.cos(roll),world[2]];
}
const smooth=t=>{t=clamp(t,0,1);return t*t*(3-2*t);};
// Yaw keeps the measured r12 surface; only pitch sees the jaw's rounded section.
function symmetricBase(point,p,relief=0){
 const [x,y]=point,[l,r]=outlineRow(y),mid=(l+r)/2,offset=mid-HEAD_CENTER[0];
 const local=correctedLift([x-offset,y],relief,p),roll=radians(p.ParamAngleZ);
 const phi=radians(clamp((p.ParamAngleY??0)*correction(p,'CorrTurn'),-30,30));
 const sPhi=clamp(Math.sin(phi)/Math.max(1e-4,Math.sin(radians(30*correction(p,'CorrTurn')))),-1.2,1.2);
 const s=clamp((x-mid)/Math.max(12,(r-l)/2),-1,1);
 local[0]+=s*10*sPhi*smooth((y-470)/70)*(1-smooth((y-575)/35));
 const q=artWorld(local,p);
 if(phi&&y>380){
  const zMid=correctedLift([HEAD_CENTER[0],Math.min(y,490)],relief,p)[2];
  const chinConv=smooth((y-525)/(616-525));
  const zRim=(28+(zMid-8-28)*chinConv)*correction(p,'CorrDepth')/.55+relief;
  const z3d=zRim+(zMid-zRim)*Math.sqrt(Math.max(.04,1-.96*s*s));
  const zPitch=local[2]+(z3d-local[2])*smooth((y-380)/70);
  const restRoll={...p,ParamAngleZ:0},before=artWorld(local,restRoll);
  const pitched=artWorld([local[0],local[1],zPitch],restRoll),dy=pitched[1]-before[1];
  q[0]-=dy*Math.sin(roll);q[1]+=dy*Math.cos(roll);
  // Keep XYZ continuous through zero pitch as well as preserving yaw-only XYZ.
  q[2]+=(pitched[2]-before[2])*smooth(Math.abs(sPhi));
 }
 q[0]+=offset*Math.cos(roll);q[1]+=offset*Math.sin(roll);
 return q;
}
export const applyOutline=(point,world,p)=>legacyOutline(point,world,p);
export function outlineWorld(point,p,relief=0){
 const plain=artWorld(correctedLift(point,relief,p),p),symmetric=symmetricBase(point,p,relief),weight=correction(p,'CorrOutline');
 return applyOutline(point,plain.map((v,i)=>v+(symmetric[i]-v)*weight),p);
}
