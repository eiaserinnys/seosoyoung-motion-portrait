import {HEAD_CENTER,NECK_PIVOT,artWorld,correction,radians,clamp,rotateXYZ} from './geometry.mjs';
import {profileExtension,earProfileTarget,beadProfileTarget} from './profile-attachments.mjs';
const ears={left:{front:[287,407,379,523],target:[324,400,442,515],angle:45},right:{front:[626,407,716,522],target:[576,385,688,511],angle:-45}};
// Recover the slanted rest plane from independently registered front/near views.
function earBase([x,y],side,p){
 const {front:f,target:t,angle}=ears[side],a=radians(angle);
 const u=(x-f[0])/(f[2]-f[0]),v=(y-f[1])/(f[3]-f[1]);
 const targetX=t[0]+u*(t[2]-t[0]),targetY=t[1]+v*(t[3]-t[1]);
 const z=(targetX-HEAD_CENTER[0]-(x-HEAD_CENTER[0])*Math.cos(a))/Math.sin(a);
 const theta=radians(p.ParamAngleX*correction(p,'CorrTurn'));
 const near=clamp(theta/a,0,1),vertical=near*near*(3-2*near)*(targetY-y);
 return artWorld([x-HEAD_CENTER[0],y-HEAD_CENTER[1]+vertical,z],p);
}
export function earWorld(point,side,p){
 const yaw=p.ParamAngleX*correction(p,'CorrTurn'),world=earBase(point,side,p);
 if(Math.abs(yaw)<=45||side!==(yaw<0?'right':'left'))return world;
 return profileExtension(world,point,side,p,earBase,a=>earProfileTarget(point,side,a));
}
export const BEADS={left:{front:[306.5,544.2],negative:[304,542.2],positive:[310.5,544.2],near:45},right:{front:[691.5,542.6],negative:[686.8,542.3],positive:[709.3,544],near:-45}};
function beadBase(side,p){
 const b=BEADS[side],a=radians(b.near),target=b.near>0?b.positive:b.negative;
 const dx=b.front[0]-HEAD_CENTER[0],z=(target[0]-HEAD_CENTER[0]-dx*Math.cos(a))/Math.sin(a);
 const theta=radians(clamp(p.ParamAngleX*correction(p,'CorrTurn'),-45,45));
 const residual=angle=>{
  const t=angle<0?b.negative:b.positive;
  return [t[0]-HEAD_CENTER[0]-dx*Math.cos(angle)-z*Math.sin(angle),t[1]-b.front[1]];
 };
 const ref=Math.PI/4,neg=residual(-ref),pos=residual(ref);
 const odd=Math.sin(theta)/Math.sin(ref),even=(1-Math.cos(theta))/(1-Math.cos(ref));
 // Both endpoint residuals are projected-space observations; apply before pitch/roll.
 const shift=neg.map((n,i)=>(pos[i]-n)*odd/2+(pos[i]+n)*even/2);
 const yaw=radians(p.ParamAngleX*correction(p,'CorrTurn'));
 return artWorld([dx*Math.cos(yaw)+z*Math.sin(yaw)+shift[0],b.front[1]-HEAD_CENTER[1]+shift[1],-dx*Math.sin(yaw)+z*Math.cos(yaw)],{...p,ParamAngleX:0});
}
export function beadRotation(side,p){
 const world=beadBase(side,p),yaw=p.ParamAngleX*correction(p,'CorrTurn');
 if(Math.abs(yaw)<=45)return world;
 return profileExtension(world,BEADS[side].front,side,p,(_,s,pose)=>beadBase(s,pose),a=>beadProfileTarget(side,a,yaw));
}
// The hanging tie belongs to the gravity frame: yaw moves its attachment,
// pitch does not. Roll is applied once, around the same neck pivot as the head.
export function beadWorld(side,p){
 if(!p.ParamAngleY)return beadRotation(side,p);
 const world=beadRotation(side,{...p,ParamAngleY:0,ParamAngleZ:0});
 if(!p.ParamAngleZ)return world;
 return rotateXYZ(world.map((v,i)=>v-NECK_PIVOT[i]),0,0,p.ParamAngleZ).map((v,i)=>v+NECK_PIVOT[i]);
}
