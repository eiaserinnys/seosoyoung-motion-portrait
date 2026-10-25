// The approved r15 3D prior stays intact; one residual chart owns all shells.
import {HAIR_SOLVE} from './hair-solve-data.mjs';
import {domeSection,sphereEase} from './hair-surface.mjs';
import {clamp,correction} from './geometry.mjs';
import {frameValue} from './pose-grid.mjs';

export function hairCoordinates([x,y,lambda],posterior=false){
 const {cx,rx}=domeSection(y);
 let s=posterior?Math.sign(lambda)*(1+Math.max(0,Math.abs(lambda)-Math.PI/2)/(Math.PI/2)):clamp((x-502-cx)/Math.max(1e-9,rx),-1,1);
 if(Math.abs(Math.abs(s)-1)<1e-12)s=Math.sign(s);
 return [s,y];
}
export function hairFade(y){const t=clamp((y-400)/35,0,1);return 1-t*t*(3-2*t);}
function linear(nodes,x){
 let i=0;while(i<nodes.length-2&&x>nodes[i+1])i++;
 return [i,clamp((x-nodes[i])/(nodes[i+1]-nodes[i]),0,1)];
}
export function hairPoseWeights(p){
 const yaw=p.ParamAngleX*correction(p,'CorrTurn'),pitch=p.ParamAngleY*correction(p,'CorrTurn');
 const x=clamp(Math.abs(yaw)/45,0,1),y=sphereEase(Math.abs(pitch)/30);
 return HAIR_SOLVE.poses.map(pose=>Math.sign(pose.pitch)!==Math.sign(pitch)?0:y*(pose.yaw===0?1-x:Math.sign(pose.yaw)===Math.sign(yaw)?x:0));
}
export function hairResidualField(s,y,p){
 if(!p.ParamAngleY||!hairFade(y))return [0,0];
 const weights=frameValue(p,'hair-solve-weights',()=>hairPoseWeights(p)),delta=[0,0];
 const [ix,tx]=linear(HAIR_SOLVE.s,s),[iy,ty]=linear(HAIR_SOLVE.y,y),nx=HAIR_SOLVE.s.length;
 for(let k=0;k<weights.length;k++)if(weights[k])for(const [x,wx] of [[ix,1-tx],[ix+1,tx]])for(const [j,wy] of [[iy,1-ty],[iy+1,ty]]){
  const v=HAIR_SOLVE.poses[k].du[j*nx+x];
  for(let d=0;d<2;d++)delta[d]+=weights[k]*hairFade(y)*wx*wy*v[d];
 }
 return delta;
}
export function hairSolveField(s,y,p){
 const delta=hairResidualField(s,y,p),roll=(p.ParamAngleZ||0)*Math.PI/180;
 return [delta[0]*Math.cos(roll)-delta[1]*Math.sin(roll),delta[0]*Math.sin(roll)+delta[1]*Math.cos(roll)];
}
export function applyHairSolve(world,point,p,posterior=false){
 const weight=correction(p,'CorrOutline');
 if(!weight||!p.ParamAngleY||!hairFade(point[1]))return world;
 const [s,y]=hairCoordinates(point,posterior),delta=hairSolveField(s,y,p);
 return [world[0]+weight*delta[0],world[1]+weight*delta[1],world[2]];
}
