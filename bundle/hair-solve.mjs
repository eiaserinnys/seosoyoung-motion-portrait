// The approved r15 3D prior stays intact; one residual chart owns all shells.
import {HAIR_SOLVE} from './hair-solve-data.mjs';
import {domeSection,sphereEase} from './hair-surface.mjs';
import {clamp,correction,artWorld} from './geometry.mjs';
import {frameValue,poseWeights,FRAME_CACHE} from './pose-grid.mjs';
import {smoothstep} from './face-profile.mjs';
import {canopyWorld,craniumWorld} from './hair-shell.mjs';

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
 const x=poseWeights(yaw,45),y=sphereEase(Math.abs(pitch)/30);
 const sideFade=1-smoothstep(60,88,Math.abs(yaw));
 return HAIR_SOLVE.poses.map(pose=>Math.sign(pose.pitch)!==Math.sign(pitch)?0:y*x[pose.yaw/45+1]*sideFade);
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
 if(!weight)return world;
 let result=world;
 if(p.ParamAngleY&&hairFade(point[1])){
  const [s,y]=hairCoordinates(point,posterior===true),delta=hairSolveField(s,y,p);
  result=[world[0]+weight*delta[0],world[1]+weight*delta[1],world[2]];
 }
 return extendProfile(result,point,p,posterior,weight);
}

// Rear canvas coordinates come from left_67/left_90 (+152,+20); the front
// stays 15..30px ahead of the approved forehead. Reflect at x=502 for right
// views. Monotone cubics and ordered knots keep the near chart oriented.
function curve(rows,x){
 let i=0;while(i<rows.length-2&&x>rows[i+1][0])i++;
 const slope=j=>(rows[j+1][1]-rows[j][1])/(rows[j+1][0]-rows[j][0]);
 const tangent=j=>{
  if(j===0)return slope(0);if(j===rows.length-1)return slope(j-1);
  const a=slope(j-1),b=slope(j);return a*b<=0?0:2*a*b/(a+b);
 };
 const [a,b]=[rows[i],rows[i+1]],h=b[0]-a[0],t=clamp((x-a[0])/h,0,1);
 return (2*t**3-3*t*t+1)*a[1]+(t**3-2*t*t+t)*h*tangent(i)+(-2*t**3+3*t*t)*b[1]+(t**3-t*t)*h*tangent(i+1);
}
const FRONT=[[20,480],[60,415],[80,382],[120,342],[160,322],[200,312],[240,306],[280,298],[320,290],[360,287],[400,286],[420,296]];
const REAR=[[34,502],[45,582],[60,627],[80,661],[120,706],[180,746],[240,772],[300,791],[350,801],[380,803],[420,801],[460,794],[500,776],[540,736],[580,700]];
const SEAM=[[34,502],[60,450],[360,450],[420,505],[460,535],[500,565],[540,605],[580,635]];
const stretch=y=>54*smoothstep(350,490,y);
function frontTarget([x,y],pose,angle,sign){
 const t=sign*(x-502),front=curve(FRONT,y),drop=smoothstep(315,415,y);
 const k1=Math.max(457-24*drop,front+58),k2=Math.max(525-50*drop,k1+42),k3=Math.max(575-70*drop,k2+30),k4=Math.max(630-85*drop,k3+40);
 let target=t<0?front+.0015*t*t:curve([[0,front],[146,k1],[201,k2],[270,k3],[360,k4]],t);
 // Join the already fitted lower lock with a Y-only weight. Sampling its
 // monotone outer branch avoids reintroducing the old inner-temple fold.
 const old=canopyWorld([502+sign*Math.max(t,100),y],pose),oldX=(sign>0?old[0]:1004-old[0])+Math.min(0,t-100)*.4;
 target+=(oldX-target)*smoothstep(405,465,y);
 const pinY=angle===90?260+(y-241)*60/70:252+(y-241)*58/70;
 const dy=(pinY-y)*smoothstep(120,230,y)*(1-smoothstep(320,407,y));
 return [sign>0?target:1004-target,y+dy];
}
function rearTarget([x,y,lambda],sign){
 const t=clamp((Math.abs(lambda)*180/Math.PI-88)/92,0,1);
 const lo=curve(SEAM,y),hi=curve(REAR,y),target=lo+(hi-lo)*Math.sin(t*Math.PI/2);
 return [sign>0?target:1004-target,y+stretch(y)];
}
function extendProfile(world,point,p,posterior,weight){
 const yaw=p.ParamAngleX*correction(p,'CorrTurn'),angle=Math.abs(yaw);
 if(angle<=45)return world;
 const sign=yaw<0?1:-1;
 if(posterior&&point[2]*yaw>=0)return world;
 const residual=a=>{
  const pose=frameValue(p,'hair-profile-pose:'+a,()=>({...p,CorrTurn:1,ParamAngleX:-sign*a,ParamAngleY:0,ParamAngleZ:0,[FRAME_CACHE]:new Map()}));
  const old=posterior?craniumWorld(point,sign>0?'right':'left',pose):canopyWorld(point,pose);
  const goal=posterior?rearTarget(point,sign):frontTarget(point,pose,a,sign);
  return [goal[0]-old[0],goal[1]-old[1]];
 };
 const a=residual(67.5),b=residual(90),w=smoothstep(45,67.5,angle),v=smoothstep(67.5,90,angle);
 const delta=[0,1].map(i=>weight*w*(a[i]+v*(b[i]-a[i])));
 // Profile residuals live in the yawed artwork plane. Pitch and roll them
 // as vectors, with the same artWorld frame used by the attachment solver.
 const frame={...p,ParamAngleX:0},zero=artWorld([0,0,0],frame),q=artWorld([...delta,0],frame);
 return world.map((value,i)=>value+q[i]-zero[i]);
}
