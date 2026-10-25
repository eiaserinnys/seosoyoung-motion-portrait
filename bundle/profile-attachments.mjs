// Actual reference canvas boxes (720px image at +152,+20); labels are canvas sides.
import {artWorld,correction,clamp} from './geometry.mjs';
import {smoothstep} from './face-profile.mjs';
const EARS={
 left:{front:[287,407,379,523],p67:[402,405,482,532],p90:[407,414,492,540]},
 right:{front:[626,407,716,522],p67:[547,405,622,530],p90:[534,412,614,538]},
};
export function profileExtension(world,point,side,p,evaluate,target){
 const yaw=p.ParamAngleX*correction(p,'CorrTurn'),angle=Math.abs(yaw);
 if(angle<=45)return world;
 const sign=Math.sign(yaw),pose=a=>({...p,CorrTurn:1,ParamAngleX:sign*a,ParamAngleY:0,ParamAngleZ:0});
 const residual=a=>{
  const old=evaluate(point,side,pose(a)),goal=target(a);
  return [goal[0]-old[0],goal[1]-old[1]];
 };
 const r67=residual(67.5),r90=residual(90),w=smoothstep(45,67.5,angle),v=smoothstep(67.5,90,angle);
 const delta=[0,1].map(i=>w*(r67[i]+v*(r90[i]-r67[i])));
 const zero=artWorld([0,0,0],{...p,ParamAngleX:0}),q=artWorld([...delta,0],{...p,ParamAngleX:0});
 return world.map((value,i)=>value+q[i]-zero[i]);
}
export function earProfileTarget(point,side,angle){
 const {front:f,p67,p90}=EARS[side],box=angle===90?p90:p67;
 return [box[0]+(point[0]-f[0])/(f[2]-f[0])*(box[2]-box[0]),box[1]+(point[1]-f[1])/(f[3]-f[1])*(box[3]-box[1])];
}
export function beadProfileTarget(side,angle,yaw){
 const near=side===(yaw<0?'right':'left');
 if(near)return side==='right'?(angle===90?[686,565]:[689,563]):(angle===90?[326,565]:[322,563]);
 return angle===90?[side==='left'?486:518,555]:[side==='left'?304:708,554];
}
const STRAND_TARGETS={
 left:{p67:[[407,468,550],[430,470,548],[460,480,545],[490,496,545],[515,510,543]],
       p90:[[407,492,570],[430,492,568],[460,496,560],[490,507,558],[515,519,552]]},
 right:{p67:[[407,462,538],[430,460,530],[460,450,515],[490,439,492],[515,430,473]],
        p90:[[407,440,520],[430,438,513],[460,435,499],[490,430,480],[515,426,460]]},
};
export function strandProfileTarget(point,side,angle,rows,yaw){
 const y=point[1],near=side===(yaw<0?'right':'left');let i=0;
 while(i<rows.length-2&&y>rows[i+1][0])i++;
 const a=rows[i],b=rows[i+1],t=clamp((y-a[0])/(b[0]-a[0]),0,1),lerp=(x,z)=>x+(z-x)*t;
 const u=(point[0]-lerp(a[1],b[1]))/(lerp(a[2],b[2])-lerp(a[1],b[1]));
 if(!near){const edge=yaw<0?(angle===90?330:321):(angle===90?674:683);return [edge+(u-.5)*(angle===90?2:14),y];}
 const targets=STRAND_TARGETS[side][angle===90?'p90':'p67'],lo=lerp(targets[i][1],targets[i+1][1]),hi=lerp(targets[i][2],targets[i+1][2]);
 return [lo+u*(hi-lo),y];
}
