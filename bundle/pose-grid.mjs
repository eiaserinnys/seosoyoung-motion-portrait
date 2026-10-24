import {clamp,radians} from './geometry.mjs';
export const FRAME_CACHE=Symbol('deformation frame');
export function frameValue(p,key,build){
 const cache=p[FRAME_CACHE];
 if(!cache)return build();
 if(!cache.has(key))cache.set(key,build());
 return cache.get(key);
}
// Trigonometric quadratic cardinal basis: exact at -limit, 0, +limit,
// with matching derivatives through the neutral row/column.
export function poseWeights(angle,limit){
 const a=radians(clamp(angle,-limit,limit)),ref=radians(limit);
 const odd=Math.sin(a)/Math.sin(ref),even=(1-Math.cos(a))/(1-Math.cos(ref));
 return [(even-odd)/2,1-even,(even+odd)/2];
}
export function gridValue(poses,yaw,pitch,value){
 const wx=poseWeights(yaw,45),wy=poseWeights(pitch,30);
 return poses.reduce((sum,pose)=>sum+wx[pose.yaw/45+1]*wy[pose.pitch/30+1]*value(pose),0);
}
