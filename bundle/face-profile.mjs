import {clamp,correction} from './geometry.mjs';
export const smoothstep=(lo,hi,value)=>{const t=clamp((value-lo)/(hi-lo),0,1);return t*t*(3-2*t);};
export function profileWeights(p){
 const yaw=p.ParamAngleX*correction(p,'CorrTurn'),angle=Math.abs(yaw);
 return {skin:smoothstep(45,67.5,angle),profile:smoothstep(60,85,angle),row:yaw<0?0:.5};
}
