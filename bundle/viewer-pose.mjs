import {correction} from './geometry.mjs';
// Keep the original slider response through 45; spend its remaining travel
// on the full profile. This is a viewer input mapping, not a second rig gain.
export function viewerPose(p){
 const angle=Math.abs(p.ParamAngleX),gain=correction(p,'CorrTurn');
 if(angle<=45||gain===1)return p;
 const yaw=Math.sign(p.ParamAngleX)*(45*gain+(angle-45)*(90-45*gain)/45);
 // Keep the value inside the rig's ±90 identity keyforms and preserve the
 // old effective pitch while passing the resolved rotation exactly once.
 return {...p,ParamAngleX:yaw,ParamAngleY:p.ParamAngleY*gain,CorrTurn:1};
}
export function sliderForYaw(yaw,gain){
 const angle=Math.abs(yaw),edge=45*gain;
 return Math.sign(yaw)*(angle<=edge?angle/gain:45+(angle-edge)*45/(90-edge));
}
