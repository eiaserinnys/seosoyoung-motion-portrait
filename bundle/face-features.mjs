import {FEATURE_FIT} from './feature-data.mjs';
import {frameValue} from './pose-grid.mjs';
import {faceSolveAnchorWorld} from './face-solve.mjs';
import {registeredFeatureWorld} from './face-feature-prior.mjs';
export {registeredFeatureWorld} from './face-feature-prior.mjs';
export function featureWorld(id,point,p){
 const q=registeredFeatureWorld(id,point,p);
 // Neutral pitch retains the authored eyelid, gaze, brow and mouth evaluator.
 if(!p.ParamAngleY)return q;
 const side=id==='closed-eyes-v1'?(point[0]<502?'left':'right'):id.includes('-left-')?'left':'right';
 const key=id.endsWith('brow')?(side==='left'?'browLeft':'browRight'):
  id.startsWith('eye-')||id==='closed-eyes-v1'?side:id==='nose-detail'?'nose':'mouth';
 const shift=frameValue(p,'solved-feature:'+key,()=>{
  const anchor=FEATURE_FIT.front[key];
  const neutral={...p,ParamEyeBallX:0,ParamEyeBallY:0,ParamEyeLOpen:1,ParamEyeROpen:1,
   ParamBrowLY:0,ParamBrowRY:0,ParamMouthOpenY:0,ParamMouthForm:0};
  const origin=registeredFeatureWorld(id,anchor,neutral),target=faceSolveAnchorWorld(key,p);
  return [target[0]-origin[0],target[1]-origin[1]];
 });
 return [q[0]+shift[0],q[1]+shift[1],q[2]];
}
