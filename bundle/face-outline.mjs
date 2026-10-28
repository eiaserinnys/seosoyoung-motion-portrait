import {correction,headBaseActive} from './geometry.mjs';
import {legacyOutlineWorld} from './face-prior.mjs';
import {faceSolveWorld} from './face-solve.mjs';
import {headSolveWorld} from './head-base.mjs';
export {OUTLINE_ROWS,faceLift,faceWorld,outlinePitchShift,applyOutline,outlinePriorWorld,legacyOutlineWorld} from './face-prior.mjs';
export {outlineRow} from './face-contour.mjs';
// The legacy field remains the hair attachment contract; skin uses the solved field.
export function outlineWorld(point,p,relief=0){
 if(headBaseActive(p))return headSolveWorld(point,p,relief);
 const weight=correction(p,'CorrOutline');
 if(!weight)return legacyOutlineWorld(point,p,relief);
 const solved=faceSolveWorld(point,p,relief);
 if(weight===1)return solved;
 const original=legacyOutlineWorld(point,{...p,CorrOutline:0},relief);
 return original.map((v,i)=>v+(solved[i]-v)*weight);
}
