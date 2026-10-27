import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
export const approvedStageA=JSON.parse(readFileSync(new URL('./face_outline_stage_a/accepted/runtime-geometry.json',import.meta.url)));
export const stageAScalars=()=>JSON.parse(readFileSync(new URL('./face_outline_stage_a/accepted/scalars.json',import.meta.url)));
export function negativeTriangleIds(rest,world,triangles){
 const area=(q,t)=>{const [a,b,c]=t.map(i=>q[i]);return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);};
 return triangles.flatMap((t,i)=>area(rest,t)*area(world,t)<0?[i]:[]);
}
export function assertApprovedReversals(name,rest,world,triangles){
 assert.deepEqual(negativeTriangleIds(rest,world,triangles),[],name+' 7.17 requires zero reversed triangles');
}
