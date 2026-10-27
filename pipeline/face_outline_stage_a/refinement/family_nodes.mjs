import {readFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import {resolve} from 'node:path';
const url=pathToFileURL(resolve(process.argv[2])+'/bundle/');
const {FACE_SOLVE}=await import(new URL('face-solve-data.mjs',url));
const {diagnosticPrior}=await import(new URL('face-solve-diag.mjs',url));
const {CORRECTION_PRESETS}=await import(new URL('geometry.mjs',url));
const {FRAME_CACHE}=await import(new URL('pose-grid.mjs',url));
const rig=JSON.parse(readFileSync(new URL('assets/seosoyoung-rig.json',url)));
const base={...Object.fromEntries(rig.parameters.map(q=>[q.id,q.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1,ParamBodyAngleX:0,ParamBreath:0,ParamAngleZ:0};
console.log(JSON.stringify(FACE_SOLVE.family45.poses.map(pose=>{
 const p={...base,ParamAngleX:pose.yaw,ParamAngleY:0,[FRAME_CACHE]:new Map()};
 return {name:pose.name,world:FACE_SOLVE.points.map((q,i)=>diagnosticPrior(q,p,FACE_SOLVE.family45,0,[[i,1]]))};
})));
