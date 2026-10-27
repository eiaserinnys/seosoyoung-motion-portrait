import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
const root=resolve(process.argv[2]),url=pathToFileURL(root+'/bundle/');
const {FACE_SOLVE}=await import(new URL('face-solve-data.mjs',url));
const {faceSolveAnchorWorld,faceSolveWorld}=await import(new URL('face-solve.mjs',url));
const {CORRECTION_PRESETS}=await import(new URL('geometry.mjs',url));
const {FRAME_CACHE}=await import(new URL('pose-grid.mjs',url));
const rig=JSON.parse(readFileSync(new URL('assets/seosoyoung-rig.json',url)));
const base={...Object.fromEntries(rig.parameters.map(q=>[q.id,q.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1,ParamAngleZ:0,ParamBodyAngleX:0,ParamBreath:0};
const list=process.argv[3]==='nodes'?FACE_SOLVE.poses:[...FACE_SOLVE.poses,...(FACE_SOLVE.family45?.poses??[])];
const poses=list.map(pose=>{
 const p={...base,ParamAngleX:pose.yaw,ParamAngleY:pose.pitch,[FRAME_CACHE]:new Map()};
 return {name:pose.name,anchors:Object.fromEntries(Object.keys(FACE_SOLVE.anchors).map(k=>[k,faceSolveAnchorWorld(k,p)])),...(process.argv[3]==='nodes'?{world:FACE_SOLVE.points.map(q=>faceSolveWorld(q,p))}:{})};
});
process.stdout.write(JSON.stringify(poses));
