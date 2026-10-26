// Paired CPU deformation cost, independent of SwiftShader and PNG encoding.
import {readFileSync} from 'node:fs';
import {performance} from 'node:perf_hooks';
import {bindRig,deform3D} from '../bundle/rig.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
import {frozenRig} from './frozen_rig.mjs';
const source=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));
const rig=structuredClone(source),priorRig=structuredClone(source);bindRig(rig);
const prior=await frozenRig(false,'1458c77');prior.bindRig(priorRig);
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1,ParamBodyAngleX:0,ParamBreath:0};
const poses=JSON.parse(readFileSync(new URL('../source/viewer/face-feature-fit.json',import.meta.url))).poses;
poses.push(...[-30,-15,15,30].map(yaw=>({name:`yaw${yaw>0?'+':''}${yaw}`,yaw,pitch:0})));
let checksum=0;
function frame(model,fn,key,pose){
 const p={...base,ParamAngleX:pose.yaw,ParamAngleY:pose.pitch,[key]:new Map()};
 const start=performance.now();
 for(const mesh of model.meshes)for(const point of mesh.positions)checksum+=fn(mesh,point,p)[0];
 return performance.now()-start;
}
try{
 const result=[];
 for(const pose of poses){
  frame(priorRig,prior.deform3D,prior.FRAME_CACHE,pose);frame(rig,deform3D,FRAME_CACHE,pose);
  const a=[],b=[];
  for(let i=0;i<5;i++){
   if(i%2){b.push(frame(rig,deform3D,FRAME_CACHE,pose));a.push(frame(priorRig,prior.deform3D,prior.FRAME_CACHE,pose));}
   else{a.push(frame(priorRig,prior.deform3D,prior.FRAME_CACHE,pose));b.push(frame(rig,deform3D,FRAME_CACHE,pose));}
  }
  const median=xs=>xs.sort((x,y)=>x-y)[2];
  result.push({name:pose.name,r24Milliseconds:median(a),candidateMilliseconds:median(b)});
 }
 process.stdout.write(JSON.stringify({method:'all mesh vertices, fresh frame cache, one warmup and five alternating paired CPU frames per pose; medians',checksum,poses:result},null,2)+'\n');
}finally{prior.dispose();}
