import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {bindRig,deform3D} from '../bundle/rig.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
import {approvedStageA,assertApprovedReversals} from './approved_stage_a.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));bindRig(rig);
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1,ParamAngleZ:0,ParamBodyAngleX:0,ParamBreath:0};
test('approved A-stage skin and both ears match the captured nine keys and eight between keys',{timeout:60000},()=>{
 for(const pose of approvedStageA.poses){
  const p={...base,ParamAngleX:pose.yaw,ParamAngleY:pose.pitch,[FRAME_CACHE]:new Map()};
  for(const saved of pose.meshes){
   const mesh=rig.meshes.find(m=>m.id===saved.id);assert.equal(mesh.positions.length,saved.world.length);
   mesh.positions.forEach((point,i)=>deform3D(mesh,point,p).forEach((value,d)=>assert(Math.abs(value-saved.world[i][d])<1e-9,`${pose.name} ${mesh.id} ${i}/${d}`)));
   if(mesh.id==='face-skin')assertApprovedReversals(pose.name,mesh.positions,saved.world,mesh.triangles);
  }
 }
});
test('approved A-stage triangle check rejects one added inversion',()=>{
 const pose=approvedStageA.poses.find(p=>p.name==='front'),mesh=rig.meshes.find(m=>m.id==='face-skin'),q=structuredClone(pose.meshes.find(m=>m.id===mesh.id).world),[a,b]=mesh.triangles[0];[q[a],q[b]]=[q[b],q[a]];
 assert.throws(()=>assertApprovedReversals('front',mesh.positions,q,mesh.triangles),assert.AssertionError);
});
