import {createHash} from 'node:crypto';
import {assertApprovedReversals} from './approved_stage_a.mjs';
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
import {FACE_SOLVE} from '../bundle/face-solve-data.mjs';
import {faceSolveWorld,faceSolveAnchorWorld} from '../bundle/face-solve.mjs';
import {FEATURE_FIT} from '../bundle/feature-data.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
import {ponytailSkinDelta} from '../bundle/ponytail-skin.mjs';
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));
const skin=rig.meshes.find(m=>m.id==='face-skin');
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1};
const at=(yaw,pitch)=>({...base,ParamAngleX:yaw,ParamAngleY:pitch,[FRAME_CACHE]:new Map()});
test('ponytail integration is present and retains the authored topology and six exact anchors',()=>{
 assert(FACE_SOLVE.ponytail);assert.equal(skin.positions.length,673);assert.equal(FACE_SOLVE.points.length,740);
 const area=(q,t)=>{const [a,b,c]=t.map(i=>q[i]);return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);};
 for(const f of FEATURE_FIT.poses){
  const p=at(f.yaw,f.pitch),q=skin.positions.map(pt=>faceSolveWorld(pt,p));
  const det=skin.triangles.map(t=>area(q,t)/area(skin.positions,t));
  assertApprovedReversals(f.name,skin.positions,q,skin.triangles);
  for(const [name,target] of Object.entries(f.centers)){
   const actual=faceSolveAnchorWorld(name,p);
   assert(Math.hypot(actual[0]-target[0],actual[1]-target[1])<1e-6,f.name+' '+name);
  }
  assert(Math.hypot(...ponytailSkinDelta([400,300],f.yaw,f.pitch,p))<1e-6,'exact key prior');
 }
 assert(FACE_SOLVE.ponytail.poses.some(p=>p.maxDisplacement>1),'new forehead observations must affect skin');
});
test('ponytail continuous prior has no lower-face support; hair modules and authored atlas stay byte exact',()=>{
 for(const y of [410,497,554.5,616])assert.deepEqual(ponytailSkinDelta([500,y],22.5,15,at(22.5,15)),[0,0]);
 for(const yaw of [-44.99,-29.25,0,22.5,44.99])assert.deepEqual(ponytailSkinDelta([400,300],yaw,0,at(yaw,0)),[0,0]);
 for(const path of ['bundle/hair-surface.mjs','bundle/hair-shell.mjs','bundle/hair-solve.mjs','bundle/hair-solve-data.mjs','bundle/dome-data.mjs','bundle/assets/seosoyoung-rig.json','bundle/assets/texture_00.png']){
  assert.equal(sha(readFileSync(new URL('../'+path,import.meta.url))),sha(execFileSync('git',['show','6dd12cf:'+path],{maxBuffer:20*1024*1024})),path);
 }
});
