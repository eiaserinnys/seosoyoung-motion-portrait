import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {faceSolveWorld,faceSolveAnchorWorld,evaluateFace45BaseWorld} from '../bundle/face-solve.mjs';
import {FACE_SOLVE} from '../bundle/face-solve-data.mjs';
import {faceSolveWorld as legacyWorld,faceSolveAnchorWorld as legacyAnchor} from '../bundle/profile-baseline/face-solve.mjs';
import {FRAME_CACHE as LEGACY_CACHE} from '../bundle/profile-baseline/pose-grid.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
import {smoothstep} from '../bundle/face-profile.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url))),skin=rig.meshes.find(m=>m.id==='face-skin');
const base={...Object.fromEntries(rig.parameters.map(q=>[q.id,q.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1,ParamBodyAngleX:0,ParamBreath:0,ParamAngleZ:0};
const at=(yaw,pitch,cache=FRAME_CACHE)=>({...base,ParamAngleX:yaw,ParamAngleY:pitch,[cache]:new Map()});
const distance=(a,b)=>Math.hypot(...a.map((v,i)=>v-b[i]));
test('final A skin and named anchors equal 6dd at every absolute profile pose',{timeout:60000},()=>{
 for(const yaw of [-90,-78.75,-67.5,67.5,78.75,90])for(const pitch of [-30,0,30]){
  const p=at(yaw,pitch),q=at(yaw,pitch,LEGACY_CACHE);
  for(const point of skin.positions)assert(distance(faceSolveWorld(point,p),legacyWorld(point,q))<1e-12,`${yaw}/${pitch} skin ${point}`);
  for(const name of Object.keys(FACE_SOLVE.anchors))assert(distance(faceSolveAnchorWorld(name,p),legacyAnchor(name,q))<1e-12,`${yaw}/${pitch} anchor ${name}`);
 }
});
test('final A interpolates B and legacy then applies the same-frame X projection',{timeout:60000},()=>{
 for(const yaw of [-56.25,-50,50,56.25])for(const pitch of [-30,0,30]){
  const p=at(yaw,pitch),b=at(yaw,pitch),q=at(yaw,pitch,LEGACY_CACHE),w=smoothstep(0,22.5,Math.abs(yaw)-45),mix=(a,z)=>a.map((v,i)=>(1-w)*v+w*z[i]);
  const check=(actual,expected,index,label)=>{
   const correction=p[FRAME_CACHE].get('face45-triangle-projection')[index];
   assert(Math.hypot(actual[1]-expected[1],actual[2]-expected[2])<1e-6,`${yaw}/${pitch} ${label} Y/Z`);
   assert(Math.abs(actual[0]-expected[0]-correction)<1e-6,`${yaw}/${pitch} ${label} projected X`);
  };
  skin.positions.forEach((point,index)=>check(faceSolveWorld(point,p),mix(evaluateFace45BaseWorld(point,b),legacyWorld(point,q)),index,`skin ${point}`));
  for(const [name,index] of Object.entries(FACE_SOLVE.anchors)){
   const prior=evaluateFace45BaseWorld(FACE_SOLVE.points[index],b,undefined,0,[[index,1]]);
   check(faceSolveAnchorWorld(name,p),mix(prior,legacyAnchor(name,q)),index,`anchor ${name}`);
  }
  const projection=p[FRAME_CACHE].get('face45-triangle-projection');
  console.log('A_MIX_PROJECTION '+JSON.stringify({yaw,pitch,maximum:Math.max(...projection.map(Math.abs)),nonzeroVertices:projection.filter(v=>v!==0).length}));
 }
});
