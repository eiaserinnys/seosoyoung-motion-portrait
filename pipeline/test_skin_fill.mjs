import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {skinFillMesh,skinFillPositions,skinFillClipMesh,skinFillClipPositions} from '../bundle/skin-fill.mjs';
import {SKIN_FILL as S} from '../bundle/skin-fill-data.mjs';
import {bindRig} from '../bundle/rig.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));bindRig(rig);
const face=rig.meshes.find(m=>m.id==='face-skin');
const params=(yaw,pitch)=>({...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1,ParamAngleX:yaw,ParamAngleY:pitch,ParamAngleZ:0,ParamBodyAngleX:0,ParamBreath:0,[FRAME_CACHE]:new Map()});
const area=(q,t)=>{const [a,b,c]=t.map(i=>q[i]);return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);};
test('97-column upper ribbon has oriented triangles throughout the registered domain',()=>{
 const mesh=skinFillMesh();assert.equal(S.columns,97);assert.equal(mesh.positions.length,582);
 const neutral=skinFillPositions(params(0,0),face);
 for(let yaw=-45;yaw<=45;yaw+=5)for(let pitch=-30;pitch<=30;pitch+=5){
  const q=skinFillPositions(params(yaw,pitch),face);assert(q.flat().every(Number.isFinite));
  for(const t of mesh.triangles)assert(area(q,t)*area(neutral,t)>0,`${yaw}/${pitch}`);
 }
});
test('ribbon and its filled silhouette mask follow body and breath exactly once',()=>{
 for(const fn of [skinFillPositions,skinFillClipPositions]){
  const a=fn(params(12,8),face),b=fn({...params(12,8),ParamBodyAngleX:7,ParamBreath:.5},face);
  for(let i=0;i<a.length;i++){
   assert(Math.abs(b[i][0]-a[i][0]-7)<1e-9);assert(Math.abs(b[i][1]-a[i][1]+2)<1e-9);
  }
 }
});
test('UVs use the actual rest ribbon; mask fills its interior; centered domes mirror exactly',()=>{
 const q=skinFillPositions(params(0,0),face),mesh=skinFillMesh();
 for(let i=0;i<q.length;i++)for(let d=0;d<2;d++)assert(Math.abs(q[i][d]/1024-mesh.uvs[i][d])<1e-9);
 assert.equal(skinFillClipMesh().triangles.length,98);
 for(const pitch of [-30,0,30]){
  const o=skinFillPositions(params(0,pitch),face).slice(485);
  for(let j=15;j<=81;j++){
   assert(Math.abs(o[j][0]+o[96-j][0]-1004)<1e-8);
   assert(Math.abs(o[j][1]-o[96-j][1])<1e-8);
  }
 }
});
test('center tracking has no jump at zero or the six-degree blend boundary',()=>{
 const outer=yaw=>skinFillPositions(params(yaw,-30),face).slice(5*S.columns);
 const samples=[-6,-3,0,3,6].map(yaw=>{const q=outer(yaw);return [q[0],q.reduce((a,b)=>a[1]<b[1]?a:b),q.at(-1)];});
 for(const index of [0,1,2])for(let i=1;i<samples.length;i++)assert(samples[i][index][0]>samples[i-1][index][0]);
 for(const yaw of [-6,0,6])for(const side of [-1,1]){
  const a=outer(yaw),b=outer(yaw+side*.001);
  assert(Math.max(...a.map((q,i)=>Math.hypot(q[0]-b[i][0],q[1]-b[i][1])))<.05,`${yaw} center discontinuity`);
 }
});
