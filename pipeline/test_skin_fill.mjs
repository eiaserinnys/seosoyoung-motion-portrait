import {test} from 'node:test';
import assert from 'node:assert/strict';
import {skinFillMesh,skinFillPositions} from '../bundle/skin-fill.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
const params=(yaw,pitch)=>({...CORRECTION_PRESETS.recommended,CorrTurn:1,ParamAngleX:yaw,ParamAngleY:pitch,ParamAngleZ:0,ParamBodyAngleX:0,ParamBreath:0});
test('exposed skin reaches the full forehead and stays oriented throughout the registered pose domain',()=>{
 const mesh=skinFillMesh(),neutral=skinFillPositions({...params(0,0),[FRAME_CACHE]:new Map()});
 assert(Math.hypot(mesh.positions[0][0]-502,mesh.positions[0][1]-430)<100,'UV material center must use canonical image coordinates');
 const top=neutral.filter(p=>p[1]<320);
 assert(Math.max(...top.map(p=>p[0]))-Math.min(...top.map(p=>p[0]))>250,'the old 138px forehead cap must be filled');
 const area=(q,t)=>{const [a,b,c]=t.map(i=>q[i]);return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);};
 for(let yaw=-45;yaw<=45;yaw+=5)for(let pitch=-30;pitch<=30;pitch+=5){
  const points=skinFillPositions({...params(yaw,pitch),[FRAME_CACHE]:new Map()});
  for(const t of mesh.triangles)assert(area(points,t)/area(neutral,t)>0,`${yaw}/${pitch} skin fill triangle`);
 }
});
test('skin backing follows the same body translation and breath as the face',()=>{
 const a=skinFillPositions(params(12,8)),b=skinFillPositions({...params(12,8),ParamBodyAngleX:7,ParamBreath:.5});
 for(let i=0;i<a.length;i++){
  assert(Math.abs(b[i][0]-a[i][0]-7)<1e-9);assert(Math.abs(b[i][1]-a[i][1]+2)<1e-9);
 }
});
