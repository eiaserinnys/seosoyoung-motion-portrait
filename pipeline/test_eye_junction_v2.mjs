import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {featureWorld,sam3EyeState} from '../bundle/face-features.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
import {EYE_SAM3} from '../bundle/eye-sam3-data.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));
const params=(yaw,pitch)=>({...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1,ParamAngleX:yaw,ParamAngleY:pitch,ParamAngleZ:0,ParamBodyAngleX:0,ParamBreath:0,[FRAME_CACHE]:new Map()});
test('requested 48 degrees still uses the fitted eye field',{timeout:60000},()=>{
 for(const side of ['left','right']){
  const p=params(48,0),point=EYE_SAM3[side].rest.cI,model=sam3EyeState(side,p);
  const actual=featureWorld('eye-'+side+'-iris',point,p);
  assert(Math.hypot(actual[0]-model.cI[0],actual[1]-model.cI[1])<1e-6);
 }
});
test('observed 45-degree seam and the new 52/67 endpoints have no vertex jump',{timeout:60000},()=>{
 const meshes=rig.meshes.filter(m=>(m.id.startsWith('eye-')&&!m.id.endsWith('brow'))||m.id==='closed-eyes-v1');
 for(const pitch of [-30,0,30])for(const sign of [-1,1])for(const seam of [45,52,67]){
  const a=params(sign*(seam-.01),pitch),b=params(sign*(seam+.01),pitch);
  for(const mesh of meshes)for(const point of [mesh.positions[0],mesh.positions.at(-1)]){
   const x=featureWorld(mesh.id,point,a),y=featureWorld(mesh.id,point,b);
   assert(Math.hypot(x[0]-y[0],x[1]-y[1])<1,mesh.id+' seam '+seam);
  }
 }
});
