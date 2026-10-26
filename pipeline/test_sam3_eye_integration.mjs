import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {bindRig,deform3D} from '../bundle/rig.mjs';
import {featureWorld,sam3EyeState} from '../bundle/face-features.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
import {EYE_SAM3 as data} from '../bundle/eye-sam3-data.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));bindRig(rig);
const params=(yaw,pitch)=>({...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1,ParamAngleX:yaw,ParamAngleY:pitch,ParamAngleZ:0,ParamBodyAngleX:0,ParamBreath:0,[FRAME_CACHE]:new Map()});
test('actual feature entry point retains neutral authored coordinates and uses SAM at nonzero yaw',{timeout:60000},()=>{
 for(const side of ['left','right']){
  const b=data[side],id=`eye-${side}-iris`,point=b.rest.cI;
  const q=featureWorld(id,point,params(0,0));
  q.slice(0,2).forEach((v,i)=>assert(Math.abs(v-point[i])<1e-6));
  const p=params(25,12),m=sam3EyeState(side,p),rendered=featureWorld(id,point,p);
  rendered.slice(0,2).forEach((v,i)=>assert(Math.abs(v-m.cI[i])<1e-6));
 }
});
test('independent blinks, pupil and highlight use the connected eye field',{timeout:60000},()=>{
 const p=params(20,-10),closed={...p,ParamEyeLOpen:0,[FRAME_CACHE]:new Map()};
 const left=rig.meshes.find(m=>m.id==='eye-left-sclera'),right=rig.meshes.find(m=>m.id==='eye-right-sclera');
 const delta=mesh=>mesh.positions.map(point=>Math.hypot(...deform3D(mesh,point,p).map((v,i)=>v-deform3D(mesh,point,closed)[i])));
 assert(Math.max(...delta(left))>1);assert(Math.max(...delta(right))<1e-6);
});
