import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {bindRig,deform3D,opacity} from '../bundle/rig.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));bindRig(rig);
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1};
const mesh=id=>rig.meshes.find(m=>m.id===id);
test('stereo ear endpoints reproduce independent near reference boxes',()=>{
 for(const [side,yaw,front,target] of [['right',-45,[626,407,716,522],[576,385,688,511]],['left',45,[287,407,379,523],[324,400,442,515]]]){
  for(const i of [0,2]){
   const q=deform3D(mesh('ear-'+side),[front[i],front[i+1]],{...base,ParamAngleX:yaw});
   assert(Math.hypot(q[0]-target[i],q[1]-target[i+1])<.01,`${side}: ${q}`);
  }
 }
});
test('bead centers reproduce both asymmetric reference poses',()=>{
 for(const [side,front,neg,pos] of [['left',[306.5,544.2],[304,542.2],[310.5,544.2]],['right',[691.5,542.6],[686.8,542.3],[709.3,544]]]){
  for(const [yaw,target] of [[-45,neg],[45,pos]]){
   const q=deform3D(mesh('tail-'+side+'-tie'),front,{...base,ParamAngleX:yaw});
   assert(Math.hypot(q[0]-target[0],q[1]-target[1])<.01,`${side} ${yaw}: ${q}`);
  }
 }
});
test('cranium is opaque while it rotates through the near horizon',()=>{
 for(const [side,sgn] of [['left',1],['right',-1]])for(const yaw of [1,10,25,45])
  assert.equal(opacity(mesh('side-cranium-'+side),{...base,ParamAngleX:sgn*yaw},{hidden:new Set()}),1);
});
test('eye, nose and mouth centers reach measured asymmetric targets',()=>{
 for(const [id,front,neg,pos] of [['eye-left-iris',[415.7,432.2],[362.4,429.3],[514.2,430.7]],['eye-right-iris',[588.1,430.8],[525.1,429.2],[670.1,428.9]],['nose-detail',[508,497],[406.9,497.1],[629.9,490.5]],['mouth-detail',[504.5,554.5],[434.1,556.2],[598.7,553.3]]]){
  for(const [yaw,target] of [[-45,neg],[45,pos]]){
   const q=deform3D(mesh(id),front,{...base,ParamAngleX:yaw});
   assert(Math.hypot(q[0]-target[0],q[1]-target[1])<1,`${id} ${yaw}: ${q}`);
  }
 }
});
test('all eye layers share the measured center-relative width ratio',()=>{
 for(const side of ['left','right'])for(const yaw of [-45,45]){
  const [cx,cy]=side==='left'?[415.7,432.2]:[588.1,430.8];
  const far=(side==='left')===(yaw<0);
  for(const kind of ['sclera','iris','pupil','highlight','upper-lashes','lower-line','brow']){
   const m=mesh(`eye-${side}-${kind}`),width=angle=>{
    const p={...base,ParamAngleX:angle};return deform3D(m,[cx+10,cy],p)[0]-deform3D(m,[cx-10,cy],p)[0];
   };
   const ratio=width(yaw)/width(0);
   assert(ratio>=(far?.719:.95)&&ratio<=(far?.85:1.08),`${side} ${kind} ${yaw}: ${ratio}`);
  }
 }
});
