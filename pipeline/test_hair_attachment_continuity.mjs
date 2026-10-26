import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {bindRig,deform3D} from '../bundle/rig.mjs';
import {CORRECTION_PRESETS,rotateXYZ,NECK_PIVOT} from '../bundle/geometry.mjs';
import {beadWorld,BEADS} from '../bundle/stereo-attachments.mjs';
import {canopyWorld,fringeSurface} from '../bundle/hair-shell.mjs';
import {domeSection} from '../bundle/hair-surface.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));bindRig(rig);
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1,ParamBodyAngleX:0,ParamBreath:0};
const near=(a,b)=>assert(Math.hypot(...a.map((v,i)=>v-b[i]))<1e-8,`${a} != ${b}`);
const signedArea=([a,b,c])=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);

test('opaque bangs-left 737 and 764 retain their forward orientation at intermediate yaw',()=>{
 const m=rig.meshes.find(m=>m.id==='bangs-left');
 for(const yaw of [-30,-15,15,30]){
  const p={...base,ParamAngleX:yaw,ParamAngleY:0};
  for(const index of [737,764]){
   const points=m.triangles[index].map(i=>m.positions[i]),rest=signedArea(points);
   assert(signedArea(points.map(q=>fringeSurface(q,p)))/rest>0,'reference is forward facing');
   assert(signedArea(points.map(q=>deform3D(m,q,p)))/rest>0,`${yaw}: bangs-left ${index} reversed`);
  }
 }
});

for(const [id,index,yaw,pitch] of [
 ['side-hair-left',182,-45,30],
 ['side-hair-right',148,45,30],
 ['hair-top',36,0,30],
])test(`opaque ${id} ${index} retains its original forward orientation`,()=>{
 const m=rig.meshes.find(m=>m.id===id),points=m.triangles[index].map(i=>m.positions[i]);
 const p={...base,ParamAngleX:yaw,ParamAngleY:pitch},rest=signedArea(points);
 assert(signedArea(points.map(q=>fringeSurface(q,p)))/rest>0,'reference is forward facing');
 assert(signedArea(points.map(q=>deform3D(m,q,p)))/rest>0,`${yaw}/${pitch}: ${id} ${index} reversed`);
});

// Exact <=45 ear behavior is now compared vertex-by-vertex with pinned main
// in test_side90.mjs; the 45..90 range intentionally extends the r15 function.

test('both hanging locks end in the neutral gravity frame at the tie height',()=>{
 for(const yaw of [-45,45]){
  const side=yaw<0?'left':'right',p={...base,ParamAngleX:yaw,ParamAngleY:30};
  const point=[side==='left'?310:691,544];
  assert.equal(canopyWorld(point,p)[1],canopyWorld(point,{...p,ParamAngleY:0})[1]);
 }
});
test('both tail layers share the corrected physical tie root',()=>{
 for(const yaw of [-45,45]){
  const side=yaw<0?'left':'right',p={...base,ParamAngleX:yaw,ParamAngleY:30},root=beadWorld(side,p);
  for(const suffix of ['tie','hair'])near(deform3D(rig.meshes.find(m=>m.id===`tail-${side}-${suffix}`),BEADS[side].front,p),root);
 }
});
test('skull-bound ties cross zero continuously and respect roll with shared frame caches',()=>{
 for(const yaw of [-45,45]){
  const side=yaw<0?'left':'right',p={...base,ParamAngleX:yaw,ParamAngleY:30};
  const unrolled=beadWorld(side,p),cached={...p,[FRAME_CACHE]:new Map()};
  beadWorld(side,cached);
  for(const roll of [-12,12]){
   const rotated=rotateXYZ(unrolled.map((v,i)=>v-NECK_PIVOT[i]),0,0,roll).map((v,i)=>v+NECK_PIVOT[i]);
   near(beadWorld(side,{...cached,ParamAngleZ:roll}),rotated);
  }
  const zero=beadWorld(side,{...p,ParamAngleY:0});
  for(const pitch of [-1e-8,1e-8])assert(Math.hypot(...beadWorld(side,{...p,ParamAngleY:pitch}).map((v,i)=>v-zero[i]))<1e-5);
 }
});
