import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {bindRig,deform3D} from '../bundle/rig.mjs';
import {CORRECTION_PRESETS,artWorld,correctedLift} from '../bundle/geometry.mjs';
import {outlineWorld,outlineRow} from '../bundle/face-outline.mjs';
import {featureWorld} from '../bundle/face-features.mjs';
import {canopyWorld} from '../bundle/hair-shell.mjs';
import {applyHairSolve} from '../bundle/hair-solve.mjs';
import {domeSection} from '../bundle/hair-surface.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));bindRig(rig);
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1};
const ids=['hair-top','bangs-left','bangs-right','side-hair-left','side-hair-right'];
const at=(id,xy,p)=>deform3D(rig.meshes.find(m=>m.id===id),xy,p);
test('neutral face and feature vertices retain every original canvas coordinate',()=>{
 for(const mesh of rig.meshes.filter(m=>m.id==='face-skin'||m.id.startsWith('eye-')||m.id==='nose-detail'||m.id.startsWith('mouth-')||m.id==='closed-eyes-v1'))for(const point of mesh.positions){
  const q=deform3D(mesh,point,base);
  assert(Math.hypot(q[0]-point[0],q[1]-point[1])<1e-9,mesh.id);
 }
});
test('skin preserves neutral symmetry and independent pitch observations continuously',()=>{
 for(const yaw of [-45,-22.5,0,22.5,45])for(const pitch of [-30,-15,0,15,30])for(const y of [465,510,570,610]){
  for(const x of outlineRow(y).slice(0,2)){
   const p={...base,ParamAngleX:yaw,ParamAngleY:pitch,[FRAME_CACHE]:new Map()};
   const q=outlineWorld([x,y],p);
   assert(q.every(Number.isFinite));
   for(const delta of [-.001,.001]){
    const near=outlineWorld([x,y],{...p,ParamAngleX:yaw+delta,[FRAME_CACHE]:new Map()});
    assert(Math.hypot(near[0]-q[0],near[1]-q[1])<.01,'continuous measured skin field');
   }
  }
 }
});
test('feature centers reach independent measured keyframes with continuous spherical intermediates',()=>{
 const fit=JSON.parse(readFileSync(new URL('../source/viewer/face-feature-fit.json',import.meta.url)));
 const ids={left:'eye-left-sclera',right:'eye-right-sclera',browLeft:'eye-left-brow',browRight:'eye-right-brow',nose:'nose-detail',mouth:'mouth-detail'};
 for(const pose of fit.poses)for(const [key,id] of Object.entries(ids)){
  const p={...base,ParamAngleX:pose.yaw,ParamAngleY:pose.pitch,[FRAME_CACHE]:new Map()};
  const q=featureWorld(id,fit.front[key],p),target=pose.centers[key];
  assert(Math.hypot(q[0]-target[0],q[1]-target[1])<1e-6,pose.name+' '+key);
 }
 for(const yaw of [-37,-22.5,0,22.5,37])for(const pitch of [-30,-15,0,18,30])for(const [key,id] of Object.entries(ids)){
  const p={...base,ParamAngleX:yaw,ParamAngleY:pitch,[FRAME_CACHE]:new Map()},q=featureWorld(id,fit.front[key],p);
  const near=featureWorld(id,fit.front[key],{...p,ParamAngleX:yaw+.001,[FRAME_CACHE]:new Map()});
  assert(Math.hypot(near[0]-q[0],near[1]-q[1])<.01,'intermediate feature continuity');
 }
});
test('upper canopy depth decreases from the center toward either outer edge',()=>{
 for(const y of [60,80,120,180,220,245,250])for(const sign of [-1,1]){
  let previous=Infinity;
  const {cx}=domeSection(y);
  for(let dx=0;dx<=280;dx+=5){const z=canopyWorld([502+cx+sign*dx,y],base)[2];assert(z<=previous+1e-9,`ridge ${sign},${y},${dx}`);previous=z;}
 }
});
test('all five flexible canopy layers share the same XYZ at every seam, including physics',()=>{
 for(const yaw of [-45,0,45])for(const pitch of [-30,0,30])for(const physics of [0,1]){
  const p={...base,ParamAngleX:yaw,ParamAngleY:pitch,ParamHairFront:physics,ParamHairSide:-physics};
  for(const xy of [[490,80],[501,120],[512,180],[502,210],[502,245],[330,180],[674,220],[380,300],[625,400]]){
   const first=at(ids[0],xy,p);
   for(const id of ids)assert(Math.hypot(...at(id,xy,p).map((v,i)=>v-first[i]))<1e-4,`${id} ${xy} ${yaw},${pitch}`);
   if(xy[1]<=245)assert.deepEqual(first,at(ids[0],xy,{...p,ParamHairFront:0,ParamHairSide:0}));
  }
 }
});
test('nine measured feature poses and normalized outline poses exist',()=>{
 const trace=JSON.parse(readFileSync(new URL('../source/viewer/outline-traces.json',import.meta.url)));
 const fit=JSON.parse(readFileSync(new URL('../source/viewer/face-feature-fit.json',import.meta.url)));
 assert.equal(trace.poses.length,9);assert.equal(fit.poses.length,9);
});
test('reference camera translation aligns binocular midpoints to the r11 pivots',()=>{
 const fit=JSON.parse(readFileSync(new URL('../source/viewer/face-feature-fit.json',import.meta.url)));
 for(const pose of fit.poses.filter(p=>p.pitch))for(const i of [0,1]){
  const mid=(pose.registeredCenters.left[i]+pose.registeredCenters.right[i])/2;
  assert(Math.abs(mid-pose.registration.eyeMid[i])<1e-8);
 }
});
test('opening envelopes preserve each independently measured pitch scale',()=>{
 const fit=JSON.parse(readFileSync(new URL('../source/viewer/face-feature-fit.json',import.meta.url)));
 for(const pose of fit.poses)for(const side of ['left','right']){
  assert(pose.scales[side].every(v=>Number.isFinite(v)&&v>0));
  if(pose.pitch)pose.scales[side].forEach((v,i)=>assert(Math.abs(v-pose.measuredScales[side][i]*pose.registration.scale)<1e-12));
 }
});
test('fringe receives its own hair solve instead of the cheek residual',()=>{
 for(const yaw of [-45,45])for(const pitch of [-30,30]){
  const p={...base,ParamAngleX:yaw,ParamAngleY:pitch};
  for(const xy of [[400,300],[600,330],[502,245],[674,330]]){
   assert.deepEqual(at('bangs-left',xy,p),applyHairSolve(canopyWorld(xy,p),xy,p));
   assert.deepEqual(at('bangs-left',xy,{...p,CorrOutline:0}),canopyWorld(xy,{...p,CorrOutline:0}));
  }
 }
});
