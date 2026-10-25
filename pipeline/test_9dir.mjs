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
const ids=['hair-top','bangs-left','bangs-right','side-hair-left','side-hair-right','hairpin'];
const at=(id,xy,p)=>deform3D(rig.meshes.find(m=>m.id===id),xy,p);
test('neutral face and feature vertices retain every original canvas coordinate',()=>{
 for(const mesh of rig.meshes.filter(m=>m.id==='face-skin'||m.id.startsWith('eye-')||m.id==='nose-detail'||m.id.startsWith('mouth-')||m.id==='closed-eyes-v1'))for(const point of mesh.positions){
  const q=deform3D(mesh,point,base);
  assert(Math.hypot(q[0]-point[0],q[1]-point[1])<1e-9,mesh.id);
 }
});
test('skin retains key-pose symmetry and bounded continuous intermediate 3D symmetry',()=>{
 for(const yaw of [10,22.5,37,45])for(const pitch of [-30,-15,0,18,30])for(const y of [465,510,570,610])for(const u of [0,.25,.5,.75,1]){
  // Pitched interior now follows the six authored feature anchors. Only the
  // measured outline endpoints have a mirror contract in the shared field.
  if(pitch&&u!==0&&u!==1)continue;
  if(pitch&&yaw!==45&&![-15,18].includes(pitch))continue;
  const [l,r]=outlineRow(y),a=[l+(r-l)*u,y],b=[r-(r-l)*u,y];
  const p={...base,ParamAngleX:-yaw,ParamAngleY:pitch,[FRAME_CACHE]:new Map()},q={...p,ParamAngleX:yaw,[FRAME_CACHE]:new Map()};
  const left=outlineWorld(a,p),right=outlineWorld(b,q);
  const restMid=(a[0]+b[0])/2;
  const even=(1-Math.cos(pitch*Math.PI/180))/(1-Math.cos(Math.PI/6));
  const targetMid=restMid+(502-restMid)*even;
  // Neutral interior queries have the same permitted chord approximation as
  // the r12 non-vertex contract; measured rim nodes remain exact.
  const limit=!pitch?(u===0||u===1?1e-6:1.2):yaw===45&&Math.abs(pitch)===30?1e-6:1;
  assert(Math.abs(left[0]+right[0]-targetMid*2)<limit,`outline X ${yaw},${pitch},${y},${u}`);
  assert(Math.abs(left[1]-right[1])<limit,`outline Y ${yaw},${pitch},${y},${u}`);
  for(const sign of [-1,1]){
   const angle=-yaw+sign*.001,near=outlineWorld(a,{...p,ParamAngleX:angle,[FRAME_CACHE]:new Map()});
   assert(Math.hypot(near[0]-left[0],near[1]-left[1])<.01,'intermediate skin continuity');
  }
 }
});
test('feature centers retain key symmetry and bounded intermediate 3D symmetry with unchanged scales',()=>{
 const fit=JSON.parse(readFileSync(new URL('../source/viewer/face-feature-fit.json',import.meta.url)));
 const ids={left:'eye-left-upper-lashes',right:'eye-right-upper-lashes',browLeft:'eye-left-brow',browRight:'eye-right-brow',nose:'nose-detail',mouth:'mouth-detail'};
 for(const [key,id] of Object.entries(ids)){
  const front=fit.front[key],a=featureWorld(id,front,base);
  assert(Math.hypot(a[0]-front[0],a[1]-front[1])<1e-9);
 }
 for(const yaw of [10,22.5,37,45])for(const pitch of [-30,-15,0,18,30])for(const [a,b] of [['left','right'],['browLeft','browRight'],['nose','nose'],['mouth','mouth']]){
  if(pitch&&yaw!==45&&![-15,18].includes(pitch))continue;
  const p={...base,ParamAngleX:-yaw,ParamAngleY:pitch,[FRAME_CACHE]:new Map()},q={...p,ParamAngleX:yaw,[FRAME_CACHE]:new Map()};
  const l=featureWorld(ids[a],fit.front[a],p),r=featureWorld(ids[b],fit.front[b],q);
  const limit=!pitch||yaw===45&&Math.abs(pitch)===30?1e-6:1;
  assert(Math.abs(l[0]-fit.front[a][0]+r[0]-fit.front[b][0])<limit,`${a} X ${yaw},${pitch}`);
  assert(Math.abs(l[1]-fit.front[a][1]-r[1]+fit.front[b][1])<limit,`${a} Y ${yaw},${pitch}`);
  const near=featureWorld(ids[a],fit.front[a],{...p,ParamAngleX:-yaw+.001,[FRAME_CACHE]:new Map()});
  assert(Math.hypot(near[0]-l[0],near[1]-l[1])<.01,'intermediate feature continuity');
  if(a==='left')for(const axis of [0,1]){
   const span=(key,pose)=>{const pt=fit.front[key],lo=pt.slice(),hi=pt.slice();lo[axis]-=5;hi[axis]+=5;return featureWorld(ids[key],hi,pose)[axis]-featureWorld(ids[key],lo,pose)[axis];};
   assert(Math.abs(span(a,p)-span(b,q))<1e-6,`eye scale ${axis} ${yaw},${pitch}`);
  }
 }
});
test('upper canopy depth decreases from the center toward either outer edge',()=>{
 for(const y of [60,80,120,180,220,245,250])for(const sign of [-1,1]){
  let previous=Infinity;
  const {cx}=domeSection(y);
  for(let dx=0;dx<=280;dx+=5){const z=canopyWorld([502+cx+sign*dx,y],base)[2];assert(z<=previous+1e-9,`ridge ${sign},${y},${dx}`);previous=z;}
 }
});
test('all six canopy layers share the same XYZ at every seam, including physics',()=>{
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
test('opening envelopes stay finite without height clamps and central pitch pairs are symmetric',()=>{
 const fit=JSON.parse(readFileSync(new URL('../source/viewer/face-feature-fit.json',import.meta.url)));
 for(const pose of fit.poses){
  for(const side of ['left','right'])assert(pose.scales[side].every(v=>Number.isFinite(v)&&v>0));
  if(!pose.yaw)assert.equal(pose.scales.left[1],pose.scales.right[1]);
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
