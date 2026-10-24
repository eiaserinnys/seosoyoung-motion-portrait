import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {bindRig,deform3D} from '../bundle/rig.mjs';
import {CORRECTION_PRESETS,artWorld,correctedLift} from '../bundle/geometry.mjs';
import {outlineWorld,outlineRow,outlinePitchShift} from '../bundle/face-outline.mjs';
import {featureWorld} from '../bundle/face-features.mjs';
import {canopyWorld} from '../bundle/hair-shell.mjs';
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
test('non-node face displacements mirror exactly around each rest row',()=>{
 for(const yaw of [10,22.5,37,45])for(const pitch of [-30,-15,0,18,30])for(const y of [465,510,570,610])for(const u of [0,.25,.5,.75,1]){
  const [l,r]=outlineRow(y),a=[l+(r-l)*u,y],b=[r-(r-l)*u,y];
  const p={...base,ParamAngleX:-yaw,ParamAngleY:pitch},q={...p,ParamAngleX:yaw};
  const left=outlineWorld(a,p),right=outlineWorld(b,q),da=outlinePitchShift(a,p),db=outlinePitchShift(b,q);
  assert(Math.abs(left[0]-a[0]+right[0]-b[0])<1e-6,`outline X ${yaw},${pitch},${y},${u}`);
  assert(Math.abs(left[1]-right[1])<1e-6,`outline Y ${yaw},${pitch},${y},${u}`);
  assert(Math.abs(da[0]+db[0])<1e-9);assert(Math.abs(da[1]-db[1])<1e-9);
 }
});
test('non-node feature centers and scales mirror while neutral art stays fixed',()=>{
 const fit=JSON.parse(readFileSync(new URL('../source/viewer/face-feature-fit.json',import.meta.url)));
 const ids={left:'eye-left-upper-lashes',right:'eye-right-upper-lashes',browLeft:'eye-left-brow',browRight:'eye-right-brow',nose:'nose-detail',mouth:'mouth-detail'};
 for(const [key,id] of Object.entries(ids)){
  const front=fit.front[key],a=featureWorld(id,front,base);
  assert(Math.hypot(a[0]-front[0],a[1]-front[1])<1e-9);
 }
 for(const yaw of [10,22.5,37,45])for(const pitch of [-30,-15,0,18,30])for(const [a,b] of [['left','right'],['browLeft','browRight'],['nose','nose'],['mouth','mouth']]){
  const p={...base,ParamAngleX:-yaw,ParamAngleY:pitch},q={...p,ParamAngleX:yaw};
  const l=featureWorld(ids[a],fit.front[a],p),r=featureWorld(ids[b],fit.front[b],q);
  assert(Math.abs(l[0]-fit.front[a][0]+r[0]-fit.front[b][0])<1e-6,`${a} X ${yaw},${pitch}`);
  assert(Math.abs(l[1]-fit.front[a][1]-r[1]+fit.front[b][1])<1e-6,`${a} Y ${yaw},${pitch}`);
  if(a==='left')for(const axis of [0,1]){
   const span=(key,pose)=>{const pt=fit.front[key],lo=pt.slice(),hi=pt.slice();lo[axis]-=5;hi[axis]+=5;return featureWorld(ids[key],hi,pose)[axis]-featureWorld(ids[key],lo,pose)[axis];};
   assert(Math.abs(span(a,p)-span(b,q))<1e-6,`eye scale ${axis} ${yaw},${pitch}`);
  }
 }
});
test('upper canopy depth decreases from the center toward either outer edge',()=>{
 for(const y of [60,80,120,180,220,245,250])for(const sign of [-1,1]){
  let previous=Infinity;
  for(let dx=0;dx<=280;dx+=5){const z=canopyWorld([502+sign*dx,y],base)[2];assert(z<=previous+1e-9,`ridge ${sign},${y},${dx}`);previous=z;}
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
test('all pitched mask edges and feature anchors reach their measured keyforms',()=>{
 const trace=JSON.parse(readFileSync(new URL('../source/viewer/outline-traces.json',import.meta.url)));
 const fit=JSON.parse(readFileSync(new URL('../source/viewer/face-feature-fit.json',import.meta.url)));
 const source=trace.rows.filter(r=>r[0]>=465);
 for(const pose of trace.poses.filter(p=>p.pitch)){
  const p={...base,ParamAngleX:pose.yaw,ParamAngleY:pose.pitch};
  pose.rows.forEach(([ty,l,r],i)=>{
   for(const [edge,[x,target]] of [[source[i][1],l],[source[i][2],r]].entries()){
    const q=at('face-skin',[x,source[i][0]],p);
    const s=(source[i][0]-465)/151;
    const expectedY=ty+(pose.registration.topEdgeY[edge]-pose.yTop)*(1-s);
    assert(Math.hypot(q[0]-target,q[1]-expectedY)<1e-6,`${pose.name} row ${i}`);
   }
  });
 }
 for(const pose of fit.poses.filter(p=>p.pitch))for(const [key,id] of Object.entries({left:'eye-left-upper-lashes',right:'eye-right-upper-lashes',browLeft:'eye-left-brow',browRight:'eye-right-brow',nose:'nose-detail',mouth:'mouth-detail'})){
  const q=at(id,fit.front[key],{...base,ParamAngleX:pose.yaw,ParamAngleY:pose.pitch});
  assert(Math.hypot(q[0]-pose.centers[key][0],q[1]-pose.centers[key][1])<1e-6,`${pose.name} ${key}`);
 }
});
test('reference camera translation aligns binocular midpoints to the r11 pivots',()=>{
 const fit=JSON.parse(readFileSync(new URL('../source/viewer/face-feature-fit.json',import.meta.url)));
 for(const pose of fit.poses.filter(p=>p.pitch))for(const i of [0,1]){
  const mid=(pose.registeredCenters.left[i]+pose.registeredCenters.right[i])/2;
  assert(Math.abs(mid-pose.registration.eyeMid[i])<1e-8);
 }
});
test('occlusion never collapses an eye keyform and central pitch pairs are symmetric',()=>{
 const fit=JSON.parse(readFileSync(new URL('../source/viewer/face-feature-fit.json',import.meta.url)));
 for(const pose of fit.poses){
  for(const side of ['left','right'])assert(pose.scales[side][1]>=.88&&pose.scales[side][1]<=1.02);
  if(!pose.yaw)assert.equal(pose.scales.left[1],pose.scales.right[1]);
 }
});
test('pitched cheek baseline has zero vertical residual, while neck cap follows the chin',()=>{
 for(const yaw of [-45,0,45])for(const pitch of [-30,30]){
  const p={...base,ParamAngleX:yaw,ParamAngleY:pitch},pure={...p,CorrOutline:0};
  for(const x of [346,502,663]){
   const [l,r]=outlineRow(465),offset=(l+r)/2-502;
   const q=at('face-skin',[x,465],p),plain=artWorld(correctedLift([x-offset,465],0,p),p);
   assert(Math.abs(q[1]-plain[1])<1e-8,'s=0 preserves rigid cheek height');
  }
  const chin=at('face-skin',[502.5,616],p),chinPlain=at('face-skin',[502.5,616],pure);
  const neck=at('neck',[502,540],p),neckPlain=at('neck',[502,540],pure);
  assert(Math.abs((neck[1]-neckPlain[1])-(chin[1]-chinPlain[1]))<1e-8);
 }
});
test('forehead and fringe never inherit cheek residuals',()=>{
 for(const yaw of [-45,45])for(const pitch of [-30,30]){
  const p={...base,ParamAngleX:yaw,ParamAngleY:pitch};
  for(const xy of [[400,300],[600,330],[502,245],[674,330]]){
   assert.deepEqual(at('bangs-left',xy,p),at('bangs-left',xy,{...p,CorrOutline:0}));
  }
 }
});
