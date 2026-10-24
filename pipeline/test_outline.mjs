import './test_stereo.mjs';
import {craniumFacing,domeSection} from '../bundle/hair-surface.mjs';
import {DOME} from '../bundle/dome-data.mjs';
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {bindRig,deform3D,opacity} from '../bundle/rig.mjs';
import {CORRECTION_PRESETS,correctedLift} from '../bundle/geometry.mjs';
import {hairLift} from '../bundle/hair-shell.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));bindRig(rig);
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended};
const mesh=id=>rig.meshes.find(m=>m.id===id);
const state={hidden:new Set(),outfit:'sweater'};
test('front ears lie behind cheek and side hair',()=>{
 for(const side of ['left','right'])for(const y of [420,460,500]){
  const x=side==='left'?360:640,p=base;
  assert(deform3D(mesh('ear-'+side),[x,y],p)[2]<deform3D(mesh('side-hair-'+side),[x,y],p)[2]);
  assert(deform3D(mesh('ear-'+side),[x,y],p)[2]<correctedLift([x,y],0,p)[2]);
 }
});
test('far ear is fully hidden at 45 degrees',()=>{
 for(const yaw of [-45,45])assert.equal(opacity(mesh('ear-'+(yaw<0?'left':'right')),{...base,ParamAngleX:yaw,CorrTurn:1},state),0);
});
test('far back hair contracts at least as much as side hair',()=>{
 for(const gain of [.65,1])for(const yaw of [-50,-25,25,50])for(const y of [350,440,530]){
  const side=yaw<0?'left':'right',x=yaw<0?210:775,p={...base,ParamAngleX:yaw,CorrTurn:gain};
  const back=deform3D(mesh('back-hair-'+side),[x,y],p)[0],front=deform3D(mesh('side-hair-'+side),[x,y],p)[0];
  assert(yaw<0?back>=front-1e-8:back<=front+1e-8,`${yaw} ${gain} ${y}: ${back} / ${front}`);
 }
});
test('upper added depth has negative horizontal second derivative',()=>{
 const extra=x=>hairLift([x,180],base)[2]-hairLift([x,180],{...base,CorrHairDepth:.4})[2];
 assert(extra(402)+extra(602)-2*extra(502)<-1);
});
test('45 degree chin reaches the measured target',()=>{
 const trace=JSON.parse(readFileSync(new URL('../source/viewer/outline-traces.json',import.meta.url))),row=trace.rows.at(-1);
 for(const yaw of [-45,45]){
  const q=deform3D(mesh('face-skin'),[502.5,616],{...base,CorrTurn:1,ParamAngleX:yaw});
  const target=yaw<0?(row[3]+row[4])/2:(row[5]+row[6])/2;
  // The edge fit retains the curved 3D interior, rather than flattening it.
  assert(Math.abs(q[0]-target)<1,`${q}`);
 }
});

test('opaque cranium emerges geometrically from the near horizon',()=>{
 for(const side of ['left','right']){
  const m=mesh('side-cranium-'+side);assert(m&&m.triangles.length>0);
  const nearYaw=side==='right'?-45:45;
  for(const yaw of [0,-nearYaw,nearYaw]){
   const p={...base,ParamAngleX:yaw,CorrTurn:1};
   assert.equal(opacity(m,p,state),1);
   assert(m.positions.every(q=>Number.isFinite(craniumFacing(q,p))));
   const visible=m.positions.filter(q=>craniumFacing(q,p)>1e-6).length;
   assert(yaw===nearYaw?visible>0:visible===0,`${side} ${yaw}: ${visible}`);
   // The two-degree overlap folds onto the horizon at rest; actual raster
   // occlusion is checked independently in the browser.
  }
 }
});
test('lower skull fits the ear attachment and measured 45-degree silhouette together',()=>{
 const {rx,rz,cz}=domeSection(465);
 assert(Math.abs(cz+rz*Math.sqrt(1-(DOME.lower.attachmentX/rx)**2)-DOME.lower.attachmentZ)<1e-8);
 assert(Math.abs((Math.hypot(rx,rz)-cz)/Math.sqrt(2)-DOME.lower.extent45)<1e-8);
 assert(rx<domeSection(330).rx);
});

test('all sampled cheek and jaw edges match independent SAM 3 targets at 45 degrees',()=>{
 const trace=JSON.parse(readFileSync(new URL('../source/viewer/outline-traces.json',import.meta.url)));
 for(const yaw of [-45,45])for(const [y,l,r,ll,lr,rl,rr,ty] of trace.rows.filter(row=>row[0]>=465)){
  const expected=yaw<0?[ll,lr]:[rl,rr];
  for(const [i,x] of [l,r].entries()){
   const q=deform3D(mesh('face-skin'),[x,y],{...base,ParamAngleX:yaw,CorrTurn:1});
   assert(Math.abs(q[0]-expected[i])<.001,`${yaw} ${y}: ${q[0]} vs ${expected[i]}`);
   assert(Math.abs(q[1]-ty)<.001);
  }
 }
});

test('inverse spherical longitude restores both reference meshes at ±45 degrees',()=>{
 for(const [side,yaw] of [['right',-45],['left',45]]){
  const m=mesh('side-cranium-'+side);
  for(const xy of m.positions){
   const q=deform3D(m,xy,{...base,ParamAngleX:yaw,CorrTurn:1});
   assert(Math.hypot(q[0]-xy[0],q[1]-xy[1])<1e-8);
  }
 }
});
