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
  assert(yaw<0?back>=front:back<=front,`${yaw} ${gain} ${y}: ${back} / ${front}`);
 }
});
test('upper added depth has negative horizontal second derivative',()=>{
 const extra=x=>hairLift([x,180],base)[2]-hairLift([x,180],{...base,CorrHairDepth:.4})[2];
 assert(extra(402)+extra(602)-2*extra(502)<-1);
});
test('45 degree chin reaches the measured target',()=>{
 for(const yaw of [-45,45]){
  const q=deform3D(mesh('face-skin'),[502.5,616],{...base,CorrTurn:1,ParamAngleX:yaw});
  assert(Math.abs(q[0]-(yaw<0?441.5:586))<4,`${q}`);
 }
});

test('cranium unfolds only on the near side, with both parts in the atlas',()=>{
 for(const side of ['left','right']){
  const m=mesh('side-cranium-'+side);assert(m&&m.triangles.length>0);
  assert.equal(opacity(m,base,state),0);
  const nearYaw=side==='right'?-45:45;
  assert.equal(opacity(m,{...base,ParamAngleX:nearYaw,CorrTurn:1},state),1);
  assert.equal(opacity(m,{...base,ParamAngleX:-nearYaw,CorrTurn:1},state),0);
  const [x,y,w]=m.bbox;
  const width=p=>deform3D(m,[x+w,y+100],p)[0]-deform3D(m,[x,y+100],p)[0];
  assert(width({...base,ParamAngleX:nearYaw,CorrTurn:1})>width(base));
 }
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
