import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
import {bindRig,deform3D,opacity} from '../bundle/rig.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
import {frozenRig} from './frozen_rig.mjs';
import {profileWeights} from '../bundle/face-profile.mjs';
import {BEADS,beadWorld} from '../bundle/stereo-attachments.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));bindRig(rig);
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1};
const at=yaw=>({...base,ParamAngleX:yaw,[FRAME_CACHE]:new Map()});
const state={hidden:new Set(),outfit:'sweater'};
const mesh=id=>rig.meshes.find(m=>m.id===id);
test('profile skin weights are exactly zero through 45 and reach the side references',()=>{
 for(const yaw of [-45,-44.99,0,44.99,45])assert.equal(profileWeights(at(yaw)).skin,0);
 for(const sign of [-1,1]){
  assert.equal(profileWeights(at(sign*67.5)).skin,1);
  assert.equal(profileWeights(at(sign*90)).profile,1);
  assert(profileWeights(at(sign*45.001)).skin>0);
 }
});

test('every vertex and alpha in the original nine poses equals the pinned main baseline',{timeout:60000},async()=>{
 const old=await frozenRig(false,'0b75a0211691965ca234ad9114073ba934568394');
 try{
  for(const yaw of [-45,0,45])for(const pitch of [-30,0,30]){
   const p={...at(yaw),ParamAngleY:pitch},q={...p,[old.FRAME_CACHE]:new Map()};delete q[FRAME_CACHE];
   for(const m of rig.meshes){
    assert.equal(opacity(m,p,state),old.opacity(m,q,state),`${yaw}/${pitch} ${m.id} opacity`);
    for(const point of m.positions)assert.deepEqual(deform3D(m,point,p),old.deform3D(m,point,q),`${yaw}/${pitch} ${m.id}`);
   }
  }
  for(const file of ['texture_00.png','texture_before_00.png','seosoyoung-rig.json']){
   assert.deepEqual(readFileSync(new URL('../bundle/assets/'+file,import.meta.url)),execFileSync('git',['show',`0b75a0211691965ca234ad9114073ba934568394:bundle/assets/${file}`],{maxBuffer:16*1024*1024}));
  }
 }finally{old.dispose();}
});
test('far eyes and brows fade on the specified side, and the frontal nose disappears',()=>{
 for(const sign of [-1,1]){
  const far=sign<0?'left':'right',near=sign<0?'right':'left';
  for(const kind of ['sclera','iris','pupil','highlight','upper-lashes','lower-line','brow']){
   const m=mesh(`eye-${far}-${kind}`);
   assert.equal(opacity(m,at(sign*60),state),1);
   assert(opacity(m,at(sign*67.5),state)>0&&opacity(m,at(sign*67.5),state)<1);
   assert.equal(opacity(m,at(sign*90),state),0);
   assert.equal(opacity(mesh(`eye-${near}-${kind}`),at(sign*90),state),1);
  }
  assert.equal(opacity(mesh('nose-detail'),at(sign*50),state),1);
  assert.equal(opacity(mesh('nose-detail'),at(sign*90),state),0);
 }
});
test('all modified attachment vertices meet the 45 degree seam continuously',{timeout:60000},()=>{
 for(const sign of [-1,1]){
  const p=at(sign*45),q=at(sign*45.000001);
  for(const m of rig.meshes.filter(m=>/^(ear-|side-hair-|tail-)/.test(m.id)))for(const point of m.positions){
   const a=deform3D(m,point,p),b=deform3D(m,point,q);
   assert(Math.hypot(...a.map((v,i)=>v-b[i]))<.001,`${m.id} seam`);
  }
 }
});
test('profile near ears continue toward the registered anatomical boxes',()=>{
 for(const [side,sign,point,target] of [['right',-1,[671,464.5],[574,475]],['left',1,[333,465],[449.5,477]]]){
  const q=deform3D(mesh('ear-'+side),point,at(sign*90));
  assert(Math.abs(q[0]-target[0])<2&&Math.abs(q[1]-target[1])<2,`${side}: ${q}`);
 }
});
test('profile eyelids and eye layers still respond to independent blink controls',()=>{
 for(const yaw of [-90,-67.5,67.5,90])for(const side of ['left','right']){
  const p=at(yaw),key=side==='left'?'ParamEyeLOpen':'ParamEyeROpen';
  assert.equal(opacity(mesh(`eye-${side}-sclera`),{...p,[key]:0},state),0);
  assert.equal(opacity(mesh('closed-eyes-v1'),{...p,ParamEyeLOpen:0,ParamEyeROpen:0},state),1);
 }
});
test('both tail layers retain the shared root while the hidden side passes behind the skull',()=>{
 for(const yaw of [-90,-67.5,67.5,90])for(const side of ['left','right']){
  const p=at(yaw),root=beadWorld(side,p);
  for(const suffix of ['tie','hair'])assert.deepEqual(deform3D(mesh(`tail-${side}-${suffix}`),BEADS[side].front,p),root);
  if(Math.abs(yaw)===90&&side===(yaw<0?'left':'right'))assert.equal(opacity(mesh(`tail-${side}-hair`),p,state),0);
 }
});
