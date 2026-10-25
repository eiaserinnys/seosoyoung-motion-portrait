import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
import {bindRig,deform3D,opacity,HAIRPIN_ANCHOR} from '../bundle/rig.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
import {frozenRig} from './frozen_rig.mjs';
import {profileWeights} from '../bundle/face-profile.mjs';
import {BEADS,beadWorld} from '../bundle/stereo-attachments.mjs';
import {applyHairSolve,applyTailSolve} from '../bundle/hair-solve.mjs';
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

test('side90 keeps all nine poses unchanged apart from the approved face calibration and hair residual',{timeout:60000},async()=>{
 // Compare the side90 code against the old runtime using the same approved
 // face targets; the independent face/hair tests verify those calibrations.
 const old=await frozenRig(false,'0b75a0211691965ca234ad9114073ba934568394',
  ['face-solve-data.mjs','feature-data.mjs','outline-data.mjs','face-feature-prior.mjs','face-features.mjs','face-profile.mjs']);
 try{
  for(const yaw of [-45,0,45])for(const pitch of [-30,0,30]){
   const p={...at(yaw),ParamAngleY:pitch},q={...p,[old.FRAME_CACHE]:new Map()};delete q[FRAME_CACHE];
   for(const m of rig.meshes){
    assert.equal(opacity(m,p,state),old.opacity(m,q,state),`${yaw}/${pitch} ${m.id} opacity`);
    const hair=m.id==='hair-top'||m.id.startsWith('bangs')||m.id.startsWith('side-hair')||m.id==='hairpin'||m.id.startsWith('back-hair')||m.id.startsWith('side-cranium');
    for(const point of m.positions){
     const prior=old.deform3D(m,point,q);let expected;
     if(m.id==='hairpin'){
      const a=old.deform3D(m,HAIRPIN_ANCHOR,q),b=applyHairSolve(a,HAIRPIN_ANCHOR,p);
      expected=[prior[0]+(b[0]-a[0]),prior[1]+(b[1]-a[1]),prior[2]];
     }else expected=hair?applyHairSolve(prior,point,p,m.id.startsWith('side-cranium')):m.id.startsWith('tail-')?applyTailSolve(prior,point,m.id.includes('-left-')?'left':'right',p):prior;
     const actual=deform3D(m,point,p);
     if(m.id==='hairpin'&&pitch)assert(Math.hypot(...actual.map((v,i)=>v-expected[i]))<1e-10,`${yaw}/${pitch} rigid pin`);
     else assert.deepEqual(actual,expected,`${yaw}/${pitch} ${m.id}`);
    }
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
   assert.equal(opacity(m,at(sign*50),state),1);
   assert(opacity(m,at(sign*58),state)>0&&opacity(m,at(sign*58),state)<1);
   assert.equal(opacity(m,at(sign*67.5),state),0);
   assert.equal(opacity(m,at(sign*90),state),0);
   assert.equal(opacity(mesh(`eye-${near}-${kind}`),at(sign*90),state),1);
  }
  assert.equal(opacity(mesh('nose-detail'),at(sign*50),state),1);
  assert.equal(opacity(mesh('nose-detail'),at(sign*90),state),0);
 }
});
test('distant side locks and tails disappear before the rejected 67.5 degree poses',()=>{
 for(const yaw of [-67.5,-90,67.5,90]){
  const far=yaw<0?'left':'right';
  for(const id of [`side-hair-${far}`,`tail-${far}-tie`,`tail-${far}-hair`])assert.equal(opacity(mesh(id),at(yaw),state),0,id);
 }
});
test('all modified attachment vertices meet the 45 degree seam continuously',{timeout:60000},()=>{
 for(const sign of [-1,1]){
  const p=at(sign*45),q=at(sign*45.000001);
  for(const m of rig.meshes.filter(m=>/^(ear-|hair-top$|bangs-|side-hair-|side-cranium-|back-hair-|hairpin$|tail-)/.test(m.id)))for(const point of m.positions){
   const a=deform3D(m,point,p),b=deform3D(m,point,q);
   assert(Math.hypot(...a.map((v,i)=>v-b[i]))<.001,`${m.id} seam`);
  }
 }
});
test('bead split preserves non-hair vertices, alpha and all neutral geometry',{timeout:60000},async()=>{
 const old=await frozenRig(false,'3ba80b1fb3ed36e87258367fdaabab8deb0973bc',
  ['face-solve-data.mjs','feature-data.mjs','outline-data.mjs','face-feature-prior.mjs','face-features.mjs','face-profile.mjs']);
 try{
  const poses=[...[-45,0,45].flatMap(y=>[-30,0,30].map(p=>[y,p])),[-44.99,0],[44.99,0],
   ...[-30,-22.5,-15,15,22.5,30].flatMap(y=>(Math.abs(y)===22.5?[-15,15]:[-20,20]).map(p=>[y,p]))];
  for(const [yaw,pitch]of poses){
   const p={...at(yaw),ParamAngleY:pitch},q={...p,[old.FRAME_CACHE]:new Map()};delete q[FRAME_CACHE];
   for(const m of rig.meshes){
    assert.equal(opacity(m,p,state),old.opacity(m,q,state),m.id);
    const hair=/^(hair-top$|bangs-|side-hair-|hairpin$|back-hair-|side-cranium-|tail-)/.test(m.id);
    if(!pitch||!hair)for(const point of m.positions)assert.deepEqual(deform3D(m,point,p),old.deform3D(m,point,q),`${yaw}/${pitch} ${m.id}`);
   }
  }
 }finally{old.dispose();}
 for(const name of ['hair-shell.mjs','hair-surface.mjs','dome-data.mjs','assets/texture_00.png','assets/texture_before_00.png','assets/seosoyoung-rig.json'])
  assert.deepEqual(readFileSync(new URL('../bundle/'+name,import.meta.url)),execFileSync('git',['show',`0b75a0211691965ca234ad9114073ba934568394:bundle/${name}`],{maxBuffer:16*1024*1024}));
});
const bounds=points=>[Math.min(...points.map(p=>p[0])),Math.min(...points.map(p=>p[1])),Math.max(...points.map(p=>p[0])),Math.max(...points.map(p=>p[1]))];
test('profile rabbit pin matches reference canvas boxes and far attachments disappear',()=>{
 for(const [yaw,target]of [[-67.5,[457,252,525,310]],[-90,[457,260,525,320]]]){
  const actual=bounds(mesh('hairpin').positions.map(v=>deform3D(mesh('hairpin'),v,at(yaw))));
  actual.forEach((v,i)=>assert(Math.abs(v-target[i])<=4,`${yaw} pin ${actual}`));
 }
 for(const yaw of [-90,-67.5,67.5,90]){
  for(const id of ['back-hair-left','back-hair-center','back-hair-right'])assert.equal(opacity(mesh(id),at(yaw),state),0,`${yaw} ${id}`);
  if(yaw>0)assert.equal(opacity(mesh('hairpin'),at(yaw),state),0);
 }
});
test('profile posterior follows the rounded reference and reaches the bead top',()=>{
 for(const yaw of [-90,-67.5,67.5,90]){
  const sign=yaw<0?1:-1,m=mesh('side-cranium-'+(sign>0?'right':'left'));
  for(const [y,x]of [[420,801],[460,794],[500,776],[540,736]]){
   const q=deform3D(m,[502,y,sign*Math.PI],at(yaw));
   assert(Math.abs(q[0]-(sign>0?x:1004-x))<=3,`${yaw}/${y}: ${q}`);
  }
  const bottom=deform3D(m,[502,500,sign*Math.PI],at(yaw));
  assert(bottom[1]>=535,`${yaw} posterior ends ${bottom}`);
  for(const [y,x]of [[300,450],[420,505],[460,535],[500,565],[540,605]]){
   const q=deform3D(m,[502,y,sign*88*Math.PI/180],at(yaw));
   assert(Math.abs(q[0]-(sign>0?x:1004-x))<=3,`${yaw}/${y} front ${q}`);
  }
 }
});
test('overlapping flexible front hair layers share the profile field including pitch and roll',{timeout:60000},()=>{
 // The rigid pin follows one anchor translation; it is not a flexible seam.
 const layers=rig.meshes.filter(m=>/^(hair-top$|bangs-|side-hair-)/.test(m.id)),groups=new Map();
 for(const m of layers)for(const v of m.positions){
  const key=v.slice(0,2).join(',');if(!groups.has(key))groups.set(key,[]);groups.get(key).push([m,v]);
 }
 const shared=[...groups.values()].filter(g=>g.length>1);assert(shared.length>0);
 for(const yaw of [-90,-67.5,67.5,90])for(const [pitch,roll]of [[0,0],[15,10],[-15,-10]]){
  const p={...at(yaw),ParamAngleY:pitch,ParamAngleZ:roll};
  for(const group of shared){
   const [m,v]=group[0],expected=deform3D(m,v,p);
   for(const [layer,point]of group.slice(1))assert.deepEqual(deform3D(layer,point,p),expected,`${yaw}/${pitch}/${roll} ${m.id}/${layer.id} ${v}`);
  }
 }
});
test('profile bangs precede the forehead and the near front chart stays oriented',()=>{
 for(const yaw of [-90,-67.5,67.5,90]){
  const p=at(yaw),sign=yaw<0?1:-1;
  for(const y of [260,300,340,380]){
   const hair=deform3D(mesh('hair-top'),[502,y],p)[0];
   const skin=mesh('face-skin').positions.filter(v=>Math.abs(v[1]-y)<12).map(v=>deform3D(mesh('face-skin'),v,p)[0]);
   assert(skin.length);
   const diff=sign>0?Math.min(...skin)-hair:hair-Math.max(...skin);
   assert(diff>=15&&diff<=30,`${yaw}/${y}: forehead clearance ${diff}px`);
  }
  for(const m of [mesh('hair-top'),mesh('bangs-'+(sign>0?'right':'left')),mesh('side-hair-'+(sign>0?'right':'left')),mesh('hairpin')]){
   let checked=0;
   for(const ids of m.triangles){
    const rest=ids.map(i=>m.positions[i]);
    if(!rest.every(([x])=>sign*(x-502)>=0))continue;
    const q=rest.map(v=>deform3D(m,v,p)),area=([a,b,c])=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
    assert(area(q)/area(rest)>0,`${yaw} ${m.id} inverted ${JSON.stringify(rest)}`);checked++;
   }
   if(m.id!=='hairpin'||sign>0)assert(checked>0);
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
 for(const yaw of [-90,-67.5,-45,0,45,67.5,90])for(const pitch of [-30,0,30])for(const side of ['left','right']){
  const p={...at(yaw),ParamAngleY:pitch},root=beadWorld(side,p);
  for(const suffix of ['tie','hair'])assert.deepEqual(deform3D(mesh(`tail-${side}-${suffix}`),BEADS[side].front,p),root);
  if(Math.abs(yaw)===90&&side===(yaw<0?'left':'right')){
   assert.equal(opacity(mesh(`tail-${side}-hair`),p,state),0);
   assert(Math.abs(root[0]-502)>180,'hidden root stays behind the side of the skull');
  }
 }
});
