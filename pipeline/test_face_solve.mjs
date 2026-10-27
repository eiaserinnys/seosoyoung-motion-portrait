import {approvedStageA,assertApprovedReversals} from './approved_stage_a.mjs';
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
import {faceSolveWorld,faceSolveAnchorWorld,evaluateFace45World} from '../bundle/face-solve.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
import {OUTLINE_POSES,OUTLINE_ROWS} from '../bundle/outline-data.mjs';
import {FEATURE_FIT} from '../bundle/feature-data.mjs';
import {featureWorld} from '../bundle/face-features.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
import {legacyOutlineWorld} from '../bundle/face-outline.mjs';
import {FACE_SOLVE} from '../bundle/face-solve-data.mjs';
import {bindRig,deform3D} from '../bundle/rig.mjs';
import {smoothstep} from '../bundle/face-profile.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));
bindRig(rig);
const skin=rig.meshes.find(m=>m.id==='face-skin');
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1};
const at=(yaw,pitch=0)=>({...base,ParamAngleX:yaw,ParamAngleY:pitch,[FRAME_CACHE]:new Map()});
const near=(a,b,tol=1)=>assert(Math.hypot(a[0]-b[0],a[1]-b[1])<=tol,`${a} vs ${b}`);
const ids={browLeft:'eye-left-brow',browRight:'eye-right-brow',left:'eye-left-sclera',right:'eye-right-sclera',nose:'nose-detail',mouth:'mouth-detail'};
function minDet(points){
 const area=(q,t)=>{const [a,b,c]=t.map(i=>q[i]);return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);};
 return Math.min(...skin.triangles.map(t=>area(points,t)/area(skin.positions,t)));
}
test('nine key poses retain all six measured feature anchors after the A-stage solve',{timeout:60000},()=>{
 for(const pose of OUTLINE_POSES){
  const p=at(pose.yaw,pose.pitch),fit=FEATURE_FIT.poses.find(f=>f.name===pose.name);
  for(const [key,id] of Object.entries(ids)){
   near(faceSolveAnchorWorld(key,p),fit.centers[key],1e-6);
   near(featureWorld(id,FEATURE_FIT.front[key],p),fit.centers[key],1e-6);
  }
 }
});
test('A-stage keys retain the approved reversal set and eight between keys remain oriented',{timeout:60000},()=>{
 for(const pose of approvedStageA.poses){const p=at(pose.yaw,pose.pitch);assertApprovedReversals(pose.name,skin.positions,skin.positions.map(pt=>faceSolveWorld(pt,p)),skin.triangles);}
});
test('orientation gate rejects an injected triangle reversal',()=>{
 const points=skin.positions.map(p=>p.slice()),[a,b]=skin.triangles[0];
 [points[a],points[b]]=[points[b],points[a]];assert(minDet(points)<0);
});
test('face-45 rest mesh meets face-0 at both yaw boundaries',{timeout:60000},()=>{
 for(const sign of [-1,1])for(const pitch of [-30,0,30])for(const point of skin.positions){
  const boundary=faceSolveWorld(point,at(sign*45,pitch));
  near(evaluateFace45World(point,at(sign*45,pitch)),boundary,1e-6);
  for(const [delta,limit] of [[.001,.01],[.01,.1]]){
   near(faceSolveWorld(point,at(sign*(45-delta),pitch)),boundary,limit);
   near(faceSolveWorld(point,at(sign*(45+delta),pitch)),boundary,limit);
  }
 }
});
test('cached and uncached arbitrary queries agree, including chin half-pixel and rim samples',()=>{
 const p=at(22.5,15),q={...p};delete q[FRAME_CACHE];
 for(const point of [[502,616],[502.5,616],[451.25,501.5],[300,200]])assert.deepEqual(faceSolveWorld(point,p),faceSolveWorld(point,q));
});
test('face-45 key poses reproduce every observed node and six feature centers',{timeout:60000},()=>{
 const inputs=JSON.parse(execFileSync('node',[new URL('./face_solve_inputs.mjs',import.meta.url).pathname],{timeout:60000}));
 assert.deepEqual(FACE_SOLVE.family45.poses.map(p=>p.name),['left67','left90','right67','right90']);
 for(const pose of inputs.family45.poses){
  const p=at(pose.yaw);
  FACE_SOLVE.points.forEach((pt,i)=>{
   const anchor=Object.entries(FACE_SOLVE.anchors).find(([,index])=>index===i);
   near(anchor?faceSolveAnchorWorld(anchor[0],p):faceSolveWorld(pt,p),pose.obs[i],1e-6);
  });
  for(const [key,id] of Object.entries(ids))near(featureWorld(id,FEATURE_FIT.front[key],p),pose.obs[FACE_SOLVE.anchors[key]],1e-6);
 }
});
test('face-45 anchors and skin meet face-0 at pitched seams',{timeout:60000},()=>{
 for(const sign of [-1,1])for(const pitch of [-30,0,30]){
  for(const [key,i] of Object.entries(FACE_SOLVE.anchors)){
   const boundary=faceSolveAnchorWorld(key,at(sign*45,pitch));
   near(evaluateFace45World(FACE_SOLVE.points[i],at(sign*45,pitch),undefined,0,[[i,1]]),boundary,1e-6);
   near(featureWorld(ids[key],FEATURE_FIT.front[key],at(sign*45,pitch)),boundary,1e-6);
   for(const delta of [-.001,.001])near(faceSolveAnchorWorld(key,at(sign*(45+delta),pitch)),boundary,.01);
  }
 }
});
test('face-45 intermediate turns and extreme diagonals keep every triangle oriented',{timeout:60000},()=>{
 const poses=[...[-90,-78.75,-67.5,-56.25,56.25,67.5,78.75,90].map(y=>[y,0]),...[-90,90].flatMap(y=>[-30,30].map(p=>[y,p]))];
 for(const [yaw,pitch] of poses){const p=at(yaw,pitch);assert(minDet(skin.positions.map(pt=>faceSolveWorld(pt,p)))>0,`${yaw},${pitch}`);}
});
test('side composition transports the pitched row while projection fixes only free skin X',()=>{
 for(const yaw of [-90,-67.5,67.5,90])for(const pitch of [-30,-15,0,15,30]){
  const p=at(yaw,pitch),edge=at(Math.sign(yaw)*45,pitch),flat=at(yaw),flatEdge=at(Math.sign(yaw)*45);
  const rows=FACE_SOLVE.samples.map(s=>{
   const a=[s.left,s.right].map(i=>faceSolveWorld(FACE_SOLVE.points[i],flatEdge));
   const b=[s.left,s.right].map(i=>faceSolveWorld(FACE_SOLVE.points[i],flat));
   return [(a[0][1]+a[1][1])/2,a[0][0],a[1][0],b[0][0],b[1][0]];
  });
  // Rim samples are fixed by the discrete projection; use them to test the
  // exact row map independently of the runtime's implementation helpers.
  const fixed=FACE_SOLVE.samples.flatMap(s=>[s.left,s.right,s.center]);
  for(const point of [...skin.positions,...fixed.map(i=>FACE_SOLVE.points[i])]){
   const actual=faceSolveWorld(point,p),boundary=faceSolveWorld(point,edge),turned=faceSolveWorld(point,flat),origin=faceSolveWorld(point,flatEdge);
   assert(Math.abs(actual[1]-boundary[1]-(turned[1]-origin[1]))<1e-6);
   if(!pitch){near(actual,turned,1e-6);continue;}
   let i=0;while(i<rows.length-2&&boundary[1]>rows[i+1][0])i++;
   const a=rows[i],b=rows[i+1],t=Math.max(0,Math.min(1,(boundary[1]-a[0])/(b[0]-a[0])));
   const row=a.map((v,k)=>v+t*(b[k]-v)),expected=row[3]+(boundary[0]-row[1])*(row[4]-row[3])/(row[2]-row[1]);
   if(fixed.some(i=>FACE_SOLVE.points[i]===point))assert(Math.abs(actual[0]-expected)<1e-6,'fixed rim changed by projection');
  }
 }
});
test('eyes, brows, iris and closed lids use the additional measured width ratios',()=>{
 for(const pose of FACE_SOLVE.family45.poses)for(const side of ['left','right']){
  const key=side==='left'?'browLeft':'browRight';
  for(const id of [`eye-${side}-sclera`,`eye-${side}-iris`,`eye-${side}-brow`,'closed-eyes-v1']){
   const anchor=FEATURE_FIT.front[id.endsWith('brow')?key:side],a=[anchor[0]-10,anchor[1]],b=[anchor[0]+10,anchor[1]];
   const width=p=>featureWorld(id,b,p)[0]-featureWorld(id,a,p)[0];
   const edge=width(at(Math.sign(pose.yaw)*45));
   const near=side===(pose.yaw<0?'right':'left');
   const expected=id.endsWith('brow')&&near?(Math.abs(pose.yaw)===67.5?.70:.48):pose.eyeWidthScale[side];
   // The shared nonlinear profile field contracts only the nasal corner.
   const ratio=width(at(pose.yaw))/edge;
   if(id.endsWith('brow')||!near)assert(Math.abs(ratio-expected)<1e-8,`${pose.name} ${id}`);
   else{
    assert(ratio<=expected+1e-8&&ratio>=expected*.85,`${pose.name} ${id}: ${ratio}`);
    assert(smoothstep(65,80,Math.abs(pose.yaw))>0);
   }
  }
 }
});
test('profile nose and mouth use approved inset centers and perspective widths',()=>{
 const targets={
  left67:{nose:[302,491],mouth:[316,548]},left90:{nose:[293,490],mouth:[308,548]},
  right67:{nose:[702,491],mouth:[688,548]},right90:{nose:[711,490],mouth:[696,548]},
 };
 for(const pose of FACE_SOLVE.family45.poses)for(const key of ['nose','mouth']){
  const id=key+'-detail',anchor=FEATURE_FIT.front[key],p=at(pose.yaw);
  near(featureWorld(id,anchor,p),targets[pose.name][key],1e-6);
  const half=key==='nose'?14:23.5;
  const ends=[[anchor[0]-half,anchor[1]],[anchor[0]+half,anchor[1]]];
  const edge=ends.map(pt=>featureWorld(id,pt,at(Math.sign(pose.yaw)*45)));
  const actual=ends.map(pt=>featureWorld(id,pt,p));
  const scale=Math.abs(pose.yaw)===67.5?.55:.35;
  assert(Math.abs((actual[1][0]-actual[0][0])/(edge[1][0]-edge[0][0])-scale)<1e-8,`${pose.name} ${key}`);
 }
});
test('profile mouth compression stays local to mouth-detail in a shared render frame',()=>{
 const anchor=FEATURE_FIT.front.mouth,point=[anchor[0]+23.5,anchor[1]];
 for(const yaw of [-90,-67.5,67.5,90])for(const first of ['mouth-detail','mouth-open-v1','mouth-round-v1']){
  const cached=at(yaw);featureWorld(first,point,cached);
  for(const id of ['mouth-detail','mouth-open-v1','mouth-round-v1']){
   const plain=at(yaw);delete plain[FRAME_CACHE];
   near(featureWorld(id,point,cached),featureWorld(id,point,plain),1e-6);
  }
 }
});
test('actual-angle controls span [-90,90] without changing recommended correction',()=>{
 const angle=rig.parameters.find(p=>p.id==='ParamAngleX');
 assert.equal(angle.min,-90);assert.equal(angle.max,90);assert.equal(CORRECTION_PRESETS.recommended.CorrTurn,.65);
});
test('renderer deformer chains pass the full angle domain to skin and features',{timeout:60000},()=>{
 for(const yaw of [-90,-67.5,-45,0,45,67.5,90]){
  for(const mesh of rig.meshes.filter(m=>m.id==='face-skin'||m.id.startsWith('eye-')||m.id==='nose-detail'||m.id==='mouth-detail')){
   const rendered=at(yaw),expected=at(yaw);
   for(const pt of mesh.positions){
    const wanted=mesh.id==='face-skin'?faceSolveWorld(pt,expected):featureWorld(mesh.id,pt,expected);
    near(deform3D(mesh,pt,rendered),wanted,1e-6);
   }
  }
 }
});
