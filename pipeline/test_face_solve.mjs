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
test('nine key poses reproduce measured silhouette and all six feature anchors',{timeout:60000},()=>{
 for(const pose of OUTLINE_POSES){
  const p=at(pose.yaw,pose.pitch),fit=FEATURE_FIT.poses.find(f=>f.name===pose.name);
  let squared=0,count=0;
  OUTLINE_ROWS.forEach(([y,l,r],i)=>{
   for(const [x,observed] of [[l,[pose.rows[i][2],pose.rows[i][1]]],[r,[pose.rows[i][4],pose.rows[i][3]]]]){
    const target=!pose.pitch&&y<465?legacyOutlineWorld([x,y],p):observed;
    const actual=faceSolveWorld([x,y],p);squared+=(actual[0]-target[0])**2+(actual[1]-target[1])**2;count++;
   }
  });
  assert(Math.sqrt(squared/count)<=1,pose.name);
  for(const [key,id] of Object.entries(ids)){
   near(faceSolveAnchorWorld(key,p),fit.centers[key]);
   near(featureWorld(id,FEATURE_FIT.front[key],p),fit.centers[key]);
  }
 }
});
test('all skin triangles retain positive orientation at key and sixteen intermediate poses',{timeout:60000},()=>{
 const poses=[...OUTLINE_POSES.map(p=>[p.yaw,p.pitch]),...[-30,-15,15,30].flatMap(y=>[-20,-10,10,20].map(p=>[y,p]))];
 for(const [yaw,pitch] of poses){const p=at(yaw,pitch);assert(minDet(skin.positions.map(pt=>faceSolveWorld(pt,p)))>0,`${yaw},${pitch}`);}
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
 for(const [yaw,pitch] of poses){const p=at(yaw,pitch);assert(minDet(skin.positions.map(pt=>faceSolveWorld(pt,p)))>=.15,`${yaw},${pitch}`);}
});
test('approved pitch attenuation scales only the additional horizontal turn displacement',()=>{
 for(const yaw of [-90,-67.5,67.5,90])for(const pitch of [-30,-15,0,15,30]){
  const p=at(yaw,pitch),edge=at(Math.sign(yaw)*45,pitch),flat=at(yaw),flatEdge=at(Math.sign(yaw)*45);
  const scale=1-.15*Math.min(1,Math.abs(pitch)/30);
  for(const point of skin.positions){
   const actual=faceSolveWorld(point,p),boundary=faceSolveWorld(point,edge),turned=faceSolveWorld(point,flat),origin=faceSolveWorld(point,flatEdge);
   assert(Math.abs(actual[0]-boundary[0]-scale*(turned[0]-origin[0]))<1e-6);
   assert(Math.abs(actual[1]-boundary[1]-(turned[1]-origin[1]))<1e-6);
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
   assert(Math.abs(width(at(pose.yaw))/edge-pose.eyeWidthScale[side])<1e-8,`${pose.name} ${id}`);
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
