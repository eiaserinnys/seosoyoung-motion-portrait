import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {faceSolveWorld,faceSolveAnchorWorld,evaluateFace45World} from '../bundle/face-solve.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
import {OUTLINE_POSES,OUTLINE_ROWS} from '../bundle/outline-data.mjs';
import {FEATURE_FIT} from '../bundle/feature-data.mjs';
import {featureWorld} from '../bundle/face-features.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
import {legacyOutlineWorld} from '../bundle/face-outline.mjs';
import {FACE_SOLVE} from '../bundle/face-solve-data.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));
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
test('a future face-45 depth/residual family is zero at its rest boundary and active beyond it',()=>{
 const zeros=FACE_SOLVE.points.map(()=>[0,0]);
 const family={Z:FACE_SOLVE.points.map(()=>50),poses:[
  {yaw:0,pitch:0,du:zeros},{yaw:45,pitch:0,du:FACE_SOLVE.points.map(()=>[4,-2])},
 ]};
 for(const sign of [-1,1]){
  const point=skin.positions[0];
  near(evaluateFace45World(point,at(sign*45),family),faceSolveWorld(point,at(sign*45)),1e-6);
  const extended=faceSolveWorld(point,at(sign*60),0,family),empty=faceSolveWorld(point,at(sign*60));
  assert(Math.hypot(extended[0]-empty[0],extended[1]-empty[1])>1);
 }
});
