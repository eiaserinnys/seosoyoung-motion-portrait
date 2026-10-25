// Read the approved measurements and neutral r11 geometry through their public evaluators.
import {readFileSync} from 'node:fs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
import {faceLift,outlineRow,legacyOutlineWorld} from '../bundle/face-prior.mjs';
import {registeredFeatureWorld} from '../bundle/face-feature-prior.mjs';
import {OUTLINE_ROWS,OUTLINE_POSES} from '../bundle/outline-data.mjs';
import {FEATURE_FIT} from '../bundle/feature-data.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
import {contourCoordinates} from '../bundle/face-contour.mjs';
import {FACE_DEPTH} from '../bundle/face-depth.mjs';
import assert from 'node:assert/strict';
const source=name=>JSON.parse(readFileSync(new URL('../source/viewer/'+name+'.json',import.meta.url)));
const trace=source('outline-traces'),features=source('face-feature-fit');
// Reject stale compiled inputs rather than attributing a solve to different source hashes.
assert.deepEqual(trace.rows,OUTLINE_ROWS);assert.deepEqual(trace.poses,OUTLINE_POSES);
assert.deepEqual(features,FEATURE_FIT);assert.deepEqual(source('face-depth'),FACE_DEPTH);
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));
const skin=rig.meshes.find(m=>m.id==='face-skin');
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1};
const points=skin.positions.map(p=>p.slice()),lookup=new Map(points.map((p,i)=>[p.join(','),i]));
const add=p=>{const key=p.join(',');if(!lookup.has(key)){lookup.set(key,points.length);points.push(p);}return lookup.get(key);};
// The same fade/upper-skin prior used by hair also constrains the query rim
// above the first visible-cheek observation (needed at the pitched ear height).
const samples=[218,347,380,400,...OUTLINE_ROWS.map(r=>r[0])].map(y=>{
 const [l,r]=outlineRow(y);
 return {y,left:add([l,y]),right:add([r,y]),center:add([502,y])};
});
add([502.5,616]);
const anchors=Object.fromEntries(Object.entries(FEATURE_FIT.front).map(([key,p])=>[key,add(p)]));
const edges=new Map();
for(const t of skin.triangles)for(let k=0;k<3;k++){
 const a=t[k],b=t[(k+1)%3],key=[a,b].sort((x,y)=>x-y).join(',');edges.set(key,(edges.get(key)||0)+1);
}
const boundary=new Set([...edges].filter(([,n])=>n===1).flatMap(([key])=>key.split(',').map(Number)));
const ids={browLeft:'eye-left-brow',browRight:'eye-right-brow',left:'eye-left-sclera',right:'eye-right-sclera',nose:'nose-detail',mouth:'mouth-detail'};
const poses=OUTLINE_POSES.map(pose=>{
 const p={...base,ParamAngleX:pose.yaw,ParamAngleY:pose.pitch,[FRAME_CACHE]:new Map()};
 const obs=points.map(pt=>legacyOutlineWorld(pt,p).slice(0,2));
 const weights=points.map((pt,i)=>!pose.pitch?4:boundary.has(i)||pt[1]<=347?4:0);
 for(const i of boundary)if(!pose.pitch)weights[i]=8;
 samples.forEach(s=>{
  const r=pose.rows.find(r=>r[0]===s.y);
  // Above restY465 the yaw SAM contour is occluding hair, not observed skin.
  if(r&&(pose.pitch||s.y>=465)){obs[s.left]=[r[2],r[1]];obs[s.right]=[r[4],r[3]];}
  weights[s.left]=weights[s.right]=pose.pitch?4:8;
 });
 const fit=FEATURE_FIT.poses.find(f=>f.name===pose.name);
 for(const [key,i] of Object.entries(anchors)){
  obs[i]=pose.pitch?fit.centers[key]:registeredFeatureWorld(ids[key],points[i],p).slice(0,2);
  weights[i]=pose.pitch?4:8;
  const nearest=skin.positions.map((pt,j)=>[j,Math.hypot(pt[0]-points[i][0],pt[1]-points[i][1])]).sort((a,b)=>a[1]-b[1]).slice(0,3);
  for(const [j] of nearest)if(!weights[j])weights[j]=2;
 }
 const pins=pose.pitch?[...samples.flatMap(s=>[s.left,s.right]),...Object.values(anchors)]:points.map((_,i)=>i);
 return {name:pose.name,yaw:pose.yaw,pitch:pose.pitch,obs,weights,pins};
});
process.stdout.write(JSON.stringify({points,evaluationPoints:points.map(contourCoordinates),triangles:skin.triangles,meshCount:skin.positions.length,
 samples,anchors,prior:points.map(pt=>faceLift(pt,0,base)[2]),poses}));
