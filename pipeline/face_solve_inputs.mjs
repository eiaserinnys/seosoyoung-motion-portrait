// Read the approved measurements and neutral r11 geometry through their public evaluators.
import {readFileSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
import {faceSolveWorld,faceSolveAnchorWorld} from '../bundle/face-solve.mjs';
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
const p45=Object.fromEntries([-1,1].map(sign=>{
 const p={...base,ParamAngleX:sign*45,ParamAngleY:0,[FRAME_CACHE]:new Map()};
 return [sign,points.map((pt,i)=>{
  const anchor=Object.entries(anchors).find(([,index])=>index===i);
  return (anchor?faceSolveAnchorWorld(anchor[0],p):faceSolveWorld(pt,p)).slice(0,2);
 })];
}));
const maskRows=JSON.parse(execFileSync('python3',[new URL('./face45_masks.py',import.meta.url).pathname],{
 input:JSON.stringify(samples.map(s=>s.y)),timeout:60000,
}));
const centers={
 left67:{left:[309,429.8],right:[404,429.2],nose:[291,491],mouth:[310,550],browLeft:[304,345.4],browRight:[412,344.3]},
 left90:{left:[313,429.8],right:[382,429.2],nose:[283,490],mouth:[300,550],browLeft:[310,345.4],browRight:[390,344.3]},
 right67:{left:[600,430.6],right:[695,428.4],nose:[713,491],mouth:[694,550],browLeft:[592,346.3],browRight:[700,343.5]},
 right90:{left:[622,430.6],right:[691,428.4],nose:[721,490],mouth:[704,550],browLeft:[614,346.3],browRight:[694,343.5]},
};
function rowAt(rows,y){
 if(y<=rows[0][0])return rows[0].slice(1);
 for(let i=1;i<rows.length;i++)if(y<=rows[i][0]){
  const a=rows[i-1],b=rows[i],t=(y-a[0])/(b[0]-a[0]);
  return [1,2].map(d=>a[d]+t*(b[d]-a[d]));
 }
 return rows.at(-1).slice(1);
}
const family45={p45Left:p45[-1],p45Right:p45[1],poses:[
 ['left67',-67.5,-22.5,.36,.72],['left90',-90,-45,.02,.45],
 ['right67',67.5,22.5,.72,.36],['right90',90,45,.45,.02],
].map(([name,yaw,delta,left,right])=>{
 const rest=p45[Math.sign(yaw)],rows=samples.map(s=>[s.y,rest[s.left][0],rest[s.right][0]]);
 const obs=points.map((_,i)=>{
  const y=points[i][1],[l,r]=rowAt(rows,y),[lt,rt]=rowAt(maskRows[name],y);
  return [lt+(rest[i][0]-l)/Math.max(1,r-l)*(rt-lt),rest[i][1]];
 });
 const weights=points.map(()=>4);
 for(const s of samples)weights[s.left]=weights[s.right]=8;
 for(const [key,i] of Object.entries(anchors)){obs[i]=centers[name][key];weights[i]=8;}
 return {name,yaw,delta,pitch:0,obs,weights,pins:points.map((_,i)=>i),eyeWidthScale:{left,right}};
})};
process.stdout.write(JSON.stringify({points,evaluationPoints:points.map(contourCoordinates),triangles:skin.triangles,meshCount:skin.positions.length,
 samples,anchors,prior:points.map(pt=>faceLift(pt,0,base)[2]),poses,family45}));
