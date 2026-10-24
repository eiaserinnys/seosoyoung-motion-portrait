// Only face skin and facial features enter these measurements and captures.
import {readFileSync} from 'node:fs';
import {bindRig,deform3D} from '../bundle/rig.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
import {outlineRow,outlineWorld} from '../bundle/face-outline.mjs';
import {featureWorld} from '../bundle/face-features.mjs';
import {FEATURE_FIT} from '../bundle/feature-data.mjs';
import {OUTLINE_POSES} from '../bundle/outline-data.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
import {sphericalFaceError} from './hairless_sphere_contract.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));
bindRig(rig);
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1};
const keys={left:'eye-left-sclera',right:'eye-right-sclera',browLeft:'eye-left-brow',browRight:'eye-right-brow',nose:'nose-detail',mouth:'mouth-detail'};
export const faceOnly=id=>id==='face-skin'||id.startsWith('eye-')||id==='nose-detail'||id==='mouth-detail';
function boundaryEdges(mesh){
 const edges=new Map();
 for(const tri of mesh.triangles)for(let i=0;i<3;i++){
  const a=tri[i],b=tri[(i+1)%3],key=[a,b].sort((a,b)=>a-b).join(':');
  const old=edges.get(key);edges.set(key,old?{...old,count:old.count+1}:{a,b,count:1});
 }
 return [...edges.values()].filter(e=>e.count===1).map(({a,b})=>[a,b]);
}
export function geometry(){
 return FEATURE_FIT.poses.map(f=>{
  const p={...base,ParamAngleX:f.yaw,ParamAngleY:f.pitch,[FRAME_CACHE]:new Map()};
  const meshes=rig.meshes.filter(m=>faceOnly(m.id)).sort((a,b)=>a.order-b.order).map(m=>({
   id:m.id,bbox:m.bbox,triangles:m.triangles,rest:m.positions,positions:m.positions.map(pt=>deform3D(m,pt,p)),
   boundary:m.id==='face-skin'?boundaryEdges(m):undefined,
  }));
  const landmarks=Object.fromEntries(Object.entries(keys).map(([k,id])=>[k,featureWorld(id,FEATURE_FIT.front[k],p)]));
  const boundary=[];
  for(let y=440;y<=616;y+=.5)boundary.push({y,points:outlineRow(y).slice(0,2).map(x=>outlineWorld([x,y],p))});
  const cheeks={};
  for(const u of [.25,.375,.5,.625,.75]){
   const [l,r]=outlineRow(497),pt=[l+(r-l)*u,497];
   cheeks[u]=outlineWorld(pt,p)[1]-outlineWorld(pt,{...p,ParamAngleY:0,[FRAME_CACHE]:new Map()})[1];
  }
  const eyes={};
  for(const side of ['left','right']){
   const [x,y]=FEATURE_FIT.front[side],id='eye-'+side+'-sclera';
   const corners=[[x-40,y-25],[x+40,y-25],[x+40,y+25],[x-40,y+25]].map(pt=>featureWorld(id,pt,p));
   const span=(id,q)=>featureWorld(id,[x,y+20],q)[1]-featureWorld(id,[x,y-20],q)[1];
   const neutral={...p,ParamAngleY:0,[FRAME_CACHE]:new Map()};
   eyes[side]={corners,shear:Math.max(Math.abs(corners[1][1]-corners[0][1]),Math.abs(corners[2][1]-corners[3][1])),
    heightRatio:span(id,p)/span(id,neutral),measuredHeightRatio:f.scales[side][1],
    ballRatios:Object.fromEntries(['iris','pupil','highlight'].map(part=>{const id='eye-'+side+'-'+part;return [part,span(id,p)/span(id,neutral)];}))};
  }
  return {name:f.name,yaw:f.yaw,pitch:f.pitch,meshes,landmarks,boundary,cheeks,eyes,
   sphericalError:!f.yaw&&f.pitch?sphericalFaceError(p):0,feature:f,outline:OUTLINE_POSES.find(o=>o.name===f.name)};
 });
}
if(process.argv[1]===new URL(import.meta.url).pathname)process.stdout.write(JSON.stringify(geometry()));
