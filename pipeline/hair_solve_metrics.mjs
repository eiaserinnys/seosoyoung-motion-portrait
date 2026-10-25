// Residual charts exclude horizon foreshortening in the 3D prior.
import {readFileSync} from 'node:fs';
import {hairResidualField} from '../bundle/hair-solve.mjs';
import {domeSection} from '../bundle/hair-surface.mjs';
import {HAIR_SOLVE} from '../bundle/hair-solve-data.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
import {pathToFileURL} from 'node:url';
import {bindRig,deform3D} from '../bundle/rig.mjs';
export function hairSolveMetrics(){
 const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));
 const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1};
 const poses=[];
 for(const pose of HAIR_SOLVE.poses){
  const p={...base,ParamAngleX:pose.yaw,ParamAngleY:pose.pitch,[FRAME_CACHE]:new Map()};
  let minDet=Infinity,reversedCells=0,worstCell=null;
  for(let j=0;j<15;j++)for(let i=0;i<11;i++){
   // A local tangent chart uses its cell's half-width, keeping the cell
   // enclosing the pole finite instead of dividing by the pole's zero radius.
   const h=domeSection(Math.max(60,(HAIR_SOLVE.y[j]+HAIR_SOLVE.y[j+1])/2)).rx;
   let localMin=Infinity;
   for(const a of [.001,.5,.999])for(const b of [.001,.5,.999]){
    const s=HAIR_SOLVE.s[i]+a*(HAIR_SOLVE.s[i+1]-HAIR_SOLVE.s[i]);
    const y=HAIR_SOLVE.y[j]+b*(HAIR_SOLVE.y[j+1]-HAIR_SOLVE.y[j]);
    const ds=1e-5,dy=1e-3,l=hairResidualField(s-ds,y,p),r=hairResidualField(s+ds,y,p),t=hairResidualField(s,y-dy,p),v=hairResidualField(s,y+dy,p);
    const xs=(r[0]-l[0])/(2*ds*h),ys=(r[1]-l[1])/(2*ds),xy=(v[0]-t[0])/(2*dy*h),yy=(v[1]-t[1])/(2*dy);
    localMin=Math.min(localMin,(1+xs)*(1+yy)-xy*ys);
   }
   if(localMin<minDet){minDet=localMin;worstCell={i,j,s:HAIR_SOLVE.s[i],y:HAIR_SOLVE.y[j]};}if(localMin<=0)reversedCells++;
  }
  poses.push({name:pose.name,minDet,reversedCells,worstCell});
 }
 const lower=lowerSolveMetrics(rig,base);
 return {pass:poses.every(p=>p.minDet>=.25&&p.reversedCells===0)&&lower.pass,definition:'residual-only local tangent chart (s + dx/h0_cell, restY + dy), 11x15 cells, 9 derivative samples/cell; cell half-width extended from the first authored section y60 above the pole; 3D silhouette projection excluded',poses,lower};
}
export function lowerSolveMetrics(rig,base){
 bindRig(rig);
 const area=([a,b,c])=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]),poses=[];
 for(const pitch of [-30,-15,0,15,30])for(const yaw of [-90,-67.5,-45,-22.5,0,22.5,45,67.5,90]){
  const p={...base,ParamAngleX:yaw,ParamAngleY:pitch,[FRAME_CACHE]:new Map()},prior={...p,CorrOutline:0,[FRAME_CACHE]:new Map()};
  let minDet=Infinity,reversedTriangles=0;
  for(const mesh of rig.meshes.filter(m=>m.id.startsWith('tail-'))){
   const before=mesh.positions.map(pt=>deform3D(mesh,pt,prior)),after=mesh.positions.map(pt=>deform3D(mesh,pt,p));
   for(const ids of mesh.triangles){
    const det=area(ids.map(i=>before[i]));if(Math.abs(det)<1e-9)continue;
    const ratio=area(ids.map(i=>after[i]))/det;minDet=Math.min(minDet,ratio);if(ratio<=0)reversedTriangles++;
   }
  }
  poses.push({yaw,pitch,minDet,reversedTriangles});
 }
 return {pass:poses.every(p=>p.minDet>=.25&&!p.reversedTriangles),definition:'all four tail mesh XY triangles relative to unchanged pendulum; 45 key/intermediate/profile poses',poses};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href)console.log(JSON.stringify(hairSolveMetrics()));
