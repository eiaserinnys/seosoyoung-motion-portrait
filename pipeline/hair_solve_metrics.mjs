// Residual charts exclude horizon foreshortening in the 3D prior.
import {readFileSync} from 'node:fs';
import {hairResidualField} from '../bundle/hair-solve.mjs';
import {domeSection} from '../bundle/hair-surface.mjs';
import {HAIR_SOLVE} from '../bundle/hair-solve-data.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
import {pathToFileURL} from 'node:url';
export function hairSolveMetrics(){
 const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));
 const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1};
 const poses=[];
 for(const pose of HAIR_SOLVE.poses){
  const p={...base,ParamAngleX:pose.yaw,ParamAngleY:pose.pitch,[FRAME_CACHE]:new Map()};
  let minDet=Infinity,reversedCells=0;
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
   minDet=Math.min(minDet,localMin);if(localMin<=0)reversedCells++;
  }
  poses.push({name:pose.name,minDet,reversedCells});
 }
 return {pass:poses.every(p=>p.minDet>.25&&p.reversedCells===0),definition:'residual-only local tangent chart (s + dx/h0_cell, restY + dy), 11x15 cells, 9 derivative samples/cell; cell half-width extended from the first authored section y60 above the pole; 3D silhouette projection excluded',poses};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href)console.log(JSON.stringify(hairSolveMetrics()));
