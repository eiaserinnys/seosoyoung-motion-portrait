import {readFileSync} from 'node:fs';
import {bindRig,deform3D} from '../bundle/rig.mjs';
import {fringeSurface,canopyWorld,attachmentWorld,rawAttachmentWorld,flareDisplacement} from '../bundle/hair-shell.mjs';
import {hairCoordinates,hairSolveField} from '../bundle/hair-solve.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));bindRig(rig);
const features=JSON.parse(readFileSync(new URL('../source/viewer/face-feature-fit.json',import.meta.url))).poses;
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1,ParamBodyAngleX:0,ParamBreath:0};
const poses=[...features,...[-30,-15,15,30].map(yaw=>({name:`yaw${yaw>0?'+':''}${yaw}`,yaw,pitch:0}))];
const meshes=rig.meshes.filter(m=>m.id==='hair-top'||m.id.startsWith('bangs')||m.id.startsWith('side-hair'));
const area=([a,b,c])=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
const output=poses.map(({name,yaw,pitch})=>{
 const p={...base,ParamAngleX:yaw,ParamAngleY:pitch,[FRAME_CACHE]:new Map()};
 return {name,yaw,pitch,meshes:meshes.map(m=>{
  const fringe=m.positions.map(q=>fringeSurface(q,p)),canopy=m.positions.map(q=>attachmentWorld(q,p)),attachment=m.positions.map(q=>canopyWorld(q,p)),final=m.positions.map(q=>deform3D(m,q,p));
  return {id:m.id,rest:m.positions,uvs:m.uvs,points:{fringe,canopy,rawCanopy:m.positions.map(q=>rawAttachmentWorld(q,p)),flare:m.positions.map(q=>flareDisplacement(q,p)),residual:m.positions.map(q=>hairSolveField(...hairCoordinates(q),p)),limitedFlare:attachment,final},indices:m.triangles,triangles:m.triangles.map((t,index)=>{
   const rest=area(t.map(i=>m.positions[i]));
   return {index,uv:[0,1].map(d=>t.reduce((sum,i)=>sum+m.uvs[i][d],0)/3),
    rest,fringe:area(t.map(i=>fringe[i]))/rest,canopy:area(t.map(i=>canopy[i]))/rest,attachment:area(t.map(i=>attachment[i]))/rest,final:area(t.map(i=>final[i]))/rest};
  })};
 })};
});
process.stdout.write(JSON.stringify(output));
