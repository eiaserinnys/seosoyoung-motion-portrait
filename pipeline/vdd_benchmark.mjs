// Evaluate the real before/after residual modules on the same chart samples.
import {readFileSync,writeFileSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {hairResidualField} from '../bundle/hair-solve.mjs';
import {HAIR_SOLVE} from '../bundle/hair-solve-data.mjs';
import {faceSolveWorld} from '../bundle/face-solve.mjs';
import {domeSection} from '../bundle/hair-surface.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
import {research} from './vdd_research_notes.mjs';

const root=new URL('../',import.meta.url),baseline=process.argv[2]||'17032ea';
const read=name=>readFileSync(new URL(name,root));
const prior=name=>execFileSync('git',['show',`${baseline}:${name}`],{cwd:root,maxBuffer:32*1024*1024});
const dataUrl=source=>'data:text/javascript;base64,'+Buffer.from(source).toString('base64');
const oldSource=prior('bundle/hair-solve.mjs').toString().replace(/from '(\.\/[^']+)'/g,(_,name)=>{
 const url=name==='./hair-solve-data.mjs'?dataUrl(prior('bundle/hair-solve-data.mjs')):new URL('bundle/'+name.slice(2),root).href;
 return `from '${url}'`;
});
const old=await import(dataUrl(oldSource));
const rig=JSON.parse(read('bundle/assets/seosoyoung-rig.json')),skin=rig.meshes.find(m=>m.id==='face-skin');
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1};
const at=(yaw,pitch)=>({...base,ParamAngleX:yaw,ParamAngleY:pitch,[FRAME_CACHE]:new Map()});
const mean=v=>v.reduce((a,b)=>a+b,0)/v.length;
const rms=v=>Math.sqrt(mean(v));
const area=([a,b,c])=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
const chartSamples=HAIR_SOLVE.y.slice(0,-1).flatMap((y,j)=>HAIR_SOLVE.s.slice(0,-1).flatMap((s,i)=>
 [.001,.5,.999].flatMap(a=>[.001,.5,.999].map(b=>({s:s+a*(HAIR_SOLVE.s[i+1]-s),y:y+b*(HAIR_SOLVE.y[j+1]-y),h:domeSection(Math.max(60,(y+HAIR_SOLVE.y[j+1])/2)).rx})))));

function hairMetrics(field,yaw,pitch){
 const p=at(yaw,pitch),mirror=at(-yaw,pitch);let minDetJ=Infinity;const energy=[],symmetry=[];
 for(const {s,y,h} of chartSamples){
  const ds=1e-5,dy=1e-3,l=field(s-ds,y,p),r=field(s+ds,y,p),t=field(s,y-dy,p),b=field(s,y+dy,p);
  const ux=(r[0]-l[0])/(2*ds*h),vx=(r[1]-l[1])/(2*ds*h),uy=(b[0]-t[0])/(2*dy),vy=(b[1]-t[1])/(2*dy);
  minDetJ=Math.min(minDetJ,(1+ux)*(1+vy)-uy*vx);
  energy.push((ux-vy)**2+(uy+vx)**2);
  const a=field(s,y,p),m=field(-s,y,mirror);symmetry.push((a[0]+m[0])**2+(a[1]-m[1])**2);
 }
 return {minDetJ,conformalEnergy:mean(energy),mirrorRms:rms(symmetry)};
}
function faceMetrics(yaw,pitch){
 const p=at(yaw,pitch),m=at(-yaw,pitch),points=skin.positions.map(v=>faceSolveWorld(v,p));
 const mirrorRms=rms(skin.positions.map((v,i)=>{const q=faceSolveWorld([1004-v[0],v[1]],m);return (points[i][0]+q[0]-1004)**2+(points[i][1]-q[1])**2;}));
 let minDetJ=Infinity;const energy=[];
 for(const ids of skin.triangles){
  const rest=ids.map(i=>skin.positions[i]),actual=ids.map(i=>points[i]),det=area(rest);
  minDetJ=Math.min(minDetJ,area(actual)/det);
  const [a,b,c]=rest,[u,v,w]=actual;
  const xx=((v[0]-u[0])*(c[1]-a[1])-(w[0]-u[0])*(b[1]-a[1]))/det;
  const xy=(-(v[0]-u[0])*(c[0]-a[0])+(w[0]-u[0])*(b[0]-a[0]))/det;
  const yx=((v[1]-u[1])*(c[1]-a[1])-(w[1]-u[1])*(b[1]-a[1]))/det;
  const yy=(-(v[1]-u[1])*(c[0]-a[0])+(w[1]-u[1])*(b[0]-a[0]))/det;
  energy.push((xx-yy)**2+(xy+yx)**2);
 }
 return {minDetJ,conformalEnergy:mean(energy),mirrorRms};
}
function velocityJump(field,epsilon){
 const errors=[];
 for(const pitch of [-15,15,-30,30]){
  const l=at(-epsilon,pitch),c=at(0,pitch),r=at(epsilon,pitch);
  for(const {s,y} of chartSamples){const a=field(s,y,l),b=field(s,y,c),d=field(s,y,r);errors.push((d[0]-2*b[0]+a[0])**2/epsilon**2+(d[1]-2*b[1]+a[1])**2/epsilon**2);}
 }
 return rms(errors);
}

// Twelve combinations exercise all requested yaw values and all four pitches.
const angles=[-30,-22.5,-15,15,22.5,30].flatMap(yaw=>(Math.abs(yaw)===22.5?[-15,15]:[-20,20]).map(pitch=>[yaw,pitch]));
const intermediate=angles.map(([yaw,pitch])=>{
 const face=faceMetrics(yaw,pitch);
 return {yaw,pitch,before:{face,hair:hairMetrics(old.hairResidualField,yaw,pitch)},after:{face,hair:hairMetrics(hairResidualField,yaw,pitch)}};
});
const sweep=[];
for(const pitch of [-15,0,15])for(let yaw=-90;yaw<=90;yaw+=2.5){
 const face=faceMetrics(yaw,pitch);
 sweep.push({yaw,pitch,faceMinDetJ:face.minDetJ,before:hairMetrics(old.hairResidualField,yaw,pitch),after:hairMetrics(hairResidualField,yaw,pitch)});
}
const crossValidation=JSON.parse(execFileSync(new URL('.venv/bin/python',root).pathname,[new URL('pipeline/fit_hair_solve.py',root).pathname,'--evaluate',baseline],{cwd:root,timeout:120000,env:{...process.env,OPENBLAS_NUM_THREADS:'1'}}));
const poses=crossValidation.rasterCaptures.after;
const main=crossValidation.rasterCaptures.main;
const pitchNames=new Set(HAIR_SOLVE.poses.map(p=>p.name));
const beforeMean=mean(main.poses.filter(p=>pitchNames.has(p.name)).map(p=>p.correctedRms));
const afterMean=mean(poses.poses.filter(p=>pitchNames.has(p.name)).map(p=>p.correctedRms));
const protectedPaths=['bundle/hair-shell.mjs','bundle/hair-surface.mjs','bundle/dome-data.mjs','source/viewer/face-solve.json','bundle/face-solve-data.mjs','bundle/assets/seosoyoung-rig.json','bundle/assets/texture_00.png'];
const unchanged=protectedPaths.every(name=>read(name).equals(prior(name)));
const continuity=[.01,.001,.0001].map(epsilon=>({epsilon,before:velocityJump(old.hairResidualField,epsilon),after:velocityJump(hairResidualField,epsilon)}));
const keys=HAIR_SOLVE.poses.map(p=>({name:p.name,yaw:p.yaw,pitch:p.pitch,before:hairMetrics(old.hairResidualField,p.yaw,p.pitch),after:hairMetrics(hairResidualField,p.yaw,p.pitch),duRms:Math.sqrt(mean(p.du.map(d=>d[0]**2+d[1]**2))),duMax:Math.max(...p.du.map(d=>Math.hypot(...d)))}));
const report={baseline,research,definitions:{hair:'local residual tangent chart (s + dx/h_cell, y + dy), 11x15 cells, nine derivative samples/cell; Jacobian excludes 3D silhouette foreshortening',conformalEnergy:'mean squared Cauchy-Riemann residual of pixel-coordinate Jacobian',face:'actual skin XY triangle Jacobian relative to rest mesh; identical before/after',mirrorRms:'RMS of mirrored screen displacements; original asymmetric artwork can have nonzero error',sweep:'219 numeric poses, yaw -90..90 step2.5 at pitch -15/0/15; sampled continuity and orientation, not a global injectivity proof'},
 protectedPriorUnchanged:unchanged,continuity,crossValidation,keys,intermediate,sweep,
 raster:{...poses,mainMean:beforeMean,afterMean,meanImproved:afterMean<=beforeMean},
 sources:Object.fromEntries(['bundle/hair-solve.mjs','bundle/hair-solve-data.mjs','source/viewer/hair-solve.json','pipeline/fit_hair_solve.py'].map(name=>[name,createHash('sha256').update(read(name)).digest('hex')]))};
report.pass=unchanged&&poses.pass&&afterMean<=beforeMean&&poses.poses.filter(p=>pitchNames.has(p.name)).every(p=>p.correctedRms<p.priorRms)&&keys.every(p=>p.after.minDetJ>=.25&&p.duRms<=45&&p.duMax<=115)&&intermediate.every(p=>p.after.face.minDetJ>0&&p.after.hair.minDetJ>=.25)&&sweep.every(p=>p.faceMinDetJ>0&&p.after.minDetJ>=.25)&&continuity.at(-1).after<1e-5;
writeFileSync(new URL('docs/vdd-research-evaluation.json',root),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({pass:report.pass,raster:report.raster,continuity,crossValidation:crossValidation.hair.mean,minHair:Math.min(...sweep.map(p=>p.after.minDetJ)),minFace:Math.min(...sweep.map(p=>p.faceMinDetJ))}));
if(!report.pass)process.exitCode=1;
