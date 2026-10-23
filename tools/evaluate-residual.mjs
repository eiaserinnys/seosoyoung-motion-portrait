// Private inputs. Output is only aggregate errors, never a mesh or fitted residual field.
import fs from 'node:fs';
import zlib from 'node:zlib';
import {lift,project,bakeResidual,deform} from '../src/head-turn.mjs';
const [sampleFile,configFile,outFile]=process.argv.slice(2);
const s=JSON.parse(zlib.gunzipSync(fs.readFileSync(sampleFile))),config=JSON.parse(fs.readFileSync(configFile));
const points=(pose,index)=>{const v=s.poses[pose].vertices[index];return Array.from({length:v.length/2},(_,j)=>[v[j*2],-v[j*2+1]]);};
const face=points('neutral',s.meshIds.indexOf(config.face));
const lo=[0,1].map(k=>Math.min(...face.map(p=>p[k]))),hi=[0,1].map(k=>Math.max(...face.map(p=>p[k])));
const width=hi[0]-lo[0],center=lo.map((v,k)=>(v+hi[k])/2),radius=.6*Math.max(...lo.map((v,k)=>hi[k]-v));
const angles=[-30,0,30],result={};
for(const [name,ids] of Object.entries(config.regions)){
 const head={center,radius,depthSign:name==='hair_back'?-1:1};let baseError=0,correctedError=0,count=0,knotMaxError=0;
 for(const id of ids){
  const index=s.meshIds.indexOf(id),rest=points('neutral',index);
  const knots=angles.map(y=>angles.map(x=>points(`xy_${x}_${y}`,index)));
  const fields=rest.map((p,j)=>bakeResidual(p,head,knots.map(row=>row.map(v=>v[j])),30));
  for(const x of [-30,-15,0,15,30])for(const y of [-30,-15,0,15,30]){
   const actual=points(`xy_${x}_${y}`,index),isKnot=angles.includes(x)&&angles.includes(y);
   rest.forEach((p,j)=>{
    const base=project(lift(p,head),head,[x,y,0]),corrected=deform(p,head,[x,y,0],fields[j],30);
    const e=Math.hypot(...actual[j].map((v,k)=>v-corrected[k]));
    if(isKnot){knotMaxError=Math.max(knotMaxError,e);return;}
    correctedError+=e*e;baseError+=actual[j].reduce((sum,v,k)=>sum+(v-base[k])**2,0);count++;
   });
  }
 }
 result[name]={heldoutVertexSamples:count,baselineRmsOverFaceWidth:Math.sqrt(baseError/count)/width,correctedRmsOverFaceWidth:Math.sqrt(correctedError/count)/width,knotMaxError};
}
fs.writeFileSync(outFile,JSON.stringify(result,null,2));console.log(JSON.stringify(result));
