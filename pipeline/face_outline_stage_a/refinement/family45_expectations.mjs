import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';

// Separate contract generation keeps stored postprocessing out of the next solve.
export function applyFamily45Projection(inputs,record){
 const poses=inputs.family45.poses.map(pose=>{
  const saved=record.poses.find(p=>p.name===pose.name);assert(saved,'Missing projection record: '+pose.name);
  const obs=pose.obs.map(q=>q.slice()),seen=new Set();
  for(const move of saved.moves){
   assert(!seen.has(move.index),'Duplicate projection vertex');seen.add(move.index);assert(obs[move.index],'Unknown projection vertex');
   const error=Math.hypot(...move.delta.map((v,d)=>v-(move.duAfter[d]-move.duBefore[d])));
   assert(error<=1e-6,'Stored projection displacement disagrees with du history');
   obs[move.index]=obs[move.index].map((v,d)=>v+move.delta[d]);
  }
  return {...pose,rawObs:pose.obs,obs,projectionMoves:saved.moves};
 });
 return {...inputs,family45:{...inputs.family45,poses}};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){
 const input=JSON.parse(execFileSync('node',[new URL('../../face_solve_inputs.mjs',import.meta.url).pathname],{timeout:60000,maxBuffer:16*1024*1024}));
 const record=JSON.parse(readFileSync(new URL('../accepted/family45-projection.json',import.meta.url)));
 process.stdout.write(JSON.stringify(applyFamily45Projection(input,record)));
}
