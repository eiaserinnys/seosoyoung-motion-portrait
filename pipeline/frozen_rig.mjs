// Read the approved baseline without switching a shared checkout.
import {execFileSync} from 'node:child_process';
import {mkdtempSync,writeFileSync,readFileSync,rmSync,existsSync,mkdirSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join,dirname} from 'node:path';
import {pathToFileURL} from 'node:url';
export const BASELINE='83d4a80cfe2cd419d8daae90ee2783c2b9b1cbf9';
export async function frozenRig(currentHairSurface=false,baseline=BASELINE,calibrationFiles=[]){
 const dir=mkdtempSync(join(tmpdir(),'hair-baseline-'));
 try{
  const files=execFileSync('git',['ls-tree','--name-only',baseline,'bundle/'],{encoding:'utf8'}).trim().split('\n').filter(p=>p.endsWith('.mjs'));
  for(const path of files)writeFileSync(join(dir,path.slice(7)),execFileSync('git',['show',baseline+':'+path]));
  // Benchmark only P1/P2; approved volume and face calibration inputs must
  // be identical on both sides, including a later reference/face refit.
  if(currentHairSurface)for(const name of ['hair-shell.mjs','hair-surface.mjs','dome-data.mjs','feature-data.mjs','outline-data.mjs'])writeFileSync(join(dir,name),readFileSync(new URL('../bundle/'+name,import.meta.url)));
  for(const name of calibrationFiles)writeFileSync(join(dir,name),readFileSync(new URL('../bundle/'+name,import.meta.url)));
  // New calibration modules may import files absent in an older runtime.
  const pending=[...calibrationFiles],seen=new Set();while(pending.length){const name=pending.pop();if(seen.has(name))continue;seen.add(name);const source=readFileSync(join(dir,name),'utf8');for(const m of source.matchAll(/from '\.\/([^']+)'/g)){const child=join(dirname(name),m[1]);if(!existsSync(join(dir,child))){mkdirSync(dirname(join(dir,child)),{recursive:true});writeFileSync(join(dir,child),readFileSync(new URL('../bundle/'+child,import.meta.url)));}pending.push(child);}}
  const rig=await import(pathToFileURL(join(dir,'rig.mjs')));
  const pose=await import(pathToFileURL(join(dir,'pose-grid.mjs')));
  return {...rig,...pose,dispose:()=>rmSync(dir,{recursive:true,force:true})};
 }catch(error){rmSync(dir,{recursive:true,force:true});throw error;}
}
