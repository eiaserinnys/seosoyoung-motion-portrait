import {readFileSync,writeFileSync,mkdirSync,existsSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {featureWorld,sam3FeatureWorld,legacyFeatureWorld} from '../bundle/face-features.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
import {CORRECTION_PRESETS} from '../bundle/geometry.mjs';
import {smoothstep} from '../bundle/face-profile.mjs';
export function junctionVerdict(own,fine,degree,legacyError){
 return {J1:fine<=own/100,J2:degree<=own,J6:legacyError<=1e-6};
}
const root=new URL('../',import.meta.url);
const read=name=>JSON.parse(readFileSync(new URL(name,root)));
const inputHash=createHash('sha256');
for(const name of ['face-features.mjs','eye-sam3-model.mjs','eye-sam3-data.mjs','face-feature-prior.mjs','face-solve.mjs','face-solve-data.mjs','feature-data.mjs','geometry.mjs','corrections.mjs','pose-grid.mjs','face-profile.mjs','assets/seosoyoung-rig.json'])inputHash.update(readFileSync(new URL('bundle/'+name,root)));
const inputSha=inputHash.digest('hex');
const rig=read('bundle/assets/seosoyoung-rig.json');
const defaults={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,
 CorrTurn:1,ParamAngleZ:0,ParamBodyAngleX:0,ParamBreath:0,ParamEyeBallX:0,ParamEyeBallY:0,ParamEyeLOpen:1,ParamEyeROpen:1};
const points=rig.meshes.filter(m=>(m.id.startsWith('eye-')&&!m.id.endsWith('brow'))||m.id==='closed-eyes-v1')
 .flatMap(m=>m.positions.map((point,vertex)=>({id:m.id,point,vertex})));
const p=(yaw,pitch,r24Cache)=>({...defaults,ParamAngleX:yaw,ParamAngleY:pitch,[FRAME_CACHE]:new Map(),[r24Cache]:new Map()});
const positions=(evaluate,parameters)=>new Float32Array(points.flatMap(v=>evaluate(v.id,v.point,parameters).slice(0,2)));
function maximum(a,b,where){
 let max={distancePx:0};
 for(let i=0;i<points.length;i++){
  const distancePx=Math.hypot(a[i*2]-b[i*2],a[i*2+1]-b[i*2+1]);
  if(distancePx>max.distancePx)max={distancePx,...where,mesh:points[i].id,vertex:points[i].vertex,rest:points[i].point};
 }
 return max;
}
const larger=(a,b)=>a.distancePx>b.distancePx?a:b;
export async function measure(){
 const {featureWorld:r24World}=await import('../debug/r24/bundle/face-features.mjs');
 const {FRAME_CACHE:R24_CACHE}=await import('../debug/r24/bundle/pose-grid.mjs');
 const cache=new URL('debug/sam3-eye-v293/junction/',root);mkdirSync(cache,{recursive:true});
 const signature=new URL('inputs.sha256',cache);
 if(existsSync(signature)&&readFileSync(signature,'utf8').trim()!==inputSha)throw Error('Junction cache belongs to different runtime inputs');
 writeFileSync(signature,inputSha+'\n');
 const baselinePath=new URL('baseline.json',cache);
 if(!existsSync(baselinePath))writeFileSync(baselinePath,readFileSync(new URL('docs/sam3-eye-junction-v2.json',root)));
 const baseline=JSON.parse(readFileSync(baselinePath)),own=baseline.J0.own;
 const V_own=Math.max(own.sam3.distancePx,own.legacy.distancePx);
 console.log('J0',JSON.stringify({V_own,own}));
 let fine={distancePx:0},degree={distancePx:0},legacy={distancePx:0},over1=0,samples=0;
 const cases=[];
 for(const pitch of [-30,0,30])for(const sign of [-1,1]){
  const file=new URL(`case-${pitch}-${sign}.json`,cache);
  if(existsSync(file)){
   const c=JSON.parse(readFileSync(file));cases.push(c);fine=larger(fine,c.peak);degree=larger(degree,c.degree);legacy=larger(legacy,c.legacy);over1+=c.over1;samples+=9001;continue;
  }
  if(process.argv[2]!==`${pitch},${sign}`)continue;
  let previous,previousDegree;let peak={distancePx:0},caseDegree={distancePx:0},caseLegacy={distancePx:0},caseOver1=0;
  const progress=new URL(`progress-${pitch}-${sign}.json`,cache);let firstTick=0;
  if(existsSync(progress)){
   const c=JSON.parse(readFileSync(progress));firstTick=c.tick+1;previous=new Float32Array(c.previous);previousDegree=new Float32Array(c.previousDegree);
   peak=c.peak;caseDegree=c.degree;caseLegacy=c.legacy;caseOver1=c.over1;
   fine=larger(fine,peak);degree=larger(degree,caseDegree);legacy=larger(legacy,caseLegacy);over1+=caseOver1;samples+=firstTick;
  }
  for(let tick=firstTick;tick<=9000;tick++){
   const yaw=sign*tick/100,params=p(yaw,pitch,R24_CACHE),now=positions(featureWorld,params);samples++;
   if(previous){
    const step=maximum(now,previous,{pitch,fromYaw:sign*(tick-1)/100,toYaw:yaw});
    peak=larger(peak,step);fine=larger(fine,step);
    if(step.distancePx>1){over1++;caseOver1++;}
   }
   if(tick%100===0){
    if(previousDegree){const m=maximum(now,previousDegree,{pitch,fromYaw:yaw-sign,toYaw:yaw});degree=larger(degree,m);caseDegree=larger(caseDegree,m);}
    previousDegree=now;
   }
   previous=now;
   if(tick%1000===0){
    writeFileSync(progress,JSON.stringify({tick,previous:Array.from(previous),previousDegree:Array.from(previousDegree),peak,degree:caseDegree,legacy:caseLegacy,over1:caseOver1})+'\n');
    console.log('J1 PROGRESS',pitch,sign,tick,'/9000',peak.distancePx);
   }
  }
  cases.push({pitch,sign,peak,degree:caseDegree,legacy:caseLegacy,over1:caseOver1});
  writeFileSync(file,JSON.stringify(cases.at(-1))+'\n');console.log('J1 CASE',JSON.stringify(cases.at(-1)));
 }
 const reference=read('docs/sam3-eye-far-clip-contract.json').poses.slice(0,24).map(row=>({name:row.name,requested:row.requested,s:smoothstep(52,67,Math.abs(row.requested[0]))}));
 const J7={maxWeight:Math.max(...reference.map(r=>r.s)),limit:.05,poses:reference};J7.passed=J7.maxWeight<=J7.limit;
 const result={complete:cases.length===6,inputSha,J0:{V_own,own},J1:{...fine,limitPx:V_own/100,over1pxSteps:over1,samples,vertexCount:points.length,cases},
  J2:{...degree,limitPx:V_own},J6:baseline.J6,J7:baseline.J7,reusedFrom:'798df34: J0,J6,J7; Fable 1.3 only J1/J2 geometry sweep rerun',
  passed:junctionVerdict(V_own,fine.distancePx,degree.distancePx,baseline.J6.xyzMaxErrorPx)};
 writeFileSync(new URL('docs/sam3-eye-junction-v2.json',root),JSON.stringify(result,null,2)+'\n');
 console.log(JSON.stringify(result,null,2));
 return result;
}
if(process.argv[1]===new URL(import.meta.url).pathname)await measure();
