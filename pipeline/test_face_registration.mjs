import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {bindRig,deform3D} from '../bundle/rig.mjs';
import {CORRECTION_PRESETS,eyeRootWorld,artWorld} from '../bundle/geometry.mjs';
import {outlineRow,outlineWorld,outlinePitchShift,faceWorld} from '../bundle/face-outline.mjs';
import {FEATURE_FIT} from '../bundle/feature-data.mjs';
import {OUTLINE_ROWS,OUTLINE_POSES} from '../bundle/outline-data.mjs';
import {featureWorld} from '../bundle/face-features.mjs';
import {strandDepth,strandWorld} from '../bundle/side-strands.mjs';
import {earWorld} from '../bundle/stereo-attachments.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));bindRig(rig);
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrTurn:1};
const at=(yaw,pitch)=>({...base,ParamAngleX:yaw,ParamAngleY:pitch,[FRAME_CACHE]:new Map()});
const mesh=id=>rig.meshes.find(m=>m.id===id);
const near=(a,b,msg)=>assert(Math.abs(a-b)<1e-7,`${msg}: ${a} vs ${b}`);
const ids={left:'eye-left-sclera',right:'eye-right-sclera',browLeft:'eye-left-brow',browRight:'eye-right-brow',nose:'nose-detail',mouth:'mouth-detail'};
// Alpha >127 outer spans measured from the project's own parts, excluding the
// transparent padding of the rectangular ArtMesh. No third-party mesh data.
const scleraBoundary=JSON.parse(readFileSync(new URL('./fixtures/sclera-alpha-boundary.json',import.meta.url)));
test('all pitch registrations retain the approved yaw binocular X and relative 3D Y',()=>{
 for(const p of FEATURE_FIT.poses.filter(p=>p.pitch)){
  const rest=FEATURE_FIT.poses.find(q=>q.yaw===p.yaw&&!q.pitch);
  const mid=i=>(rest.centers.left[i]+rest.centers.right[i])/2;
  const sphere=pitch=>['left','right'].reduce((s,side)=>s+eyeRootWorld(side,at(p.yaw,pitch))[1]/2,0);
  near(p.registration.eyeMid[0],mid(0),p.name+' X');
  near(p.registration.eyeMid[1],mid(1)+sphere(p.pitch)-sphere(0),p.name+' Y');
  assert(p.registration.binocularRatio>0);
  assert.deepEqual(OUTLINE_POSES.find(q=>q.name===p.name).registration.delta,p.registration.delta);
 }
});
test('outline tables cover the complete sixteen rest rows with independent edge heights',()=>{
 for(const p of OUTLINE_POSES){
  assert.deepEqual(p.rows.map(r=>r[0]),OUTLINE_ROWS.map(r=>r[0]),p.name);
  assert(p.rows.every(r=>r.length===5&&r.every(Number.isFinite)),p.name);
  if(p.pitch){
   assert.deepEqual(p.mapping.restY,[431.5,616]);
   const mapping=p.mapping;
   assert.equal(mapping.sFloor,p.pitch>0?.06:0);
   assert.equal(mapping.s3d.length,OUTLINE_ROWS.length);
   const y3d=y=>outlineRow(y).slice(0,2).reduce((sum,x)=>sum+faceWorld([x,y],0,at(0,p.pitch))[1]/2,0);
   const top=y3d(431.5),span=y3d(616)-top;
   OUTLINE_ROWS.forEach(([y],i)=>near(mapping.s3d[i],(y3d(y)-top)/span,p.name+' s3d '+y));
   assert.deepEqual(mapping.clampedRows,OUTLINE_ROWS.filter((r,i)=>mapping.s3d[i]<mapping.sFloor).map(r=>r[0]));
   for(const side of ['left','right']){
    const feature=FEATURE_FIT.poses.find(f=>f.name===p.name);
    assert.deepEqual(mapping.rawYSpan[side],[feature.rawCenters[side][1],p.rawYChin]);
    near(mapping.canvasYSpan[side][0],feature.centers[side][1],p.name+' canvas eye');
    near(mapping.canvasYSpan[side][1],feature.centers.mouth[1]+(p.pitch>0?-12+64:6+50),p.name+' canvas chin');
   }
   assert(p.fitRmsPixels<=(p.name==='up'?4.5:5.5),`${p.name} RMS ${p.fitRmsPixels}`);
   if(p.yaw===0){const chin=p.rows.find(r=>r[0]===616);assert(chin[4]-chin[2]>=11-1e-7,`${p.name} chin width ${chin[4]-chin[2]}`);}
   assert.equal(p.samplingSpan,undefined);assert.equal(p.targetSpan,undefined);
  }
 }
});
test('upper diagonal far cheeks converge monotonically toward the chin',()=>{
 for(const name of ['up-left','up-right']){
  const pose=OUTLINE_POSES.find(p=>p.name===name),index=name==='up-left'?2:4,sign=name==='up-left'?1:-1;
  const xs=[465,510,550,570,590,616].map(y=>pose.rows.find(r=>r[0]===y)[index]);
  for(let i=1;i<xs.length;i++)assert(sign*(xs[i]-xs[i-1])>=0,`${name} nonconvergent ${xs}`);
 }
});
test('pitch field reaches every measured outline endpoint without width suppression',()=>{
 for(const p of OUTLINE_POSES.filter(p=>p.pitch))for(const [y,yL,xL,yR,xR] of p.rows){
  const [l,r]=outlineRow(y),pose=at(p.yaw,p.pitch),a=outlineWorld([l,y],pose),b=outlineWorld([r,y],pose);
  near(a[0],xL,p.name+' left X '+y);near(a[1],yL,p.name+' left Y '+y);
  near(b[0],xR,p.name+' right X '+y);near(b[1],yR,p.name+' right Y '+y);
 }
});
test('frame cache keeps neutral feature controls independent of the pitched frame',()=>{
 for(const yaw of [-45,0,45])for(const pitch of [-30,30]){
  const p=at(yaw,pitch),uncached={...p};delete uncached[FRAME_CACHE];
  for(const point of [[415.7,432.2],[508,497],[504.5,554.5]])assert.deepEqual(outlinePitchShift(point,p),outlinePitchShift(point,uncached));
 }
});
test('skin feature controls share pitch residual within the agreed anchor tolerances',()=>{
 for(const yaw of [-45,0,45])for(const pitch of [-30,30])for(const [key,id] of Object.entries(ids)){
  const point=FEATURE_FIT.front[key],p=at(yaw,pitch),p0=at(yaw,0);
  // Compare the complete outline shift against feature-minus-legacy-skin.
  const skin=outlineWorld(point,p),skin0=outlineWorld(point,p0);
  const f=featureWorld(id,point,p),f0=featureWorld(id,point,p0);
  for(const i of [0,1]){
   const error=Math.abs(skin[i]-skin0[i]-(f[i]-f0[i]));
   assert(error<(key==='nose'||key==='mouth'?1e-6:.02),`${key} ${yaw},${pitch} axis ${i}: ${error}`);
  }
 }
});
// Intersect the deformed boundary at the eye/ear canvas Y, rather than compare
// unrelated source rows. This also catches a deliberately shifted eye below.
function boundaryX(side,y,p){
 const index=side==='left'?0:1,cache=p[FRAME_CACHE],key='test-boundary:'+side;
 if(!cache.has(key)){
  const points=[];
  for(let restY=347;restY<=616;restY+=.5){const x=outlineRow(restY)[index];points.push(deform3D(mesh('face-skin'),[x,restY],p));}
  cache.set(key,points);
 }
 const points=cache.get(key);
 const hits=[];
 for(let i=1;i<points.length;i++){
  const a=points[i-1],b=points[i];
  if((y-a[1])*(y-b[1])<=0&&a[1]!==b[1])hits.push(a[0]+(b[0]-a[0])*(y-a[1])/(b[1]-a[1]));
 }
 assert(hits.length,`no ${side} boundary at ${y}`);
 return side==='left'?Math.min(...hits):Math.max(...hits);
}
function eyeMargins(p,offset=0){
 return ['left','right'].map(side=>{
  const m=mesh(`eye-${side}-sclera`),sign=side==='left'?1:-1;
  return Math.min(...scleraBoundary[side].map(pt=>deform3D(m,pt,p)).map(q=>sign*(q[0]-boundaryX(side,q[1],p))-offset));
 });
}
function assertEyeMargins(yaw,margins,baseline){
 margins.forEach((margin,i)=>{
  if(yaw===0)assert(margin>=2,`front eye ${i} margin ${margin}`);
  else assert(margin>=2||(margin>=baseline[i]-1.5&&margin>=-4.75),`${yaw} eye ${i} margin ${margin} vs r11 ${baseline[i]}`);
 });
}
function assertIntermediateMargins(yaw,margins,endpoint){
 margins.forEach((margin,i)=>assert(margin>=(Math.abs(yaw)<20?2:endpoint[i]),`${yaw} intermediate eye ${i} margin ${margin} vs endpoint ${endpoint[i]}`));
}
test('sclera preserves the r11 margin in nine key and sixteen intermediate poses',()=>{
 const failed=[];
 for(const yaw of [-45,0,45])for(const pitch of [-30,0,30]){
  const margins=eyeMargins(at(yaw,pitch)),baseline=eyeMargins(at(yaw,0));
  try{assertEyeMargins(yaw,margins,baseline);}catch(error){failed.push({yaw,pitch,margins,baseline,error:error.message});}
 }
 for(const yaw of [-33.75,-11.25,11.25,33.75])for(const pitch of [-22.5,-7.5,7.5,22.5]){
  const margins=eyeMargins(at(yaw,pitch)),endpoint=eyeMargins(at(Math.sign(yaw)*45,0));
  try{assertIntermediateMargins(yaw,margins,endpoint);}catch(error){failed.push({yaw,pitch,margins,endpoint,error:error.message});}
 }
 assert.deepEqual(failed,[],'sclera baseline violations');
});
test('sclera margin gate detects an injected protrusion',()=>{
 const baseline=eyeMargins(at(0,0));
 assert.throws(()=>assertEyeMargins(0,eyeMargins(at(0,0),1000),baseline),assert.AssertionError);
 assert.doesNotThrow(()=>assertEyeMargins(45,[2,2],[20,20]));
 assert.doesNotThrow(()=>assertEyeMargins(-45,[-4.685,2],[-3.2388,20]));
 assert.throws(()=>assertEyeMargins(0,[1.99,2],[-3.2388,20]),assert.AssertionError);
 assert.throws(()=>assertEyeMargins(-45,[-4.76,2],[-100,20]),assert.AssertionError);
 assert.throws(()=>assertIntermediateMargins(-33.75,[-3.3,20],[-3.2388,15]),assert.AssertionError);
 assert.throws(()=>assertIntermediateMargins(-11.25,[1.99,20],[-3.2388,15]),assert.AssertionError);
});
test('yaw endpoints sweep through pitch without a binocular X jump or discontinuous outline',()=>{
 for(const yaw of [-45,45]){
  const xs=[-30,-15,0,15,30].map(pitch=>['left','right'].reduce((s,k)=>s+featureWorld(ids[k],FEATURE_FIT.front[k],at(yaw,pitch))[0]/2,0));
  assert(Math.max(...xs)-Math.min(...xs)<.3,`${yaw} midpoint sweep ${xs}`);
  for(const y of [410,465,530,570,616])for(const x of outlineRow(y).slice(0,2)){
   const a=outlineWorld([x,y],at(yaw,-1e-6)),b=outlineWorld([x,y],at(yaw,1e-6));
   assert(Math.hypot(...a.map((v,i)=>v-b[i]))<1e-4);
  }
 }
});
test('diagonal near cheeks retain the r11 endpoint envelope through rest Y570',()=>{
 for(const p of OUTLINE_POSES.filter(p=>p.pitch&&p.yaw)){
  const rest=OUTLINE_POSES.find(q=>q.yaw===p.yaw&&!q.pitch),index=p.yaw<0?4:2,sign=p.yaw<0?1:-1;
  p.rows.forEach((row,i)=>{if(row[0]<=570)assert(sign*(row[index]-rest.rows[i][index])>=-1e-6,`${p.name} near cheek ${row[0]}`);});
 }
});
test('near ear overlap stays within two pixels of the r11 baseline at six turned poses',()=>{
 const failed=[];
 for(const yaw of [-45,45])for(const pitch of [-30,0,30]){
  const side=yaw<0?'right':'left',x=side==='right'?626:379,p=at(yaw,pitch),p0=at(yaw,0);
  for(const y of [440,465,490]){
   const q=deform3D(mesh('ear-'+side),[x,y],p),edge=boundaryX(side,q[1],p);
   const overlap=side==='right'?edge-q[0]:q[0]-edge;
   const q0=deform3D(mesh('ear-'+side),[x,y],p0),edge0=boundaryX(side,q0[1],p0);
   const baseline=side==='right'?edge0-q0[0]:q0[0]-edge0;
   if(overlap<baseline-2)failed.push({yaw,pitch,y,overlap,baseline});
  }
 }
 assert.deepEqual(failed,[],'ear overlap baseline violations');
});
test('hanging locks use their cheek attachment depth for pitch while preserving yaw XY',()=>{
 for(const [side,xs] of [['left',[224,250,353]],['right',[655,750,777]]])for(const yaw of [-45,0,45]){
  const inner=side==='left'?353:655,y=430,rest=at(0,0),zAttach=Math.max(strandDepth([inner,y],side),earWorld([inner,y],side,rest)[2]+8);
  for(const x of xs)for(const pitch of [-30,30]){
   const p=at(yaw,pitch),zero=at(yaw,0),z=Math.max(strandDepth([x,y],side),earWorld([x,y],side,rest)[2]+4);
   const local=[x-502,y-326,z],attachment=[x-502,y-326,zAttach];
   const expected=artWorld(attachment,p),neutral=artWorld(attachment,zero),actual=strandWorld([x,y],side,p),old=strandWorld([x,y],side,zero);
   near(actual[1]-old[1],expected[1]-neutral[1],`${side} ${yaw},${pitch},${x} pitch Y`);
   if(yaw===0||Math.abs(yaw)===45){
    if(yaw*(side==='left'?1:-1)>=0)near(old[0],artWorld(local,zero)[0],'yaw X preserved');
   }
  }
 }
});
