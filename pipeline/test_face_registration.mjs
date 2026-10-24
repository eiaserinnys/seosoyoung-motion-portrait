import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {bindRig,deform3D} from '../bundle/rig.mjs';
import {CORRECTION_PRESETS,eyeRootWorld,artWorld} from '../bundle/geometry.mjs';
import {outlineRow,outlineWorld,outlinePitchShift} from '../bundle/face-outline.mjs';
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
   assert.deepEqual(p.mapping.restY,[431.5,497,554.5,616]);
   assert.equal(p.samplingSpan,undefined);assert.equal(p.targetSpan,undefined);
  }
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
test('skin feature controls share the exact pitch residual of eyes, brows, nose and mouth',()=>{
 for(const yaw of [-45,0,45])for(const pitch of [-30,30])for(const [key,id] of Object.entries(ids)){
  const point=FEATURE_FIT.front[key],p=at(yaw,pitch),p0=at(yaw,0);
  // Compare the complete outline shift against feature-minus-legacy-skin.
  const skin=outlineWorld(point,p),skin0=outlineWorld(point,p0);
  const f=featureWorld(id,point,p),f0=featureWorld(id,point,p0);
  for(const i of [0,1])near(skin[i]-skin0[i],f[i]-f0[i],`${key} ${yaw},${pitch} axis ${i}`);
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
test('sclera stays at least two pixels inside the skin in nine key and sixteen intermediate poses',()=>{
 const poses=[...[-45,0,45].flatMap(x=>[-30,0,30].map(y=>[x,y])),...[-33.75,-11.25,11.25,33.75].flatMap(x=>[-22.5,-7.5,7.5,22.5].map(y=>[x,y]))];
 const failed=poses.map(([yaw,pitch])=>({yaw,pitch,margins:eyeMargins(at(yaw,pitch))})).filter(r=>Math.min(...r.margins)<2);
 assert.deepEqual(failed,[],'sclera margins below 2px');
});
test('sclera margin gate detects an injected protrusion',()=>{
 assert(eyeMargins(at(0,0),1000).some(v=>v<2),'injected protrusion must fail the margin gate');
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
test('near ear inner edge overlaps the cheek by at least ten pixels at six turned poses',()=>{
 for(const yaw of [-45,45])for(const pitch of [-30,0,30]){
  const side=yaw<0?'right':'left',x=side==='right'?626:379,p=at(yaw,pitch);
  for(const y of [440,465,490]){
   const q=deform3D(mesh('ear-'+side),[x,y],p),edge=boundaryX(side,q[1],p);
   const overlap=side==='right'?edge-q[0]:q[0]-edge;
   assert(overlap>=10,`${yaw},${pitch},${y}: ${overlap}`);
  }
 }
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
