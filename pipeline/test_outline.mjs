import {stageAScalars} from './approved_stage_a.mjs';
import './test_stereo.mjs';
import {craniumFacing,domeSection,craniumPosition} from '../bundle/hair-surface.mjs';
import {DOME} from '../bundle/dome-data.mjs';
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {bindRig,deform3D,opacity} from '../bundle/rig.mjs';
import {CORRECTION_PRESETS,correctedLift} from '../bundle/geometry.mjs';
import {hairLift} from '../bundle/hair-shell.mjs';
import {mainOpaqueRuns,projectedHeadInterval,headInteriorInterval,rowAlphaGaps,foreheadBoundary,foreheadIntrusion} from './raster_interior.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));bindRig(rig);
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended};
const mesh=id=>rig.meshes.find(m=>m.id===id);
test('forehead intrusion follows the projected lock attachment and rejects inward pixels',()=>{
 for(const yaw of [-45,45])for(const y of [280,310,344,360]){
  const p={...base,CorrTurn:1,ParamAngleX:yaw},boundary=foreheadBoundary(y,p),sign=Math.sign(yaw);
  assert.equal(foreheadIntrusion(boundary-sign,boundary,yaw),false);
  assert.equal(foreheadIntrusion(boundary+sign,boundary,yaw),true);
 }
});
test('raster interior rejects injected closed and open gaps while excluding exterior background',()=>{
 const width=32,rgba=new Uint8Array(width*4);
 for(let x=8;x<24;x++)rgba[x*4+3]=255;
 assert.equal(rowAlphaGaps(rgba,width,1,0,8,24).count,0);
 rgba[16*4+3]=0;
 assert.equal(rowAlphaGaps(rgba,width,1,0,8,24).count,1);
 for(let x=14;x<18;x++)rgba[x*4+3]=0;
 assert.equal(rowAlphaGaps(rgba,width,1,0,8,24).count,4);
});
test('near-side bridged coverage remains strict while the far boundary follows source alpha',()=>{
 const y=300,scale=2,ox=30;
 for(const yaw of [-45,0,45]){
  const p={...base,CorrTurn:1,ParamAngleX:yaw},a=headInteriorInterval(y,p,100,1900,ox,scale);
  const [left,right]=projectedHeadInterval(y,p);
  const {rx,rz,cz,cx}=domeSection(y),angle=yaw*Math.PI/180;
  const center=502+cx*Math.cos(angle)+cz*Math.sin(angle),radius=Math.hypot(rx*Math.cos(angle),rz*Math.sin(angle));
  assert.equal(a[0],yaw>0?ox+(left+15)*scale:Math.max(130,yaw<0?ox+(center-radius+15)*scale:0));
  assert.equal(a[1],yaw<0?ox+(right-15)*scale:Math.min(1870,yaw>0?ox+(center+radius-15)*scale:Infinity));
  const shifted=headInteriorInterval(y,p,110,1890,ox,scale);
  if(yaw>0)assert.equal(a[0],shifted[0]);
  if(yaw<0)assert.equal(a[1],shifted[1]);
 }
});
test('main opaque spans ignore five-pixel wisps but retain holes between head regions',()=>{
 const width=200,rgba=new Uint8Array(width*4);
 const fill=(a,b)=>{for(let x=a;x<b;x++)rgba[x*4+3]=255;};
 fill(0,5);fill(40,95);fill(105,160);fill(185,190);
 const span=mainOpaqueRuns(rgba,width,1,0,1);
 assert.deepEqual(span,[40,160]);
 assert.equal(rowAlphaGaps(rgba,width,1,0,span[0]+15,span[1]-15).count,10);
 fill(95,105);
 assert.equal(rowAlphaGaps(rgba,width,1,0,span[0]+15,span[1]-15).count,0);
 assert.deepEqual(mainOpaqueRuns(rgba,width,1,0,2),[40,160]);
});
test('far ear notch outside the dome is excluded while an internal gap still fails',()=>{
 const width=2200,y=414,rgba=new Uint8Array(width*4);
 for(let x=0;x<width;x++)rgba[x*4+3]=255;
 for(const yaw of [-45,45]){
  const p={...base,CorrTurn:1,ParamAngleX:yaw},[left,right]=headInteriorInterval(y,p,0,width,0,2);
  assert(left>30&&right<width-30,'far ear notch lies beyond the dome intersection');
  const outside=yaw<0?Math.floor(left)-5:Math.ceil(right)+5;
  rgba[outside*4+3]=0;
  assert.equal(rowAlphaGaps(rgba,width,1,0,left,right).count,0);
  const inside=Math.floor((left+right)/2);rgba[inside*4+3]=0;
  assert.equal(rowAlphaGaps(rgba,width,1,0,left,right).count,1);
  rgba[outside*4+3]=255;rgba[inside*4+3]=255;
 }
});
const state={hidden:new Set(),outfit:'sweater'};
test('front ears lie behind the cheek surface',()=>{
 for(const side of ['left','right'])for(const y of [420,460,500]){
  const x=side==='left'?360:640,p=base;
  // Near side hair now paints over the ear independently of depth (M6).
  assert(deform3D(mesh('ear-'+side),[x,y],p)[2]<correctedLift([x,y],0,p)[2]);
 }
});
test('far ear is fully hidden at 45 degrees',()=>{
 for(const yaw of [-45,45])assert.equal(opacity(mesh('ear-'+(yaw<0?'left':'right')),{...base,ParamAngleX:yaw,CorrTurn:1},state),0);
});
test('far back hair contracts at least as much as side hair',()=>{
 for(const gain of [.65,1])for(const yaw of [-50,-25,25,50])for(const y of [350,440,530]){
  const side=yaw<0?'left':'right',x=yaw<0?210:775,p={...base,ParamAngleX:yaw,CorrTurn:gain};
  const back=deform3D(mesh('back-hair-'+side),[x,y],p)[0],front=deform3D(mesh('side-hair-'+side),[x,y],p)[0];
  assert(yaw<0?back>=front-1e-8:back<=front+1e-8,`${yaw} ${gain} ${y}: ${back} / ${front}`);
 }
});
test('upper added depth has negative horizontal second derivative',()=>{
 const extra=x=>hairLift([x,180],base)[2]-hairLift([x,180],{...base,CorrHairDepth:.4})[2];
 assert(extra(402)+extra(602)-2*extra(502)<-1);
});
test('45 degree chin reaches the approved A-stage position',()=>{
 const trace=JSON.parse(readFileSync(new URL('../source/viewer/outline-traces.json',import.meta.url))),row=trace.rows.at(-1);
 for(const yaw of [-45,45]){
  const q=deform3D(mesh('face-skin'),[502.5,616],{...base,CorrTurn:1,ParamAngleX:yaw});
  const target=stageAScalars().chin[[yaw,0]].center[0];
  // The edge fit retains the curved 3D interior, rather than flattening it.
  assert(Math.abs(q[0]-target)<1,`${q}`);
 }
});

test('opaque cranium emerges geometrically from the near horizon',()=>{
 for(const side of ['left','right']){
  const m=mesh('side-cranium-'+side);assert(m&&m.triangles.length>0);
  const nearYaw=side==='right'?-45:45;
  for(const yaw of [0,-nearYaw,nearYaw]){
   const p={...base,ParamAngleX:yaw,CorrTurn:1};
   assert.equal(opacity(m,p,state),1);
   assert(m.positions.every(q=>Number.isFinite(craniumFacing(q,p))));
   const visible=m.positions.filter(q=>craniumFacing(q,p)>1e-6).length;
   assert(yaw===nearYaw?visible>0:visible===0,`${side} ${yaw}: ${visible}`);
   // The two-degree overlap folds onto the horizon at rest; actual raster
   // occlusion is checked independently in the browser.
  }
 }
});
test('lower skull fits the ear attachment and measured 45-degree silhouette together',()=>{
 const {rx,rz,cz}=domeSection(465);
 assert(Math.abs(cz+rz*Math.sqrt(1-(DOME.lower.attachmentX/rx)**2)-DOME.lower.attachmentZ)<1e-8);
 assert(Math.abs((Math.hypot(rx,rz)-cz)/Math.sqrt(2)-DOME.lower.extent45)<1e-8);
 assert(rx<domeSection(330).rx);
});

test('sampled cheek and jaw edges match the approved A-stage positions at 45 degrees',()=>{
 const trace=JSON.parse(readFileSync(new URL('../source/viewer/outline-traces.json',import.meta.url)));
 for(const yaw of [-45,45])for(const [y,l,r,ll,lr,rl,rr,ty] of trace.rows.filter(row=>row[0]>=465)){
  const expected=[l,r].map(x=>stageAScalars().outline[[yaw,0,x,y]]);
  for(const [i,x] of [l,r].entries()){
   const q=deform3D(mesh('face-skin'),[x,y],{...base,ParamAngleX:yaw,CorrTurn:1});
   assert(Math.abs(q[0]-expected[i][0])<.001,`${yaw} ${y}: ${q[0]} vs ${expected[i][0]}`);
   assert(Math.abs(q[1]-expected[i][1])<.001);
  }
 }
});

test('unbridged inverse spherical longitude restores both reference meshes at ±45 degrees',()=>{
 for(const [side,yaw] of [['right',-45],['left',45]]){
  const m=mesh('side-cranium-'+side);
  for(const xy of m.positions){
   // The authored canopy bridge deliberately shifts the inner collar. This
   // contract concerns its unmodified spherical reference, tested separately
   // from final seam matching and silhouette preservation in test_hair_shell.
   const q=craniumPosition(xy,{...base,ParamAngleX:yaw,CorrTurn:1});
   assert(Math.hypot(q[0]-xy[0],q[1]-xy[1])<1e-8);
  }
 }
});
