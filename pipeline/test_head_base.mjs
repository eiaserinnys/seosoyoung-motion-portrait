import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {configureHeadBase,projectModel,modelAngles,phi,fieldAt,headBaseVertices,headSolveWorld} from '../bundle/head-base.mjs';
import * as headBase from '../bundle/head-base.mjs';
import {CORRECTION_PRESETS} from '../bundle/corrections.mjs';
import {FRAME_CACHE} from '../bundle/pose-grid.mjs';
import {FACE_SOLVE} from '../bundle/face-solve-data.mjs';
import {outlineWorld} from '../bundle/face-outline.mjs';
import {bindRig,deform3D,opacity} from '../bundle/rig.mjs';
import {backgroundLayerPolicy,neckDepthPositions} from '../bundle/renderer.mjs';
import * as renderer from '../bundle/renderer.mjs';
import {skinFillPositions} from '../bundle/skin-fill.mjs';
import {sideWorld} from '../bundle/hair-shell.mjs';
import {earWorld} from '../bundle/stereo-attachments.mjs';

const read=path=>readFileSync(new URL(path,import.meta.url));
const metadata=JSON.parse(read('../bundle/assets/head-base.json'));
const binary=read('../bundle/assets/head-base.bin');
configureHeadBase(metadata,binary.buffer.slice(binary.byteOffset,binary.byteOffset+binary.byteLength));
const fixture=JSON.parse(read('./headmodel/fit/results/fit2e/head-base-fixture.json'));
const rig=JSON.parse(read('../bundle/assets/seosoyoung-rig.json'));bindRig(rig);
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended,CorrHeadBase:1,CorrOutline:1,CorrTurn:1};
const pose=(yaw=0,pitch=0,extra={})=>({...base,ParamAngleX:yaw,ParamAngleY:pitch,...extra,[FRAME_CACHE]:new Map()});
const skin=rig.meshes.find(m=>m.id==='face-skin');
const feature=rig.meshes.filter(m=>m.id.startsWith('eye-')||m.id==='closed-eyes-v1'||m.id==='nose-detail'||m.id.startsWith('mouth'));
const close=(a,b,tol,label)=>assert(Math.abs(a-b)<=tol,`${label}: ${a} vs ${b}`);
const digest=v=>createHash('sha256').update(JSON.stringify(v)).digest('hex');
const oldFeatures={
 '1:0':'80c0e03a75e7d8fdc530f7814323c25a933708fa6c4d6e69a68f824e9326f7f2',
 '1:0.5':'bcdbe31ed6c048b067628dfb4fb7e936524a7f2045a4d782f3baa3543709cf88',
 '1:1':'aa6b7136d84fc56dcba3d375d34c871f142eee458fcf9774ba50899a13a1c51a',
 '0.65:0':'29f256bdce53066bb90bcebe355239c01c19b1cb6d05a662aec93c98ef436f84',
 '0.65:0.5':'f7d7592cc233828d99f27a709c1b4f87d4a3d58bba48ef8c21e372c095885138',
 '0.65:1':'9cfe292c3370db2ba277f50d34bf5323268b5052a2a3ff94e3311b4205d5df70'};
const oldSkin={'1':'643b11c1d22b621f598f6633185d2cd82ee78b03b01f040cb0a4bef5b3048631','0.65':'80388b8f6df4482094cc9e250cde58010b3020732108f266714c148f4241c02c'};

test('ear angle margin retains the near ear and fades to two pixels',{timeout:30000},()=>{
 for(const side of ['left','right']){
  const b=metadata.ear_depth_bias_px[side],sign=side==='left'?-1:1;
  for(const far of [-45,0])close(renderer.earDepthMargin(side,pose(sign*far),metadata.ear_depth_bias_px),b,1e-9,'near/rest');
  for(const far of [20,45,90])close(renderer.earDepthMargin(side,pose(sign*far),metadata.ear_depth_bias_px),2,1e-9,'far');
  close(renderer.earDepthMargin(side,pose(sign*10),metadata.ear_depth_bias_px),b*.5,1e-9,'midpoint');
 }
});
test('neck extension reaches the projected head ring and keeps cylinder depth',{timeout:30000},()=>{
 const neck=rig.meshes.find(m=>m.id==='neck'),head=headBase.getHeadBase(),p=pose();
 const extended=renderer.createNeckExtension(neck,head,p);
 const target=Math.min(...head.open_boundary.map(i=>head.reference[i*3+1]));
 const first=neck.positions.length;
 assert(extended.positions.length>first);
 for(const point of extended.positions.slice(first))close(deform3D(extended,point,p)[1],target,1e-6,'rest top');
 const pts=extended.positions.map(point=>deform3D(extended,point,p));
 const depth=neckDepthPositions(extended,pts,head.reference,head.open_boundary);
 const zc=head.open_boundary.reduce((s,i)=>s+head.reference[i*3+2],0)/head.open_boundary.length;
 for(const point of depth.slice(first))assert(point[2]>=zc-1e-9,'extension must use neck cylinder depth');
 assert.equal(backgroundLayerPolicy(neck,p,{hidden:new Set(),outfit:'sweater',bangs:false},true).compareHead,true);
});

test('N7: Python projection fixture, 100 vertices and 13 poses',{timeout:30000},()=>{
 const vertices=new Float32Array(binary.buffer,binary.byteOffset+metadata.offsets.vertices.offset,metadata.offsets.vertices.length/4);
 for(let k=0;k<fixture.poses.length;k++)for(let j=0;j<fixture.indices.length;j++){
  const i=fixture.indices[j],q=projectModel(Array.from(vertices.slice(i*3,i*3+3)),fixture.poses[k]);
  q.forEach((x,c)=>close(x,fixture.projections[k][j][c],1e-6,`pose ${k} point ${j}`));
 }
});
test('angle field fixture at all nine nodes',{timeout:30000},()=>{
 metadata.theta_k.forEach((theta,k)=>fixture.indices.forEach((index,j)=>fieldAt(theta,index).forEach((x,c)=>close(x,fixture.fields[k][j][c],1e-6,`field ${k} point ${j}`))));
});
test('N1: all rest skin and solve points retain their coordinates',{timeout:30000},()=>{
 const p=pose();for(const point of [...skin.positions,...FACE_SOLVE.points]){
  const q=headSolveWorld(point,p);close(q[0],point[0],1e-3,'rest x');close(q[1],point[1],1e-3,'rest y');
  const via=outlineWorld(point,p);close(via[0],point[0],1e-3,'outline x');close(via[1],point[1],1e-3,'outline y');
 }
});
test('N2: legacy skin and all feature vertices match main hashes',{timeout:30000},()=>{
 for(const gain of [1,.65])for(const outline of [0,.5,1])for(const enabled of [0,1]){
  const values=[],skinValues=[];
  for(const pitch of [-30,0,30])for(const yaw of [-45,0,45]){
   const p=pose(yaw,pitch,{CorrTurn:gain,CorrOutline:outline,CorrHeadBase:enabled});
   for(const mesh of feature)for(const point of mesh.positions)values.push(deform3D(mesh,point,p));
   if(!enabled&&outline===1)for(const point of skin.positions)skinValues.push(deform3D(skin,point,p));
  }
  assert.equal(digest(values),oldFeatures[`${gain}:${outline}`],`features ${gain}/${outline}/${enabled}`);
  if(!enabled&&outline===1)assert.equal(digest(skinValues),oldSkin[gain],`skin ${gain}`);
 }
});
test('N6: skin and head surface are continuous at both 45 degree boundaries',{timeout:30000},()=>{
 for(const sign of [-1,1])for(const pitch of [-30,0,30]){
  const a=pose(sign*44.99,pitch),b=pose(sign*45.01,pitch),va=headBaseVertices(a).surface,vb=headBaseVertices(b).surface;
  for(let i=0;i<va.length;i+=3)assert(Math.hypot(va[i]-vb[i],va[i+1]-vb[i+1])<=.5,`head ${sign}/${pitch}/${i/3}`);
  for(const point of skin.positions){const x=headSolveWorld(point,a),y=headSolveWorld(point,b);assert(Math.hypot(x[0]-y[0],x[1]-y[1])<=.5,`skin ${sign}/${pitch}/${point}`);}
 }
});
test('ear root preserves rest coordinates and matches the Kabsch fixture',{timeout:30000},()=>{
 for(const [side,anchor] of [['left',[369,465]],['right',[636,464.5]]]){
  const p=pose(),frame=headBase.attachedTransform(anchor,p);
  for(let i=0;i<3;i++)for(let j=0;j<3;j++)close(frame.R[i][j],Number(i===j),1e-9,`rest rotation ${side}`);
  const mesh=rig.meshes.find(m=>m.id==='ear-'+side),root=metadata.ear_roots[side],ref=headBase.getHeadBase().reference;
  const depth=root.ring.reduce((sum,i)=>sum+ref[i*3+2],0)/root.ring.length;
  for(const point of mesh.positions){const q=headBase.headEarWorld(point,side,p);close(q[0],point[0],1e-6,`${side} x`);close(q[1],point[1],1e-6,`${side} y`);close(q[2],depth+5,1e-6,`${side} z`);}
  for(const item of fixture.ear_roots){
   const q=pose(item.yaw,item.pitch),expected=item.expected[side],origin=headBase.headEarWorld(anchor,side,q);
   for(let i=0;i<3;i++)close(origin[i],expected.anchor[i],1e-6,`${side} key ${item.frame} anchor`);
   for(let j=0;j<2;j++){const point=anchor.map((x,i)=>x+Number(i===j)),value=headBase.headEarWorld(point,side,q);
    for(let i=0;i<3;i++)close(value[i]-origin[i],expected.R[i][j],1e-6,`${side} key ${item.frame} rotation`);
   }
  }
  const point=mesh.positions[0],angle=17*Math.PI/180,c=Math.cos(angle),s=Math.sin(angle);
  const rolled=headBase.headEarWorld(point,side,pose(0,0,{ParamAngleZ:17})),x=point[0]-502,y=point[1]-568;
  close(rolled[0],502+c*x-s*y,1e-6,`${side} roll x`);close(rolled[1],568+s*x+c*y,1e-6,`${side} roll y`);
  close(rolled[2],depth+5,1e-6,`${side} roll z`);
 }
});
test('ear routing preserves the legacy path whenever the new surface is disabled',{timeout:30000},()=>{
 for(const [side,point] of [['left',[369,465]],['right',[636,464.5]]])for(const pitch of [-30,0,30])for(const yaw of [-45,0,45]){
  for(const [enabled,outline] of [[0,0],[0,.5],[0,1],[1,0],[1,.5]]){
   const p=pose(yaw,pitch,{CorrHeadBase:enabled,CorrOutline:outline});
   assert.deepEqual(sideWorld(point,side,p,true),earWorld(point,side,p));
  }
 }
});
test('neck cap closes the original boundary without adding a skirt',{timeout:30000},()=>{
 const ring=metadata.open_boundary,count=ring.length;
 assert.equal(metadata.vertexCount,12258);assert.equal(metadata.triangleCount,24389+count);
 const edges=new Map(),triangles=headBase.getHeadBase().triangles;
 for(let i=0;i<triangles.length;i+=3)for(const [a,b] of [[triangles[i],triangles[i+1]],[triangles[i+1],triangles[i+2]],[triangles[i+2],triangles[i]]]){
  const key=[Math.min(a,b),Math.max(a,b)].join(',');edges.set(key,(edges.get(key)||0)+1);
 }
 const boundary=[...edges].filter(([,count])=>count===1).map(([key])=>key);
 assert.equal(boundary.length,0);
});
test('legacy ear alpha and neck mask are preserved when the new surface is inactive',{timeout:30000},()=>{
 const state={hidden:new Set(),outfit:'sweater',bangs:false};
 const ear=rig.meshes.find(m=>m.id==='ear-left'),neck=rig.meshes.find(m=>m.id==='neck');
 for(const [enabled,outline,visible] of [[0,0,true],[0,1,true],[1,0,true],[1,.5,true],[1,1,false]])for(const yaw of [-45,-15,0,15,45]){
  const p=pose(yaw,0,{CorrHeadBase:enabled,CorrOutline:outline});
  const ep=backgroundLayerPolicy(ear,p,state,visible),np=backgroundLayerPolicy(neck,p,state,visible);
  assert.equal(ep.alpha,opacity(ear,p,state));assert.equal(ep.compareHead,false);
  assert.equal(np.alpha,opacity(neck,p,state));assert.equal(np.compareHead,false);
  assert.equal(np.neckMask,true);
 }
 for(const yaw of [-45,-15,0,15,45]){
  const p=pose(yaw),ep=backgroundLayerPolicy(ear,p,state,true),np=backgroundLayerPolicy(neck,p,state,true);
  assert.equal(ep.alpha,1);assert.equal(ep.compareHead,true);
  assert.equal(np.compareHead,true);assert.equal(np.neckMask,false);
 }
 const hidden={...state,hidden:new Set(['ear-left'])};
 assert.equal(backgroundLayerPolicy(ear,pose(-15),hidden,true).alpha,0);
});
test('neck depth uses the original xy and the rest ring cylinder',{timeout:30000},()=>{
 const neck=rig.meshes.find(m=>m.id==='neck'),p=pose(),points=neck.positions.map(point=>deform3D(neck,point,p));
 const ref=headBase.getHeadBase().reference,values=neckDepthPositions(neck,points,ref,metadata.open_boundary);
 const ring=metadata.open_boundary,zc=ring.reduce((sum,i)=>sum+ref[i*3+2],0)/ring.length;
 points.forEach((v,i)=>{close(values[i][0],v[0],1e-10,'neck x');close(values[i][1],v[1],1e-10,'neck y');assert(values[i][2]>=zc-1e-10);});
});
test('ear bias covers each rest ear depth and preserves legacy neck and fill positions',{timeout:30000},()=>{
 const evidence=JSON.parse(read('../docs/captures/fix4-baseline/rest_analysis.json'));
 for(const side of ['left','right'])close(metadata.ear_depth_bias_px[side],evidence.ears[side].min_bias_px+1,1e-9,'ear rest margin');
 const neck=rig.meshes.find(m=>m.id==='neck'),values=[];
 for(const pitch of [-30,0,30])for(const yaw of [-45,0,45]){
  const p=pose(yaw,pitch,{CorrHeadBase:0,ParamAngleZ:0,ParamBodyAngleX:0,ParamBreath:0});
  values.push({neck:neck.positions.map(q=>deform3D(neck,q,p)),fill:skinFillPositions(p,skin)});
 }
 assert.equal(digest(values),'5e43bf076d88b50f7a325f9e2bbdb983ef00fe1ab3f2b1d52ac5dad17a4c2b7a');
});
