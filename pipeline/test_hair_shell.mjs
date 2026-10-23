import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {bindRig,deform3D} from '../bundle/rig.mjs';
import {CORRECTION_PRESETS,correctedLift,artWorld} from '../bundle/geometry.mjs';
import {hairLift,hairWorld,sideWorld} from '../bundle/hair-shell.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));bindRig(rig);
const base={...Object.fromEntries(rig.parameters.map(p=>[p.id,p.default])),...CORRECTION_PRESETS.recommended};
const mesh=id=>rig.meshes.find(m=>m.id===id);
const point=(id,xy,p)=>deform3D(mesh(id),xy,p);
const near=(a,b)=>assert(Math.max(...a.map((x,i)=>Math.abs(x-b[i])))<1e-8);
let checks=0;
for(const [yaw,pitch] of [[0,0],[-25,0],[25,0],[-50,0],[50,0],[0,-20],[0,20]]){
 const p={...base,CorrOutline:0,ParamAngleX:yaw,ParamAngleY:pitch};
 for(const xy of [[330,530],[502,552],[502,617]]){
  near(point('face-skin',xy,p),artWorld(correctedLift(xy,0,p),p));checks++;
 }
 for(const xy of [[502,64],[502,150],[502,230]]){
  const front=point('hair-top',xy,p),back=point('back-hair-center',xy,p);
  assert(Math.hypot(...front.map((v,i)=>v-back[i]))<=6.000001,'crown layers share one surface');checks++;
 }
 for(const [side,x] of [['left',379],['right',626]]){
  near(sideWorld([x,438],side,p),point('face-skin',[x,438],p));checks++;
 }
 if(yaw||pitch){
  assert.notDeepEqual(point('hair-top',[502,150],p),point('hair-top',[502,150],{...p,CorrHairDepth:.5}));checks++;
 }
 for(const id of ['bangs-left','bangs-right','hairpin']){
  const relief=id==='hairpin'?8:6;
  near(point(id,[502,310],p),artWorld(correctedLift([502,310],relief,p),p));checks++;
 }
}
assert.equal(hairLift([502,64],base)[2],0,'north pole has zero depth');checks++;
for(const y of [150,230]){
 assert(hairLift([502,y],base)[2]>hairLift([502,y],{...base,CorrHairDepth:.4})[2],'volume control deepens upper cap');checks++;
}
const capHeight=p=>hairWorld([502,245],p)[1]-hairWorld([502,64],p)[1];
assert(capHeight({...base,ParamAngleY:-20})>capHeight(base));checks++;
assert(capHeight({...base,ParamAngleY:20})<capHeight(base));checks++;
for(const side of ['left','right']){
 const yaw=side==='left'?-50:50,p={...base,ParamAngleX:yaw};
 const id='ear-'+side,[x,y,w]=mesh(id).bbox;
 const width=q=>Math.abs(point(id,[x+w,y+50],q)[0]-point(id,[x,y+50],q)[0]);
 assert(width(p)<width({...p,CorrHairOcclusion:0}),'far ear narrows');checks++;
 const nearPose={...p,ParamAngleX:-yaw};
 assert(width(nearPose)>width(p),'near ear remains wider');checks++;
}
console.log(`PASS ${checks} hair shell and preserved face checks`);
