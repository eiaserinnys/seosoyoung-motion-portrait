import assert from 'node:assert/strict';
import {test} from 'node:test';
import {readFileSync} from 'node:fs';
import {bindRig,deform3D,opacity} from '../bundle/rig.mjs';
import {CORRECTION_PRESETS,correctedLift,artWorld} from '../bundle/geometry.mjs';
import {domeSection,craniumPosition} from '../bundle/hair-surface.mjs';
import {hairLift,hairWorld,sideWorld,backWorld,craniumWorld} from '../bundle/hair-shell.mjs';
import {craniumWorld as exportedCranium} from '../bundle/cranium-surface.mjs';
import './test_9dir.mjs';
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
  const rest={...p,ParamAngleX:0,ParamAngleY:0},front=point('hair-top',xy,rest),back=point('back-hair-center',xy,rest);
  near(front.slice(0,2),back.slice(0,2));
  near(back,artWorld(hairLift(xy,rest,-4),rest));checks++;
 }
 for(const [side,x] of [['left',379],['right',626]]){
  const original=sideWorld([x,438],side,p),morphed=sideWorld([x,438],side,{...p,CorrOutline:1});
  assert.equal(original[2],morphed[2],'outline displacement never changes lock depth');checks++;
 }
 if(yaw||pitch){
  assert.notDeepEqual(point('hair-top',[502,150],p),point('hair-top',[502,150],{...p,CorrHairDepth:.5}));checks++;
 }
 for(const id of ['bangs-left','bangs-right','hairpin']){
  near(point(id,[502,310],p),hairWorld([502,310],p));checks++;
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
 assert(opacity(mesh(id),p,{hidden:new Set()})<opacity(mesh(id),{...p,CorrHairOcclusion:0},{hidden:new Set()}),'far ear occlusion control');checks++;
 const nearPose={...p,ParamAngleX:-yaw};
 assert(width(nearPose)>width(p),'near ear remains wider');checks++;
}
// The temple lock must not overtake its fringe into the forehead arch.
for(const [side,x,yaw,edge] of [['right',674,-45,535],['left',330,45,469]]){
 const q=sideWorld([x,330],side,{...base,CorrTurn:1,ParamAngleX:yaw});
 assert(side==='right'?q[0]>=edge:q[0]<=edge,`${side} temple crosses forehead: ${q[0]}`);checks++;
}
for(const side of ['left','right'])for(const y of [200,260,330,350])for(const yaw of [-45,0,45]){
 const {cx,rx}=domeSection(y),xy=[502+cx+(side==='left'?-rx:rx),y],p={...base,CorrTurn:1,ParamAngleX:yaw};
 near(sideWorld(xy,side,p),hairWorld(xy,p));checks++;
}
console.log(`PASS ${checks} hair shell and preserved face checks`);

// The backing stays on the fringe hemisphere through y380, then joins the
// lower shell with a smooth tangent. The former sign flip made a 390px ridge.
for(const yaw of [-45,0,45])for(const pitch of [-30,0,30]){
 const p={...base,CorrTurn:1,ParamAngleX:yaw,ParamAngleY:pitch};
 for(const x of [300,400,502,600,700]){
  for(const y of [60,180,245,300,350,380])near(backWorld([x,y],p),artWorld(hairLift([x,y],p,-4),p));
  for(const y of [245,350,380,465]){
   const a=backWorld([x,y-.001],p),b=backWorld([x,y+.001],p);
   assert(Math.hypot(...a.map((v,i)=>v-b[i]))<.1,'backing contains a depth step');
  }
 }
}

test('cranium bridge meets the shared canopy seam at all nine poses',()=>{
 assert.equal(exportedCranium,craniumWorld,'one implementation owns both exports');
 for(const yaw of [-45,0,45])for(const pitch of [-30,0,30])for(const sign of [-1,1])for(const y of [60,120,180,245,300,350,400,465]){
  const p={...base,CorrTurn:1,ParamAngleX:yaw,ParamAngleY:pitch};
  const {cx,rx}=domeSection(y),side=sign<0?'left':'right';
  const a=craniumWorld([0,y,sign*Math.PI/2],side,p),b=hairWorld([502+cx+sign*rx,y],p);
  assert(Math.hypot(...a.map((v,i)=>v-b[i]))<1e-14,`${yaw},${pitch},${sign},${y}`);
 }
});
test('cranium bridge preserves the outer silhouette at and beyond 135 degrees',()=>{
 for(const yaw of [-45,0,45])for(const pitch of [-30,0,30])for(const sign of [-1,1])for(const y of [60,180,300,400,465])for(const angle of [135,150,180]){
  const p={...base,CorrTurn:1,ParamAngleX:yaw,ParamAngleY:pitch},pt=[0,y,sign*angle*Math.PI/180];
  assert.deepEqual(craniumWorld(pt,sign<0?'left':'right',p),craniumPosition(pt,p));
 }
});
test('upper diagonal fringe strands remain wholly on the fringe surface',()=>{
 for(const yaw of [-45,45])for(const pitch of [0,30])for(const y of [350,380,400,405])for(const x of [606,615,624]){
  const p={...base,CorrTurn:1,ParamAngleX:yaw,ParamAngleY:pitch};
  near(hairWorld([x,y],p),artWorld(hairLift([x,y],p,0),p));
 }
});
