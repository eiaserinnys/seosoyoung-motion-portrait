import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {eyeClipControls,eyeMaskId,faceMaskSource} from '../bundle/renderer.mjs';
const rig=JSON.parse(readFileSync(new URL('../bundle/assets/seosoyoung-rig.json',import.meta.url)));
test('frontal artwork and near eyes retain their original masks; far eyes fade into clipping',{timeout:60000},()=>{
 for(const mesh of rig.meshes){
  const zero=eyeClipControls(mesh.id,0);assert.equal(zero.eyeFaceClip,0);assert.equal(zero.profileFaceClip,false);
 }
 for(const side of ['left','right']){
  const ids=rig.meshes.filter(m=>m.id.startsWith('eye-'+side+'-')).map(m=>m.id);
  assert.equal(ids.length,7);
  const sign=side==='left'?-1:1;
  for(const id of ids){
   assert.equal(eyeClipControls(id,-sign*20).eyeFaceClip,0);
   assert.equal(eyeClipControls(id,sign*15).eyeFaceClip,1);
   assert.equal(eyeClipControls(id,sign*7.5).eyeFaceClip,.5);
  }
 }
});
test('J4: every eye mask stays fixed and clip strengths are continuous at all junctions',{timeout:60000},()=>{
 for(const mesh of rig.meshes.filter(m=>m.id.startsWith('eye-')||m.id==='closed-eyes-v1')){
  for(const yaw of [-90,-67,-52,-45,-15,0,15,45,52,67,90]){
   const a=eyeClipControls(mesh.id,yaw-1e-7),b=eyeClipControls(mesh.id,yaw+1e-7);
   assert.equal(a.profileFaceClip,false);assert.equal(b.profileFaceClip,false);
   assert(Math.abs(a.eyeFaceClip-b.eyeFaceClip)<1e-6);
   assert.equal(eyeMaskId(mesh,yaw),mesh.masks[0]);
   assert.equal(faceMaskSource(mesh.id,yaw),'skin-silhouette');
  }
 }
 assert.equal(faceMaskSource('face-skin',0),'exposed-skin-clip');
});
test('profile eye and closed-lid clipping stays soft while nose and mouth retain hard clipping',{timeout:60000},()=>{
 for(const yaw of [-90,-67.5,67.5,90])for(const mesh of rig.meshes){
  const eye=mesh.id.startsWith('eye-')||mesh.id==='closed-eyes-v1';
  const expected=mesh.id==='nose-detail'||mesh.id.startsWith('mouth-');
  const c=eyeClipControls(mesh.id,yaw);assert.equal(c.profileFaceClip,expected);assert.equal(c.eyeFaceClip,eye?1:0);
 }
});
