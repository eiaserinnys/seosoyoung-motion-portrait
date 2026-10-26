import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {eyeClipControls} from '../bundle/renderer.mjs';
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
test('the profile branch exactly retains the original family of clipping targets',{timeout:60000},()=>{
 for(const yaw of [-90,-67.5,67.5,90])for(const mesh of rig.meshes){
  const expected=mesh.id.startsWith('eye-')||mesh.id==='closed-eyes-v1'||mesh.id==='nose-detail'||mesh.id.startsWith('mouth-');
  const c=eyeClipControls(mesh.id,yaw);assert.equal(c.profileFaceClip,expected);assert.equal(c.eyeFaceClip,0);
 }
});
