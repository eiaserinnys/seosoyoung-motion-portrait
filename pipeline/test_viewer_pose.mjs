import {test} from 'node:test';
import assert from 'node:assert/strict';
import {viewerPose,sliderForYaw} from '../bundle/viewer-pose.mjs';
test('viewer preserves its original rotation through the 45 degree slider seam',()=>{
 for(const gain of [.4,.65,1])for(const yaw of [-45,-44.99,0,44.99,45]){
  const p={ParamAngleX:yaw,ParamAngleY:20,CorrTurn:gain};
  assert.strictEqual(viewerPose(p),p);
 }
});
test('recommended and pure viewer sliders reach both profiles continuously',()=>{
 for(const gain of [.4,.65,1])for(const sign of [-1,1]){
  const p={ParamAngleX:sign*90,ParamAngleY:20,CorrTurn:gain},q=viewerPose(p);
  assert.equal(q.ParamAngleX*q.CorrTurn,sign*90);
  assert.equal(q.ParamAngleY*q.CorrTurn,20*gain);
  const edge=viewerPose({...p,ParamAngleX:sign*45.000001});
  assert(Math.abs(edge.ParamAngleX*edge.CorrTurn-sign*45*gain)<.00001);
  for(const target of [67.5,90]){
   const actual=viewerPose({...p,ParamAngleX:sliderForYaw(sign*target,gain)});
   assert(Math.abs(actual.ParamAngleX*actual.CorrTurn-sign*target)<1e-10);
  }
 }
});
