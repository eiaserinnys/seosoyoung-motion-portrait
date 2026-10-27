import {test} from 'node:test';
import assert from 'node:assert/strict';
import {applyFamily45Projection} from './face_outline_stage_a/refinement/family45_expectations.mjs';
const inputs={family45:{poses:[{name:'left67',obs:[[1,2],[3,4]]}]}};
const record={poses:[{name:'left67',moves:[{index:1,duBefore:[0,0],duAfter:[.25,0],delta:[.25,0]}]}]};
test('profile expectations move only recorded vertices and leave raw solve inputs unchanged',()=>{
 const result=applyFamily45Projection(inputs,record);
 assert.deepEqual(result.family45.poses[0].obs,[[1,2],[3.25,4]]);
 assert.deepEqual(inputs.family45.poses[0].obs,[[1,2],[3,4]]);
});
test('profile expectations reject an injected displacement that disagrees with saved du',()=>{
 const broken=structuredClone(record);broken.poses[0].moves[0].delta[0]+=1;
 assert.throws(()=>applyFamily45Projection(inputs,broken),/displacement disagrees/);
});
