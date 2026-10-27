import {test} from 'node:test';
import assert from 'node:assert/strict';
import {profileRows} from './face_outline_stage_a/refinement/profile_rows.mjs';
test('profile mapping uses mesh intersections even when the old rim moved less than 40px',()=>{
 const p=[[0,0],[10,0],[10,10],[0,10],[2,5],[8,5],[2,-1],[8,-1]];
 const world=p.map(([x,y])=>[x+1,y]),old=p.map(q=>q.slice());
 const rows=profileRows([{y:5,left:4,right:5},{y:-1,left:6,right:7}],world,old,p,[[0,1,2],[0,2,3]]);
 assert.deepEqual(rows[0].selected,[1,11]);assert.equal(rows[0].definition,'mesh-boundary');
 assert.deepEqual(rows[1].selected,[3,9]);assert.equal(rows[1].definition,'unchanged-upper-rim');
});
