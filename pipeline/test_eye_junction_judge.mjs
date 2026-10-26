import {test} from 'node:test';
import assert from 'node:assert/strict';
import {junctionVerdict} from './eye_junction_quality.mjs';
test('junction judge rejects injected excessive speed and a broken legacy join',{timeout:60000},()=>{
 assert.deepEqual(junctionVerdict(7.24,.069,6.92,0),{J1:true,J2:true,J6:true});
 assert.deepEqual(junctionVerdict(7.24,1,8,.001),{J1:false,J2:false,J6:false});
});
