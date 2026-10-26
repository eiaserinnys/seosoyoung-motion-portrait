import {test} from 'node:test';
import assert from 'node:assert/strict';
import {outsideVisibleOutline} from './eye_visible_outline.mjs';
test('R4 detects an injected eye pixel on actual transparent background, independent of face masks',{timeout:60000},()=>{
 const eye=new Uint8Array([255,255,255,255,144,144,144,220,255,255,255,127]);
 const outline=new Uint8Array([0,0,0,0,0,0,0,1,0,0,0,0]);
 assert.deepEqual(outsideVisibleOutline(eye,outline,3),{pixels:1,coordinates:[[0,0]]});
 outline[3]=255;
 assert.deepEqual(outsideVisibleOutline(eye,outline,3),{pixels:0,coordinates:[]});
});
