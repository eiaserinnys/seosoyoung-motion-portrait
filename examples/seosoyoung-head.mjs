// Reference geometry in the portrait's 1024-wide artwork coordinates.
// These are initial scene dimensions, not parameters recovered from a third-party model.
import {lift,project,eyePoint,deform,attachNeck} from '../src/head-turn.mjs';
export const head={center:[502,326],radius:262};
export const eyeRadius=54;
export const eyeCenters={left:[406,438],right:[597,438]};
export function headVertex(point,angles,residual) {
  return residual ? deform(point,head,angles,residual,30) : project(lift(point,head),head,angles);
}
export function eyeVertex(point,side,gazeAngles,headAngles) {
  const center=eyeCenters[side];if(!center)throw Error('unknown eye');
  const root=lift(center,head);root[2]-=eyeRadius-3;
  const dx=point[0]-center[0],dy=point[1]-center[1];
  const local=[dx,dy,Math.sqrt(Math.max(0,eyeRadius**2-dx**2-dy**2))];
  const world=eyePoint(local,root,gazeAngles,headAngles);
  return [head.center[0]+world[0],head.center[1]+world[1]];
}
export function neckVertex(point,jawPoint,angles,weight) {
  const moved=headVertex(jawPoint,angles);
  return attachNeck(point,moved.map((v,i)=>v-jawPoint[i]),weight);
}
// Consumer smoke example: node examples/seosoyoung-head.mjs
if(typeof process!=='undefined' && process.argv[1] && import.meta.url===new URL(process.argv[1],'file:').href){
  console.log(JSON.stringify({neutral:headVertex([502,500],[0,0,0]),turn:headVertex([502,500],[25,15,0]),
    eye:eyeVertex([417,438],'left',[0,0,0],[25,15,0]),neckBase:neckVertex([502,650],[502,570],[25,15,0],0)}));
}
