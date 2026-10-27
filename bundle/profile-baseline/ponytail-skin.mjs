import {FACE_SOLVE} from './face-solve-data.mjs';
import {gridValue,frameValue} from './pose-grid.mjs';
import {clamp,radians} from './geometry.mjs';

// Only the difference from the exact nine-key field is added. Feature controls
// keep their authored sphere/arch evaluator; hair uses its original field.
export function ponytailSkinDelta(point,yaw,pitch,p){
 const fade=clamp((410-point[1])/80,0,1)*clamp((pitch/30)**2,0,1);
 if(!fade)return [0,0];
 const values=frameValue(p,'ponytail-skin-continuous',()=>{
  const shell=FACE_SOLVE.ponytail.continuous;
  const evaluate=(yaw,pitch)=>{
   const y=clamp(yaw/45,-1,1),v=clamp(pitch/30,-1,1),a=radians(y*45),b=radians(v*30);
   const B=[1,y,v,y*y,y*v,v*v,y*y*y,y*y*v,y*v*v,v*v*v];
   return [-Math.PI,-Math.PI/2,0].map((theta,i)=>{
    const x=shell.front[i]*Math.cos(theta),h=shell.front[i]*Math.sin(theta),z=shell.depth[i];
    const xx=x*Math.cos(a)+h*Math.sin(a)*Math.sin(b)+z*Math.sin(a)*Math.cos(b),yy=h*Math.cos(b)-z*Math.sin(b);
    return Math.hypot(xx,yy)+B.reduce((s,c,j)=>s+c*shell.residual[j][i],0);
   });
  };
  const current=evaluate(yaw,pitch);
  return current.map((v,i)=>v-gridValue(FACE_SOLVE.poses,yaw,pitch,pose=>evaluate(pose.yaw,pose.pitch)[i]));
 });
 const u=clamp((point[0]-310)/384,0,1);
 return [fade*((1-u)*-values[0]+u*values[2]),-fade*values[1]];
}
