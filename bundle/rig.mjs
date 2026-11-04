const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const mix=(a,b,t)=>a+(b-a)*t;

function weights(u,v){
  u=clamp(u,-1,1);v=clamp(v,-1,1);
  return {left:Math.max(0,-u),center:1-Math.abs(u),right:Math.max(0,u),down:Math.max(0,-v),middle:1-Math.abs(v),up:Math.max(0,v)};
}

export class Rig {
  constructor(data){
    this.data=data;
    this.meshes=[...data.meshes].sort((a,b)=>a.drawOrder-b.drawOrder);
    this.inverse=new Map(Object.entries(data.keys).map(([key,value])=>[value.join(','),key]));
  }

  key(u,v){
    const x=u<0?-1:u>0?1:0,y=v<0?-1:v>0?1:0;
    return this.inverse.get(`${x},${y}`);
  }

  delta(mesh,u,v){
    const w=weights(u,v),out=mesh.positions.map(()=>[0,0]);
    for(const [x,wx] of [[-1,w.left],[0,w.center],[1,w.right]])for(const [y,wy] of [[-1,w.down],[0,w.middle],[1,w.up]]){
      const amount=wx*wy;if(amount<=0)continue;
      const key=this.inverse.get(`${x},${y}`),values=mesh.deltas[key];
      if(!values)continue;
      for(let i=0;i<out.length;i++){out[i][0]+=values[i][0]*amount;out[i][1]+=values[i][1]*amount;}
    }
    return out;
  }

  eyeMatrix(mesh,u,v){
    const w=weights(u,v),out=[[0,0,0],[0,0,0]],identity=[[1,0,0],[0,1,0]];
    for(const [x,wx] of [[-1,w.left],[0,w.center],[1,w.right]])for(const [y,wy] of [[-1,w.down],[0,w.middle],[1,w.up]]){
      const amount=wx*wy;if(amount<=0)continue;
      const key=this.inverse.get(`${x},${y}`),matrix=key==='front'?identity:mesh.transforms[key];
      if(!matrix)continue;
      for(let row=0;row<2;row++)for(let col=0;col<3;col++)out[row][col]+=matrix[row][col]*amount;
    }
    return out;
  }

  positions(mesh,p){
    let pos=mesh.positions.map(point=>point.slice());
    if(mesh.group==='eye'){
      const side=mesh.id.includes('-left')?'left':'right',open=side==='left'?p.openL:p.openR,kind=mesh.id.split('-').at(-1);
      if(!mesh.id.startsWith('closed')){
        const s=Math.min(1,(1-open)/(1-this.data.blinkSquashEnd)),curve=this.data.blinkCurves[side],ix=point=>clamp(Math.round(point[0])-curve.x0,0,curve.top.length-1),at=(name,i)=>curve[name][i];
        pos=pos.map(point=>{
          const i=ix(point),closed=at('closed',i),next=point.slice();
          if(kind==='lashes'||kind==='line'){
            const top=at(kind==='lashes'?'lashTop':'lowerTop',i),bottom=at(kind==='lashes'?'lashBot':'lowerBot',i),ct=at('closedTop',i),cb=at('closedBot',i);
            const k=clamp((cb-ct)/Math.max(bottom-top,1),0,1),target=(ct+cb)/2+(point[1]-(top+bottom)/2)*k;
            next[1]+=s*(target-point[1]);
          }else{
            next[1]=closed+(point[1]-closed)*(1-s);
            if(['iris','pupil','highlight'].includes(kind)){next[0]+=(p.gazeX||0)*this.data.gazePx;next[1]+=(p.gazeY||0)*this.data.gazePx;}
          }
          return next;
        });
      }
      const m=this.eyeMatrix(mesh,p.u,p.v);
      pos=pos.map(([x,y])=>[m[0][0]*x+m[0][1]*y+m[0][2],m[1][0]*x+m[1][1]*y+m[1][2]]);
    }else{
      const delta=this.delta(mesh,p.u,p.v);
      pos=pos.map(([x,y],i)=>[x+delta[i][0],y+delta[i][1]]);
    }
    if(p.roll&&mesh.group!=='body'){
      const [cx,cy]=this.data.neckPivot,a=p.roll*Math.PI/180,c=Math.cos(a),s=Math.sin(a);
      pos=pos.map(([x,y],i)=>{
        const weight=['neck','fill'].includes(mesh.group)?clamp((this.data.neckRamp[1]-mesh.positions[i][1])/(this.data.neckRamp[1]-this.data.neckRamp[0]),0,1):1;
        const rx=(x-cx)*c-(y-cy)*s+cx,ry=(x-cx)*s+(y-cy)*c+cy;
        return [x+(rx-x)*weight,y+(ry-y)*weight];
      });
    }
    if(mesh.group!=='body'&&(p.bodyAngleX||p.breath))pos=pos.map(([x,y])=>{
      const upper=clamp((1536-y)/960,0,1);
      return [x+(p.bodyAngleX||0)*upper,y-(p.breath||0)*4*upper];
    });
    return pos;
  }

  opacity(mesh,p){
    const id=mesh.id;
    if((this.data.hair.includes(id)||id.startsWith('still-fill-hair-'))&&!p.hair)return 0;
    if(this.data.bareOnlyParts.includes(id)&&p.hair)return 0;
    const [low,high]=this.data.blinkFade;
    if(mesh.group==='eye'&&!id.startsWith('closed')){
      const open=id.includes('-left')?p.openL:p.openR;
      return clamp((open-low)/(high-low),0,1);
    }
    if(id.startsWith('closed-eyes')){
      const open=id.endsWith('left')?p.openL:p.openR,[closedLow,closedHigh]=this.data.blinkClosedFade;
      return 1-clamp((open-closedLow)/(closedHigh-closedLow),0,1);
    }
    return 1;
  }

  order(){return this.meshes;}

  curve(name,p){
    const source=name==='silhouette'?this.data.silhouette:this.data.facePolygon,w=weights(p.u,p.v),out=source.front.map(()=>[0,0]);
    for(const [x,wx] of [[-1,w.left],[0,w.center],[1,w.right]])for(const [y,wy] of [[-1,w.down],[0,w.middle],[1,w.up]]){
      const amount=wx*wy;if(amount<=0)continue;
      const key=this.inverse.get(`${x},${y}`),curve=source[key];
      if(!curve)continue;
      for(let i=0;i<out.length;i++){out[i][0]+=curve[i][0]*amount;out[i][1]+=curve[i][1]*amount;}
    }
    return out;
  }

  offsetOutward(points,amount){
    const p=points,n=p.length,normals=p.map((_,i)=>{
      const next=p[(i+1)%n],prev=p[(i+n-1)%n],dx=next[0]-prev[0],dy=next[1]-prev[1],length=Math.max(Math.hypot(dx,dy),1e-6);
      return [dy/length,-dx/length];
    });
    const area=poly=>Math.abs(poly.reduce((sum,[x,y],i)=>{const [nx,ny]=poly[(i+1)%poly.length];return sum+x*ny-nx*y;},0)/2);
    const a=p.map(([x,y],i)=>[x+normals[i][0]*amount,y+normals[i][1]*amount]);
    const b=p.map(([x,y],i)=>[x-normals[i][0]*amount,y-normals[i][1]*amount]);
    return ((area(a)>=area(b))===(amount>=0))?a:b;
  }

}

export const defaultPose=()=>({u:0,v:0,roll:0,openL:1,openR:1,gazeX:0,gazeY:0,breath:0,bodyAngleX:0,hair:false});
