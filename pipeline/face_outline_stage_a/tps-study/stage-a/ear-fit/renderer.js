async()=>{
 studio.state.auto=false;studio.state.track=false;studio.renderer.render=()=>{};
 const {Renderer}=await import('./renderer.mjs'),{EAR_TRANSFORMS}=await import('./ear-transform-data.mjs'),{deform3D}=await import('./rig.mjs');
 const c=document.createElement('canvas'),o=document.createElement('canvas');Object.assign(c.style,{width:'1100px',height:'1600px'});document.body.append(c);
 const renderer=new Renderer(c,o,studio.rig,studio.renderer.images);
 for(const k of EAR_TRANSFORMS)for(const side of ['left','right'])k[side]={t:[0,0],sx:1,s:1,theta:0};
 window.renderEar=({theta,side,yaw,pitch,targetBox,frontSource})=>{
  const key=EAR_TRANSFORMS.find(k=>k.yaw===yaw&&k.pitch===pitch);if(!key)throw Error('ear key missing');
  key[side]={t:theta.slice(0,2),sx:theta[2],s:theta[3],theta:theta[4]};
  const mesh=studio.rig.meshes.find(m=>m.id==='ear-'+side),p={...studio.p,ParamAngleX:yaw,ParamAngleY:pitch,ParamAngleZ:0,CorrTurn:1,ParamBodyAngleX:0,ParamBreath:0};
  const state={...studio.state,auto:false,track:false,hidden:new Set(studio.rig.meshes.filter(m=>m.id!=='ear-'+side).map(m=>m.id)),mode:'normal',wire:false,grid:false,before:false,surfaceOnly:false,identify:null};
  renderer.render(p,state);const gl=renderer.gl;gl.finish();const world=mesh.positions.map(q=>deform3D(mesh,q,p));
  const x0=frontSource?0:Math.max(0,Math.floor(Math.min(targetBox[0],...world.map(q=>q[0])))-2),y0=frontSource?0:Math.max(0,Math.floor(Math.min(targetBox[1],...world.map(q=>q[1])))-2);
  const x1=frontSource?1024:Math.min(1024,Math.ceil(Math.max(targetBox[2],...world.map(q=>q[0])))+2),y1=frontSource?1024:Math.min(1024,Math.ceil(Math.max(targetBox[3],...world.map(q=>q[1])))+2);
  const w=x1-x0,h=y1-y0,rgba=new Uint8Array(w*h*4),alpha=new Uint8Array(w*h);
  gl.readPixels(38+x0,renderer.canvas.height-32-y1,w,h,gl.RGBA,gl.UNSIGNED_BYTE,rgba);if(gl.getError())throw Error('ear GL read error');
  for(let y=0;y<h;y++)for(let x=0;x<w;x++)alpha[y*w+x]=rgba[((h-1-y)*w+x)*4+3];
  let text='';for(let i=0;i<alpha.length;i+=16384)text+=String.fromCharCode(...alpha.subarray(i,i+16384));
  return {box:[x0,y0,x1,y1],bits:btoa(text)};
 };
}
