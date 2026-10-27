import {craniumFacing,craniumSeam,referenceHairSample} from './hair-surface.mjs';
import {SKIN_FILL} from './skin-fill-data.mjs';
import {FRAME_CACHE,frameValue} from './pose-grid.mjs';
import {profileWeights,smoothstep} from './face-profile.mjs';
import {skinFillMesh,skinFillPositions,skinFillClipMesh,skinFillClipPositions} from './skin-fill.mjs';
import {deform3D,opacity,bindRig} from './rig.mjs';
import {headWorld,eyeWorld,eyeCenter,eyeRootWorld,makeSphereMesh,sphereVertex,projectWorld,HEAD_CENTER,HEAD_RADIUS,NECK_PIVOT,EYE_BALL_R,correction} from './geometry.mjs';
import {getHeadBase,headBaseVertices} from './head-base.mjs';

function faceBoundaryWeights(mesh){
  const edges=new Map();
  for(const tri of mesh.triangles)for(const [a,b] of [[tri[0],tri[1]],[tri[1],tri[2]],[tri[2],tri[0]]]){
    const key=[Math.min(a,b),Math.max(a,b)].join(','),edge=edges.get(key);
    if(edge)edge.count++;else edges.set(key,{a,b,count:1});
  }
  const boundary=[...edges.values()].filter(edge=>edge.count===1);
  return mesh.positions.map(([x,y])=>{
    let distance=Infinity;
    for(const {a,b} of boundary){
      const p=mesh.positions[a],q=mesh.positions[b],dx=q[0]-p[0],dy=q[1]-p[1];
      const t=Math.max(0,Math.min(1,((x-p[0])*dx+(y-p[1])*dy)/(dx*dx+dy*dy)));
      distance=Math.min(distance,Math.hypot(x-p[0]-t*dx,y-p[1]-t*dy));
    }
    return Math.max(0,Math.min(1,1-distance/12));
  });
}

export class Renderer {
  constructor(canvas,overlay,rig,images) {
    bindRig(rig);this.shell={...makeSphereMesh(rig.geometry.head.radius),id:'head-surface',surface:true,masks:[]};this.canvas=canvas;this.overlay=overlay;this.rig=rig;this.images=images;this.headBase=getHeadBase();
    this.faceRim=faceBoundaryWeights(rig.meshes.find(m=>m.id==='face-skin'));
    const gl=canvas.getContext('webgl',{alpha:true,antialias:true,preserveDrawingBuffer:true});
    if(!gl)throw new Error('WebGL을 사용할 수 없습니다');this.gl=gl;
    const shader=(type,source)=>{const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;};
    this.program=gl.createProgram();
    gl.attachShader(this.program,shader(gl.VERTEX_SHADER,`attribute vec3 a_pos;attribute vec2 a_uv;attribute float a_rawX;attribute float a_rawY;attribute float a_facing;attribute float a_seam;attribute vec2 a_refUV;attribute float a_refMix;uniform vec4 u_view;varying vec2 v_uv;varying float v_rawX;varying float v_rawY;varying float v_facing;varying float v_seam;varying vec2 v_refUV;varying float v_refMix;void main(){gl_Position=vec4(a_pos.xy*u_view.xy+u_view.zw,-a_pos.z/1024.,1.);v_uv=a_uv;v_rawX=a_rawX;v_rawY=a_rawY;v_facing=a_facing;v_seam=a_seam;v_refUV=a_refUV;v_refMix=a_refMix;}`));
    gl.attachShader(this.program,shader(gl.FRAGMENT_SHADER,`precision highp float;uniform sampler2D u_tex;uniform sampler2D u_faceMask;uniform bool u_faceClip;uniform sampler2D u_rim;uniform bool u_repairRim;uniform bool u_upperClip;uniform float u_earHeight;uniform sampler2D u_profileTex;uniform vec3 u_profileMix;uniform vec2 u_lidAlpha;uniform sampler2D u_mask;uniform bool u_neck;uniform bool u_solid;uniform bool u_headBase;uniform vec4 u_baseColor;uniform bool u_clip;uniform bool u_rimFade;uniform float u_rimStep;uniform bool u_closed;uniform bool u_tint;uniform vec2 u_eye;uniform vec2 u_res;uniform float u_opacity;uniform float u_farBack;uniform float u_backStart;uniform float u_identify;uniform vec4 u_refRect;varying vec2 v_uv;varying float v_rawX;varying float v_rawY;varying float v_facing;varying float v_seam;varying vec2 v_refUV;varying float v_refMix;void main(){if(v_facing<=0.000001)discard;vec4 c=u_headBase?u_baseColor:(u_solid?vec4(.28,.35,.42,1.):texture2D(u_tex,v_uv));c.a*=v_seam;if(!u_solid&&u_profileMix.x>0.){vec2 uv=vec2(v_rawX/2048.,v_rawY/2048.+u_profileMix.z);vec3 c67=texture2D(u_profileTex,uv).rgb;vec3 c90=texture2D(u_profileTex,uv+vec2(.5,0.)).rgb;c.rgb=mix(c.rgb,mix(c67,c90,u_profileMix.y),u_profileMix.x);}if(!u_solid&&v_refMix>0.&&all(greaterThanEqual(v_refUV,u_refRect.xy))&&all(lessThanEqual(v_refUV,u_refRect.zw)))c.rgb=mix(c.rgb,texture2D(u_tex,v_refUV).rgb,v_refMix);if(u_repairRim){vec4 rim=texture2D(u_rim,vec2(v_rawX,v_rawY)/1024.);c.rgb=mix(c.rgb,rim.rgb,rim.a);}if(u_closed)c.a*=v_rawX<502.?u_lidAlpha.x:u_lidAlpha.y;if(u_closed&&((v_rawX<502.&&u_eye.x>=.22)||(v_rawX>=502.&&u_eye.y>=.22)))discard;if(u_neck)c.a*=mix(texture2D(u_mask,gl_FragCoord.xy/u_res).a,1.,smoothstep(550.,590.,v_rawY));if(u_clip)c.a*=texture2D(u_mask,gl_FragCoord.xy/u_res).a;if(u_rimFade){vec2 q=gl_FragCoord.xy;float m=min(min(texture2D(u_mask,(q+vec2(u_rimStep,0.))/u_res).a,texture2D(u_mask,(q-vec2(u_rimStep,0.))/u_res).a),min(texture2D(u_mask,(q+vec2(0.,u_rimStep))/u_res).a,texture2D(u_mask,(q-vec2(0.,u_rimStep))/u_res).a));c.a*=1.-v_refMix*m;}if(u_farBack!=0.&&(v_rawX-502.)*u_farBack>0.&&v_rawY>u_backStart)c.a*=mix(1.,texture2D(u_mask,gl_FragCoord.xy/u_res).a,abs(u_farBack));if(u_upperClip&&v_rawY<u_earHeight&&texture2D(u_faceMask,gl_FragCoord.xy/u_res).a<.99)discard;if(u_faceClip&&texture2D(u_faceMask,gl_FragCoord.xy/u_res).a<.999)discard;if(c.a*u_opacity<.01)discard;if(u_identify>0.)c.rgb=vec3(u_identify>1.5?1.:0.);if(u_tint)c.rgb=mix(c.rgb,vec3(.1,1.,.8),.7);gl_FragColor=vec4(c.rgb,c.a*u_opacity);}`));
    gl.linkProgram(this.program);if(!gl.getProgramParameter(this.program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(this.program));gl.useProgram(this.program);
    this.loc={};for(const name of ['neck','solid','headBase','baseColor','view','tex','mask','clip','rimFade','rimStep','closed','tint','eye','res','opacity','farBack','backStart','identify','refRect','profileTex','profileMix'])this.loc[name]=gl.getUniformLocation(this.program,'u_'+name);
    this.buffer=gl.createBuffer();this.indices=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,this.buffer);
    for(const [name,size,offset] of [['pos',3,0],['uv',2,12],['rawX',1,20],['rawY',1,24],['facing',1,28],['seam',1,32],['refUV',2,36],['refMix',1,44]]){const loc=gl.getAttribLocation(this.program,'a_'+name);gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,size,gl.FLOAT,false,48,offset);}
    this.textures=images.map(im=>{const tex=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,tex);this.textureParams();gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,im);return tex;});
    this.maskTexture=gl.createTexture();this.framebuffer=gl.createFramebuffer();this.width=0;
    this.faceMaskTexture=gl.createTexture();this.faceFramebuffer=gl.createFramebuffer();
    this.loc.faceMask=gl.getUniformLocation(this.program,'u_faceMask');this.loc.faceClip=gl.getUniformLocation(this.program,'u_faceClip');this.loc.upperClip=gl.getUniformLocation(this.program,'u_upperClip');this.loc.earHeight=gl.getUniformLocation(this.program,'u_earHeight');this.loc.rim=gl.getUniformLocation(this.program,'u_rim');this.loc.repairRim=gl.getUniformLocation(this.program,'u_repairRim');
    gl.depthFunc(gl.LEQUAL);gl.frontFace(gl.CW);gl.enable(gl.BLEND);gl.blendFuncSeparate(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA,gl.ONE,gl.ONE_MINUS_SRC_ALPHA);
    this.loc.lidAlpha=gl.getUniformLocation(this.program,'u_lidAlpha');
    gl.uniform1i(this.loc.tex,0);gl.uniform1i(this.loc.mask,1);gl.uniform1i(this.loc.profileTex,2);gl.uniform1i(this.loc.faceMask,3);gl.uniform1i(this.loc.rim,4);
  }
  textureParams(){const gl=this.gl;gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);}
  resize(){
    const dpr=Math.min(devicePixelRatio,2),r=this.canvas.getBoundingClientRect();const w=Math.round(r.width*dpr),h=Math.round(r.height*dpr);
    if(w===this.canvas.width&&h===this.canvas.height)return;
    this.canvas.width=this.overlay.width=w;this.canvas.height=this.overlay.height=h;
    const gl=this.gl;gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_2D,this.maskTexture);this.textureParams();gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,w,h,0,gl.RGBA,gl.UNSIGNED_BYTE,null);
    gl.bindFramebuffer(gl.FRAMEBUFFER,this.framebuffer);gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,this.maskTexture,0);gl.bindFramebuffer(gl.FRAMEBUFFER,null);
    gl.activeTexture(gl.TEXTURE3);gl.bindTexture(gl.TEXTURE_2D,this.faceMaskTexture);this.textureParams();gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,w,h,0,gl.RGBA,gl.UNSIGNED_BYTE,null);
    gl.bindFramebuffer(gl.FRAMEBUFFER,this.faceFramebuffer);gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,this.faceMaskTexture,0);gl.bindFramebuffer(gl.FRAMEBUFFER,null);
  }
  draw(mesh,p,alpha,{clip=false,tint=false,faceClip=false}={}) {
    this.gl.uniform1i(this.loc.faceClip,faceClip);this.gl.uniform1i(this.loc.upperClip,!!this.skinClip&&mesh.id==='face-skin');this.gl.uniform1f(this.loc.earHeight,SKIN_FILL.earHeight);this.gl.uniform1i(this.loc.repairRim,!!this.repairRim&&mesh.id==='face-skin');
    const weights=profileWeights(p);
    this.gl.uniform3f(this.loc.profileMix,mesh.id==='face-skin'?weights.skin:0,weights.profile,weights.row);
    const gl=this.gl;gl.uniform1f(this.loc.backStart,mesh.id==='back-hair-center'?407:310);gl.uniform1f(this.loc.identify,this.identify?((Array.isArray(this.identify)?this.identify.includes(mesh.id):mesh.id===this.identify)?2:1):0);gl.uniform1f(this.loc.farBack,mesh.id.startsWith('back-hair')?Math.sign(p.ParamAngleX)*Math.min(1,correction(p,'CorrHairOcclusion')/.8)*Math.min(1,Math.abs(p.ParamAngleX)/25)**2*(3-2*Math.min(1,Math.abs(p.ParamAngleX)/25)):0);
    const points=frameValue(p,'pos:'+mesh.id,()=>mesh.id==='exposed-skin-clip'?skinFillClipPositions(p,this.rig.meshes.find(m=>m.id==='face-skin')):mesh.id==='exposed-skin'?skinFillPositions(p,this.rig.meshes.find(m=>m.id==='face-skin')):mesh.positions.map(pt=>mesh.surface?headWorld(pt,p):deform3D(mesh,pt,p)));
    const ref=this.rig.meshes.find(m=>m.id==='side-cranium-'+(p.ParamAngleX<0?'right':'left'));
    const [ax,ay,bw,bh]=ref.atlas,size=this.rig.atlasSize;
    gl.uniform4f(this.loc.refRect,(ax+.5)/size,(ay+.5)/size,(ax+bw-.5)/size,(ay+bh-.5)/size);
    const reference=xy=>{
      if(!['hair-top','back-hair-center','side-hair-left','side-hair-right'].includes(mesh.id))return [0,0,0];
      const sample=referenceHairSample(xy,p);
      if(!sample.weight)return [0,0,0];
      const [x,y]=sample.point,[bx,by]=ref.bbox,[ax,ay]=ref.atlas;
      return [(ax+x-bx)/this.rig.atlasSize,(ay+y-by)/this.rig.atlasSize,sample.weight];
    };
    const data=new Float32Array(points.flatMap((pt,i)=>[...pt,...mesh.uvs[i],mesh.positions[i][0],mesh.positions[i][1],mesh.id.startsWith('side-cranium')?craniumFacing(mesh.positions[i],p):1,mesh.id.startsWith('side-cranium')?craniumSeam(mesh.positions[i]):1,...(mesh.id==='face-skin'?[0,0,this.faceRim[i]]:reference(mesh.positions[i]))]));
    gl.bindBuffer(gl.ARRAY_BUFFER,this.buffer);gl.bufferData(gl.ARRAY_BUFFER,data,gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,this.indices);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,new Uint16Array(mesh.triangles.flat()),gl.STATIC_DRAW);
    gl.uniform1i(this.loc.neck,mesh.id==='neck');gl.uniform1f(this.loc.opacity,alpha);gl.uniform1i(this.loc.clip,clip);gl.uniform1i(this.loc.tint,tint);gl.uniform1i(this.loc.closed,mesh.id==='closed-eyes-v1');gl.drawElements(gl.TRIANGLES,mesh.triangles.length*3,gl.UNSIGNED_SHORT,0);
    return points;
  }
  drawHeadBase(p){
    const gl=this.gl,points=headBaseVertices(p).inset,n=this.headBase.vertexCount,data=new Float32Array(n*12);
    for(let i=0;i<n;i++){data.set(points.subarray(i*3,i*3+3),i*12);data[i*12+7]=1;data[i*12+8]=1;}
    gl.bindBuffer(gl.ARRAY_BUFFER,this.buffer);gl.bufferData(gl.ARRAY_BUFFER,data,gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,this.indices);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,new Uint16Array(this.headBase.triangles),gl.STATIC_DRAW);
    gl.uniform1i(this.loc.headBase,1);gl.uniform4f(this.loc.baseColor,...this.headBase.color,1);
    for(const name of ['faceClip','upperClip','repairRim','neck','clip','closed','tint'])gl.uniform1i(this.loc[name],0);
    gl.uniform3f(this.loc.profileMix,0,0,0);gl.uniform1f(this.loc.farBack,0);gl.uniform1f(this.loc.opacity,1);
    gl.uniform1f(this.loc.identify,this.identify?((Array.isArray(this.identify)?this.identify.includes('head-base'):this.identify==='head-base')?2:1):0);
    gl.disable(gl.DEPTH_TEST);gl.depthMask(false);gl.frontFace(gl.CCW);gl.enable(gl.CULL_FACE);
    const count=this.headBase.triangles.length,first=Math.floor(count/6)*3;
    gl.drawElements(gl.TRIANGLES,first,gl.UNSIGNED_SHORT,0);gl.drawElements(gl.TRIANGLES,count-first,gl.UNSIGNED_SHORT,first*2);
    gl.frontFace(gl.CW);gl.disable(gl.CULL_FACE);gl.depthMask(true);gl.uniform1i(this.loc.headBase,0);
    return points;
  }
  render(parameters,state) {
    const p={...parameters,[FRAME_CACHE]:new Map()};
    this.identify=state.identify;this.skinClip=false;this.repairRim=!state.before&&!state.surfaceOnly;
    this.overlay.hidden=!(state.wire||state.grid);
    this.resize();const gl=this.gl,w=this.canvas.width,h=this.canvas.height;
    const scale=Math.min(w/1100,h/1600),ox=(w-1024*scale)/2,oy=(h-1536*scale)/2;
    this.view={scale,ox,oy};
    gl.viewport(0,0,w,h);gl.uniform4f(this.loc.view,scale*2/w,-scale*2/h,ox*2/w-1,1-oy*2/h);gl.uniform2f(this.loc.res,w,h);gl.uniform2f(this.loc.eye,p.ParamEyeLOpen,p.ParamEyeROpen);
    gl.uniform1i(this.loc.solid,!!state.surfaceOnly);gl.uniform1i(this.loc.rimFade,0);
    const yaw=p.ParamAngleX*correction(p,'CorrTurn'),farLid=1-smoothstep(50,66,Math.abs(yaw));
    gl.uniform2f(this.loc.lidAlpha,yaw<0?farLid:1,yaw>0?farLid:1);
    gl.activeTexture(gl.TEXTURE4);gl.bindTexture(gl.TEXTURE_2D,this.textures[5]);
    gl.activeTexture(gl.TEXTURE2);gl.bindTexture(gl.TEXTURE_2D,this.textures[4]);gl.activeTexture(gl.TEXTURE0);
    gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.clearColor(0,0,0,0);gl.clearDepth(1);gl.depthMask(true);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
    const ctx=this.overlay.getContext('2d');ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,w,h);ctx.setTransform(scale,0,0,scale,ox,oy);ctx.lineWidth=1/scale;
    const texture=index=>{gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,this.textures[index]);};
    texture(state.before?1:0);
    const background=m=>m.deformerId==='WarpBody'||m.id==='neck'||m.id.startsWith('back-hair')||m.id.startsWith('side-cranium')||m.id.startsWith('ear-');
    const visible=m=>opacity(m,p,state)>0&&(!m.id.startsWith('side-cranium')||m.positions.some(q=>craniumFacing(q,p)>1e-6))&&(state.mode!=='solo'||state.selectedIds.has(m.id));
    const wire=(mesh,points)=>{
      if(!state.wire)return;ctx.strokeStyle=state.selectedIds.has(mesh.id)?'#5fffd7':'#64d8ff66';ctx.beginPath();
      for(const tri of mesh.triangles){tri.forEach((n,i)=>{const q=projectWorld(points[n]);i?ctx.lineTo(...q):ctx.moveTo(...q);});ctx.closePath();}ctx.stroke();
    };
    let filled=false;
    const drawFill=()=>{
      if(filled)return;filled=true;
      if(state.surfaceOnly||state.before||state.hidden.has('face-skin')||(state.mode==='solo'&&!state.selectedIds.has('face-skin')))return;
      const alpha=(1-smoothstep(45,50,Math.abs(yaw)))*(state.mode==='dim'&&!state.selectedIds.has('face-skin')?.18:1);
      if(!alpha)return;
      gl.disable(gl.DEPTH_TEST);gl.disable(gl.CULL_FACE);gl.depthMask(false);
      texture(6);const fill=skinFillMesh();wire(fill,this.draw(fill,p,alpha));
      texture(0);gl.depthMask(true);
    };
    gl.disable(gl.DEPTH_TEST);gl.disable(gl.CULL_FACE);
    const profile=Math.abs(yaw)>45;
    if(!profile&&!state.surfaceOnly&&!state.before){
      gl.activeTexture(gl.TEXTURE3);gl.bindTexture(gl.TEXTURE_2D,null);
      gl.bindFramebuffer(gl.FRAMEBUFFER,this.faceFramebuffer);gl.clear(gl.COLOR_BUFFER_BIT);
      gl.uniform1i(this.loc.solid,1);gl.depthMask(false);texture(6);this.draw(skinFillClipMesh(),p,1);
      gl.uniform1i(this.loc.solid,0);gl.depthMask(true);
      gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.activeTexture(gl.TEXTURE3);gl.bindTexture(gl.TEXTURE_2D,this.faceMaskTexture);
      texture(0);this.skinClip=true;
    }
    if(profile&&!state.surfaceOnly){
      // A separate immutable face mask preserves the existing child-eye mask.
      // Sampling an attachment's own framebuffer would be a WebGL feedback loop.
      gl.activeTexture(gl.TEXTURE3);gl.bindTexture(gl.TEXTURE_2D,null);
      gl.bindFramebuffer(gl.FRAMEBUFFER,this.faceFramebuffer);gl.clear(gl.COLOR_BUFFER_BIT);
      gl.enable(gl.CULL_FACE);this.draw(this.rig.meshes.find(m=>m.id==='face-skin'),p,1);gl.disable(gl.CULL_FACE);
      gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.bindTexture(gl.TEXTURE_2D,this.faceMaskTexture);gl.activeTexture(gl.TEXTURE0);
    }
    if(!state.surfaceOnly&&p.ParamAngleX!==0){
      // Far back ribbons share the cheek/side-hair occlusion mask. Keep their
      // physical contraction too; clipping only removes the exposed fringe.
      gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_2D,null);
      gl.bindFramebuffer(gl.FRAMEBUFFER,this.framebuffer);gl.clear(gl.COLOR_BUFFER_BIT);
      const side=p.ParamAngleX<0?'left':'right';
      for(const id of ['face-skin','side-hair-'+side]){gl.enable(gl.CULL_FACE);this.draw(this.rig.meshes.find(m=>m.id===id),p,1);gl.disable(gl.CULL_FACE);}
      gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.bindTexture(gl.TEXTURE_2D,this.maskTexture);gl.activeTexture(gl.TEXTURE0);
    }
    const headVisible=this.headBase&&correction(p,'CorrHeadBase')===1&&!state.hidden.has('head-base')&&!state.surfaceOnly;
    if(headVisible){
      const points=this.drawHeadBase(p);
      if(state.wire){ctx.strokeStyle=state.selectedIds.has('head-base')?'#5fffd7':'#64d8ff66';ctx.beginPath();
        const tri=this.headBase.triangles;for(let i=0;i<tri.length;i+=3){for(let j=0;j<3;j++){const k=tri[i+j]*3;j?ctx.lineTo(points[k],points[k+1]):ctx.moveTo(points[k],points[k+1]);}ctx.closePath();}ctx.stroke();}
    }
    if(!state.surfaceOnly)for(const mesh of this.rig.meshes.filter(background).sort((a,b)=>Number(!a.id.startsWith('side-cranium'))-Number(!b.id.startsWith('side-cranium')))){
      const alpha=opacity(mesh,p,state)*(state.mode==='dim'&&!state.selectedIds.has(mesh.id)?.18:1);
      if(!visible(mesh))continue;
      if(mesh.id==='neck'){
        gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_2D,null);gl.bindFramebuffer(gl.FRAMEBUFFER,this.framebuffer);gl.clear(gl.COLOR_BUFFER_BIT);
        gl.enable(gl.CULL_FACE);this.draw(this.rig.meshes.find(m=>m.id==='face-skin'),p,1);gl.disable(gl.CULL_FACE);
        if(headVisible)this.drawHeadBase(p);
        gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.bindTexture(gl.TEXTURE_2D,this.maskTexture);gl.activeTexture(gl.TEXTURE0);
      }
      if(mesh.id.startsWith('ear-')){gl.enable(gl.DEPTH_TEST);gl.depthMask(true);}
      if(mesh.id.startsWith('back-hair'))gl.enable(gl.CULL_FACE);
      wire(mesh,this.draw(mesh,p,alpha));gl.disable(gl.CULL_FACE);gl.disable(gl.DEPTH_TEST);
    }
    drawFill();
    gl.enable(gl.DEPTH_TEST);gl.enable(gl.CULL_FACE);
    // The preserved illustration contour owns the corrected silhouette. The full
    // sphere is an inspection/reference surface, not a second opaque hairstyle.
    const shellAlpha=state.surfaceOnly?1:1-correction(p,'CorrJaw');
    if((state.mode!=='solo'||state.surfaceOnly)&&shellAlpha>0){gl.depthMask(!!state.surfaceOnly);texture(state.before?3:2);this.draw(this.shell,p,shellAlpha*(state.mode==='dim'?.18:1));texture(state.before?1:0);gl.depthMask(true);}
    // ArtMesh layers are two-sided illustration ribbons, unlike the closed backing sphere.
    gl.disable(gl.CULL_FACE);
    let currentMask=null;
    // The face crop extends under the crown. Its unmodified r9 depth must not
    // punch through the shallower north pole of the front hair artwork.
    const hairOrder={'side-hair-left':1,'side-hair-right':2,'hair-top':3,'bangs-left':4,'bangs-right':5,'hairpin':6};
    const frontHair=m=>hairOrder[m.id]>0;
    const foreground=this.rig.meshes.filter(m=>!background(m)).sort((a,b)=>(hairOrder[a.id]||0)-(hairOrder[b.id]||0));
    if(!state.surfaceOnly)for(const mesh of foreground) {
      if(!visible(mesh))continue;
      const alpha=opacity(mesh,p,state)*(state.mode==='dim'&&!state.selectedIds.has(mesh.id)?.18:1);
      if(mesh.id==='face-skin'&&headVisible){
        gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_2D,null);gl.bindFramebuffer(gl.FRAMEBUFFER,this.framebuffer);gl.clear(gl.COLOR_BUFFER_BIT);
        this.drawHeadBase(p);
        gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.bindTexture(gl.TEXTURE_2D,this.maskTexture);gl.activeTexture(gl.TEXTURE0);
        currentMask=null;gl.uniform1f(this.loc.rimStep,6*this.view.scale);gl.uniform1i(this.loc.rimFade,1);
      }
      const faceClip=profile&&(mesh.id.startsWith('eye-')||mesh.id==='closed-eyes-v1'||mesh.id==='nose-detail'||mesh.id.startsWith('mouth-'));
      const maskId=mesh.masks[0]||(faceClip?'face-skin':null);
      if(maskId){
        if(currentMask!==maskId){
          gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_2D,null);
          gl.bindFramebuffer(gl.FRAMEBUFFER,this.framebuffer);gl.disable(gl.DEPTH_TEST);gl.clear(gl.COLOR_BUFFER_BIT);
          this.draw(this.rig.meshes.find(m=>m.id===maskId),p,1);
          gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.bindTexture(gl.TEXTURE_2D,this.maskTexture);gl.activeTexture(gl.TEXTURE0);currentMask=maskId;
        }
        // An eye opening reveals the embedded child sphere through the skin surface.
        gl.disable(gl.DEPTH_TEST);gl.depthMask(false);
      // Measured feature anchors define motion, not a second skin surface.
      // Paint facial details over the preserved face; hair still draws later.
      }else if(mesh.id.startsWith('side-hair')){
        const farSideHair=(mesh.id==='side-hair-left'&&p.ParamAngleX<0)||(mesh.id==='side-hair-right'&&p.ParamAngleX>0);
        if(farSideHair)gl.enable(gl.DEPTH_TEST);else gl.disable(gl.DEPTH_TEST);
        gl.depthMask(false);
      }
      else if(frontHair(mesh)||mesh.id.startsWith('eye-')||mesh.id==='closed-eyes-v1'||mesh.id==='nose-detail'||mesh.id.startsWith('mouth-')){gl.disable(gl.DEPTH_TEST);gl.depthMask(false);}
      else{gl.enable(gl.DEPTH_TEST);gl.depthMask(true);}
      if(frontHair(mesh)||mesh.id==='face-skin')gl.enable(gl.CULL_FACE);else gl.disable(gl.CULL_FACE);
      wire(mesh,this.draw(mesh,p,alpha,{clip:!!maskId,faceClip,tint:state.masks&&mesh.id.endsWith('sclera')}));
      if(mesh.id==='face-skin')gl.uniform1i(this.loc.rimFade,0);
    }
    gl.depthMask(true);gl.disable(gl.CULL_FACE);gl.disable(gl.DEPTH_TEST);
    if(state.grid)this.drawGlobe(ctx,p,scale);
    ctx.setTransform(1,0,0,1,0,0);
  }
  drawGlobe(ctx,p,scale) {
    const curve=(points,color,back=false)=>{
      ctx.strokeStyle=color;ctx.setLineDash(back?[3/scale,4/scale]:[]);ctx.beginPath();let pen=false;
      for(const pt of points){
        if((pt[2]<0)!==back){pen=false;continue;}
        const q=projectWorld(pt);pen?ctx.lineTo(...q):ctx.moveTo(...q);pen=true;
      }
      ctx.stroke();ctx.setLineDash([]);
    };
    const steps=160;
    const drawSphere=(radius,world,color)=>{
      for(const back of [true,false]){
        for(let lat=-2;lat<=2;lat++)curve(Array.from({length:steps+1},(_,i)=>world(sphereVertex(-Math.PI+i/steps*Math.PI*2,lat*Math.PI/6,radius))),back?'#ffce6530':color,back);
        for(let lon=0;lon<7;lon++)curve(Array.from({length:steps+1},(_,i)=>{
          const t=i/steps*2*Math.PI;return world([radius*Math.cos(t)*Math.sin(lon*Math.PI/7),radius*Math.sin(t),radius*Math.cos(t)*Math.cos(lon*Math.PI/7)]);
        }),back?'#5fffd730':'#5fffd7aa',back);
      }
    };
    drawSphere(HEAD_RADIUS,q=>headWorld(q,p),'#ffce65cc');
    // A camera-facing great circle is the silhouette of a full sphere in any orientation.
    const ring=(center,radius,color)=>curve(Array.from({length:steps+1},(_,i)=>{const a=i/steps*Math.PI*2;return [center[0]+radius*Math.cos(a),center[1]+radius*Math.sin(a),Math.max(0,center[2])];}),color);
    ring(headWorld([0,0,0],p),HEAD_RADIUS,'#ffce65ee');
    ctx.strokeStyle='#79dfcf';ctx.beginPath();ctx.arc(NECK_PIVOT[0],NECK_PIVOT[1],5/scale,0,Math.PI*2);ctx.moveTo(NECK_PIVOT[0]-8/scale,NECK_PIVOT[1]);ctx.lineTo(NECK_PIVOT[0]+8/scale,NECK_PIVOT[1]);ctx.stroke();
    for(const side of ['left','right']){
      const center=eyeRootWorld(side,p);
      // Classify front and back relative to the child center, not the parent's origin.
      const eyeCurve=(points,color)=>{ctx.strokeStyle=color;ctx.beginPath();points.forEach((q,i)=>i?ctx.lineTo(...projectWorld(q)):ctx.moveTo(...projectWorld(q)));ctx.stroke();};
      ring(center,EYE_BALL_R,'#ffd84dee');
      for(let j=-1;j<=1;j++){
        eyeCurve(Array.from({length:steps+1},(_,i)=>eyeWorld(sphereVertex(-Math.PI+i/steps*2*Math.PI,j*Math.PI/6,EYE_BALL_R),side,p)),'#ffd84d88');
        eyeCurve(Array.from({length:steps+1},(_,i)=>{const t=i/steps*Math.PI*2;return eyeWorld([EYE_BALL_R*Math.cos(t)*Math.sin(j*Math.PI/6),EYE_BALL_R*Math.sin(t),EYE_BALL_R*Math.cos(t)*Math.cos(j*Math.PI/6)],side,p);}),'#ffd84d88');
      }
      const iris=this.rig.meshes.find(m=>m.id==='eye-'+side+'-iris');
      const cx=side==='left'?417:587,cy=side==='left'?438:437.5;
      eyeCurve(Array.from({length:steps+1},(_,i)=>{const a=i/steps*Math.PI*2;return deform3D(iris,[cx+30*Math.cos(a),cy+30*Math.sin(a)],p);}),'#5fffd7ff');
    }
  }
}
