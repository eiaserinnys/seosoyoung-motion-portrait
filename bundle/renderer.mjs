import {deform3D,opacity,bindRig} from './rig.mjs';
import {headWorld,eyeWorld,eyeCenter,eyeRootWorld,makeSphereMesh,sphereVertex,projectWorld,HEAD_CENTER,HEAD_RADIUS,NECK_PIVOT,EYE_BALL_R,correction} from './geometry.mjs';
export class Renderer {
  constructor(canvas,overlay,rig,images) {
    bindRig(rig);this.shell={...makeSphereMesh(rig.geometry.head.radius),id:'head-surface',surface:true,masks:[]};this.canvas=canvas;this.overlay=overlay;this.rig=rig;this.images=images;
    const gl=canvas.getContext('webgl',{alpha:true,antialias:true,preserveDrawingBuffer:true});
    if(!gl)throw new Error('WebGL을 사용할 수 없습니다');this.gl=gl;
    const shader=(type,source)=>{const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;};
    this.program=gl.createProgram();
    gl.attachShader(this.program,shader(gl.VERTEX_SHADER,`attribute vec3 a_pos;attribute vec2 a_uv;attribute float a_rawX;attribute float a_rawY;uniform vec4 u_view;varying vec2 v_uv;varying float v_rawX;varying float v_rawY;void main(){gl_Position=vec4(a_pos.xy*u_view.xy+u_view.zw,-a_pos.z/1024.,1.);v_uv=a_uv;v_rawX=a_rawX;v_rawY=a_rawY;}`));
    gl.attachShader(this.program,shader(gl.FRAGMENT_SHADER,`precision mediump float;uniform sampler2D u_tex;uniform sampler2D u_mask;uniform bool u_neck;uniform bool u_solid;uniform bool u_clip;uniform bool u_closed;uniform bool u_tint;uniform vec2 u_eye;uniform vec2 u_res;uniform float u_opacity;uniform float u_farBack;uniform bool u_ear;varying vec2 v_uv;varying float v_rawX;varying float v_rawY;void main(){vec4 c=u_solid?vec4(.28,.35,.42,1.):texture2D(u_tex,v_uv);if(u_closed&&((v_rawX<502.&&u_eye.x>=.22)||(v_rawX>=502.&&u_eye.y>=.22)))discard;if(u_neck)c.a*=mix(texture2D(u_mask,gl_FragCoord.xy/u_res).a,1.,smoothstep(550.,590.,v_rawY));if(u_clip)c.a*=texture2D(u_mask,gl_FragCoord.xy/u_res).a;if(u_ear)c.a*=1.-texture2D(u_mask,gl_FragCoord.xy/u_res).a;if(u_farBack!=0.&&(v_rawX-502.)*u_farBack>0.&&v_rawY>310.)c.a*=mix(1.,texture2D(u_mask,gl_FragCoord.xy/u_res).a,abs(u_farBack));if(c.a*u_opacity<.01)discard;if(u_tint)c.rgb=mix(c.rgb,vec3(.1,1.,.8),.7);gl_FragColor=vec4(c.rgb,c.a*u_opacity);}`));
    gl.linkProgram(this.program);if(!gl.getProgramParameter(this.program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(this.program));gl.useProgram(this.program);
    this.loc={};for(const name of ['neck','solid','view','tex','mask','clip','closed','tint','eye','res','opacity','farBack','ear'])this.loc[name]=gl.getUniformLocation(this.program,'u_'+name);
    this.buffer=gl.createBuffer();this.indices=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,this.buffer);
    for(const [name,size,offset] of [['pos',3,0],['uv',2,12],['rawX',1,20],['rawY',1,24]]){const loc=gl.getAttribLocation(this.program,'a_'+name);gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,size,gl.FLOAT,false,28,offset);}
    this.textures=images.map(im=>{const tex=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,tex);this.textureParams();gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,im);return tex;});
    this.maskTexture=gl.createTexture();this.framebuffer=gl.createFramebuffer();this.width=0;
    gl.depthFunc(gl.LEQUAL);gl.frontFace(gl.CW);gl.enable(gl.BLEND);gl.blendFuncSeparate(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA,gl.ONE,gl.ONE_MINUS_SRC_ALPHA);
    gl.uniform1i(this.loc.tex,0);gl.uniform1i(this.loc.mask,1);
  }
  textureParams(){const gl=this.gl;gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);}
  resize(){
    const dpr=Math.min(devicePixelRatio,2),r=this.canvas.getBoundingClientRect();const w=Math.round(r.width*dpr),h=Math.round(r.height*dpr);
    if(w===this.canvas.width&&h===this.canvas.height)return;
    this.canvas.width=this.overlay.width=w;this.canvas.height=this.overlay.height=h;
    const gl=this.gl;gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_2D,this.maskTexture);this.textureParams();gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,w,h,0,gl.RGBA,gl.UNSIGNED_BYTE,null);
    gl.bindFramebuffer(gl.FRAMEBUFFER,this.framebuffer);gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,this.maskTexture,0);gl.bindFramebuffer(gl.FRAMEBUFFER,null);
  }
  draw(mesh,p,alpha,{clip=false,tint=false}={}) {
    const gl=this.gl;gl.uniform1f(this.loc.farBack,mesh.id.startsWith('back-hair')?Math.sign(p.ParamAngleX)*Math.min(1,Math.abs(p.ParamAngleX)/25)**2*(3-2*Math.min(1,Math.abs(p.ParamAngleX)/25)):0);
    gl.uniform1i(this.loc.ear,mesh.id.startsWith('ear-'));
    const points=mesh.positions.map(pt=>mesh.surface?headWorld(pt,p):deform3D(mesh,pt,p));
    const data=new Float32Array(points.flatMap((pt,i)=>[...pt,...mesh.uvs[i],mesh.positions[i][0],mesh.positions[i][1]]));
    gl.bindBuffer(gl.ARRAY_BUFFER,this.buffer);gl.bufferData(gl.ARRAY_BUFFER,data,gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,this.indices);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,new Uint16Array(mesh.triangles.flat()),gl.STATIC_DRAW);
    gl.uniform1i(this.loc.neck,mesh.id==='neck');gl.uniform1f(this.loc.opacity,alpha);gl.uniform1i(this.loc.clip,clip);gl.uniform1i(this.loc.tint,tint);gl.uniform1i(this.loc.closed,mesh.id==='closed-eyes-v1');gl.drawElements(gl.TRIANGLES,mesh.triangles.length*3,gl.UNSIGNED_SHORT,0);
    return points;
  }
  render(p,state) {
    this.overlay.hidden=!(state.wire||state.grid);
    this.resize();const gl=this.gl,w=this.canvas.width,h=this.canvas.height;
    const scale=Math.min(w/1100,h/1600),ox=(w-1024*scale)/2,oy=(h-1536*scale)/2;
    this.view={scale,ox,oy};
    gl.viewport(0,0,w,h);gl.uniform4f(this.loc.view,scale*2/w,-scale*2/h,ox*2/w-1,1-oy*2/h);gl.uniform2f(this.loc.res,w,h);gl.uniform2f(this.loc.eye,p.ParamEyeLOpen,p.ParamEyeROpen);
    gl.uniform1i(this.loc.solid,!!state.surfaceOnly);
    gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.clearColor(0,0,0,0);gl.clearDepth(1);gl.depthMask(true);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
    const ctx=this.overlay.getContext('2d');ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,w,h);ctx.setTransform(scale,0,0,scale,ox,oy);ctx.lineWidth=1/scale;
    const texture=index=>{gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,this.textures[index]);};
    texture(state.before?1:0);
    const background=m=>m.deformerId==='WarpBody'||m.id==='neck'||m.id.startsWith('back-hair')||m.id.startsWith('side-cranium')||m.id.startsWith('ear-');
    const visible=m=>opacity(m,p,state)>0&&(state.mode!=='solo'||state.selectedIds.has(m.id));
    const wire=(mesh,points)=>{
      if(!state.wire)return;ctx.strokeStyle=state.selectedIds.has(mesh.id)?'#5fffd7':'#64d8ff66';ctx.beginPath();
      for(const tri of mesh.triangles){tri.forEach((n,i)=>{const q=projectWorld(points[n]);i?ctx.lineTo(...q):ctx.moveTo(...q);});ctx.closePath();}ctx.stroke();
    };
    gl.disable(gl.DEPTH_TEST);gl.disable(gl.CULL_FACE);
    if(!state.surfaceOnly&&p.ParamAngleX!==0){
      // Far back ribbons share the cheek/side-hair occlusion mask. Keep their
      // physical contraction too; clipping only removes the exposed fringe.
      gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_2D,null);
      gl.bindFramebuffer(gl.FRAMEBUFFER,this.framebuffer);gl.clear(gl.COLOR_BUFFER_BIT);
      const side=p.ParamAngleX<0?'left':'right';
      for(const id of ['face-skin','side-hair-'+side])this.draw(this.rig.meshes.find(m=>m.id===id),p,1);
      gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.bindTexture(gl.TEXTURE_2D,this.maskTexture);gl.activeTexture(gl.TEXTURE0);
    }
    if(!state.surfaceOnly)for(const mesh of this.rig.meshes.filter(background)){
      const alpha=opacity(mesh,p,state)*(state.mode==='dim'&&!state.selectedIds.has(mesh.id)?.18:1);
      if(mesh.id==='ear-left'){
        gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_2D,null);
        gl.bindFramebuffer(gl.FRAMEBUFFER,this.framebuffer);gl.clear(gl.COLOR_BUFFER_BIT);
        for(const id of ['face-skin','side-hair-left','side-hair-right'])this.draw(this.rig.meshes.find(m=>m.id===id),p,1);
        gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.bindTexture(gl.TEXTURE_2D,this.maskTexture);gl.activeTexture(gl.TEXTURE0);
      }
      if(!visible(mesh))continue;
      if(mesh.id==='neck'){
        gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_2D,null);gl.bindFramebuffer(gl.FRAMEBUFFER,this.framebuffer);gl.clear(gl.COLOR_BUFFER_BIT);
        this.draw(this.rig.meshes.find(m=>m.id==='face-skin'),p,1);
        gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.bindTexture(gl.TEXTURE_2D,this.maskTexture);gl.activeTexture(gl.TEXTURE0);
      }
      if(mesh.id.startsWith('side-cranium'))gl.enable(gl.CULL_FACE);
      wire(mesh,this.draw(mesh,p,alpha));gl.disable(gl.CULL_FACE);
    }
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
    const frontHair=m=>m.id==='hair-top'||m.id.startsWith('bangs')||m.id==='hairpin';
    const foreground=this.rig.meshes.filter(m=>!background(m)).sort((a,b)=>Number(frontHair(a))-Number(frontHair(b)));
    if(!state.surfaceOnly)for(const mesh of foreground) {
      if(!visible(mesh))continue;
      const alpha=opacity(mesh,p,state)*(state.mode==='dim'&&!state.selectedIds.has(mesh.id)?.18:1);
      if(mesh.masks.length){
        if(currentMask!==mesh.masks[0]){
          gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_2D,null);
          gl.bindFramebuffer(gl.FRAMEBUFFER,this.framebuffer);gl.disable(gl.DEPTH_TEST);gl.clear(gl.COLOR_BUFFER_BIT);
          this.draw(this.rig.meshes.find(m=>m.id===mesh.masks[0]),p,1);
          gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.bindTexture(gl.TEXTURE_2D,this.maskTexture);gl.activeTexture(gl.TEXTURE0);currentMask=mesh.masks[0];
        }
        // An eye opening reveals the embedded child sphere through the skin surface.
        gl.disable(gl.DEPTH_TEST);gl.depthMask(false);
      }else if(frontHair(mesh)){gl.disable(gl.DEPTH_TEST);gl.depthMask(false);}
      else{gl.enable(gl.DEPTH_TEST);gl.depthMask(true);}
      wire(mesh,this.draw(mesh,p,alpha,{clip:mesh.masks.length>0,tint:state.masks&&mesh.id.endsWith('sclera')}));
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
