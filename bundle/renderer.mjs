const WIDTH=1024,HEIGHT=1536;
const EDGE_AA_PX=1.0;

const vertexSource=`#version 300 es
in vec2 a_position;
in vec2 a_uv;
uniform vec2 u_size;
out vec2 v_uv;
void main(){
  gl_Position=vec4(a_position.x*2.0/u_size.x-1.0,1.0-a_position.y*2.0/u_size.y,0.0,1.0);
  v_uv=a_uv;
}`;
const colorSource=`#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_atlas;
uniform sampler2D u_mask;
uniform vec2 u_size;
uniform vec2 u_atlas_size;
uniform vec4 u_color;
uniform float u_opacity;
uniform int u_solid;
uniform int u_apply_mask;
out vec4 out_color;
vec4 sample_atlas(vec2 uv){
  vec2 point=uv*u_atlas_size-vec2(0.5);
  ivec2 p0=ivec2(floor(point)),p1=p0+ivec2(1);
  vec2 f=fract(point);
  ivec2 hi=ivec2(u_atlas_size)-ivec2(1);
  vec4 c00=texelFetch(u_atlas,clamp(p0,ivec2(0),hi),0);
  vec4 c10=texelFetch(u_atlas,clamp(ivec2(p1.x,p0.y),ivec2(0),hi),0);
  vec4 c01=texelFetch(u_atlas,clamp(ivec2(p0.x,p1.y),ivec2(0),hi),0);
  vec4 c11=texelFetch(u_atlas,clamp(p1,ivec2(0),hi),0);
  c00.rgb*=c00.a;c10.rgb*=c10.a;c01.rgb*=c01.a;c11.rgb*=c11.a;
  return mix(mix(c00,c10,f.x),mix(c01,c11,f.x),f.y);
}
void main(){
  vec4 color=u_solid==1?u_color:(u_solid==2?vec4(u_color.rgb*v_uv.x,v_uv.x):sample_atlas(v_uv));
  if(u_apply_mask==1)color*=texture(u_mask,gl_FragCoord.xy/u_size).r;
  color*=u_opacity;
  out_color=color;
}`;
const maskSource=`#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_atlas;
uniform int u_textured;
uniform vec2 u_atlas_size;
out vec4 out_color;
float sample_alpha(vec2 uv){
  vec2 point=uv*u_atlas_size-vec2(0.5);ivec2 p0=ivec2(floor(point)),p1=p0+ivec2(1);vec2 f=fract(point);ivec2 hi=ivec2(u_atlas_size)-ivec2(1);
  float c00=texelFetch(u_atlas,clamp(p0,ivec2(0),hi),0).a;
  float c10=texelFetch(u_atlas,clamp(ivec2(p1.x,p0.y),ivec2(0),hi),0).a;
  float c01=texelFetch(u_atlas,clamp(ivec2(p0.x,p1.y),ivec2(0),hi),0).a;
  float c11=texelFetch(u_atlas,clamp(p1,ivec2(0),hi),0).a;
  return mix(mix(c00,c10,f.x),mix(c01,c11,f.x),f.y);
}
void main(){float a=u_textured==1?sample_alpha(v_uv):1.0;out_color=vec4(a,a,a,a);}`;
const copyVertex=`#version 300 es
in vec2 a_position;
in vec2 a_uv;
out vec2 v_uv;
void main(){gl_Position=vec4(a_position,0.0,1.0);v_uv=a_uv;}`;
const copyFragment=`#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_image;
out vec4 out_color;
void main(){out_color=texture(u_image,v_uv);}`;

function shader(gl,type,source){
  const value=gl.createShader(type);gl.shaderSource(value,source);gl.compileShader(value);
  if(!gl.getShaderParameter(value,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(value)||'셰이더 생성 실패');
  return value;
}
function program(gl,vertex,fragment){
  const value=gl.createProgram();gl.attachShader(value,shader(gl,gl.VERTEX_SHADER,vertex));gl.attachShader(value,shader(gl,gl.FRAGMENT_SHADER,fragment));gl.linkProgram(value);
  if(!gl.getProgramParameter(value,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(value)||'화면 프로그램 생성 실패');
  return value;
}
function texture(gl,internal,width,height,format,type,data=null){
  const value=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,value);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D,0,internal,width,height,0,format,type,data);return value;
}
function framebuffer(gl,image){
  const value=gl.createFramebuffer();gl.bindFramebuffer(gl.FRAMEBUFFER,value);gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,image,0);
  return gl.checkFramebufferStatus(gl.FRAMEBUFFER)===gl.FRAMEBUFFER_COMPLETE?value:null;
}

function earclip(points){
  const p=points.map(([x,y])=>[x,y]),indices=p.map((_,i)=>i);
  const area=p.reduce((sum,[x,y],i)=>{const [nx,ny]=p[(i+1)%p.length];return sum+x*ny-nx*y;},0);
  if(area<0)indices.reverse();
  const cross=(a,b,c)=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
  const inside=(a,b,c,q)=>cross(a,b,q)>=0&&cross(b,c,q)>=0&&cross(c,a,q)>=0;
  const triangles=[];let guard=0;
  while(indices.length>3&&guard<10000){
    guard++;let removed=false;
    for(let k=0;k<indices.length;k++){
      const i0=indices[(k+indices.length-1)%indices.length],i1=indices[k],i2=indices[(k+1)%indices.length],a=p[i0],b=p[i1],c=p[i2];
      if(cross(a,b,c)<=1e-9)continue;
      if(indices.some(i=>![i0,i1,i2].includes(i)&&inside(a,b,c,p[i])))continue;
      triangles.push([a,b,c]);indices.splice(k,1);removed=true;break;
    }
    if(!removed)break;
  }
  if(indices.length===3)triangles.push(indices.map(i=>p[i]));
  if(indices.length>3)throw new Error('실루엣 삼각화 실패');
  return triangles;
}

function bandTriangles(points,outer){
  const triangles=[];
  for(let i=0;i<points.length;i++){
    const j=(i+1)%points.length,a=points[i],b=points[j],c=outer[i],d=outer[j];
    triangles.push([[a[0],a[1],1],[c[0],c[1],0],[d[0],d[1],0]],[[a[0],a[1],1],[d[0],d[1],0],[b[0],b[1],1]]);
  }
  return triangles;
}

export class Renderer{
  constructor(canvas,rig,atlasImage){
    this.canvas=canvas;this.rig=rig;this.gl=canvas.getContext('webgl2',{alpha:true,antialias:false,preserveDrawingBuffer:true,premultipliedAlpha:true});
    if(!this.gl)throw new Error('WebGL 2를 사용할 수 없습니다');
    this.meshes=new Map(rig.meshes.map(mesh=>[mesh.id,mesh]));
    this.glPrograms={};this.setup(atlasImage);
  }

  setup(atlasImage){
    const gl=this.gl;
    this.glPrograms.color=program(gl,vertexSource,colorSource);this.glPrograms.mask=program(gl,vertexSource,maskSource);this.glPrograms.copy=program(gl,copyVertex,copyFragment);
    this.buffers={vertices:gl.createBuffer(),indices:gl.createBuffer(),copy:gl.createBuffer(),copyIndices:gl.createBuffer()};
    gl.bindBuffer(gl.ARRAY_BUFFER,this.buffers.copy);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,0,0,1,-1,1,0,1,1,1,1,-1,1,0,1]),gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,this.buffers.copyIndices);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,new Uint16Array([0,1,2,0,2,3]),gl.STATIC_DRAW);
    gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL,gl.NONE);
    this.atlas=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,this.atlas);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,atlasImage);
    const maskImage=texture(gl,gl.R8,WIDTH,HEIGHT,gl.RED,gl.UNSIGNED_BYTE);this.maskFramebuffer=framebuffer(gl,maskImage);this.maskTexture=maskImage;
    const colorFloat=gl.getExtension('EXT_color_buffer_float')&&gl.getExtension('EXT_float_blend');
    this.floatTarget=null;
    if(colorFloat){
      const image=texture(gl,gl.RGBA32F,WIDTH,HEIGHT,gl.RGBA,gl.FLOAT);const fbo=framebuffer(gl,image);
      if(fbo)this.floatTarget={image,framebuffer:fbo};else gl.deleteTexture(image);
    }
    this.float32=!!this.floatTarget;
    gl.bindFramebuffer(gl.FRAMEBUFFER,null);this.canvas.width=WIDTH;this.canvas.height=HEIGHT;this.targetSize=[WIDTH,HEIGHT];
    gl.disable(gl.DITHER);gl.disable(gl.DEPTH_TEST);gl.disable(gl.CULL_FACE);gl.disable(gl.SCISSOR_TEST);gl.enable(gl.BLEND);gl.blendFunc(gl.ONE,gl.ONE_MINUS_SRC_ALPHA);
    this.uniforms={};
    for(const name of ['color','mask','copy']){
      const p=this.glPrograms[name];this.uniforms[name]={
        position:gl.getAttribLocation(p,'a_position'),uv:gl.getAttribLocation(p,'a_uv'),
        size:gl.getUniformLocation(p,'u_size'),atlasSize:gl.getUniformLocation(p,'u_atlas_size'),atlas:gl.getUniformLocation(p,'u_atlas'),mask:gl.getUniformLocation(p,'u_mask'),
        solid:gl.getUniformLocation(p,'u_solid'),applyMask:gl.getUniformLocation(p,'u_apply_mask'),color:gl.getUniformLocation(p,'u_color'),
        textured:gl.getUniformLocation(p,'u_textured'),image:gl.getUniformLocation(p,'u_image'),
        opacity:gl.getUniformLocation(p,'u_opacity')
      };
    }
  }

  bindGeometry(points,uvs,triangles){
    const gl=this.gl,data=new Float32Array(points.length*4);
    for(let i=0;i<points.length;i++){data[i*4]=points[i][0];data[i*4+1]=points[i][1];data[i*4+2]=uvs?.[i]?.[0]||0;data[i*4+3]=uvs?.[i]?.[1]||0;}
    gl.bindBuffer(gl.ARRAY_BUFFER,this.buffers.vertices);gl.bufferData(gl.ARRAY_BUFFER,data,gl.DYNAMIC_DRAW);
    const indices=new Uint16Array(triangles.flat());gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,this.buffers.indices);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,indices,gl.DYNAMIC_DRAW);
    return indices.length;
  }

  attributes(programInfo){
    const gl=this.gl;gl.bindBuffer(gl.ARRAY_BUFFER,this.buffers.vertices);gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,this.buffers.indices);
    gl.enableVertexAttribArray(programInfo.position);gl.vertexAttribPointer(programInfo.position,2,gl.FLOAT,false,16,0);
    gl.enableVertexAttribArray(programInfo.uv);gl.vertexAttribPointer(programInfo.uv,2,gl.FLOAT,false,16,8);
  }

  drawMesh(mesh,points,applyMask=false,opacity=1){
    const gl=this.gl,info=this.uniforms.color;this.bindGeometry(points,mesh.uvs,mesh.triangles);gl.useProgram(this.glPrograms.color);this.attributes(info);
    gl.uniform2f(info.size,WIDTH,HEIGHT);gl.uniform2f(info.atlasSize,this.rig.data.atlasSize,this.rig.data.atlasSize);gl.uniform1i(info.atlas,0);gl.uniform1i(info.mask,1);gl.uniform1i(info.solid,0);gl.uniform1i(info.applyMask,applyMask?1:0);gl.uniform1f(info.opacity,opacity);
    gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,this.atlas);gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_2D,this.maskTexture);
    gl.drawElements(gl.TRIANGLES,mesh.triangles.length*3,gl.UNSIGNED_SHORT,0);
  }

  drawTriangles(triangles,color,{maskOnly=false,vertexAlpha=false}={}){
    const vertices=[],uvs=[],indices=[];
    for(const triangle of triangles){
      const start=vertices.length;
      for(const point of triangle){vertices.push([point[0],point[1]]);uvs.push([vertexAlpha?point[2]:0,0]);}
      indices.push([start,start+1,start+2]);
    }
    const gl=this.gl,count=this.bindGeometry(vertices,uvs,indices),info=maskOnly?this.uniforms.mask:this.uniforms.color;
    gl.useProgram(maskOnly?this.glPrograms.mask:this.glPrograms.color);this.attributes(info);gl.uniform2f(info.size,WIDTH,HEIGHT);
    if(maskOnly){gl.uniform2f(info.atlasSize,this.rig.data.atlasSize,this.rig.data.atlasSize);gl.uniform1i(info.atlas,0);gl.uniform1i(info.textured,0);}
    else{gl.uniform2f(info.atlasSize,this.rig.data.atlasSize,this.rig.data.atlasSize);gl.uniform1i(info.solid,vertexAlpha?2:1);gl.uniform1i(info.applyMask,0);gl.uniform4f(info.color,color[0]/255,color[1]/255,color[2]/255,1);gl.uniform1f(info.opacity,1);}
    gl.drawElements(gl.TRIANGLES,count,gl.UNSIGNED_SHORT,0);
  }

  drawSilhouette(points,color){
    this.drawTriangles(earclip(points),color);
    const outer=this.rig.offsetOutward(points,EDGE_AA_PX);
    this.drawTriangles(bandTriangles(points,outer),color,{vertexAlpha:true});
  }

  drawMaskMesh(mesh,pose){
    const gl=this.gl,info=this.uniforms.mask,points=this.rig.positions(mesh,pose);this.bindGeometry(points,mesh.uvs,mesh.triangles);gl.useProgram(this.glPrograms.mask);this.attributes(info);
    gl.uniform2f(info.size,WIDTH,HEIGHT);gl.uniform2f(info.atlasSize,this.rig.data.atlasSize,this.rig.data.atlasSize);gl.uniform1i(info.atlas,0);gl.uniform1i(info.textured,1);gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,this.atlas);
    gl.drawElements(gl.TRIANGLES,mesh.triangles.length*3,gl.UNSIGNED_SHORT,0);
  }

  renderMask(points,margin){
    const gl=this.gl;gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_2D,null);gl.activeTexture(gl.TEXTURE0);gl.bindFramebuffer(gl.FRAMEBUFFER,this.maskFramebuffer);gl.viewport(0,0,WIDTH,HEIGHT);gl.disable(gl.BLEND);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);
    const outer=this.rig.offsetOutward(points,margin);this.drawTriangles([...earclip(points),...bandTriangles(points,outer)],null,{maskOnly:true});gl.enable(gl.BLEND);gl.blendFunc(gl.ONE,gl.ONE_MINUS_SRC_ALPHA);gl.bindFramebuffer(gl.FRAMEBUFFER,this.floatTarget?.framebuffer||null);
  }

  renderEyeMask(mesh,pose){
    const gl=this.gl;gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_2D,null);gl.activeTexture(gl.TEXTURE0);gl.bindFramebuffer(gl.FRAMEBUFFER,this.maskFramebuffer);gl.viewport(0,0,WIDTH,HEIGHT);gl.disable(gl.BLEND);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);
    this.drawMaskMesh(mesh,pose);gl.enable(gl.BLEND);gl.blendFunc(gl.ONE,gl.ONE_MINUS_SRC_ALPHA);gl.bindFramebuffer(gl.FRAMEBUFFER,this.floatTarget?.framebuffer||null);
  }

  render(pose,{hidden=new Set(),hair=false}={}){
    const gl=this.gl,p={...pose,hair};this.currentPose=p;this.currentHidden=new Set(hidden);gl.bindFramebuffer(gl.FRAMEBUFFER,this.floatTarget?.framebuffer||null);gl.viewport(0,0,WIDTH,HEIGHT);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);gl.enable(gl.BLEND);gl.blendFunc(gl.ONE,gl.ONE_MINUS_SRC_ALPHA);
    const silhouette=this.rig.curve('silhouette',p);this.renderMask(silhouette,this.rig.data.clipMarginPx+this.rig.data.maskRasterMarginPx);
    let eyeMask=null;
    for(const mesh of this.rig.order(p)){
      if(mesh.id==='skull-plate'){
        if(!hair&&!hidden.has(mesh.id))this.drawSilhouette(silhouette,this.rig.data.skinColor);
        continue;
      }
      const opacity=this.rig.opacity(mesh,p);
      if(hidden.has(mesh.id)||opacity<=0)continue;
      if(this.rig.data.clips[mesh.id]){
        const clipId=this.rig.data.clips[mesh.id];
        if(eyeMask!==clipId){this.renderEyeMask(this.meshes.get(clipId),p);eyeMask=clipId;}
        this.drawMesh(mesh,this.rig.positions(mesh,p),true,opacity);
        continue;
      }
      const clipped=this.rig.data.clipParts.includes(mesh.id);
      this.drawMesh(mesh,this.rig.positions(mesh,p),clipped,opacity);
    }
    if(this.floatTarget)this.copyFloatToCanvas();
    return this.canvas;
  }

  copyFloatToCanvas(){
    const gl=this.gl,info=this.uniforms.copy;gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.viewport(0,0,this.canvas.width,this.canvas.height);gl.disable(gl.BLEND);
    gl.useProgram(this.glPrograms.copy);gl.bindBuffer(gl.ARRAY_BUFFER,this.buffers.copy);gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,this.buffers.copyIndices);
    gl.enableVertexAttribArray(info.position);gl.vertexAttribPointer(info.position,2,gl.FLOAT,false,16,0);gl.enableVertexAttribArray(info.uv);gl.vertexAttribPointer(info.uv,2,gl.FLOAT,false,16,8);
    gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,this.floatTarget.image);gl.uniform1i(info.image,0);gl.drawElements(gl.TRIANGLES,6,gl.UNSIGNED_SHORT,0);gl.enable(gl.BLEND);gl.blendFunc(gl.ONE,gl.ONE_MINUS_SRC_ALPHA);
  }

  readPixels(){
    const gl=this.gl;gl.bindFramebuffer(gl.FRAMEBUFFER,this.floatTarget?.framebuffer||null);const data=this.float32?new Float32Array(WIDTH*HEIGHT*4):new Uint8Array(WIDTH*HEIGHT*4);
    gl.readPixels(0,0,WIDTH,HEIGHT,gl.RGBA,this.float32?gl.FLOAT:gl.UNSIGNED_BYTE,data);gl.bindFramebuffer(gl.FRAMEBUFFER,null);return {width:WIDTH,height:HEIGHT,float32:this.float32,data};
  }
}
