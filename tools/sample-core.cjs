/* Local-only measurement. Supply your own authorized Core, MOC and pose JSON.
 * Output contains model geometry: keep it outside any public repository. */
const fs=require('node:fs'),vm=require('node:vm'),path=require('node:path'),zlib=require('node:zlib'),crypto=require('node:crypto');
async function main(){
  const [coreFile,mocFile,poseFile,outFile]=process.argv.slice(2);
  if(!outFile)throw Error('usage: node sample-core.cjs CORE.js MODEL.moc3 POSES.json OUTPUT.json.gz');
  const sandbox={console,require,process,Buffer,setTimeout,clearTimeout,__dirname:path.dirname(path.resolve(coreFile))};
  vm.createContext(sandbox);vm.runInContext(fs.readFileSync(coreFile,'utf8'),sandbox);
  const core=sandbox.Live2DCubismCore;
  // The distributed Core initializes asynchronously; the documented version query is our readiness probe.
  const deadline=Date.now()+5000;
  for(;;){try{core.Version.csmGetVersion();break;}catch(e){if(Date.now()>deadline)throw e;await new Promise(r=>setTimeout(r,10));}}
  const bytes=fs.readFileSync(mocFile),moc=core.Moc.fromArrayBuffer(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.length));
  if(!moc)throw Error('Core rejected MOC');
  const model=core.Model.fromMoc(moc);if(!model)throw Error('Core rejected model');
  const p=model.parameters,d=model.drawables,poses={};
  const spec={neutral:{},...JSON.parse(fs.readFileSync(poseFile,'utf8')),neutralRepeat:{}};
  for(const [name,values] of Object.entries(spec)){
    p.values.set(p.defaultValues);
    for(const [id,value] of Object.entries(values)){
      const i=p.ids.indexOf(id);
      if(i<0||!Number.isFinite(value)||value<p.minimumValues[i]||value>p.maximumValues[i])throw Error(`invalid channel/value: ${id}`);
      p.values[i]=value;
    }
    model.update();
    poses[name]={parameters:values,vertices:d.vertexPositions.map(v=>Array.from(v)),opacities:Array.from(d.opacities)};
  }
  const original=poses.neutral.vertices.flat(),repeat=poses.neutralRepeat.vertices.flat();
  const drift=Math.max(...original.map((v,i)=>Math.abs(v-repeat[i])));
  if(!Number.isFinite(drift)||drift>1e-6)throw Error(`neutral drift: ${drift}`);
  const out={sourceSha256:crypto.createHash('sha256').update(bytes).digest('hex'),coreVersion:core.Version.csmGetVersion(),meshIds:d.ids,indices:d.indices.map(v=>Array.from(v)),poses,neutralDrift:drift};
  fs.writeFileSync(outFile,zlib.gzipSync(JSON.stringify(out)));
  console.log(JSON.stringify({meshes:d.count,poses:Object.keys(poses).length,neutralDrift:drift}));
  model.release();moc._release();
}
main().catch(e=>{console.error(e.message);process.exitCode=1;});
