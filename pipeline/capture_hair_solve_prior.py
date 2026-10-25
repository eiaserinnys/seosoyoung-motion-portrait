"""Capture the unchanged 3D prior with the residual disabled in an isolated server."""
import hashlib
import json
import shutil
from pathlib import Path
import probe_hair_only as probe

ROOT=Path(__file__).resolve().parents[1]
original=probe.Handler
data=json.loads((ROOT/'source/viewer/hair-solve.json').read_text())


class PriorHandler(original):
    def do_GET(self):
        if self.path.split('?')[0].endswith('hair-solve-data.mjs'):
            value={**data,'poses':[{**p,'du':[[0,0]]*192} for p in data['poses']]}
            content=('export const HAIR_SOLVE='+json.dumps(value,ensure_ascii=False)+';').encode()
            self.send_response(200);self.send_header('Content-Type','text/javascript')
            self.send_header('Content-Length',str(len(content)));self.end_headers();self.wfile.write(content)
            return
        return super().do_GET()


def main():
    probe.Handler=PriorHandler
    poses=sorted(json.loads((ROOT/'source/viewer/face-feature-fit.json').read_text())['poses'],key=lambda p:(-p['pitch'],p['yaw']))
    current=probe.capture(poses,False)
    out=ROOT/'debug/hair_solve_prior';shutil.copytree(current,out,dirs_exist_ok=True)
    paths=['bundle/hair-shell.mjs','bundle/hair-surface.mjs','bundle/dome-data.mjs','source/viewer/face-feature-fit.json']
    (out/'sources.json').write_text(json.dumps({p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
    print('PASS captured nine unchanged prior poses with the residual disabled')


if __name__=='__main__':main()
