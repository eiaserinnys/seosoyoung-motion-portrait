"""Freeze main and the zero-residual r15 prior without changing checkout."""
import json
import shutil
import subprocess
from urllib.parse import unquote,urlsplit
import probe_hair_only as probe
import capture_hair_solve_prior as prior

BASE='6915b2751ef94ff80d3b86456befda5a9a958f0c'


class MainHandler(probe.Handler):
    def do_GET(self):
        path=unquote(urlsplit(self.path).path).lstrip('/') or 'index.html'
        try:data=subprocess.check_output(['git','show',BASE+':bundle/'+path],cwd=probe.ROOT,stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type',self.guess_type(path))
        self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)


def main():
    poses=json.loads((probe.ROOT/'source/viewer/face-feature-fit.json').read_text())['poses']
    original=probe.Handler
    try:
        probe.Handler=MainHandler
        current=probe.capture(poses,False)
        shutil.copytree(current,probe.ROOT/'debug/vdd_main',dirs_exist_ok=True)
    finally:probe.Handler=original
    prior.main()
    print('PASS frozen main and zero-residual prior captured')


if __name__=='__main__':main()
