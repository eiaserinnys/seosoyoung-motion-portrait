#!/usr/bin/env python3
"""Bounded 2d experiment; stop at camera, global-shape and B3 review gates."""
import argparse
import time
import cv2
import numpy as np
import torch
from fit_core import load_targets, write_json, read_json
from fit2d_core import RESULTS, pose_affine, frame_sets, template
from fit2d_sheets import init_sheet
from run_falsify import initial_poses


def init():
    started=time.perf_counter()
    a33,allframes,six=frame_sets()
    v,tri,_,_,faces=template()
    # Shared camera gauge is anchored on front key 3 for every frame set.
    selected=[3]+six
    targets=load_targets(selected,pose_affine)
    poses,gauge=initial_poses(v,tri,faces,targets)
    checks=init_sheet(v,tri,faces,poses[1:],targets[1:])
    passed=all(c['center_error_px']<=30 for c in checks)
    write_json(RESULTS/'init.json',{'frames_a33':a33,'frames_all':allframes,
               'sample_checks':checks,'camera_gauge':gauge,'passed':passed,
               'elapsed_seconds':time.perf_counter()-started})
    print('INIT',checks,'passed',passed,flush=True)
    if not passed: raise SystemExit('STOP: initial head center exceeds 30px')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['init'])
    args=parser.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1); cv2.setNumThreads(2)
    {'init':init}[args.phase]()
