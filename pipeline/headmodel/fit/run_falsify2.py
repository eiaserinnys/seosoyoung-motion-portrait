#!/usr/bin/env python3
"""2b camera checkpoint: render the neutral mesh before any optimization."""
import cv2
import numpy as np
import torch

from fit_core import FIT, FRAMES, load_model, define_regions, load_targets, write_json
from fit_sheets import shade, source_image, label, save_image
from run_falsify import initial_poses, input_hashes

RESULTS = FIT/'results/falsify2'


def init_checkpoint():
    neutral,tri,_ = load_model()
    _,faces = define_regions(neutral,tri)
    targets = load_targets()
    poses,gauge = initial_poses(neutral,tri,faces,targets)
    panels=[]
    for frame,target,pose in zip(FRAMES,targets,poses):
        # Keep the verified key-normalized yaw/pitch/roll exactly. The previous
        # origin/units conversion changes only translation and logscale.
        np.testing.assert_allclose(pose[:3],target['normalized_pose'][:3],atol=2e-6)
        original = source_image(target)
        gray,mask,_ = shade(neutral,tri[faces['head']],pose)
        image = original.copy()
        image[mask] = (.48*original[mask]+.52*gray[mask]).astype(np.uint8)
        label(image,f'frame {frame}: ICT NEUTRAL / NO OPTIMIZATION')
        label(image,f'yaw {pose[0]:.2f}  pitch {pose[1]:.2f}  roll {pose[2]:.2f}',(12,50))
        panels.append(image)
    save_image('init-overlay.png',np.hstack(panels))
    write_json(RESULTS/'init.json',{
        'status':'awaiting_assignee_camera_check',
        'frames':{str(frame):pose.tolist() for frame,pose in zip(FRAMES,poses)},
        'camera_gauge':gauge,'input_hashes':input_hashes(),
        'shape':'ICT neutral; beta=0, scale_xyz=[1,1,1]',
        'optimization_iterations':0,
        'planned_fit':{'rotation':'fixed to these registered yaw/pitch/roll values',
                       'free_camera':['logscale','tx','ty'],
                       'beta_abs_limit':3.,'beta_L2_sum_weight':.1,
                       'shape_sheet':'same ten views; neck triangles excluded in both rows'}
    })
    print('No optimization run; awaiting assignee camera check.',flush=True)


if __name__=='__main__':
    torch.set_num_threads(2); torch.set_num_interop_threads(1); cv2.setNumThreads(2)
    init_checkpoint()
