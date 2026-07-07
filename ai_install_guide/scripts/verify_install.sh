#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/submodules/multi-hmr:$REPO_ROOT/submodules/CameraHMR"
export PYOPENGL_PLATFORM=egl

conda run -n annyfit python -c "
import os
import sys
import pickle
import numpy as np
import torch

print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), 'count', torch.cuda.device_count())
if torch.cuda.is_available():
    print('gpu0', torch.cuda.get_device_name(0))

required = [
    'checkpoints/multiHMR_672_L_anny.pt',
    'checkpoints/sam2.1_hiera_large.pt',
    'checkpoints/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth',
    'checkpoints/hrnetv2_w18_coco_wholebody_hand_256x256_dark-a9228c9c_20210908.pth',
    'checkpoints/cam_model_cleaned.ckpt',
    'checkpoints/camerahmr_checkpoint_cleaned.ckpt',
    'checkpoints/densekp.ckpt',
    'checkpoints/model_final_f05665.pkl',
    'checkpoints/vitpose_backbone.pth',
    'checkpoints/vitpose_huge_wholebody.pth',
    'checkpoints/body_models/SMPL_NEUTRAL.pkl',
    'checkpoints/body_models/joint_regressors/J_regressor_coco_hip_smpl.npy',
    'checkpoints/body_models/joint_regressors/smplx2smpl.pkl',
    'checkpoints/body_models/joint_regressors/downsample_mat.pkl',
]

for path in required:
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise FileNotFoundError(path)
    print(path, os.path.getsize(path))

print('J', np.load('checkpoints/body_models/joint_regressors/J_regressor_coco_hip_smpl.npy').shape)
print('smplx2smpl', pickle.load(open('checkpoints/body_models/joint_regressors/smplx2smpl.pkl', 'rb'))['matrix'].shape)
print('downsample', pickle.load(open('checkpoints/body_models/joint_regressors/downsample_mat.pkl', 'rb')).shape)

import mmcv, mmpose, detectron2, unidepth, sam2, groundingdino
print('core_imports_ok')

os.chdir('annyfit')
sys.path.insert(0, os.getcwd())
from anny_wrapper import Anny
m = Anny(batch_size=1)
print('anny_wrapper_ok', tuple(m.smpl2coco.shape), tuple(m.smplx2smpl.shape), tuple(m.smpl2dense.shape))
"
