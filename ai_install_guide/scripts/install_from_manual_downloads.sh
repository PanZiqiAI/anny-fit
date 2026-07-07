#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DOWNLOAD_DIR="${1:-}"
if [[ -z "$DOWNLOAD_DIR" ]]; then
    if [[ -d "$HOME/下载" ]]; then
        DOWNLOAD_DIR="$HOME/下载"
    elif [[ -d "$HOME/Downloads" ]]; then
        DOWNLOAD_DIR="$HOME/Downloads"
    else
        echo "ERROR: pass the directory containing manually downloaded files." >&2
        exit 1
    fi
fi

require_file() {
    local path="$1"
    if [[ ! -s "$path" ]]; then
        echo "ERROR: missing required file: $path" >&2
        exit 1
    fi
}

download_public() {
    local path="$1"
    local url="$2"
    if [[ -s "$path" ]]; then
        echo "exists: $path"
    else
        echo "downloading: $path"
        wget -q --show-progress -O "$path" "$url"
    fi
}

mkdir -p checkpoints/body_models/joint_regressors

download_public checkpoints/multiHMR_672_L_anny.pt \
    https://download.europe.naverlabs.com/ComputerVision/MultiHMR/multiHMR_672_L_anny.pt

download_public checkpoints/sam2.1_hiera_large.pt \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

download_public checkpoints/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth \
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth

download_public checkpoints/hrnetv2_w18_coco_wholebody_hand_256x256_dark-a9228c9c_20210908.pth \
    https://download.openmmlab.com/mmpose/hand/dark/hrnetv2_w18_coco_wholebody_hand_256x256_dark-a9228c9c_20210908.pth

require_file "$DOWNLOAD_DIR/cam_model_cleaned.ckpt"
require_file "$DOWNLOAD_DIR/camerahmr_checkpoint_cleaned.ckpt"
require_file "$DOWNLOAD_DIR/densekp.ckpt"
require_file "$DOWNLOAD_DIR/model_final_f05665.pkl"
require_file "$DOWNLOAD_DIR/SMPL_NEUTRAL.pkl"
require_file "$DOWNLOAD_DIR/train-eval-utils.zip"

cp -f "$DOWNLOAD_DIR/cam_model_cleaned.ckpt" checkpoints/cam_model_cleaned.ckpt
cp -f "$DOWNLOAD_DIR/camerahmr_checkpoint_cleaned.ckpt" checkpoints/camerahmr_checkpoint_cleaned.ckpt
cp -f "$DOWNLOAD_DIR/densekp.ckpt" checkpoints/densekp.ckpt
cp -f "$DOWNLOAD_DIR/model_final_f05665.pkl" checkpoints/model_final_f05665.pkl
cp -f "$DOWNLOAD_DIR/SMPL_NEUTRAL.pkl" checkpoints/body_models/SMPL_NEUTRAL.pkl

unzip -p "$DOWNLOAD_DIR/train-eval-utils.zip" 'train-eval-utils/vitpose_backbone.pth' > checkpoints/vitpose_backbone.pth
unzip -p "$DOWNLOAD_DIR/train-eval-utils.zip" 'train-eval-utils/smplx2smpl.pkl' > checkpoints/body_models/joint_regressors/smplx2smpl.pkl
unzip -p "$DOWNLOAD_DIR/train-eval-utils.zip" 'train-eval-utils/downsample_mat.pkl' > checkpoints/body_models/joint_regressors/downsample_mat.pkl

if unzip -l "$DOWNLOAD_DIR/train-eval-utils.zip" | grep -q 'train-eval-utils/J_regressor_coco_hip_smpl.npy'; then
    unzip -p "$DOWNLOAD_DIR/train-eval-utils.zip" 'train-eval-utils/J_regressor_coco_hip_smpl.npy' > checkpoints/body_models/joint_regressors/J_regressor_coco_hip_smpl.npy
else
    echo "WARNING: train-eval-utils.zip has no J_regressor_coco_hip_smpl.npy; using J_regressor_h36m.npy as 17-joint compatible fallback."
    unzip -p "$DOWNLOAD_DIR/train-eval-utils.zip" 'train-eval-utils/J_regressor_h36m.npy' > checkpoints/body_models/joint_regressors/J_regressor_coco_hip_smpl.npy
fi

if [[ ! -s checkpoints/vitpose_huge_wholebody.pth ]]; then
    VITPOSE_SOURCE=""
    if [[ -s "$DOWNLOAD_DIR/vitpose _huge.pth" ]]; then
        VITPOSE_SOURCE="$DOWNLOAD_DIR/vitpose _huge.pth"
    elif [[ -s "$DOWNLOAD_DIR/vitpose_huge.pth" ]]; then
        VITPOSE_SOURCE="$DOWNLOAD_DIR/vitpose_huge.pth"
    fi

    if [[ -z "$VITPOSE_SOURCE" ]]; then
        echo "ERROR: missing ViTPose source checkpoint. Expected 'vitpose _huge.pth' or 'vitpose_huge.pth'." >&2
        exit 1
    fi

    PYTHON_CODE=$(cat <<'PY'
import copy
import os
import sys
import torch

src, out = sys.argv[1], sys.argv[2]
ckpt = torch.load(src, map_location="cpu")
new_ckpt = copy.deepcopy(ckpt)
state_dict = new_ckpt["state_dict"]
experts = {k: v for k, v in state_dict.items() if "mlp.experts" in k}

for key in list(state_dict):
    if "mlp.fc2" in key:
        expert_key = key.replace("fc2.", "experts.5.")
        if expert_key in experts:
            state_dict[key] = torch.cat([state_dict[key], experts[expert_key]], dim=0)

weight_names = [
    "keypoint_head.deconv_layers.0.weight",
    "keypoint_head.deconv_layers.1.weight",
    "keypoint_head.deconv_layers.1.bias",
    "keypoint_head.deconv_layers.1.running_mean",
    "keypoint_head.deconv_layers.1.running_var",
    "keypoint_head.deconv_layers.1.num_batches_tracked",
    "keypoint_head.deconv_layers.3.weight",
    "keypoint_head.deconv_layers.4.weight",
    "keypoint_head.deconv_layers.4.bias",
    "keypoint_head.deconv_layers.4.running_mean",
    "keypoint_head.deconv_layers.4.running_var",
    "keypoint_head.deconv_layers.4.num_batches_tracked",
    "keypoint_head.final_layer.weight",
    "keypoint_head.final_layer.bias",
]

for tensor_name in weight_names:
    state_dict[tensor_name] = state_dict[tensor_name.replace("keypoint_head", "associate_keypoint_heads.4")]

for tensor_name in ["keypoint_head.final_layer.weight", "keypoint_head.final_layer.bias"]:
    state_dict[tensor_name] = state_dict[tensor_name][:133]

for i in range(5):
    for tensor_name in weight_names:
        state_dict.pop(tensor_name.replace("keypoint_head", f"associate_keypoint_heads.{i}"), None)

for key in list(state_dict):
    if "expert" in key:
        state_dict.pop(key)

os.makedirs(os.path.dirname(out), exist_ok=True)
torch.save(new_ckpt, out)
print(out, os.path.getsize(out))
PY
)
    conda run -n annyfit python -c "$PYTHON_CODE" "$VITPOSE_SOURCE" checkpoints/vitpose_huge_wholebody.pth
fi

echo "Checkpoint layout:"
find checkpoints -maxdepth 4 -type f \
    \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.pkl' -o -name '*.npy' \) \
    -printf '%p %s bytes\n' | sort
