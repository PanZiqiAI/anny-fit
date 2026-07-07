#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SRC="${1:-}"
if [[ -z "$SRC" ]]; then
    echo "Usage: bash ai_install_guide/scripts/copy_prepared_checkpoints.sh /path/to/prepared/checkpoints" >&2
    exit 1
fi

if [[ ! -d "$SRC" ]]; then
    echo "ERROR: source checkpoints directory does not exist: $SRC" >&2
    exit 1
fi

required=(
    "multiHMR_672_L_anny.pt"
    "sam2.1_hiera_large.pt"
    "hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"
    "hrnetv2_w18_coco_wholebody_hand_256x256_dark-a9228c9c_20210908.pth"
    "cam_model_cleaned.ckpt"
    "camerahmr_checkpoint_cleaned.ckpt"
    "densekp.ckpt"
    "model_final_f05665.pkl"
    "vitpose_backbone.pth"
    "vitpose_huge_wholebody.pth"
    "body_models/SMPL_NEUTRAL.pkl"
    "body_models/joint_regressors/J_regressor_coco_hip_smpl.npy"
    "body_models/joint_regressors/smplx2smpl.pkl"
    "body_models/joint_regressors/downsample_mat.pkl"
)

for rel in "${required[@]}"; do
    if [[ ! -s "$SRC/$rel" ]]; then
        echo "ERROR: missing required checkpoint file: $SRC/$rel" >&2
        exit 1
    fi
done

mkdir -p checkpoints

if command -v rsync >/dev/null 2>&1; then
    rsync -a --info=progress2 "$SRC"/ checkpoints/
else
    cp -a "$SRC"/. checkpoints/
fi

echo "Copied prepared checkpoints from: $SRC"
echo "Checkpoint layout:"
find checkpoints -maxdepth 4 -type f \
    \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.pkl' -o -name '*.npy' \) \
    -printf '%p %s bytes\n' | sort
