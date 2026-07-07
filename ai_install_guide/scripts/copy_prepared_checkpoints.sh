#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SRC="${1:-}"
if [[ -z "$SRC" ]]; then
    echo "Usage: bash ai_install_guide/scripts/copy_prepared_checkpoints.sh <source_repo_or_checkpoints>" >&2
    echo "" >&2
    echo "Examples:" >&2
    echo "  Local repo:        /path_to_repo" >&2
    echo "  Local checkpoints: /path_to_repo/checkpoints" >&2
    echo "  Remote repo:       192.168.x.xxx:/path_to_repo" >&2
    echo "  Remote repo:       user@192.168.x.xxx:/path_to_repo" >&2
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

is_remote=0
remote_host=""
remote_path=""
src_checkpoints=""

if [[ "$SRC" == *:* && "$SRC" != /* ]]; then
    is_remote=1
    remote_host="${SRC%%:*}"
    remote_path="${SRC#*:}"
    if [[ -z "$remote_host" || -z "$remote_path" ]]; then
        echo "ERROR: invalid remote source: $SRC" >&2
        exit 1
    fi

    if ssh "$remote_host" "test -d '$remote_path/checkpoints'"; then
        src_checkpoints="$remote_path/checkpoints"
    elif ssh "$remote_host" "test -d '$remote_path'"; then
        src_checkpoints="$remote_path"
    else
        echo "ERROR: remote path does not exist or is not a directory: $SRC" >&2
        exit 1
    fi

    for rel in "${required[@]}"; do
        if ! ssh "$remote_host" "test -s '$src_checkpoints/$rel'"; then
            echo "ERROR: missing required remote checkpoint file: $remote_host:$src_checkpoints/$rel" >&2
            exit 1
        fi
    done
else
    if [[ -d "$SRC/checkpoints" ]]; then
        src_checkpoints="$SRC/checkpoints"
    elif [[ -d "$SRC" ]]; then
        src_checkpoints="$SRC"
    else
        echo "ERROR: source path does not exist or is not a directory: $SRC" >&2
        exit 1
    fi

    for rel in "${required[@]}"; do
        if [[ ! -s "$src_checkpoints/$rel" ]]; then
            echo "ERROR: missing required checkpoint file: $src_checkpoints/$rel" >&2
            exit 1
        fi
    done
fi

mkdir -p checkpoints

if [[ "$is_remote" == "1" ]]; then
    if command -v rsync >/dev/null 2>&1; then
        rsync -a --info=progress2 -e ssh "$remote_host:$src_checkpoints"/ checkpoints/
    else
        scp -r "$remote_host:$src_checkpoints"/. checkpoints/
    fi
else
    if command -v rsync >/dev/null 2>&1; then
        rsync -a --info=progress2 "$src_checkpoints"/ checkpoints/
    else
        cp -a "$src_checkpoints"/. checkpoints/
    fi
fi

echo "Copied prepared checkpoints from: $SRC"
echo "Checkpoint layout:"
find checkpoints -maxdepth 4 -type f \
    \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.pkl' -o -name '*.npy' \) \
    -printf '%p %s bytes\n' | sort
