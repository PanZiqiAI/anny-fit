#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PATCH="$REPO_ROOT/ai_install_guide/patches/vitpose-model-split-expert-key.patch"
VITPOSE_DIR="$REPO_ROOT/submodules/ViTPose"

if [[ ! -d "$VITPOSE_DIR/.git" && ! -f "$VITPOSE_DIR/.git" ]]; then
    echo "ERROR: ViTPose submodule is missing. Run: git submodule update --init --recursive" >&2
    exit 1
fi

if git -C "$VITPOSE_DIR" apply --check "$PATCH" 2>/dev/null; then
    git -C "$VITPOSE_DIR" apply "$PATCH"
    echo "Applied ViTPose model_split patch."
elif git -C "$VITPOSE_DIR" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "ViTPose model_split patch is already applied."
else
    echo "ERROR: ViTPose patch cannot be applied cleanly." >&2
    echo "Inspect: $VITPOSE_DIR/tools/model_split.py" >&2
    exit 1
fi
