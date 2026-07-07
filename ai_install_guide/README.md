# Anny-Fit 跨环境复现安装指南（给 AI/自动化助手）

本文面向“目标环境”中的 AI/自动化助手。目标是在目标环境中复现 Anny-Fit 的可运行安装状态，包括依赖、按当前仓库记录初始化子模块、从源环境复制已整理 checkpoints，以及最终验证。

术语：

- 源环境：已经完成安装、并且拥有完整 `checkpoints/` 目录的机器或路径。
- 目标环境：当前正在执行本指南、准备安装 Anny-Fit 的机器。

源环境参考记录：

- Conda 环境名：`annyfit`
- 源环境应已验证 PyTorch CUDA build 可用，或至少完成 CPU 级导入验证。
- 源环境中 Anny-Fit wrapper 应能成功读取 checkpoints 并初始化。

目标环境不要求 GPU 型号与源环境一致。目标环境只需要满足项目运行所需的 CUDA/PyTorch/GPU 驱动条件；如果只做 CPU 级导入验证，也可以没有可见 GPU。

注意：不要把 `checkpoints/` 下的权重提交到 Git。项目 `.gitignore` 已忽略 `checkpoints/`。

## 1. 克隆与子模块

```bash
git clone --recurse-submodules <repo-url> anny-fit
cd anny-fit
git submodule update --init --recursive
```

如果已经 clone 但没拉子模块，执行：

```bash
git submodule update --init --recursive
```

如果这是一个已经存在的旧 clone，并且 `.gitmodules` 最近被更新过，先同步子模块 URL：

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

## 2. 子模块来源与固定 commit

当前 repo 通过 `.gitmodules` 记录子模块 URL，并通过顶层 Git commit 固定每个子模块应 checkout 的具体 commit。目标环境不需要手动修改子模块，也不应依赖文档中写死的 branch 名。

安装助手应从当前仓库读取子模块来源：

```bash
git config -f .gitmodules --get-regexp '^submodule\..*\.url$'
```

确认子模块不是本地临时脏改动，而是顶层 repo 已固定的子模块 commit：

```bash
git submodule status --recursive
```

输出中的 commit hash 才是跨环境复现依据。`.gitmodules` 中如果存在 `branch = ...`，它只是维护分支提示，主要影响 `git submodule update --remote`，不是普通安装复现的依据。

如果这是旧 clone，或者子模块 URL 与当前 `.gitmodules` 不一致，执行：

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

## 3. 安装 Conda 环境与依赖

优先按项目官方脚本：

```bash
bash scripts/install.sh
```

源环境安装时遇到过的兼容问题和处理方式；目标环境中如果遇到同类错误，可按以下方式处理：

1. Anaconda TOS 未接受时，按 conda 提示接受对应 channel 的 TOS 后重跑。
2. `mmcv` / `albumentations` 构建遇到 `pkg_resources` / setuptools 问题时：

   ```bash
   conda run -n annyfit pip install 'setuptools<81'
   bash scripts/install.sh
   ```

3. `detectron2` 构建缺少 C++ 编译器时：

   ```bash
   conda install -n annyfit -y gxx_linux-64 ninja
   ```

4. 默认沙箱可能看不到 `/dev/nvidia*`；GPU 验证需要在宿主环境执行。

## 4. 准备已整理好的 checkpoints 目录（推荐）

推荐不要在目标环境重新下载或解压权重。目标环境应直接从源环境复制已经整理好的 `checkpoints/` 目录，并放到目标环境 repo 的根目录下。

源环境需要提供的目录结构是：

```text
checkpoints/
├── multiHMR_672_L_anny.pt
├── sam2.1_hiera_large.pt
├── hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth
├── hrnetv2_w18_coco_wholebody_hand_256x256_dark-a9228c9c_20210908.pth
├── cam_model_cleaned.ckpt
├── camerahmr_checkpoint_cleaned.ckpt
├── densekp.ckpt
├── model_final_f05665.pkl
├── vitpose_backbone.pth
├── vitpose_huge_wholebody.pth
└── body_models/
    ├── SMPL_NEUTRAL.pkl
    └── joint_regressors/
        ├── J_regressor_coco_hip_smpl.npy
        ├── smplx2smpl.pkl
        └── downsample_mat.pkl
```

在目标环境中，AI 应先询问用户源环境的 repo/checkpoints 在哪里，并使用以下提示语：

```text
如果是另一台机器，请给出诸如 ip地址:/path_to_repo 或 user@ip地址:/path_to_repo 的格式；
如果是本机，请给出诸如 /path_to_repo 的格式。
```

复制脚本同时支持源环境 repo 根目录和源环境 checkpoints 目录。例如：

```bash
# 源环境与目标环境在同一台机器：传源 repo 根目录
bash ai_install_guide/scripts/copy_prepared_checkpoints.sh /path_to_repo

# 源环境与目标环境在同一台机器：直接传源 checkpoints 目录
bash ai_install_guide/scripts/copy_prepared_checkpoints.sh /path_to_repo/checkpoints

# 源环境在局域网另一台机器：传源 repo 根目录
bash ai_install_guide/scripts/copy_prepared_checkpoints.sh 192.168.x.xxx:/path_to_repo

# 源环境在局域网另一台机器：带用户名
bash ai_install_guide/scripts/copy_prepared_checkpoints.sh user@192.168.x.xxx:/path_to_repo
```

如果源环境已整理好的 checkpoints 位于 `/path/to/prepared/checkpoints`，运行：

```bash
bash ai_install_guide/scripts/copy_prepared_checkpoints.sh /path/to/prepared/checkpoints
```

这一步只复制文件，不下载、不解压、不重新生成 ViTPose 权重。远程路径会使用 `ssh` 校验文件，并优先使用 `rsync -e ssh` 复制；没有 `rsync` 时回退到 `scp -r`。

如果用户已经把 `checkpoints/` 目录直接拷贝到了目标环境 repo 根目录，也可以跳过这个脚本，直接进入验证步骤。

## 5. 从原始下载文件整理 checkpoints（备用，不推荐重复执行）

只有在没有已整理好的 `checkpoints/` 目录时，才使用这个备用脚本：

```bash
bash ai_install_guide/scripts/install_from_manual_downloads.sh /path/to/downloaded/files
```

该脚本会从原始下载文件中抽取/生成所需权重，包括从 `train-eval-utils.zip` 抽取文件、从 `vitpose _huge.pth` 生成 `vitpose_huge_wholebody.pth`。如果你已经有完整 `checkpoints/` 目录，不要使用它。

备用脚本需要的原始文件：

```text
cam_model_cleaned.ckpt
camerahmr_checkpoint_cleaned.ckpt
densekp.ckpt
model_final_f05665.pkl
SMPL_NEUTRAL.pkl
train-eval-utils.zip
vitpose _huge.pth
```

如果源环境已经完成了这一步，跨环境复现时优先复制源环境中整理后的 `checkpoints/`。

## 6. 验证安装

```bash
bash ai_install_guide/scripts/verify_install.sh
```

预期输出包含：

```text
core_imports_ok
anny_wrapper_ok (17, 6890) (6890, 10475) (138, 6890)
```

如果在默认沙箱中运行，`torch.cuda.is_available()` 可能是 `False`；在宿主环境运行时应按目标环境实际硬件显示 GPU。目标环境 GPU 型号不需要与源环境一致。

## 7. 使用方式

交互 shell 中：

```bash
source setup.sh
cd annyfit
python optimize.py --config configs/demo/multihmr.yaml
```

非交互 shell 中建议使用：

```bash
PYTHONPATH="$PWD:$PWD/submodules/multi-hmr:$PWD/submodules/CameraHMR" \
PYOPENGL_PLATFORM=egl \
conda run -n annyfit python <your_command.py>
```

## 8. Git 注意事项

- 不要 `git add checkpoints/`。
- 不要在目标环境临时修改子模块来完成安装；需要长期保留的子模块改动应提交到对应 fork，并由顶层 repo 记录新的 submodule commit。
- 顶层 repo 中的 `git add submodules/<name>` 只更新子模块指针，不会把子模块内部文件 diff 直接提交进顶层 repo。
- 安装复现以顶层 repo 记录的 submodule commit 为准，不以 README 中的分支名、当前本地分支名或远端默认分支为准。
- 跨环境复现应依赖 fork 中已提交的子模块 commit。
