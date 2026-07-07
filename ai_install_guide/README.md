# Anny-Fit 跨环境复现安装指南（给 AI/自动化助手）

目标：在新环境中复现当前机器上的安装状态，包括依赖、子模块补丁、已整理 checkpoints 的拷贝和最终验证。

当前环境已验证通过：

- Conda 环境：`annyfit`
- PyTorch：CUDA build 可用
- GPU：RTX 3090 / RTX 2070 SUPER 在宿主环境下可见
- Anny-Fit wrapper 能成功读取 checkpoints 并初始化

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

## 2. 应用子模块补丁

当前顶层 repo 不能直接提交 ViTPose 子模块内部文件 diff，因此这里用顶层 patch 复现该改动。

```bash
bash ai_install_guide/scripts/apply_submodule_patches.sh
```

该脚本会修复：

```text
submodules/ViTPose/tools/model_split.py
```

核心改动是把错误的：

```python
if 'expert' in keys:
```

改成：

```python
if 'expert' in key:
```

## 3. 安装 Conda 环境与依赖

优先按项目官方脚本：

```bash
bash scripts/install.sh
```

本机安装时遇到过的兼容问题和处理方式：

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

推荐不要在新环境重新下载或解压权重。直接把当前机器已经整理好的 `checkpoints/` 目录复制到新环境，然后放到新 repo 的根目录下。

当前机器需要保留/拷贝的目录结构是：

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

在新环境中，如果已整理好的 checkpoints 位于 `/path/to/prepared/checkpoints`，运行：

```bash
bash ai_install_guide/scripts/copy_prepared_checkpoints.sh /path/to/prepared/checkpoints
```

这一步只复制文件，不下载、不解压、不重新生成 ViTPose 权重。

如果你是把 `checkpoints/` 目录直接拷贝到了 repo 根目录，也可以跳过这个脚本，直接进入验证步骤。

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

当前这台机器已经完成了这一步；跨环境复现时优先复制整理后的 `checkpoints/`。

## 6. 验证安装

```bash
bash ai_install_guide/scripts/verify_install.sh
```

预期输出包含：

```text
core_imports_ok
anny_wrapper_ok (17, 6890) (6890, 10475) (138, 6890)
```

如果在默认沙箱中运行，`torch.cuda.is_available()` 可能是 `False`；在宿主环境运行时应能看到 GPU。

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
- 子模块内部补丁不应直接依赖本地脏工作区；用 `ai_install_guide/patches/` 里的 patch 复现。
- 如果必须提交子模块修改，需要 fork ViTPose 并让顶层 repo 指向 fork 中可访问的 commit；否则其他环境无法获取该子模块 commit。
