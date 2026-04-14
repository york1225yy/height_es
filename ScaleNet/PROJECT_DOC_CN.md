# ScaleNet — 野外单视图测量（Single View Metrology in the Wild）

> **论文**：*Single View Metrology in the Wild*, Zhu et al., ECCV 2020
> **仓库**：[https://github.com/Jerrypiglet/ScaleNet](https://github.com/Jerrypiglet/ScaleNet)

---

## 目录

1. [项目简介](#1-项目简介)
2. [核心功能](#2-核心功能)
3. [整体架构](#3-整体架构)
4. [目录结构](#4-目录结构)
5. [环境依赖](#5-环境依赖)
6. [安装步骤](#6-安装步骤)
7. [数据准备](#7-数据准备)
8. [模型检查点下载](#8-模型检查点下载)
9. [快速开始（推理演示）](#9-快速开始推理演示)
10. [模块详解](#10-模块详解)
11. [配置说明](#11-配置说明)
12. [训练流程](#12-训练流程)
13. [当前进度与待办事项](#13-当前进度与待办事项)
14. [注意事项](#14-注意事项)
15. [引用](#15-引用)

---

## 1. 项目简介

ScaleNet 是一个从**单张图像**中恢复真实物理尺度与相机参数的深度学习系统。它联合：

- **相机标定网络**（Camera Calibration Network）：从图像中估计相机内参（焦距、视场角）与外参（俯仰角 pitch、滚转角 roll）。
- **尺度估计网络**（Scale Estimation Network）：结合人体先验与 PointNet，估计场景中物体的物理高度（摄像机高度）。

系统无需专用硬件（深度传感器、IMU 等），仅凭一张 RGB 图像即可完成场景测量，具有广泛的实际应用价值。

---

## 2. 核心功能

| 功能 | 说明 |
|------|------|
| 相机垂直视场角（vFoV）估计 | 预测图像垂直方向的视角范围 |
| 俯仰角（Pitch）估计 | 预测摄像机与水平面的夹角 |
| 滚转角（Roll）估计 | 预测摄像机绕光轴的旋转角 |
| 焦距（Focal Length）估计 | 以像素和35mm等效焦距输出 |
| 地平线估计 | 由 pitch/roll/vFoV 几何推导 |
| 摄像机高度（Camera Height）估计 | 通过 PointNet 从 ROI 特征中回归 |
| 人员高度（Person Height）估计 | 多层 PointNet 细化人体像素高度 |

---

## 3. 整体架构

```
输入图像
    │
    ▼
┌──────────────────────────────────────┐
│  Backbone: FBNet (cham_v1a)          │  ← 特征提取
│  + FPN (Feature Pyramid Network)     │
└──────────────────┬───────────────────┘
                   │特征图
         ┌─────────┴──────────┐
         ▼                    ▼
┌─────────────────┐  ┌──────────────────────────────┐
│ 相机标定分类头   │  │  Mask R-CNN 检测头（ROI Head）│
│ (CLS Head)      │  │  - RPN + ROI Align            │
│ 估计：          │  │  - Bbox / Keypoint 分支        │
│  • horizon      │  │  - 人体检测 & 关键点           │
│  • pitch        │  └─────────────┬────────────────┘
│  • roll         │                │ROI 特征 + Bbox
│  • vfov         │                ▼
└────────┬────────┘  ┌──────────────────────────────┐
         │           │ PointNet (CamH / PersonH)     │
         │           │  - 输入: 人体 bbox 几何特征    │
         │           │  - 多层迭代细化               │
         └─────────► │  - 输出: 摄像机高度 yc         │
                     │         人体像素高度 h_px      │
                     └──────────────────────────────┘
```

**几何关系**（内参之间的约束）：

$$f_{\text{pix}} = \frac{H}{2 \cdot \tan(\text{vfov}/2)}$$

$$f_{\text{mm}} = \frac{f_{\text{pix}}}{H} \times \text{sensor\_size}$$

---

## 4. 目录结构

```
ScaleNet/
├── README.md                          # 英文原版说明
├── requirements.txt                   # Python 依赖
├── index.html                         # 项目主页（HTML）
├── ScaleNet_files/                    # 项目主页资源文件（图片、视频、CSS 等）
│   ├── teaser.png                     # 论文 teaser 图
│   ├── 1337-shortVideo.mp4            # 演示短视频
│   └── ...
└── RELEASE_ScaleNet_minimal/          # 核心代码发布目录
    ├── demo-evalCameraCalib-RELEASE-V2.ipynb      # 相机标定推理 Demo
    ├── demo-evalScaleNet-COCOScale-RELEASE.ipynb  # 尺度估计推理 Demo
    ├── train_batch_combine_RCNNOnly_v5_pose_multiCat.py  # 训练批次函数
    ├── dataset_coco_pickle_eccv.py    # COCO 数据集加载器
    ├── dataset_cvpr.py                # SUN360 数据集加载器
    ├── coco_config_small_RCNNOnly.yaml           # 主配置（纯 Scale）
    ├── coco_config_small_synBN1108_kps.yaml      # 含关键点配置
    ├── setup_maskrcnn_rui.py          # 自定义 Mask R-CNN 安装脚本
    ├── SUN360_mini_crops_dataset_cvpr_myDistNarrowerLarge1105.json  # SUN360 标注
    ├── maskrcnn_rui/                  # 自定义 Mask R-CNN 编译库
    │   ├── _C.cpython-36m-x86_64-linux-gnu.so
    │   └── _C.cpython-37m-x86_64-linux-gnu.so
    ├── models/                        # 模型定义
    │   ├── model_RCNNOnly_combine_indeptPointnet_maskrcnnPose_discount.py  # 主模型
    │   ├── model_RCNN_only.py         # 纯 RCNN 模型
    │   ├── model_part_GeneralizedRCNNRuiMod_cameraCalib_sep.py            # 相机标定子模块
    │   ├── model_part_GeneralizedRCNNRuiMod_cameraCalib_sep_maskrcnnPose_hybrid.py  # 混合子模块
    │   ├── model_part_pointnet_cls.py  # PointNet 分类（摄像机高度）
    │   └── model_part_pointnet_seg.py  # PointNet 分割（人体高度）
    ├── utils/                         # 工具函数
    │   ├── checkpointer.py            # 模型权重加载/保存
    │   ├── compute_vectors.py         # 几何向量计算
    │   ├── data_utils.py              # 数据处理工具
    │   ├── eval_save_utils_combine_RCNNONly.py  # 评估与结果保存
    │   ├── geo_utils.py               # 几何运算工具
    │   ├── logger.py                  # 日志工具
    │   ├── model_serialization.py     # 模型序列化
    │   ├── model_utils.py             # 模型辅助函数（Bbox 转换等）
    │   ├── train_utils.py             # 训练辅助函数（焦距换算等）
    │   ├── utils_coco.py              # COCO 数据集工具
    │   ├── utils_misc.py              # 杂项工具
    │   └── vis_utils.py               # 可视化工具
    ├── rendering/                     # 渲染相关
    │   ├── render_coco_rui_cylinder_all_fix_final.py  # Blender 渲染脚本
    │   └── scene_chair_fix.blend      # Blender 场景文件
    └── demo/                          # 演示图片
        ├── fall-cmu-700x700.jpg
        └── white-house.jpg
```

---

## 5. 环境依赖

| 依赖包 | 版本要求 |
|--------|----------|
| Python | 3.6（推荐） |
| PyTorch | 1.7.0 |
| torchvision | 与 PyTorch 对应版本 |
| CUDA | 与 PyTorch 兼容版本 |
| Pillow | 6.1 |
| opencv-python | 最新稳定版 |
| scikit-image | 最新稳定版 |
| tqdm | 最新稳定版 |
| tensorboard / tensorboardX | 最新稳定版 |
| yacs | 最新稳定版 |
| pycocotools | 最新稳定版 |
| jupyterlab | 最新稳定版 |
| NVIDIA apex | 需手动编译安装 |
| maskrcnn-benchmark | Facebook 定制版（仓库内附） |

完整依赖见 [requirements.txt](requirements.txt)。

---

## 6. 安装步骤

### 6.1 创建 Conda 环境

```bash
conda create -y -n scalenet python=3.6
conda activate scalenet
pip install -r requirements.txt
conda install nb_conda
```

### 6.2 安装 maskrcnn-benchmark

```bash
cd maskrcnn-benchmark
conda install cudatoolkit
python setup.py build develop
cd ..
```

### 6.3 安装 NVIDIA apex（混合精度训练）

```bash
git clone https://github.com/NVIDIA/apex.git
cd apex
# 若出现 IF 语句报错，按提示注释掉对应代码行
python setup.py install --cuda_ext --cpp_ext
cd ..
```

### 6.4 编译自定义 Mask R-CNN 扩展

```bash
python setup_maskrcnn_rui.py build develop
```

### 6.5 启动 Jupyter Notebook

```bash
jupyter notebook
# 在 Kernel 菜单 -> Change Kernel -> 选择 scalenet
```

---

## 7. 数据准备

### 7.1 COCOScale 数据集

下载以下内容并解压到 `data/results_coco/`：

- [COCOScale zip 文件（Google Drive）](https://drive.google.com/drive/folders/1yew9ol6w_T83fLVMQ34AHCu6k5eLArWs?usp=sharing)

解压后目录结构：

```
data/results_coco/
├── results_test_20200302_Car_noSmall-ratio1-35-mergeWith-results_with_kps_20200225_train2017_detOnly_filtered_2-8_moreThan2
├── results_with_kps_20200208_morethan2_2-8
└── results_with_kps_20200225_val2017_test_detOnly_filtered_2-8_moreThan2
```

### 7.2 COCO 图像

从 [COCO 官网](https://cocodataset.org/#download) 下载 2017 年 train/val 图片，存放到 `/data/COCO/`：

```
/data/COCO/
├── train2017/
└── val2017/
```

> 路径可在 `dataset_coco_pickle_eccv.py` 中自定义配置。

### 7.3 SUN360 数据集

> **注意**：由于 Adobe 版权限制，SUN360 训练数据无法公开发布。仓库仅提供相机标定推理 Demo 及预训练权重。

---

## 8. 模型检查点下载

从 Google Drive 下载以下两个检查点，分别放入 `checkpoint/` 目录：

| 检查点名称 | 用途 | 下载链接 |
|-----------|------|----------|
| `1109-0141-mm1_SUN360RCNN-...` | 相机标定网络权重 | [Google Drive](https://drive.google.com/drive/folders/111hCohH_X5TjOQKRx5P1w8Ow_7Od_P6Q?usp=sharing) |
| `20200222-162430_pod_backCompat_...` | 尺度估计网络权重 | 同上 |

下载后目录结构：

```
checkpoint/
├── 1109-0141-mm1_SUN360RCNN-HorizonPitchRollVfovNET_myDistNarrowerLarge1105_bs16on4_le1e-5_indeptClsHeads_synBNApex_valBS1_yannickTransformAug
└── 20200222-162430_pod_backCompat_adam_wPerson05_720-540_REafterDeathV_afterFaster_bs16_fix3_nokpsLoss_personLoss3Layers_loss3layers
```

---

## 9. 快速开始（推理演示）

### 9.1 相机标定推理

打开并运行：

```
RELEASE_ScaleNet_minimal/demo-evalCameraCalib-RELEASE-V2.ipynb
```

输入任意图像，输出：
- 垂直视场角 `vfov`
- 俯仰角 `pitch`
- 滚转角 `roll`
- 焦距（像素单位 & 35mm 等效）

### 9.2 COCOScale 尺度估计推理

打开并运行：

```
RELEASE_ScaleNet_minimal/demo-evalScaleNet-COCOScale-RELEASE.ipynb
```

输入 COCOScale 数据集图像，输出：
- 摄像机高度估计
- 人体检测与高度可视化
- 场景测量结果

---

## 10. 模块详解

### 10.1 主模型（`RCNNOnly_combine`）

文件：`models/model_RCNNOnly_combine_indeptPointnet_maskrcnnPose_discount.py`

核心组件：

| 子模块 | 类名 | 作用 |
|--------|------|------|
| 检测骨干 | `GeneralizedRCNNRuiMod_cameraCalib_maskrcnnPose` | FBNet + FPN + ROI Head |
| 相机高度估计 | `CamHPointNet` | PointNet，输入 7+n 维 bbox 特征 |
| 人体高度细化 | `CamHPersonHPointNet`（多层） | PointNet Seg，逐层迭代 |
| 分类头 | `FCPredictorRui` | 256 分类估计 pitch/roll/vfov 分布 |

**PointNet 输入特征维度**（默认 7+额外维度）：

- 人体 ROI 的几何特征（bbox 坐标、大小、位置等）
- 可选额外输入：人体像素高度（`person_H`）及折扣项（`person_H_discount`）
- 可选 ROI 特征（`pointnet_roi_feat_input`，16 维）

### 10.2 训练批次函数（`train_batch_combine`）

文件：`train_batch_combine_RCNNOnly_v5_pose_multiCat.py`

训练时同时使用两路数据：

| 数据路 | 数据集 | 监督信号 |
|--------|--------|----------|
| COCO 路 | COCOScale | bbox、关键点、摄像机高度、人体高度 |
| SUN360 路 | SUN360（仅推理可用） | horizon/pitch/roll/vfov 分布（KL 散度损失） |

**损失函数**：

| 损失项 | 说明 |
|--------|------|
| `loss_vt` | 摄像机高度损失（PointNet 各层均值或最后一层） |
| `loss_person` | 人体高度损失（PointNet 各层均值或最后一层） |
| `loss_kp` | 关键点损失（可选，受 `weight_kps` 控制） |
| `loss_bbox_cls/reg` | 检测框分类/回归损失（可选） |
| `loss_horizon/pitch/roll/vfov` | 相机参数 KL 散度损失（SUN360 路，`weight_SUN360` 控制） |

### 10.3 数据集加载器

| 文件 | 数据集 | 说明 |
|------|--------|------|
| `dataset_coco_pickle_eccv.py` | COCOScale | 加载 COCO 图像与预检测结果；生成非均匀区间 bins（正态分布加权） |
| `dataset_cvpr.py` | SUN360 | 加载相机标定训练数据（版权受限，不公开） |

### 10.4 几何工具（`utils/geo_utils.py`）

提供摄像机几何数学工具：

- 焦距单位换算（像素 ↔ 毫米）
- pitch/vfov/v0（主点纵坐标）相互转换
- 地平线估计

---

## 11. 配置说明

配置文件：`coco_config_small_RCNNOnly.yaml`

关键配置项：

```yaml
MODEL:
  BACKBONE:
    CONV_BODY: FBNet          # 骨干网络（FBNet cham_v1a）
  ROI_BOX_HEAD:
    NUM_CLASSES_h: 256        # 人体高度分箱数
  CLASSIFIER_HEAD:
    NUM_CLASSES: 256          # 相机参数分箱数
  HUMAN:
    MEAN: 1.70                # 人体平均高度先验（米）
    STD: 0.20                 # 人体高度标准差（米）

INPUT:
  MIN_SIZE_TRAIN: (600,)
  MAX_SIZE_TRAIN: 1000
  MIN_SIZE_TEST: 600
  PIXEL_MEAN: [103.53, 116.28, 123.675]

SOLVER:
  BASE_LR: 1e-3
  MAX_ITER: 900000
  IMS_PER_BATCH: 4
```

---

## 12. 训练流程

> 当前版本尚未完全公开训练代码，以下为已知训练逻辑说明。

**训练入口**（推断）：调用 `train_batch_combine` 函数，传入：

- `input_dict`：批次数据（图像、bbox、相机参数真值）
- `model`：`RCNNOnly_combine` 实例
- `opt`：命令行/配置参数
- `is_training=True`

**多 GPU 训练**：通过 NVIDIA apex 的同步 BN（`synBN`）支持分布式训练，批量大小建议为 GPU 数量的整数倍。

---

## 13. 当前进度与待办事项

| 状态 | 任务 |
|------|------|
| ✅ 已完成 | 相机标定推理 Demo（含样例图像） |
| ✅ 已完成 | COCOScale 尺度估计推理 Demo |
| ⏳ 待发布 | COCOScale 完整训练代码 |
| ⏳ 待发布 | KITTI 数据集推理 Demo 及数据 |
| ⏳ 待发布 | IMDB Celebrity 数据集推理 Demo 及数据 |

---

## 14. 注意事项

1. **SUN360 数据版权**：由于 Adobe 的版权限制，SUN360 相机标定训练数据无法公开。目前仅提供推理 Demo 及预训练权重。

2. **Python 版本**：推荐使用 Python 3.6。仓库内预编译的 `.so` 库（`maskrcnn_rui`）同时包含 cp36 和 cp37 版本。

3. **代码质量**：代码发布仍在进行中，当前版本的模型、数据加载器等实现可能较为复杂，后续会进行清理和注释完善。

4. **apex 安装**：安装 apex 时若出现关于 `IF` 语句的报错，需按提示手动注释掉对应代码行。

5. **数据路径**：COCO 图像路径 `/data/COCO` 仅为默认值，可在 `dataset_coco_pickle_eccv.py` 中修改。

---

## 15. 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@InProceedings{zhu2020singleviewmetrology,
  title     = {Single View Metrology in the Wild},
  author    = {Zhu, Rui and others},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2020}
}
```

---

*本文档由 GitHub Copilot 根据仓库代码与 README 自动整理生成。*
