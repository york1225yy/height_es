# Face Recognition 实时人脸识别系统 — 使用说明文档

> 项目地址：[vectornguyen76/face-recognition](https://github.com/vectornguyen76/face-recognition)

---

## 目录

- [项目简介](#项目简介)
- [系统架构](#系统架构)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [环境要求](#环境要求)
- [安装步骤](#安装步骤)
  - [1. 克隆项目](#1-克隆项目)
  - [2. 创建虚拟环境](#2-创建虚拟环境)
  - [3. 安装 PyTorch](#3-安装-pytorch)
  - [4. 安装依赖](#4-安装依赖)
  - [5. 下载预训练权重](#5-下载预训练权重)
- [使用流程](#使用流程)
  - [Step 1：添加人员到数据库](#step-1添加人员到数据库)
  - [Step 2：运行人脸识别](#step-2运行人脸识别)
  - [Step 3（可选）：仅运行人脸检测](#step-3可选仅运行人脸检测)
- [配置说明](#配置说明)
  - [切换人脸检测器](#切换人脸检测器)
  - [追踪参数配置](#追踪参数配置)
- [数据集目录结构](#数据集目录结构)
- [模块说明](#模块说明)
- [常见问题](#常见问题)

---

## 项目简介

本项目是一个基于深度学习的**实时人脸识别系统**，支持摄像头输入，能够同时完成人脸检测、人脸对齐、人脸跟踪和人脸身份识别。识别过程全程使用余弦相似度进行特征匹配，可在 CPU 或 GPU 上运行。

---

## 系统架构

系统的整体处理流程如下：

```
摄像头帧
   ↓
人脸检测（SCRFD / YOLOv5-Face / RetinaFace）
   ↓
人脸对齐（基于关键点仿射变换）
   ↓
人脸追踪（ByteTrack）
   ↓
特征提取（ArcFace / iResNet-100）
   ↓
余弦相似度匹配（与已有人员特征库比对）
   ↓
输出：人员姓名 + 置信度 + 追踪框
```

---

## 技术栈

| 模块 | 技术方案 | 说明 |
|------|----------|------|
| 人脸检测 | **SCRFD**（推荐） | 轻量、高速，支持多尺度人脸检测，默认使用 `scrfd_2.5g_bnkps.onnx` |
| 人脸检测 | **YOLOv5-Face** | 基于 YOLO 架构，实时高效 |
| 人脸检测 | **RetinaFace** | 精度高，适合高精度场景 |
| 人脸对齐 | 基于 5 点关键点的仿射变换 | 对齐到 112×112 的 ArcFace 标准坐标 |
| 人脸识别 | **ArcFace（iResNet-100）** | 业界主流人脸识别特征提取模型 |
| 人脸追踪 | **ByteTrack** | 轻量多目标追踪器，稳定跟踪多张人脸 |
| 相似度匹配 | **余弦相似度** | 用于特征向量比对，阈值可配置 |

---

## 项目结构

```
face-recognition/
├── add_persons.py              # 添加新人员到特征库脚本
├── recognize.py                # 实时人脸识别主程序
├── detect.py                   # 仅人脸检测演示脚本
├── tracking.py                 # 仅人脸追踪演示脚本
├── face_align.py               # 人脸对齐工具
├── requirements.txt            # 依赖列表
│
├── face_alignment/
│   └── alignment.py            # 基于关键点的人脸对齐实现
│
├── face_detection/
│   ├── scrfd/
│   │   ├── detector.py         # SCRFD 检测器封装
│   │   └── weights/            # 放置 SCRFD ONNX 模型权重
│   ├── yolov5_face/
│   │   ├── detector.py         # YOLOv5-Face 检测器封装
│   │   ├── models/             # 模型定义
│   │   └── weights/            # 放置 YOLOv5-Face 模型权重
│   └── retinaface/             # RetinaFace 检测器（含训练/转换代码）
│
├── face_recognition/
│   └── arcface/
│       ├── model.py            # iResNet 模型定义
│       ├── utils.py            # 特征读取与余弦相似度计算
│       └── weights/            # 放置 ArcFace 模型权重
│
├── face_tracking/
│   ├── tracker/
│   │   ├── byte_tracker.py     # ByteTrack 追踪器
│   │   ├── kalman_filter.py    # 卡尔曼滤波
│   │   ├── matching.py         # 目标匹配
│   │   └── visualize.py        # 追踪结果可视化
│   ├── config/
│   │   └── config_tracking.yaml # 追踪参数配置文件
│   └── pretrained/             # 放置 ByteTrack 预训练权重
│
├── datasets/
│   ├── backup/                 # 原始图片备份
│   ├── data/                   # 处理后人脸图像
│   ├── face_features/          # 提取的特征向量（feature.npz）
│   └── new_persons/            # 新增人员图片放置目录
│
└── assets/                     # 文档图片资源
```

---

## 环境要求

- Python **3.9**
- Anaconda / Miniconda（推荐）
- CUDA（可选，支持 CPU 运行）
- 摄像头（用于实时识别）

---

## 安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/vectornguyen76/face-recognition.git
cd face-recognition
```

### 2. 创建虚拟环境

```bash
conda create -n face-dev python=3.9
conda activate face-dev
```

### 3. 安装 PyTorch

**CPU 版本（推荐入门）：**

```bash
pip install torch==1.9.1+cpu torchvision==0.10.1+cpu torchaudio==0.9.1 \
    -f https://download.pytorch.org/whl/torch_stable.html
```

**GPU 版本（需要 CUDA 环境）：**

请前往 [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) 根据 CUDA 版本选择对应安装命令。

### 4. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖包括：`opencv-python`、`onnxruntime`、`numpy`、`scikit-image`、`PyYAML`、`scipy`、`torch`、`torchvision` 等。

### 5. 下载预训练权重

需要下载以下三类模型权重，并放置到对应目录：

| 模型 | 下载地址 | 目标路径 |
|------|----------|----------|
| **SCRFD**（人脸检测，推荐） | [Google Drive](https://drive.google.com/drive/folders/1C9RzReAihJQRl8EJOX6vQj7qbHBPmzME?usp=sharing) | `face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx` |
| **YOLOv5-Face**（可选） | [Google Drive](https://drive.google.com/drive/folders/1CGq-2AfcSyWGwZWs9sIzQ1BXhRkPGgxF?usp=sharing) | `face_detection/yolov5_face/weights/yolov5n-face.pt` |
| **ArcFace**（人脸识别） | [Google Drive](https://drive.google.com/drive/folders/1CHHb_7wbvfjKPFNKVBb76lL5sVfBLcv5?usp=sharing) | `face_recognition/arcface/weights/arcface_r100.pth` |
| **ByteTrack**（追踪，可选） | [Google Drive](https://drive.google.com/file/d/1uSmhXzyV1Zvb4TJJCzpsZOIcw7CCJLxj/view?usp=sharing) | `face_tracking/pretrained/bytetrack_s_mot17.pth.tar` |

> **注意**：ArcFace 权重和 SCRFD 权重是必须下载的，否则识别功能无法运行。

---

## 使用流程

### Step 1：添加人员到数据库

在识别之前，需要先将人员照片添加到人脸特征库中。

**1. 在 `datasets/new_persons/` 下创建以人员姓名命名的文件夹：**

```
datasets/
└── new_persons/
    ├── 张三/
    │   ├── photo1.jpg
    │   └── photo2.jpg
    └── 李四/
        └── photo1.jpg
```

> 每位人员建议提供 **3~5 张**不同角度的清晰正脸照片，以提高识别准确率。图片格式支持 `.jpg`、`.png`。

**2. 运行添加脚本：**

```bash
python add_persons.py
```

脚本将自动完成以下工作：
- 检测照片中的人脸
- 对人脸进行对齐裁剪，保存到 `datasets/data/人员姓名/` 目录
- 将原始图片备份到 `datasets/backup/人员姓名/` 目录
- 提取人脸特征向量，更新并保存特征库到 `datasets/face_features/feature.npz`

---

### Step 2：运行人脸识别

```bash
python recognize.py
```

程序将打开摄像头，实时进行：
- 人脸检测（默认使用 SCRFD）
- 人脸追踪（ByteTrack）
- 人脸特征提取与匹配（ArcFace + 余弦相似度）
- 在画面中显示人员姓名、追踪 ID 及帧率

按键盘 **`Q`** 键退出程序。

---

### Step 3（可选）：仅运行人脸检测

如果只需要测试人脸检测功能（无需识别）：

```bash
python detect.py
```

此模式下使用 YOLOv5-Face 检测人脸并标注关键点，结果保存为视频文件 `results/face-detection.mp4`。

---

## 配置说明

### 切换人脸检测器

在 `recognize.py` 或 `add_persons.py` 顶部，通过注释切换检测器：

```python
# 使用 SCRFD（默认，推荐）
detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")

# 或使用 YOLOv5-Face
# detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5n-face.pt")
```

### 追踪参数配置

追踪相关参数位于 `face_tracking/config/config_tracking.yaml`：

```yaml
device: cpu            # 运行设备：cpu 或 cuda
fps: 30                # 视频帧率
match_thresh: 0.8      # 人脸识别匹配阈值（越高要求越严格）
min_box_area: 10       # 最小检测框面积（过滤太小的人脸）
save_result: True      # 是否保存识别结果视频
track_buffer: 30       # 追踪缓冲帧数
track_thresh: 0.5      # 追踪置信度阈值
aspect_ratio_thresh: 1.6  # 人脸宽高比过滤（过滤非正常比例）
ckpt: bytetrack_s_mot17.pth.tar  # ByteTrack 权重文件名
fp16: True             # 是否使用半精度推理（需要 GPU 支持）
```

> **`match_thresh`**：是识别匹配核心参数。值越高，识别越严格，陌生人越不容易被错误识别；值越低，识别召回率更高但可能出现误识。建议范围：`0.6 ~ 0.9`。

---

## 数据集目录结构

```
datasets/
├── backup/             # 原始图片备份（add_persons.py 自动创建）
│   └── 张三/
│       └── photo1.jpg
├── data/               # 对齐后的人脸图像（add_persons.py 自动创建）
│   └── 张三/
│       ├── 0.jpg
│       └── 1.jpg
├── face_features/
│   └── feature.npz     # 人脸特征向量库（add_persons.py 自动生成）
└── new_persons/        # 你需要手动创建并放置图片的目录
    └── 张三/
        └── photo1.jpg
```

---

## 模块说明

### `add_persons.py`

负责将新人员照片处理并加入识别数据库：
1. 遍历 `datasets/new_persons/` 下各人员文件夹
2. 调用检测器找到人脸，进行关键点对齐（112×112）
3. 提取 ArcFace 特征向量并归一化
4. 将新特征追加到已有的 `feature.npz` 文件中

### `recognize.py`

实时识别主程序，核心逻辑：
1. 读取摄像头帧
2. 检测人脸，获取边界框和 5 个关键点
3. ByteTrack 对检测结果进行跨帧追踪，分配追踪 ID
4. 对跟踪到的人脸图像提取特征
5. 与数据库中所有人员的特征向量计算余弦相似度
6. 相似度超过 `match_thresh` 则显示对应人员名称，否则显示 `Unknown`

### `detect.py`

仅做人脸检测演示，适合测试检测器效果：
- 打开摄像头，对每帧进行人脸检测
- 绘制人脸框（矩形）和 5 个关键点（彩色圆点）
- 显示实时 FPS，结果保存为视频

### `face_alignment/alignment.py`

人脸对齐核心模块：
- 基于 5 点关键点（眼睛×2、鼻子、嘴角×2）的仿射变换
- 将人脸统一对齐到 ArcFace 标准坐标系（112×112）
- 对齐后再送入识别模型，显著提升识别精度

---

## 常见问题

**Q：运行 `recognize.py` 时报错无法找到权重文件？**

A：请确认已下载所有必要权重并放置到正确目录，详见 [下载预训练权重](#5-下载预训练权重)。

---

**Q：添加人员后识别不到？**

A：检查以下几点：
- 确认运行了 `add_persons.py` 且输出了提取特征成功的日志
- 检查 `datasets/face_features/feature.npz` 文件是否已更新
- 适当降低 `config_tracking.yaml` 中的 `match_thresh`（如改为 0.6）
- 确保添加的照片中人脸清晰、正面朝向

---

**Q：识别速度很慢？**

A：可以尝试：
- 改用更轻量的检测模型（如 `yolov5n-face.pt` 或 `scrfd_2.5g_bnkps.onnx`）
- 如有 GPU，确认 PyTorch 检测到 CUDA 并将 `config_tracking.yaml` 中 `device` 改为 `cuda`
- 降低摄像头分辨率

---

**Q：提示 `CUDA not available`，能否正常运行？**

A：可以。系统默认自动回退到 CPU 推理，功能完全正常，只是速度会慢一些。

---

**Q：如何删除已添加的人员？**

A：目前没有专用删除脚本，需手动操作：
1. 删除 `datasets/data/人员姓名/` 目录
2. 删除 `datasets/backup/人员姓名/` 目录
3. 删除 `datasets/face_features/feature.npz`
4. 重新对所有剩余人员运行 `add_persons.py` 重建特征库

---

## 参考资料

- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [YOLOv5-Face](https://github.com/deepcam-cn/yolov5-face)
- [InsightFace - ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)
- [InsightFace-REST](https://github.com/SthPhoenix/InsightFace-REST)
