# height_es — 单目摄像头人体身高测量

基于已知相机内外参数（或单应性矩阵）的传统几何算法，无需深度学习，实现单帧图像人体身高估算。

## 方法概述

1. **反投影脚底** → 与地面平面（$Z=0$）求交，得到 3D 脚底坐标
2. **反投影头顶** → 利用人体垂直站立约束（头顶在脚底正上方）求解 3D 头顶坐标
3. **身高** = 头顶 Z 坐标 − 脚底 Z 坐标

详细数学推导见 [principle.md](./principle.md)。

## 文件结构

```
height_es/
├── src/
│   ├── height_estimator.py   # 核心算法（HeightEstimator 类）
│   └── validate.py           # 合成数据验证脚本
├── output/                   # 验证结果图（运行后生成）
├── principle.md              # 算法原理说明
├── requirements.txt          # Python 依赖
└── README.md
```

## 快速开始

```bash
pip install -r requirements.txt
python src/validate.py
```

验证结果图保存在 `output/validation_result.png`。

## 使用示例

```python
import numpy as np
from src.height_estimator import HeightEstimator, build_K

# 构造内参矩阵
K = build_K(fx=1200, fy=1200, cx=960, cy=540)

# 方式1：通过相机位置+注视点构造（最简便）
est = HeightEstimator.from_look_at(
    cam_pos=[0, 0, 4.0],   # 相机安装高度 4m
    target =[0, 12, 0],    # 注视地面 12m 前方
    K=K
)

# 方式2：直接提供 R, t（已标定结果）
# est = HeightEstimator(K, R, t)

# 估计身高（输入：检测到的像素坐标）
foot_px = (960, 850)   # 脚底像素 (u, v)
head_px = (955, 460)   # 头顶像素 (u, v)

height, P_foot, P_head = est.estimate_height(foot_px, head_px)
print(f"估计身高: {height:.3f} m")
```

## 算法精度

在典型监控视角（相机高 4 m，目标距离 6–20 m，$f=1200$ px）下：

| 像素检测噪声 | 均值误差 | 最大误差 |
|---|---|---|
| 0 px（理论） | < 0.01 mm | < 0.01 mm |
| 1 px | ~1 cm | ~3 cm |
| 3 px | ~3 cm | ~9 cm |
| 5 px | ~5 cm | ~15 cm |

## 依赖

- Python ≥ 3.9
- NumPy ≥ 1.21
- Matplotlib ≥ 3.4

## 许可证

MIT License