# 使用方法说明

本文档说明如何在这个项目中进行人体身高估计，包括 Python 调用方式、命令行调用方式、参数格式说明，以及如何与上游检测器对接。

## 1. 安装依赖

在项目根目录执行：

```bash
pip install -r requirements.txt
```

当前依赖：

- Python >= 3.9
- numpy >= 1.21
- matplotlib >= 3.4

## 2. 项目入口说明

项目中最常用的几个文件如下：

- `src/height_estimator.py`：核心几何算法实现
- `src/estimate.py`：面向实际使用的直接估计接口
- `src/validate.py`：合成数据验证脚本
- `principle.md`：带数学推导的原理说明
- `principle_intuition.md`：不带公式的直觉原理说明

如果你的目标是直接输入标定参数和关键点结果来得到身高，优先使用 `src/estimate.py`。

## 3. 最快开始方式

### 3.1 直接运行默认示例

```bash
python src/estimate.py
```

这会使用脚本内置的默认参数运行一个完整示例：

- 内置相机内参 `DEFAULT_K`
- 内置外参 `DEFAULT_R` 和 `DEFAULT_T`
- 内置人物脚底像素 `DEFAULT_FOOT_PX`
- 内置人物头顶像素 `DEFAULT_HEAD_PX`

终端会打印：

- 估计身高
- 脚底三维坐标
- 头顶三维坐标
- 使用的方法名称
- 如果提供了真实身高，还会打印误差

### 3.2 运行验证脚本

```bash
python src/validate.py
```

验证脚本会完成三类测试：

- 无噪声理论精度测试
- 噪声灵敏度测试
- 地面单应性映射一致性测试

并在 `output/validation_result.png` 生成可视化结果图。

## 4. Python 中调用

### 4.1 方式一：已知 K、R、t

这是最标准的使用方式，适合你已经完成相机标定的场景。

```python
from src.estimate import estimate_height

K = [
    [1200.0, 0.0, 960.0],
    [0.0, 1200.0, 540.0],
    [0.0, 0.0, 1.0],
]

R = [
    [1.0, 0.0, 0.0],
    [0.0, 0.31622777, -0.94868330],
    [0.0, 0.94868330, 0.31622777],
]

t = [0.0, -1.26491106, 3.79473319]

foot_px = (960.0, 711.4)
head_px = (960.0, 489.8)

result = estimate_height(
    K=K,
    R=R,
    t=t,
    foot_px=foot_px,
    head_px=head_px,
    method="robust",
)

print("估计身高:", result["height"])
print("脚底3D坐标:", result["P_foot"])
print("头顶3D坐标:", result["P_head"])
```

返回结果 `result` 是一个字典，包含：

- `height`：估计身高
- `P_foot`：脚底三维坐标
- `P_head`：头顶三维坐标
- `method`：使用的估计方法
- `foot_px`：输入脚底像素
- `head_px`：输入头顶像素

### 4.2 方式二：已知相机位置和注视点

如果你暂时没有直接拿到 `R` 和 `t`，但知道相机装在哪里、镜头对准哪里，可以用这个接口。

```python
from src.estimate import estimate_height_from_lookat

cam_pos = [0.0, 0.0, 4.0]
target = [0.0, 12.0, 0.0]
K = [
    [1200.0, 0.0, 960.0],
    [0.0, 1200.0, 540.0],
    [0.0, 0.0, 1.0],
]

foot_px = (960.0, 711.4)
head_px = (960.0, 489.8)

result = estimate_height_from_lookat(
    cam_pos=cam_pos,
    target=target,
    K=K,
    foot_px=foot_px,
    head_px=head_px,
    method="robust",
)

print("估计身高:", result["height"])
```

说明：

- `target` 只用于推导相机朝向
- 真正参与身高计算的是内部推导得到的 `R` 和 `t`
- 如果你已经有准确标定结果，优先直接传 `R` 和 `t`

### 4.3 方式三：直接使用 HeightEstimator 类

如果你希望访问更底层的几何工具，比如投影、反投影、射线与地面求交，可以直接使用类接口。

```python
from src.height_estimator import HeightEstimator, build_K

K = build_K(fx=1200, fy=1200, cx=960, cy=540)

est = HeightEstimator.from_look_at(
    cam_pos=[0.0, 0.0, 4.0],
    target=[0.0, 12.0, 0.0],
    K=K,
    method="robust",
)

height, P_foot, P_head = est.estimate(
    foot_px=(960.0, 711.4),
    head_px=(960.0, 489.8),
)

print(height)
```

## 5. 命令行使用

`src/estimate.py` 支持三种命令行模式。

### 5.1 模式 0：不传参数，直接演示

```bash
python src/estimate.py
```

### 5.2 模式 A：传入 `cam_pos` 和 `target`

```bash
python src/estimate.py \
  --K "[[1200,0,960],[0,1200,540],[0,0,1]]" \
  --cam_pos "[0,0,4]" \
  --target "[0,12,0]" \
  --foot_px "[960,711.4]" \
  --head_px "[960,489.8]" \
  --method robust
```

### 5.3 模式 B：传入 `R` 和 `t`

```bash
python src/estimate.py \
  --K "[[1200,0,960],[0,1200,540],[0,0,1]]" \
  --R "[[1,0,0],[0,0.31622777,-0.9486833],[0,0.9486833,0.31622777]]" \
  --t "[0,-1.26491106,3.79473319]" \
  --foot_px "[960,711.4]" \
  --head_px "[960,489.8]" \
  --method robust
```

如果同时传入了两套相机参数：

- `--R` 和 `--t`
- `--cam_pos` 和 `--target`

程序会优先使用 `--R` 和 `--t`。

### 5.4 可选真实值对比

```bash
python src/estimate.py \
  --foot_px "[960,711.4]" \
  --head_px "[960,489.8]" \
  --true_height 1.70
```

这不会影响估计过程，只会在终端额外打印误差。

## 6. 参数格式说明

命令行参数统一使用 JSON 格式字符串。

### 6.1 相机内参矩阵 K

格式：

```text
[[fx, 0, cx],
 [0, fy, cy],
 [0,  0,  1]]
```

示例：

```text
[[1200,0,960],[0,1200,540],[0,0,1]]
```

### 6.2 旋转矩阵 R

格式：3×3 矩阵。

```text
[[r11,r12,r13],[r21,r22,r23],[r31,r32,r33]]
```

通常来自：

- OpenCV `solvePnP` 的旋转向量经 Rodrigues 转换后的结果
- 其他标定工具直接导出的旋转矩阵

### 6.3 平移向量 t

格式：

```text
[tx, ty, tz]
```

注意：

- 单位必须和你的世界坐标单位一致
- 如果世界坐标用米，输出身高也是米

### 6.4 像素坐标

脚底和头顶坐标格式统一为：

```text
[u, v]
```

其中：

- `u` 表示图像列坐标，向右增加
- `v` 表示图像行坐标，向下增加

## 7. 方法选择建议

项目支持两种身高估计方法：

- `robust`：默认方法，使用 X/Y 双轴联合最小二乘求解，真实场景更稳，推荐优先使用
- `single`：使用单个更稳定的轴求解，速度略快，适合低噪声或演示场景

建议：

- 真实部署、关键点有抖动时，用 `robust`
- 纯理论验证或需要保持和最简公式一致时，用 `single`

## 8. 如何与检测器结果对接

这个项目不负责人体检测或关键点检测，通常需要你先使用上游模型拿到脚底和头顶像素坐标，再把结果传入本项目。

典型流程如下：

1. 用检测器定位人体框或关键点
2. 提取脚底关键点像素坐标 `foot_px`
3. 提取头顶关键点像素坐标 `head_px`
4. 将标定参数和关键点坐标传给 `estimate_height()`
5. 读取返回的 `result["height"]`

对接时建议注意：

- 脚底点尽量取人体与地面的真实接触点
- 头顶点尽量取人体最高点，而不是眼睛或鼻子
- 如果头顶被遮挡，结果会明显变差
- 如果人物没有直立站立，结果会偏差较大

## 9. 验证与调试建议

如果你准备接入真实数据，建议先做下面几件事：

1. 先运行 `python src/validate.py`，确认环境和算法正常
2. 先用 `python src/estimate.py` 验证默认示例是否可正常输出
3. 再替换为你自己的 `K/R/t` 或 `cam_pos/target`
4. 最后接入真实检测器输出的脚底和头顶像素坐标

如果你希望切换验证脚本里使用的方法，可在 `src/validate.py` 中修改：

```python
ESTIMATION_METHOD = "robust"
```

可改为：

```python
ESTIMATION_METHOD = "single"
```

## 10. 常见问题

### 10.1 为什么结果单位是米

因为示例中的世界坐标单位采用米。这个算法的输出单位与输入世界坐标单位一致。

### 10.2 为什么远处的人误差更大

因为远距离目标在图像中占据的像素高度更小，同样的像素误差会被放大成更大的实际高度误差。

### 10.3 什么情况下结果会失效

以下情况会明显降低精度：

- 人不在同一地面平面上
- 人处于弯腰、跳跃、坐姿等非直立状态
- 相机外参不准确
- 镜头畸变未校正
- 脚底或头顶关键点检测错误

### 10.4 `cam_pos + target` 和 `R + t` 该用哪个

如果你已经做过严格标定，优先使用 `R + t`。如果你只有相机安装位置和朝向的近似信息，可以先用 `cam_pos + target` 快速搭建和验证。

## 11. 推荐使用顺序

如果你是第一次接触这个项目，推荐按下面顺序使用：

1. 阅读 `principle_intuition.md`，先理解算法直觉
2. 运行 `python src/validate.py`，确认理论正确性
3. 运行 `python src/estimate.py`，观察默认示例输出
4. 用你自己的相机参数和关键点结果替换示例参数
5. 如果需要进一步分析，再阅读 `principle.md`
