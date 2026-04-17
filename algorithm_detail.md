# 单目摄像头身高估计 — 算法推导详解

> **版本**: 1.0  
> **前置知识**: 线性代数、透视投影模型（见 `principle.md` 第 2–3 节）  
> **本文涵盖**: 射线方程基础 · 平面方程基础 · 射线-平面求交 · 最小二乘定位 · 完整推导与参数说明

---

## 目录

1. [基础知识一：三维射线方程](#1-基础知识一三维射线方程)
2. [基础知识二：平面方程与射线-平面求交](#2-基础知识二平面方程与射线-平面求交)
3. [基础知识三：最小二乘法与线性方程组求解](#3-基础知识三最小二乘法与线性方程组求解)
4. [基础知识四：反投影（Unprojection）](#4-基础知识四反投影unprojection)
5. [Step 1 — 脚底射线的构造](#5-step-1--脚底射线的构造)
6. [Step 2 — 脚底与地面的交点](#6-step-2--脚底与地面的交点)
7. [Step 3 — 头顶 3D 位置（垂直站立约束）](#7-step-3--头顶-3d-位置垂直站立约束)
8. [Step 4 — 身高计算](#8-step-4--身高计算)
9. [完整算法流程总览](#9-完整算法流程总览)
10. [参考公式汇总](#10-参考公式汇总)

---

## 1. 基础知识一：三维射线方程

### 1.1 射线的参数方程

三维空间中，一条**射线（Ray）**由以下两个元素唯一确定：

- **起点** $\mathbf{O}$（Origin）：射线出发的位置
- **方向向量** $\mathbf{d}$（Direction）：射线前进的方向

其参数方程为：

$$
\mathbf{P}(s) = \mathbf{O} + s \cdot \mathbf{d}, \quad s \geq 0
$$

| 符号 | 类型 | 含义 |
|------|------|------|
| $\mathbf{P}(s)$ | $\mathbb{R}^3$ 向量 | 射线上参数为 $s$ 时对应的 3D 点 |
| $\mathbf{O}$ | $\mathbb{R}^3$ 向量 | 射线起点（相机光心） |
| $\mathbf{d}$ | $\mathbb{R}^3$ 向量 | 射线方向（不要求单位向量，但通常归一化） |
| $s$ | 标量 $\geq 0$ | 沿射线前进的参数（深度尺度），$s=0$ 时退化到起点 |

**归一化方向向量**（单位射线方向）：

$$
\hat{\mathbf{d}} = \frac{\mathbf{d}}{\|\mathbf{d}\|}, \qquad \|\hat{\mathbf{d}}\| = 1
$$

此时参数 $s$ 的数值等于沿射线方向行进的**真实欧氏距离**（单位与坐标系一致）。

> **物理直觉**：在摄像头成像中，相机向场景发出无数条射线，每条对应图像上的一个像素点。三维世界中任何看得见的物体，都一定落在某条像素射线上。

### 1.2 射线方向的三个分量

将 $\mathbf{d}$ 展开为分量形式：

$$
\mathbf{d} = \begin{bmatrix} d_X \\ d_Y \\ d_Z \end{bmatrix}
$$

| 分量 | 含义（世界坐标系，$Z$ 朝上） |
|------|------|
| $d_X$ | 射线在水平横向（右）的方向分量 |
| $d_Y$ | 射线在水平纵深（前）的方向分量 |
| $d_Z$ | 射线在垂直方向（上）的方向分量；若 $d_Z < 0$ 则射线朝下，可与地面相交 |

---

## 2. 基础知识二：平面方程与射线-平面求交

### 2.1 平面的点法式方程

三维空间中，一个**平面（Plane）**由以下两个元素唯一确定：

- **法向量** $\mathbf{n}$：垂直于平面的方向向量
- **平面上一点** $\mathbf{Q}$

其**点法式方程**为：

$$
\mathbf{n} \cdot (\mathbf{P} - \mathbf{Q}) = 0 \quad \Longleftrightarrow \quad \mathbf{n} \cdot \mathbf{P} = \mathbf{n} \cdot \mathbf{Q} = d
$$

| 符号 | 类型 | 含义 |
|------|------|------|
| $\mathbf{n}$ | $\mathbb{R}^3$ 向量 | 平面法向量（垂直于平面的方向） |
| $\mathbf{Q}$ | $\mathbb{R}^3$ 向量 | 平面上任意已知点 |
| $d$ | 标量 | 法向量与已知点的内积（平面的截距） |
| $\mathbf{P}$ | $\mathbb{R}^3$ 向量 | 平面上任意点（变量） |

**本文中的地面平面（Ground Plane）**：

$$
\text{法向量} \quad \mathbf{n} = [0, 0, 1]^T, \quad \text{过原点} \quad \mathbf{Q} = [0,0,0]^T
$$

因此地面方程简化为：

$$
Z_w = 0
$$

### 2.2 射线与平面的交点推导

将射线参数方程代入平面方程：

$$
\mathbf{n} \cdot \big(\mathbf{O} + s \cdot \mathbf{d}\big) = d
$$

展开并求解参数 $s^*$：

$$
\mathbf{n} \cdot \mathbf{O} + s \cdot (\mathbf{n} \cdot \mathbf{d}) = d
\implies
\boxed{s^* = \frac{d - \mathbf{n} \cdot \mathbf{O}}{\mathbf{n} \cdot \mathbf{d}}}
$$

| 符号 | 含义 |
|------|------|
| $d$ | 平面截距（$= \mathbf{n} \cdot \mathbf{Q}$） |
| $\mathbf{n} \cdot \mathbf{O}$ | 射线起点到平面法向方向的分量 |
| $\mathbf{n} \cdot \mathbf{d}$ | 射线方向在法向量上的投影（**不能为零**，否则射线与平面平行） |
| $s^*$ | 交点所对应的射线参数；$s^* > 0$ 才是有效交点（射线朝向平面） |

**对地面（$Z_w=0$）的特化**：取 $\mathbf{n} = [0,0,1]^T$，$d = 0$，$\mathbf{O} = \mathbf{C}_w$（相机光心）：

$$
s^* = \frac{0 - C_Z}{d_Z} = -\frac{C_Z}{d_Z}
$$

**数值稳定性条件**：
- $d_Z \neq 0$：射线方向必须有垂直分量，否则射线平行于地面，无交点
- $s^* > 0$：要求 $C_Z > 0$（相机高于地面）且 $d_Z < 0$（射线朝下），符合实际监控场景

---

## 3. 基础知识三：最小二乘法与线性方程组求解

### 3.1 超定方程组

在 Step 3 中，需要用**已知的脚底位置**和**头顶射线方向**，求解头顶的 3D 坐标。  
这涉及到从多个约束方程（$X$ 方向和 $Y$ 方向各一个）解出一个未知参数 $s$，这是一个**超定线性方程组**（方程数 > 未知数）：

$$
\begin{cases}
d_{head,X} \cdot s = P_{foot,X} - C_X \\
d_{head,Y} \cdot s = P_{foot,Y} - C_Y
\end{cases}
\implies
\underbrace{\begin{bmatrix} d_{head,X} \\ d_{head,Y} \end{bmatrix}}_{\mathbf{A}} s = \underbrace{\begin{bmatrix} P_{foot,X} - C_X \\ P_{foot,Y} - C_Y \end{bmatrix}}_{\mathbf{b}}
$$

| 符号 | 维度 | 含义 |
|------|------|------|
| $\mathbf{A}$ | $2 \times 1$ | 系数矩阵（头顶射线方向的 $XY$ 分量） |
| $\mathbf{b}$ | $2 \times 1$ | 右端向量（脚底位置相对光心的 $XY$ 偏移） |
| $s$ | 标量 | 待求的射线参数 |

### 3.2 最小二乘解（Normal Equation）

对于超定系统 $\mathbf{A} s = \mathbf{b}$，**最小二乘解**最小化残差的平方和 $\|\mathbf{A}s - \mathbf{b}\|^2$，由正规方程给出：

$$
\mathbf{A}^T \mathbf{A} \, s = \mathbf{A}^T \mathbf{b}
\implies
s^* = \frac{\mathbf{A}^T \mathbf{b}}{\mathbf{A}^T \mathbf{A}} = \frac{\mathbf{A} \cdot \mathbf{b}}{\|\mathbf{A}\|^2}
$$

| 符号 | 含义 |
|------|------|
| $\mathbf{A}^T \mathbf{A}$ | 系数矩阵的 Gram 矩阵；对于 $2\times1$ 向量，等于 $d_{head,X}^2 + d_{head,Y}^2$ |
| $\mathbf{A}^T \mathbf{b}$ | 系数矩阵与右端向量的内积 |
| $s^*$ | 最优射线参数（最小化来自两个方向约束的联合误差） |

展开后：

$$
\boxed{
s^* = \frac{d_{head,X}(P_{foot,X} - C_X) + d_{head,Y}(P_{foot,Y} - C_Y)}{d_{head,X}^2 + d_{head,Y}^2}
}
$$

**直观理解**：如果两个约束方程（$X$ 和 $Y$）完全一致（理想情况下无噪声），解是精确的；有噪声时，最小二乘在两者之间取最优折中，对每个方程都贡献权重，权重正比于该方向上射线参数的"灵敏度"（即 $d_{head,i}^2$）。

### 3.3 奇异值分解（SVD）补充

对于更一般的线性系统，可使用**奇异值分解（SVD）**求最小二乘解——这也是 `np.linalg.lstsq` 的内部算法：

$$
\mathbf{A} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T
\implies
s^* = \mathbf{V} \boldsymbol{\Sigma}^+ \mathbf{U}^T \mathbf{b}
$$

| 符号 | 含义 |
|------|------|
| $\mathbf{U}$ | 左奇异向量矩阵（正交） |
| $\boldsymbol{\Sigma}$ | 对角奇异值矩阵 |
| $\mathbf{V}$ | 右奇异向量矩阵（正交） |
| $\boldsymbol{\Sigma}^+$ | 伪逆：对非零奇异值取倒数 |

> 本算法中的超定系统只有 1 个未知数，直接使用正规方程即可，SVD 作为通用背景知识补充。

---

## 4. 基础知识四：反投影（Unprojection）

### 4.1 什么是反投影

**投影（Projection）**：将三维世界点 → 二维像素坐标  
**反投影（Unprojection）**：将二维像素坐标 → 三维射线（无法唯一确定 3D 点，只能确定一条射线）

这是因为投影是多对一的映射（深度信息丢失），反投影得到的是所有可能世界点构成的**一条射线**。

### 4.2 内参矩阵 $K$ 的逆

内参矩阵 $K$ 定义了从相机坐标系到像素坐标系的线性映射：

$$
K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
\implies
K^{-1} = \begin{bmatrix} 1/f_x & 0 & -c_x/f_x \\ 0 & 1/f_y & -c_y/f_y \\ 0 & 0 & 1 \end{bmatrix}
$$

| 符号 | 含义 |
|------|------|
| $f_x, f_y$ | 水平/垂直像素焦距（单位：像素），$f_x \approx f_y$ 时为方形像素 |
| $c_x, c_y$ | 主点坐标（光轴与像平面的交点，通常接近图像中心） |
| $K^{-1}$ | 内参逆矩阵，用于从像素坐标计算相机坐标系方向向量 |

### 4.3 反投影公式

给定像素坐标 $(u, v)$，首先得到**相机坐标系下的射线方向**：

$$
\mathbf{d}_{cam} = K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \begin{bmatrix} (u - c_x)/f_x \\ (v - c_y)/f_y \\ 1 \end{bmatrix}
$$

| 符号 | 含义 |
|------|------|
| $(u, v)$ | 像素坐标（水平列索引，垂直行索引） |
| $\mathbf{d}_{cam}$ | 在相机坐标系下的方向向量，$Z$ 分量始终为 1（沿主光轴方向归一化） |

再通过旋转矩阵 $R$ 将相机坐标系方向转换为**世界坐标系方向**：

$$
\mathbf{d}_{world} = R^T \mathbf{d}_{cam}
$$

| 符号 | 含义 |
|------|------|
| $R$ | 外参旋转矩阵（$3\times3$，正交矩阵，将世界向量转到相机坐标系） |
| $R^T$ | $R$ 的转置，等于 $R^{-1}$（由于 $R$ 是正交矩阵），将相机坐标系向量转回世界坐标系 |
| $\mathbf{d}_{world}$ | 同一方向在世界坐标系下的表示 |

---

## 5. Step 1 — 脚底射线的构造

### 5.1 直觉说明

假设从相机光心向脚底像素点 $(u_f, v_f)$ 张一条射线。这条射线穿过图像平面上该像素后，继续延伸进入三维场景，最终打到地面（$Z_w = 0$）上的某一点——即**行人脚底的 3D 位置**。

```
相机光心 C
   \
    \  <-- 脚底射线
     \
      \
-------+-------  Z_w = 0 (地面)
      P_foot
```

### 5.2 计算步骤

**Step 1a：像素坐标 → 相机坐标系方向**

$$
\mathbf{d}_{cam}^{foot} = K^{-1} \begin{bmatrix} u_f \\ v_f \\ 1 \end{bmatrix}
$$

**Step 1b：相机坐标系方向 → 世界坐标系方向**

$$
\mathbf{d}_{world}^{foot} = R^T \mathbf{d}_{cam}^{foot}
$$

**Step 1c：归一化（可选，但推荐）**

$$
\hat{\mathbf{d}}_{foot} = \frac{\mathbf{d}_{world}^{foot}}{\|\mathbf{d}_{world}^{foot}\|}
$$

**Step 1d：相机光心（射线起点）**

$$
\mathbf{C}_w = -R^T \mathbf{t}
$$

| 符号 | 类型 | 含义 |
|------|------|------|
| $(u_f, v_f)$ | 像素坐标 | 脚底在图像中的位置（由姿态估计或目标检测提供） |
| $\mathbf{d}_{cam}^{foot}$ | $\mathbb{R}^3$ | 脚底射线在相机坐标系中的方向向量 |
| $\mathbf{d}_{world}^{foot}$ | $\mathbb{R}^3$ | 脚底射线在世界坐标系中的方向向量 |
| $\hat{\mathbf{d}}_{foot}$ | $\mathbb{R}^3$（单位） | 归一化后的脚底射线方向 |
| $\mathbf{C}_w$ | $\mathbb{R}^3$ | 相机光心在世界坐标系中的坐标（射线起点） |
| $R$ | $3\times3$ 矩阵 | 外参旋转矩阵 |
| $\mathbf{t}$ | $\mathbb{R}^3$ | 外参平移向量（相机坐标系下世界原点的位置） |

**完整射线方程**：

$$
\mathbf{P}(s) = \mathbf{C}_w + s \cdot \hat{\mathbf{d}}_{foot}, \quad s \geq 0
$$

---

## 6. Step 2 — 脚底与地面的交点

### 6.1 代入地面方程

将脚底射线代入地面方程 $Z_w = 0$：

$$
C_Z + s \cdot \hat{d}_{foot,Z} = 0
$$

解出参数 $s^*$：

$$
s^* = -\frac{C_Z}{\hat{d}_{foot,Z}}
$$

### 6.2 求脚底 3D 坐标

$$
\boxed{
\mathbf{P}_{foot} = \mathbf{C}_w + s^* \cdot \hat{\mathbf{d}}_{foot}
= \mathbf{C}_w - \frac{C_Z}{\hat{d}_{foot,Z}} \hat{\mathbf{d}}_{foot}
}
$$

| 符号 | 类型 | 含义 |
|------|------|------|
| $C_Z$ | 标量 | 相机光心的高度（世界坐标系 $Z$ 分量），即摄像头离地面的高度，单位：米，典型值 3–5 m |
| $\hat{d}_{foot,Z}$ | 标量 | 脚底射线方向在 $Z$ 轴上的分量；朝下时为负值 |
| $s^*$ | 标量 $> 0$ | 射线上脚底对应的参数值（距离尺度） |
| $\mathbf{P}_{foot}$ | $\mathbb{R}^3$ | 脚底在世界坐标系中的 3D 位置，第三分量 $P_{foot,Z} = 0$（在地面上） |

### 6.3 有效性检验

| 条件 | 物理含义 | 处理方式 |
|------|----------|----------|
| $C_Z > 0$ | 相机在地面上方（正常安装） | 标定时确认 |
| $\hat{d}_{foot,Z} < 0$ | 射线朝下（脚底在图像中位于地平线以下） | 若违反则跳过该目标，记录警告 |
| $s^* > 0$ | 相交点在射线前方（正常） | 由以上两条自动保证 |

---

## 7. Step 3 — 头顶 3D 位置（垂直站立约束）

### 7.1 约束来源

对头顶像素 $(u_h, v_h)$ 同理得到世界坐标系下的射线方向 $\hat{\mathbf{d}}_{head}$，头顶 3D 点满足：

$$
\mathbf{P}_{head}(s) = \mathbf{C}_w + s \cdot \hat{\mathbf{d}}_{head}
$$

但仅凭射线无法唯一确定 $s$，还需要**垂直站立约束**：

> 行人直立时，头顶与脚底具有相同的水平位置（$X$ 和 $Y$ 坐标相同），只有高度 $Z$ 不同。

即：

$$
P_{head,X} = P_{foot,X}, \qquad P_{head,Y} = P_{foot,Y}
$$

### 7.2 建立方程组

将约束代入射线方程，展开各分量：

$$
\begin{cases}
C_X + s \cdot \hat{d}_{head,X} = P_{foot,X} \\
C_Y + s \cdot \hat{d}_{head,Y} = P_{foot,Y}
\end{cases}
\implies
\begin{cases}
\hat{d}_{head,X} \cdot s = P_{foot,X} - C_X \\
\hat{d}_{head,Y} \cdot s = P_{foot,Y} - C_Y
\end{cases}
$$

| 符号 | 类型 | 含义 |
|------|------|------|
| $(u_h, v_h)$ | 像素坐标 | 头顶在图像中的位置 |
| $\hat{\mathbf{d}}_{head}$ | $\mathbb{R}^3$（单位） | 归一化后的头顶射线世界方向向量 |
| $\hat{d}_{head,X}, \hat{d}_{head,Y}$ | 标量 | 头顶射线在水平面上的方向分量（对 $s$ 的可观测贡献） |
| $P_{foot,X}, P_{foot,Y}$ | 标量 | 已由 Step 2 求得的脚底水平坐标 |
| $C_X, C_Y$ | 标量 | 相机光心在世界坐标系 $X, Y$ 方向的坐标 |
| $s$ | 标量 | 头顶射线的参数（待求） |

### 7.3 最小二乘求解

写成矩阵形式 $\mathbf{A} s = \mathbf{b}$：

$$
\mathbf{A} = \begin{bmatrix} \hat{d}_{head,X} \\ \hat{d}_{head,Y} \end{bmatrix}, \quad
\mathbf{b} = \begin{bmatrix} P_{foot,X} - C_X \\ P_{foot,Y} - C_Y \end{bmatrix}
$$

由正规方程 $s^* = (\mathbf{A}^T\mathbf{A})^{-1} \mathbf{A}^T \mathbf{b}$，展开得：

$$
\boxed{
s^* = \frac{\hat{d}_{head,X}(P_{foot,X} - C_X) + \hat{d}_{head,Y}(P_{foot,Y} - C_Y)}{\hat{d}_{head,X}^2 + \hat{d}_{head,Y}^2}
}
$$

**分母解释**：$\hat{d}_{head,X}^2 + \hat{d}_{head,Y}^2 = \|\hat{\mathbf{d}}_{head}^{XY}\|^2$，即头顶射线方向在水平面上的投影长度的平方。

**数值稳定性**：若头顶射线在水平面上投影接近零（即头顶像素接近图像中心正下方，射线几乎朝正前方），分母趋近于零，求解不稳定。实际应用中需设置最小阈值（如 $> 10^{-6}$）予以过滤。

### 7.4 求头顶 3D 坐标

$$
\mathbf{P}_{head} = \mathbf{C}_w + s^* \cdot \hat{\mathbf{d}}_{head}
$$

---

## 8. Step 4 — 身高计算

### 8.1 身高定义

身高 $H$ 定义为头顶与脚底在垂直方向（$Z$ 轴）上的差值：

$$
H = P_{head,Z} - P_{foot,Z}
$$

### 8.2 化简

由于脚底已确认在地面上，$P_{foot,Z} = 0$，故：

$$
\boxed{
H = P_{head,Z} = C_Z + s^* \cdot \hat{d}_{head,Z}
}
$$

| 符号 | 类型 | 含义 |
|------|------|------|
| $H$ | 标量（米） | 最终估计的身高 |
| $P_{foot,Z}$ | 标量 $= 0$ | 脚底的世界高度，定义为 0（地面） |
| $P_{head,Z}$ | 标量 $> 0$ | 头顶的世界高度（预期正值，约 1.5–2.1 m） |
| $C_Z$ | 标量 | 相机光心高度（安装高度） |
| $s^*$ | 标量 | Step 3 求得的头顶射线参数 |
| $\hat{d}_{head,Z}$ | 标量 | 头顶射线方向在 $Z$ 轴上的分量 |

**有效性检验**：$H$ 应在合理范围内，例如 $[1.0, 2.5]$ 米。超出范围时说明检测点异常或相机参数有误。

---

## 9. 完整算法流程总览

```
输入：
  相机参数: K, R, t
  脚底像素: (u_f, v_f)
  头顶像素: (u_h, v_h)

┌─────────────────────────────────────────────────────────────────┐
│ 预计算（每次标定后只需计算一次）                                   │
│   C_w = -R^T t              （相机光心世界坐标）                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: 构造脚底射线                                              │
│   d_cam_foot = K^{-1} [u_f, v_f, 1]^T                          │
│   d_world_foot = R^T d_cam_foot                                 │
│   d_hat_foot = d_world_foot / ||d_world_foot||                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: 脚底射线与地面求交                                        │
│   s_foot = -C_Z / d_hat_foot_Z      （需检验 d_hat_foot_Z < 0）  │
│   P_foot = C_w + s_foot * d_hat_foot                            │
│   （验证 P_foot_Z ≈ 0）                                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: 构造头顶射线 + 垂直站立约束求解                           │
│   d_cam_head = K^{-1} [u_h, v_h, 1]^T                          │
│   d_world_head = R^T d_cam_head                                 │
│   d_hat_head = d_world_head / ||d_world_head||                  │
│                                                                  │
│   s_head = (d_hat_head_X*(P_foot_X - C_X)                      │
│           + d_hat_head_Y*(P_foot_Y - C_Y))                      │
│           / (d_hat_head_X^2 + d_hat_head_Y^2)                   │
│                                                                  │
│   P_head = C_w + s_head * d_hat_head                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 4: 身高计算                                                  │
│   H = P_head_Z - P_foot_Z = P_head_Z    （因 P_foot_Z = 0）     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
输出：H（估计身高，单位：米）
```

---

## 10. 参考公式汇总

$$
\begin{aligned}
\text{相机光心:} \quad & \mathbf{C}_w = -R^T \mathbf{t} \\[6pt]
\text{反投影方向（世界系）:} \quad & \hat{\mathbf{d}} = \frac{R^T K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}}{\left\|R^T K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}\right\|} \\[10pt]
\text{脚底参数:} \quad & s_{foot}^* = -\frac{C_Z}{\hat{d}_{foot,Z}} \\[6pt]
\text{脚底3D位置:} \quad & \mathbf{P}_{foot} = \mathbf{C}_w + s_{foot}^* \hat{\mathbf{d}}_{foot} \\[6pt]
\text{头顶射线参数（最小二乘）:} \quad & s_{head}^* = \frac{\hat{\mathbf{d}}_{head}^{XY} \cdot (\mathbf{P}_{foot}^{XY} - \mathbf{C}_w^{XY})}{\left\|\hat{\mathbf{d}}_{head}^{XY}\right\|^2} \\[6pt]
\text{估计身高:} \quad & H = C_Z + s_{head}^* \cdot \hat{d}_{head,Z}
\end{aligned}
$$

其中上标 $XY$ 表示取向量的前两个分量（水平面分量）。

---

*文档结束 — 如需进一步了解误差分析或实现细节，请参阅 `principle.md` 第 6–8 节。*
