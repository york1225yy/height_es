"""
单目摄像头人体身高测量算法
=========================

【三种方案说明】
本模块提供两种身高估计方法，可通过 HeightEstimator(method=...) 参数选择：

  method="single"（单轴求解）
    - 从头顶射线的 X、Y 分量中选数值最大的一个来解方程
    - 速度快，无噪声时精度完美
    - 适合：相机已精确标定、关键点检测准确、对速度有要求的场景

  method="robust"（最小二乘，默认推荐）
    - 同时利用 X 和 Y 两个方向的约束联合求解
    - 在有检测噪声时误差更小、抖动更少
    - 适合：真实场景、视频流、关键点由检测器自动获取的通用场景

  homography_from_KRt（辅助工具函数，不是独立身高估计方法）
    - 仅计算地面平面的映射矩阵（像素 ↔ 地面坐标）
    - 不能独立完成身高估计，需配合上述两种方法使用
    - 适合：需要将地面坐标可视化、或已知单应性矩阵替代 K/R/t 的场合

坐标系约定:
  - 世界坐标系 (右手系): X 向右, Y 向前(深度), Z 向上
  - 地面平面: Z = 0（即所有脚底点的 Z 坐标为 0）
  - 相机坐标系 (OpenCV 标准): X 向右, Y 向下, Z 向前
"""

import numpy as np


# --------------------------------------------------------------------------- #
#  可选方法的合法取值，便于外部代码检查
# --------------------------------------------------------------------------- #
VALID_METHODS = ("single", "robust")


class HeightEstimator:
    """
    基于相机内外参数的人体身高估计器。

    Parameters
    ----------
    K : array-like (3, 3)
        相机内参矩阵:
            [[fx,  s, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
        fx, fy: 焦距(像素); cx, cy: 主点坐标; s: 畸变(通常为0)
    R : array-like (3, 3)
        旋转矩阵（世界坐标系 → 相机坐标系），须满足 R @ R^T = I
    t : array-like (3,) 或 (3,1)
        平移向量（相机坐标系下世界原点的位置）
    method : str, 默认 "robust"
        身高估计方法选择：
          "single" — 单轴求解（快速，无噪声时精度与 robust 相同）
          "robust" — 最小二乘双轴联合求解（有噪声时更稳定，推荐用于真实场景）
    """

    def __init__(self, K, R, t, method: str = "robust"):
        # ── 参数合法性检查 ──────────────────────────────────────────────
        if method not in VALID_METHODS:
            raise ValueError(f"method 必须是 {VALID_METHODS} 之一，收到: {method!r}")

        # ── 存储内参矩阵 ────────────────────────────────────────────────
        # K 是 3×3 矩阵，描述镜头的"放大规律"（焦距、主点）
        self.K = np.asarray(K, dtype=np.float64)

        # ── 存储旋转矩阵 ────────────────────────────────────────────────
        # R 描述相机的安装朝向（从世界坐标系旋转到相机坐标系）
        self.R = np.asarray(R, dtype=np.float64)

        # ── 存储平移向量，统一为 shape (3,) ────────────────────────────
        # t 描述相机光心的位置（用相机坐标系表达）
        self.t = np.asarray(t, dtype=np.float64).reshape(3)

        # ── 预计算内参逆矩阵 ────────────────────────────────────────────
        # K_inv 用于"反投影"：把像素坐标还原成相机坐标系中的方向向量
        # 预计算一次，后续每次反投影都复用，避免重复求逆
        self.K_inv = np.linalg.inv(self.K)

        # ── 计算相机光心在世界坐标系中的位置 ───────────────────────────
        # 推导：P_cam = R @ P_world + t
        #        当 P_world = C（光心）时 P_cam = 0（光心在相机坐标系原点）
        #        => 0 = R @ C + t  =>  C = -R^T @ t
        self.C = -self.R.T @ self.t  # shape (3,)

        # ── 记录当前使用哪种估计方法 ────────────────────────────────────
        self.method = method

    # ----------------------------------------------------------------------- #
    #  工厂方法（快速构造器）
    # ----------------------------------------------------------------------- #

    @classmethod
    def from_look_at(cls, cam_pos, target, K, method: str = "robust"):
        """
        通过"相机安装位置 + 注视点"构造估计器。

        这是最直观的构造方式：你只需告诉函数"相机装在哪里（cam_pos）"
        以及"镜头对准哪个地面位置（target）"，函数自动计算出旋转矩阵 R 和
        平移向量 t，无需手动标定外参。

        注意：target 仅用于确定相机朝向，不参与任何身高计算。

        Parameters
        ----------
        cam_pos : array-like (3,)
            相机光心在世界坐标系中的位置，格式 [x, y, z]
            例如 [0, 0, 4.0] 表示相机安装在地面正上方 4 米处
        target : array-like (3,)
            相机镜头对准的地面点（世界坐标），格式 [x, y, z]
            例如 [0, 12, 0] 表示镜头朝向正前方 12 米处的地面位置
            ★ 该点仅用于计算 R，计算完毕后不再使用
        K : array-like (3, 3)
            相机内参矩阵
        method : str, 默认 "robust"
            传递给 HeightEstimator.__init__ 的方法选择参数

        Returns
        -------
        HeightEstimator
        """
        cam_pos = np.asarray(cam_pos, dtype=np.float64)
        target  = np.asarray(target,  dtype=np.float64)

        # ── 计算相机"向前看"的方向（即相机 Z 轴） ─────────────────────
        # 用"注视点 - 光心位置"得到方向向量，再归一化为单位向量
        Z = target - cam_pos            # 方向向量（未归一化）
        norm = np.linalg.norm(Z)        # 向量的模（长度）
        if norm < 1e-9:
            # 两点重合时无法确定朝向
            raise ValueError("cam_pos 与 target 重合，无法确定相机朝向")
        Z /= norm                       # 归一化 → 相机 Z 轴（单位向量）

        # ── 计算相机"向右看"的方向（即相机 X 轴） ─────────────────────
        # 假设相机无横滚：X 轴与世界 X 方向 [1,0,0] 对齐
        # 但 X 必须与 Z 垂直，所以要减去 X 在 Z 上的投影（Gram-Schmidt 正交化）
        X = np.array([1.0, 0.0, 0.0])  # 初始取世界 X 方向
        X -= X.dot(Z) * Z              # 减去在 Z 方向上的分量，使 X ⊥ Z
        x_norm = np.linalg.norm(X)
        if x_norm < 1e-6:
            # 极端情况：相机正对世界 X 轴，X 轴退化为零向量
            raise ValueError("相机朝向与世界 X 轴平行，请调整 cam_pos/target")
        X /= x_norm                    # 归一化 → 相机 X 轴（单位向量）

        # ── 计算相机"向下看"的方向（即相机 Y 轴） ─────────────────────
        # OpenCV 相机 Y 轴朝下，由叉积 Z × X 得到（右手系）
        # 叉积自动保证 Y ⊥ X 且 Y ⊥ Z，且已经是单位向量（X、Z 都是单位向量）
        Y = np.cross(Z, X)

        # ── 构造旋转矩阵 R ─────────────────────────────────────────────
        # R 的每一行是一个相机坐标轴在世界坐标系下的方向
        # 行 0 = X（向右），行 1 = Y（向下），行 2 = Z（向前）
        R = np.array([X, Y, Z])        # shape (3, 3)

        # ── 计算平移向量 t ─────────────────────────────────────────────
        # 由 P_cam = R @ P_world + t，当 P_world = cam_pos 时 P_cam = 0
        # => t = -R @ cam_pos
        t = -(R @ cam_pos)

        return cls(K, R, t, method=method)

    # ----------------------------------------------------------------------- #
    #  公开接口：统一的估计入口
    # ----------------------------------------------------------------------- #

    def estimate(self, foot_px, head_px):
        """
        统一身高估计入口 —— 根据构造时选择的 method 自动调用对应算法。

        这是推荐使用的主接口。你不需要直接调用 _estimate_single 或
        _estimate_robust，直接调用 estimate() 即可，方法选择由初始化时的
        method 参数决定。

        Parameters
        ----------
        foot_px : (u, v)  脚底像素坐标（图像中脚与地面的接触点）
        head_px : (u, v)  头顶像素坐标（图像中人体最高点）

        Returns
        -------
        height : float         估计的身高（与世界坐标单位相同，通常为米）
        P_foot : ndarray (3,)  脚底的 3D 世界坐标 [x, y, 0]
        P_head : ndarray (3,)  头顶的 3D 世界坐标 [x, y, height]
        """
        if self.method == "single":
            return self._estimate_single(foot_px, head_px)
        else:  # "robust"
            return self._estimate_robust(foot_px, head_px)

    # ----------------------------------------------------------------------- #
    #  底层几何工具（供内部和高级用户使用）
    # ----------------------------------------------------------------------- #

    def project(self, P_world):
        """
        【正向投影】3D 世界坐标 → 2D 图像像素坐标。

        用途：主要用于验证（把已知 3D 点投影到图像，检查与检测结果是否吻合）。

        步骤：
          1. 用 R、t 把世界坐标转到相机坐标系：P_cam = R @ P_world + t
          2. 用 K 做透视除法投影到图像：[u, v] = K @ P_cam / P_cam[2]

        Parameters
        ----------
        P_world : array-like (3,)
            世界坐标 [x, y, z]

        Returns
        -------
        ndarray (2,)  像素坐标 [u, v]
        """
        P = np.asarray(P_world, dtype=np.float64)

        # 第一步：世界坐标 → 相机坐标
        # R 旋转 + t 平移，把世界坐标系里的点表示到相机坐标系中
        P_cam = self.R @ P + self.t      # shape (3,)

        # 检查点是否在相机前方（z_cam > 0 才能被看到）
        if P_cam[2] <= 0:
            raise ValueError(f"点 {P_world} 在相机后方 (z_cam={P_cam[2]:.4f})")

        # 第二步：相机坐标 → 图像像素坐标（透视投影）
        # K @ P_cam 得到齐次像素坐标 [u*w, v*w, w]，除以第三分量得到 [u, v]
        p = self.K @ P_cam               # 齐次像素坐标
        return p[:2] / p[2]              # 透视除法，返回 [u, v]

    def unproject_ray(self, u, v):
        """
        【反向投影】图像像素 (u, v) → 世界空间中的射线。

        "反投影"是投影的逆操作：一个像素对应的不是一个 3D 点，而是一条从
        相机光心出发、穿过该像素方向的无限长射线。

        射线表示为参数方程：P(s) = C + s * d，s > 0 时在相机前方。

        步骤：
          1. 用 K_inv 把像素坐标还原成相机坐标系中的方向向量 d_cam
          2. 用 R^T 把 d_cam 旋转到世界坐标系中得到 d_world
          3. 归一化 d_world 使其成为单位方向向量

        Parameters
        ----------
        u, v : float
            图像像素坐标

        Returns
        -------
        C : ndarray (3,)  射线起点 = 相机光心（世界坐标）
        d : ndarray (3,)  射线方向（单位向量，世界坐标）
        """
        # 第一步：像素坐标 → 相机坐标系中的方向向量
        # K_inv 把归一化的齐次像素 [u, v, 1] 变换回相机坐标系方向
        d_cam = self.K_inv @ np.array([u, v, 1.0])   # shape (3,)

        # 第二步：相机坐标系方向 → 世界坐标系方向
        # 旋转矩阵的转置 R^T 是 R 的逆，用它把相机方向转回世界方向
        d_world = self.R.T @ d_cam                    # shape (3,)

        # 第三步：归一化，变成单位方向向量（模长 = 1）
        # 归一化后射线参数 s 的单位与世界坐标相同（米）
        return self.C.copy(), d_world / np.linalg.norm(d_world)

    def intersect_ground(self, C, d):
        """
        【射线-地面求交】找到射线与地面平面（Z=0）的交点。

        地面是 Z=0 的水平面。射线 P(s) = C + s*d 与地面的交点满足
        P(s)[2] = 0，即 C[2] + s * d[2] = 0，解出 s = -C[2] / d[2]，
        代回射线方程即得交点坐标。

        Parameters
        ----------
        C : ndarray (3,)  射线起点（相机光心）
        d : ndarray (3,)  射线方向（单位向量）

        Returns
        -------
        P : ndarray (3,)  交点世界坐标（即脚底的 3D 位置）
        s : float         射线参数值（表示距相机光心的距离）
        """
        # 若 d[2] ≈ 0，射线近似水平，永远不会碰到地面（或极远处才相交）
        if abs(d[2]) < 1e-9:
            raise RuntimeError("射线近似平行于地面，无法求交点（相机安装过低或几乎水平）")

        # 解方程 C[2] + s * d[2] = 0，得到射线参数 s
        s = -C[2] / d[2]

        # s > 0 表示交点在相机前方；s ≤ 0 表示在相机后面，不合理
        if s <= 0:
            raise RuntimeError(f"地面交点在相机后方 (s={s:.4f})，请检查相机朝向")

        # 把 s 代回射线方程，得到 3D 交点坐标
        return C + s * d, float(s)

    # ----------------------------------------------------------------------- #
    #  两种核心估计算法（私有，通过 estimate() 调用）
    # ----------------------------------------------------------------------- #

    def _estimate_single(self, foot_px, head_px):
        """
        【方法一：单轴求解】用 X 或 Y 中数值最稳定的一个轴来解方程。

        原理：
          脚底已定位到地面，头顶在脚底正上方（X、Y 相同）。
          头顶射线方程 P_head = C + s * d_head，取其 X 或 Y 分量：
            s = (P_foot[i] - C[i]) / d_head[i]，i ∈ {X=0, Y=1}
          选 |d_head[i]| 最大的那个分量，分母越大数值越稳定（不易被噪声放大）。

        适用场景：理想条件（无噪声或噪声极小、标定精确）。

        Parameters
        ----------
        foot_px : (u, v)  脚底像素坐标
        head_px : (u, v)  头顶像素坐标

        Returns
        -------
        height : float, P_foot : ndarray(3,), P_head : ndarray(3,)
        """
        # ── Step 1：脚底射线 → 与地面相交 → 得到脚底 3D 坐标 ──────────
        C, d_foot = self.unproject_ray(*foot_px)   # 从脚底像素反投影得到射线
        P_foot, _ = self.intersect_ground(C, d_foot)  # 射线与地面 Z=0 的交点

        # ── Step 2：头顶射线方向 ────────────────────────────────────────
        _, d_head = self.unproject_ray(*head_px)   # 只需要方向，不需要起点（起点就是 C）

        # ── Step 3：用垂直约束求头顶射线参数 s ─────────────────────────
        # 枚举 X（i=0）和 Y（i=1）两个分量，分别计算对应的 s 值
        candidates = []
        for i in range(2):                         # i=0 是 X 轴，i=1 是 Y 轴
            if abs(d_head[i]) > 1e-10:             # 跳过接近零的分量（数值不稳定）
                s = (P_foot[i] - C[i]) / d_head[i] # 由该分量解出 s
                candidates.append((abs(d_head[i]), s))  # 记录 (分量绝对值, s)

        if not candidates:
            raise RuntimeError("头顶射线 XY 分量均接近零，无法求解（目标可能在图像中心正下方）")

        # 取 |d_head[i]| 最大的那组 s（分母最大 → 数值最稳定）
        s_head = max(candidates, key=lambda x: x[0])[1]

        # ── Step 4：由 s 计算头顶 3D 坐标，进而得到身高 ─────────────────
        P_head = C + s_head * d_head        # 头顶的 3D 世界坐标
        height = float(P_head[2] - P_foot[2])  # 身高 = 头顶 Z - 脚底 Z（脚底 Z=0）
        return height, P_foot, P_head

    def _estimate_robust(self, foot_px, head_px):
        """
        【方法二：最小二乘双轴联合求解（默认推荐）】

        原理：
          同时利用 X 和 Y 两个方向的约束，对 s 做加权最小二乘估计。
          当检测有像素噪声时，两个约束对同一个 s 的估计会略有偏差；
          最小二乘把两者的误差平方和最小化，得到更稳定的 s。

          数学上等价于把两个方程 d_head[0]*s ≈ Δr[0] 和 d_head[1]*s ≈ Δr[1]
          合并成一个超定方程组，用正规方程求解：
            s = (d_XY · ΔrXY) / |d_XY|²   （点积版最小二乘）

        适用场景：真实视频流，关键点由检测器自动获取（存在 1–5 px 噪声）。

        Parameters
        ----------
        foot_px : (u, v)  脚底像素坐标
        head_px : (u, v)  头顶像素坐标

        Returns
        -------
        height : float, P_foot : ndarray(3,), P_head : ndarray(3,)
        """
        # ── Step 1：脚底 → 地面交点 ────────────────────────────────────
        C, d_foot = self.unproject_ray(*foot_px)
        P_foot, _ = self.intersect_ground(C, d_foot)

        # ── Step 2：头顶射线方向 ────────────────────────────────────────
        _, d_head = self.unproject_ray(*head_px)

        # ── Step 3：最小二乘求解 s ──────────────────────────────────────
        # d_head[:2] 是头顶射线在 XY 平面上的分量（二维向量）
        # denom = |d_head_XY|² 是分母，也是最小二乘正规方程的归一化系数
        denom = np.dot(d_head[:2], d_head[:2])   # 标量：d_X² + d_Y²
        if denom < 1e-10:
            raise RuntimeError("头顶射线 XY 分量均接近零，无法求解（目标可能在图像中心正下方）")

        # P_foot[:2] - C[:2] 是脚底与光心在水平面的偏差向量（二维）
        delta_r = P_foot[:2] - C[:2]             # 从光心指向脚底的 XY 分量

        # 正规方程：s = dot(d_head_XY, delta_r) / |d_head_XY|²
        s_head = np.dot(d_head[:2], delta_r) / denom

        # ── Step 4：得到头顶坐标和身高 ─────────────────────────────────
        P_head = C + s_head * d_head             # 头顶 3D 坐标
        height = float(P_head[2] - P_foot[2])   # 身高 = Z 差值
        return height, P_foot, P_head

    # ----------------------------------------------------------------------- #
    #  向后兼容的公开别名（保留旧接口，内部转发给新方法）
    # ----------------------------------------------------------------------- #

    def estimate_height(self, foot_px, head_px):
        """向后兼容接口，等价于 method="single" 的 estimate()。"""
        return self._estimate_single(foot_px, head_px)

    def estimate_height_robust(self, foot_px, head_px):
        """向后兼容接口，等价于 method="robust" 的 estimate()。"""
        return self._estimate_robust(foot_px, head_px)

    # ----------------------------------------------------------------------- #
    #  辅助属性与魔法方法
    # ----------------------------------------------------------------------- #

    @property
    def camera_height(self):
        """相机光心距地面的高度（即 C 的 Z 分量，与世界坐标同单位）。"""
        return float(self.C[2])

    def __repr__(self):
        return (
            f"HeightEstimator(\n"
            f"  method={self.method!r},\n"
            f"  camera_pos={self.C.tolist()},\n"
            f"  K=\n{self.K}\n)"
        )


# =========================================================================== #
#  实用工具函数
# =========================================================================== #

def build_K(fx, fy, cx, cy, skew=0.0):
    """
    构造 3×3 相机内参矩阵。

    Parameters
    ----------
    fx, fy : float  水平/垂直方向的像素焦距（单位：像素）
    cx, cy : float  主点坐标（通常是图像宽/高的一半）
    skew   : float  像素畸变系数（几乎所有现代相机均为 0）

    Returns
    -------
    K : ndarray (3, 3)
    """
    return np.array([
        [fx,  skew, cx],   # 第一行：水平焦距、畸变、主点 u
        [ 0,    fy, cy],   # 第二行：垂直焦距、主点 v
        [ 0,     0,  1],   # 第三行：齐次坐标固定行
    ], dtype=np.float64)


def homography_from_KRt(K, R, t):
    """
    从内外参数计算地面平面 (Z=0) 的单应性矩阵 H。

    【这是辅助工具，不是独立的身高估计方法】
    单应性矩阵 H 只描述"地面点 ↔ 像素"的映射关系，
    不包含高度信息，不能单独用于身高估计。
    用途：地面坐标可视化、验证标定精度、配合外部单应性标定流程。

    数学推导：
      地面点满足 Z_w=0，代入投影方程后第三列消失：
        p_img ~ K @ [r1, r2, t] @ [X_w, Y_w, 1]^T
      其中 r1, r2 是旋转矩阵 R 的第一、二列。
      令 H = K @ [r1, r2, t]，则 p_img ~ H @ [X_w, Y_w, 1]^T。

    Parameters
    ----------
    K : ndarray (3,3)  内参矩阵
    R : ndarray (3,3)  旋转矩阵
    t : array-like (3,)  平移向量

    Returns
    -------
    H : ndarray (3, 3)  单应性矩阵（已归一化使 H[2,2]=1）
    """
    r1 = R[:, 0:1]          # R 的第一列（向右方向），shape (3,1)
    r2 = R[:, 1:2]          # R 的第二列（向下方向），shape (3,1)
    t_ = t.reshape(3, 1)    # 平移向量，shape (3,1)

    # 横向拼接三列，构成 3×3 矩阵，再乘以 K 得到单应性矩阵
    H = K @ np.hstack([r1, r2, t_])     # shape (3, 3)

    # 归一化：令 H[2,2]=1，保证齐次坐标表示的一致性
    return H / H[2, 2]
