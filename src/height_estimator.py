"""
单目摄像头人体身高测量算法
=========================

坐标系约定:
  - 世界坐标系 (右手系): X 向右, Y 向前(深度), Z 向上
  - 地面平面: Z = 0
  - 相机坐标系 (OpenCV 标准): X 向右, Y 向下, Z 向前

相机成像模型:
  P_cam = R @ P_world + t
  [u*w, v*w, w] = K @ P_cam
  [u, v] = 前两分量 / 第三分量
"""

import numpy as np


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
        fx, fy: 焦距 (像素); cx, cy: 主点; s: 畸变（通常为0）
    R : array-like (3, 3)
        旋转矩阵（世界 → 相机），满足 R @ R^T = I
    t : array-like (3,) 或 (3,1)
        平移向量（相机坐标系下的世界原点位置）
    """

    def __init__(self, K, R, t):
        self.K = np.asarray(K, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)
        self.t = np.asarray(t, dtype=np.float64).reshape(3)
        self.K_inv = np.linalg.inv(self.K)

        # 相机光心在世界坐标系中的位置: C_w = -R^T @ t
        self.C = -self.R.T @ self.t  # shape (3,)

    # ------------------------------------------------------------------ #
    #  工厂方法
    # ------------------------------------------------------------------ #
    @classmethod
    def from_look_at(cls, cam_pos, target, K):
        """
        通过相机安装位置和注视点构造估计器（无滚转，水平方向与世界 X 对齐）。

        Parameters
        ----------
        cam_pos : array-like (3,)
            相机光心在世界坐标系中的位置 [x, y, z]
        target  : array-like (3,)
            相机注视点的世界坐标 [x, y, z]
        K       : array-like (3, 3)
            相机内参矩阵

        Returns
        -------
        HeightEstimator
        """
        cam_pos = np.asarray(cam_pos, dtype=np.float64)
        target  = np.asarray(target,  dtype=np.float64)

        # 相机 Z 轴（向前）= 归一化方向向量
        Z = target - cam_pos
        norm = np.linalg.norm(Z)
        if norm < 1e-9:
            raise ValueError("cam_pos 与 target 重合")
        Z /= norm

        # 相机 X 轴（向右）= 世界 X 方向，正交化到 Z 轴
        X = np.array([1.0, 0.0, 0.0])
        X -= X.dot(Z) * Z
        x_norm = np.linalg.norm(X)
        if x_norm < 1e-6:
            raise ValueError("相机朝向与世界 X 轴平行，请检查 cam_pos/target 设置")
        X /= x_norm

        # 相机 Y 轴（向下）= Z × X（满足右手系 X × Y = Z）
        Y = np.cross(Z, X)

        # 构造旋转矩阵（行 = 相机基向量在世界坐标下的表示）
        R = np.array([X, Y, Z])
        t = -(R @ cam_pos)
        return cls(K, R, t)

    # ------------------------------------------------------------------ #
    #  核心投影与反投影
    # ------------------------------------------------------------------ #
    def project(self, P_world):
        """
        3D 世界坐标 → 2D 图像像素坐标。

        Parameters
        ----------
        P_world : array-like (3,)

        Returns
        -------
        (u, v) : ndarray (2,)
        """
        P = np.asarray(P_world, dtype=np.float64)
        P_cam = self.R @ P + self.t
        if P_cam[2] <= 0:
            raise ValueError(f"点 {P_world} 在相机后方 (z_cam={P_cam[2]:.4f})")
        p = self.K @ P_cam
        return p[:2] / p[2]

    def unproject_ray(self, u, v):
        """
        图像像素 (u, v) → 世界空间射线 (起点 C, 方向 d)。

        射线参数化: P(s) = C + s * d, s > 0

        Parameters
        ----------
        u, v : float
            图像像素坐标

        Returns
        -------
        C : ndarray (3,)  相机光心（世界坐标）
        d : ndarray (3,)  射线方向（已归一化，世界坐标）
        """
        d_cam = self.K_inv @ np.array([u, v, 1.0])
        d_world = self.R.T @ d_cam
        return self.C.copy(), d_world / np.linalg.norm(d_world)

    def intersect_ground(self, C, d):
        """
        射线与地面平面 Z=0 求交点。

        地面方程: n^T (P - p0) = 0, n=[0,0,1], p0=[0,0,0]
        代入射线: C[2] + s * d[2] = 0  =>  s = -C[2] / d[2]

        Returns
        -------
        P : ndarray (3,)  交点世界坐标
        s : float         射线参数（> 0 表示在相机前方）
        """
        if abs(d[2]) < 1e-9:
            raise RuntimeError("射线近似平行于地面，无法求交点")
        s = -C[2] / d[2]
        if s <= 0:
            raise RuntimeError(f"地面交点在相机后方 (s={s:.4f})")
        return C + s * d, float(s)

    # ------------------------------------------------------------------ #
    #  身高估计
    # ------------------------------------------------------------------ #
    def estimate_height(self, foot_px, head_px):
        """
        从头脚像素坐标估计人体身高（核心算法）。

        算法步骤
        --------
        1. 脚底反投影
           foot_px → 射线 → ∩ 地面 (Z=0) → P_foot (3D)

        2. 头顶反投影
           head_px → 射线方向 d_head

        3. 垂直约束（人体直立假设）
           P_head = C + s * d_head
           约束: P_head[0] = P_foot[0]  (X 相同)
                 P_head[1] = P_foot[1]  (Y 相同)
           → 选数值稳定的分量（|d_head[i]| 最大）解 s

        4. 身高 = P_head[2] - P_foot[2]

        Parameters
        ----------
        foot_px : (u, v)  脚底像素坐标
        head_px : (u, v)  头顶像素坐标

        Returns
        -------
        height  : float        估计身高（米，世界单位）
        P_foot  : ndarray(3,)  3D 脚底坐标
        P_head  : ndarray(3,)  3D 头顶坐标
        """
        # Step 1: 脚底 → 地面交点
        C, d_foot = self.unproject_ray(*foot_px)
        P_foot, _ = self.intersect_ground(C, d_foot)

        # Step 2: 头顶射线方向
        _, d_head = self.unproject_ray(*head_px)

        # Step 3: 垂直约束
        #   d_head[i] * s = P_foot[i] - C[i], i ∈ {0, 1}
        #   选 |d_head[i]| 最大的分量保证数值稳定
        candidates = []
        for i in range(2):
            if abs(d_head[i]) > 1e-10:
                s = (P_foot[i] - C[i]) / d_head[i]
                candidates.append((abs(d_head[i]), s))

        if not candidates:
            raise RuntimeError("头顶射线 XY 分量均接近零，无法求解")

        s_head = max(candidates, key=lambda x: x[0])[1]
        P_head = C + s_head * d_head

        height = float(P_head[2] - P_foot[2])
        return height, P_foot, P_head

    def estimate_height_robust(self, foot_px, head_px):
        """
        鲁棒版身高估计：同时利用 X、Y 两个约束做加权最小二乘。

        对于存在像素检测噪声的情况，比 estimate_height 更稳定。

        正规方程（1D 未知量 s）:
            s = (d_head[:2] · Δr) / ||d_head[:2]||²
            Δr = P_foot[:2] - C[:2]
        """
        C, d_foot = self.unproject_ray(*foot_px)
        P_foot, _ = self.intersect_ground(C, d_foot)

        _, d_head = self.unproject_ray(*head_px)

        denom = np.dot(d_head[:2], d_head[:2])
        if denom < 1e-10:
            raise RuntimeError("头顶射线 XY 分量均接近零")

        s_head = np.dot(d_head[:2], P_foot[:2] - C[:2]) / denom
        P_head = C + s_head * d_head

        height = float(P_head[2] - P_foot[2])
        return height, P_foot, P_head

    # ------------------------------------------------------------------ #
    #  辅助属性
    # ------------------------------------------------------------------ #
    @property
    def camera_height(self):
        """相机光心距地面高度 (Z 分量, 单位与世界坐标相同)"""
        return float(self.C[2])

    def __repr__(self):
        return (
            f"HeightEstimator(\n"
            f"  camera_pos={self.C.tolist()},\n"
            f"  K=\n{self.K}\n)"
        )


# ======================================================================= #
#  实用函数
# ======================================================================= #

def build_K(fx, fy, cx, cy, skew=0.0):
    """
    构造相机内参矩阵。

    Parameters
    ----------
    fx, fy : 像素焦距
    cx, cy : 主点坐标
    skew   : 畸变系数（通常为 0）
    """
    return np.array([
        [fx,  skew, cx],
        [ 0,    fy, cy],
        [ 0,     0,  1],
    ], dtype=np.float64)


def homography_from_KRt(K, R, t):
    """
    从内外参数计算地面平面 (Z=0) 到图像平面的单应性矩阵 H。

    H 满足: p_img ~ H @ [X_w, Y_w, 1]^T  (地面点 Z_w=0)

    推导:
      P_cam = R @ [X,Y,0]^T + t = [r1, r2, t] @ [X, Y, 1]^T
      H = K @ [r1, r2, t]     其中 r1, r2 是 R 的前两列
    """
    r1 = R[:, 0:1]  # (3,1)
    r2 = R[:, 1:2]  # (3,1)
    t_ = t.reshape(3, 1)
    H = K @ np.hstack([r1, r2, t_])  # (3,3)
    return H / H[2, 2]  # 归一化使 H[2,2]=1
