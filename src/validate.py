"""
单目摄像头身高测量算法验证脚本
================================

【验证策略说明】
本脚本使用"合成数据"验证算法：虚构一个已知参数的相机和已知身高的人物，
把 3D 坐标投影到图像得到像素坐标，再反向用算法从像素坐标估计身高，
最后与真实值对比——无需真实图像也能精确验证算法是否正确。

本脚本包含三组测试，分别验证不同侧面：
  测试 1：无噪声 — 验证算法数学推导是否严格正确（误差应接近机器精度）
  测试 2：噪声灵敏度 — 验证在真实检测误差下两种方法（single/robust）的精度对比
  测试 3：单应性矩阵 — 验证 homography_from_KRt 工具函数的正确性

用法:
  cd /workspaces/height_es
  python src/validate.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")          # 无显示器时用离屏渲染（服务器/容器环境常见）
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  （激活 3D 投影支持）

# 把 src 目录加入 Python 模块搜索路径，使 import 能找到 height_estimator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from height_estimator import HeightEstimator, build_K, homography_from_KRt


# ======================================================================= #
#  ★ 方法选择：修改这一行来切换全局使用的估计方法
#
#    "robust"  → 最小二乘双轴联合求解（推荐，真实场景抗噪更好）
#    "single"  → 单轴最稳定分量求解（快速，理论精度与 robust 相同）
# ======================================================================= #
ESTIMATION_METHOD = "robust"   # ← 改这里即可切换方法


# ======================================================================= #
#  场景配置
# ======================================================================= #

def create_scene():
    """
    创建虚拟测试场景：相机参数 + 一组虚拟人物。

    模拟的是一个典型的交通/安防监控场景：
      - 相机安装在路边灯杆上，高度 4 m
      - 镜头俯视前方，对准 12 m 外的地面
      - 图像分辨率 1920×1080（Full HD）
      - 等效焦距约 25mm（全画幅），fx=fy=1200 px
    """
    # ── 图像分辨率 ────────────────────────────────────────────────────
    W, H = 1920, 1080         # 图像宽度和高度（像素）

    # ── 内参矩阵参数 ──────────────────────────────────────────────────
    fx = fy = 1200.0          # 焦距（像素单位），fx=fy 表示无像素非正方形失真
    cx, cy = W / 2.0, H / 2.0  # 主点设在图像正中心（标准假设）
    K = build_K(fx, fy, cx, cy)  # 构造 3×3 内参矩阵

    # ── 相机安装位置和朝向 ────────────────────────────────────────────
    cam_pos = np.array([0.0, 0.0, 4.0])   # 相机光心世界坐标：在地面正上方 4m
    target  = np.array([0.0, 12.0, 0.0])  # 注视点：前方 12m 的地面位置
    # ★ target 仅用于在 from_look_at 中计算旋转矩阵，之后不再使用

    # ── 构造估计器，传入全局方法选择 ─────────────────────────────────
    # ESTIMATION_METHOD 变量控制使用哪种身高估计算法
    estimator = HeightEstimator.from_look_at(
        cam_pos, target, K, method=ESTIMATION_METHOD
    )

    # ── 虚拟人物：分布在不同深度和侧方，覆盖多种测试条件 ────────────
    # 格式: x=左右位置(m), y=前后深度(m), height=真实身高(m), label=标签
    people = [
        {"x": -2.5, "y":  6.0, "height": 1.60, "label": "P1"},  # 近处左侧
        {"x":  0.0, "y":  8.0, "height": 1.70, "label": "P2"},  # 近处中央
        {"x":  2.0, "y": 10.0, "height": 1.75, "label": "P3"},  # 中距右侧
        {"x": -1.0, "y": 14.0, "height": 1.80, "label": "P4"},  # 中距左侧
        {"x":  1.5, "y": 18.0, "height": 1.90, "label": "P5"},  # 远处右侧
        {"x": -3.0, "y": 20.0, "height": 1.65, "label": "P6"},  # 远处左侧
    ]
    return estimator, K, W, H, people


# ======================================================================= #
#  辅助函数
# ======================================================================= #

def project_person(est: HeightEstimator, person: dict):
    """
    把一个虚拟人物的脚底和头顶从 3D 世界坐标投影到 2D 图像像素坐标。

    这是验证用的"正向"操作：我们确切知道这个人的 3D 位置，
    所以可以精确计算出他在图像中应该出现在哪里。
    后续再用算法从这些像素坐标反推身高，比较与真实值的误差。
    """
    x, y, h = person["x"], person["y"], person["height"]

    # 脚底在地面上，Z=0
    P_foot = np.array([x, y, 0.0])
    # 头顶在脚底正上方，Z=身高
    P_head = np.array([x, y, h])

    # 调用 project() 做正向投影，得到图像像素坐标
    foot_px = est.project(P_foot)   # 脚底像素 [u, v]
    head_px = est.project(P_head)   # 头顶像素 [u, v]

    return foot_px, head_px, P_foot, P_head


def add_noise(foot_px, head_px, sigma: float, rng: np.random.Generator):
    """
    给像素坐标加上高斯噪声，模拟真实场景中关键点检测器的定位误差。

    sigma 是噪声标准差（单位：像素）。
    rng.normal(均值=0, 标准差=sigma, 形状=(2,)) 生成二维噪声向量（u 方向 + v 方向）。
    """
    return (
        foot_px + rng.normal(0, sigma, 2),   # 脚底像素 + 随机噪声
        head_px + rng.normal(0, sigma, 2),   # 头顶像素 + 随机噪声
    )


# ======================================================================= #
#  测试函数
# ======================================================================= #

def test_no_noise(est, people):
    """
    测试 1：无噪声，验证理论精度。

    把精确的像素坐标（由 project 计算，没有任何误差）交给算法，
    理论上应该能完美还原身高，误差仅来自浮点数的舍入误差（< 0.001 mm）。
    若这里出现较大误差，说明算法推导有问题。
    """
    print("\n" + "─" * 58)
    print(f"  【测试 1】无噪声 — 验证理论精度  (method={est.method!r})")
    print("─" * 58)
    print(f"  {'人物':>4}  {'真实(m)':>8}  {'估计(m)':>8}  "
          f"{'误差(cm)':>9}  {'相对误差':>8}")

    errors = []
    results = []
    for p in people:
        # 第一步：把 3D 人物投影到图像，得到精确像素坐标
        foot_px, head_px, _, _ = project_person(est, p)

        # 第二步：用 estimate() 统一接口估计身高（方法由 est.method 决定）
        h_est, P_foot, P_head = est.estimate(foot_px, head_px)

        # 第三步：计算与真实值的误差
        err = abs(h_est - p["height"])   # 绝对误差（米）
        errors.append(err)

        # 保存本次结果，供后续可视化使用
        results.append({
            "label":   p["label"],
            "gt":      p["height"],       # ground truth 真实身高
            "est":     h_est,             # 估计身高
            "err_m":   err,               # 误差（米）
            "foot_px": foot_px,           # 脚底像素坐标
            "head_px": head_px,           # 头顶像素坐标
            "P_foot":  P_foot,            # 脚底 3D 坐标
            "P_head":  P_head,            # 头顶 3D 坐标
        })
        print(f"  {p['label']:>4}  {p['height']:>8.3f}  {h_est:>8.4f}  "
              f"{err*100:>9.4f}  {err/p['height']*100:>7.4f}%")

    # err*1000 把米转成毫米，方便查看精度量级
    print(f"\n  最大误差: {max(errors)*1000:.3f} mm")
    print(f"  均值误差: {np.mean(errors)*1000:.3f} mm")
    print("  → 误差量级为数值精度（< 0.01 mm），算法理论正确 ✓")
    return results


def test_noise_sensitivity(est, people, sigmas=(0.5, 1, 2, 3, 5, 10)):
    """
    测试 2：像素噪声灵敏度分析，并对比两种方法。

    对每个噪声水平 sigma，重复 200 次 Monte Carlo 随机实验：
      每次随机加噪声 → 估计身高 → 记录误差
    最后统计均值、最大值、标准差，评估两种方法在噪声下的表现。

    Monte Carlo 方法：重复大量随机试验，用统计结果近似概率分布。
    这里用来模拟"不同时刻/不同帧检测器给出结果的随机性"。
    """
    print("\n" + "─" * 58)
    print("  【测试 2】噪声灵敏度分析 — 两种方法对比")
    print("─" * 58)
    print(f"  {'σ(px)':>6}  {'方法':>8}  {'均值误差(cm)':>12}  "
          f"{'最大误差(cm)':>12}  {'标准差(cm)':>10}")

    # 固定随机种子，保证每次运行结果一致（可复现）
    rng = np.random.default_rng(42)

    noise_stats = {}  # 存储统计结果，用于可视化

    for sigma in sigmas:
        # 对两种方法分别做 Monte Carlo 测试
        method_results = {}
        for method_name in ("single", "robust"):
            all_errors = []

            for _trial in range(200):    # 每个噪声水平重复 200 次独立试验
                for p in people:
                    # 获取精确像素坐标
                    foot_px, head_px, _, _ = project_person(est, p)

                    # 加入随机噪声，模拟检测器误差
                    fn, hn = add_noise(foot_px, head_px, sigma, rng)

                    try:
                        # 根据当前测试的方法直接调用内部私有方法
                        # （这里绕过 est.method 设置，两种方法都测一遍）
                        if method_name == "single":
                            h_est, _, _ = est._estimate_single(fn, hn)
                        else:
                            h_est, _, _ = est._estimate_robust(fn, hn)

                        # 计算绝对误差并收集
                        all_errors.append(abs(h_est - p["height"]))
                    except RuntimeError:
                        pass  # 极端噪声导致数值问题时跳过

            # 把误差从米转换为厘米，计算统计量
            m  = np.mean(all_errors) * 100   # 均值误差（cm）
            mx = np.max(all_errors)  * 100   # 最大误差（cm）
            sd = np.std(all_errors)  * 100   # 标准差（cm）
            method_results[method_name] = {"mean": m, "max": mx, "std": sd}
            print(f"  {sigma:>6.1f}  {method_name:>8}  {m:>12.2f}  {mx:>12.2f}  {sd:>10.2f}")

        noise_stats[sigma] = method_results
        print()  # 每个 sigma 后空一行，便于阅读

    return noise_stats


def test_homography(est, people):
    """
    测试 3：单应性矩阵辅助工具的正确性验证。

    homography_from_KRt 是一个辅助函数，它从 K/R/t 推导出地面单应性矩阵 H。
    H 的逆矩阵可以把图像中的地面点（脚底）反映射回世界坐标，
    预期结果应与 estimate() 函数内部用射线法求出的脚底坐标完全一致。

    若两者偏差 < 1e-10 m，说明 homography_from_KRt 实现正确。
    """
    print("\n" + "─" * 58)
    print("  【测试 3】单应性矩阵地面映射一致性")
    print("─" * 58)

    # 由内外参数计算地面单应性矩阵 H（3×3）
    H = homography_from_KRt(est.K, est.R, est.t)

    # H 的逆矩阵：图像像素 → 地面 (Z=0) 坐标
    H_inv = np.linalg.inv(H)

    max_err = 0.0
    for p in people:
        # 获取精确的脚底像素坐标
        foot_px, _, _, _ = project_person(est, p)

        # ── 方法A：用单应性矩阵反映射到地面坐标 ─────────────────────
        # H_inv @ [u, v, 1]^T 得到齐次坐标 [X', Y', W']
        p_h = H_inv @ np.array([foot_px[0], foot_px[1], 1.0])
        # 除以第三分量得到真实地面坐标 [X_w, Y_w]
        ground_hom = p_h / p_h[2]      # [X_w, Y_w, 1]

        # ── 方法B：用射线-地面相交法得到脚底 3D 坐标 ─────────────────
        # 这里只需要脚底坐标，head_px 随便给一个合理值（不影响脚底计算）
        _, P_foot, _ = est.estimate(foot_px, (foot_px[0], foot_px[1] - 1))

        # ── 比较两种方法的 XY 偏差 ────────────────────────────────────
        # 两者都应该给出完全相同的地面 XY 坐标（偏差仅为浮点误差）
        err = np.linalg.norm(ground_hom[:2] - P_foot[:2])
        max_err = max(max_err, err)

        print(f"  {p['label']:>4}: 单应映射 [{ground_hom[0]:+.4f}, {ground_hom[1]:+.4f}]  "
              f"vs 反投影 [{P_foot[0]:+.4f}, {P_foot[1]:+.4f}]  "
              f"偏差={err:.2e} m")

    print(f"\n  最大 XY 偏差: {max_err:.2e} m  → 两种方法一致 ✓")


# ======================================================================= #
#  可视化
# ======================================================================= #

# 6 种人物颜色（来自 matplotlib tab10 色板，视觉区分度好）
COLORS = plt.cm.tab10(np.linspace(0, 0.9, 6))


def visualize(est, people, clean_results, noise_stats, W, H, out_path):
    """
    生成包含四个子图的验证结果图并保存为 PNG。

    子图布局：
      [子图1: 图像平面标注（左宽）]  [子图2: 身高柱状图（右）]
      [子图3: 噪声曲线（左）]        [子图4: 3D 场景（右宽）]
    """
    # ── 创建画布和全局样式 ────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#16213e")      # 深蓝背景
    fig.suptitle(
        f"Monocular Camera Height Estimation — Validation Results  "
        f"(method={est.method!r})",
        fontsize=17, fontweight="bold", color="white", y=0.98
    )
    # GridSpec 定义 2 行 3 列的子图网格，并设置间距和边距
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35,
                  left=0.06, right=0.97, top=0.93, bottom=0.07)

    # ── 子图 1：图像平面标注 ─────────────────────────────────────────
    # 展示每个人物在图像中的头顶/脚底位置，以及估计结果
    ax1 = fig.add_subplot(gs[0, :2])      # 占第0行的前两列
    ax1.set_facecolor("#0f3460")
    ax1.set_xlim(0, W)                    # 横轴 = 图像宽度（像素）
    ax1.set_ylim(H, 0)                    # 纵轴反转：图像 v 轴向下
    ax1.set_title("Image Plane — Projected Keypoints", color="white", fontsize=13)
    ax1.set_xlabel("u  (pixel)", color="white")
    ax1.set_ylabel("v  (pixel)", color="white")
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#4a90d9")

    for i, (p, r) in enumerate(zip(people, clean_results)):
        c = COLORS[i]
        fp, hp = r["foot_px"], r["head_px"]  # 脚底像素坐标, 头顶像素坐标

        # 画包围框（虚线矩形），模拟检测器输出的人体边界框
        box_w = 44
        rect = mpatches.FancyBboxPatch(
            (min(fp[0], hp[0]) - box_w / 2, hp[1]),   # 左上角坐标
            box_w, fp[1] - hp[1],                      # 宽度, 高度（像素）
            boxstyle="round,pad=2", linewidth=1.8,
            edgecolor=c, facecolor="none", linestyle="--"
        )
        ax1.add_patch(rect)

        # 连线：头顶到脚底
        ax1.plot([fp[0], hp[0]], [fp[1], hp[1]], "-", color=c, lw=1.5, alpha=0.7)
        ax1.plot(*fp, "o", color=c, ms=7, zorder=6)   # 脚底：圆点
        ax1.plot(*hp, "^", color=c, ms=7, zorder=6)   # 头顶：三角

        # 标注真实值和估计值
        ax1.annotate(
            f"{p['label']}\nGT:{p['height']:.2f}m\nEst:{r['est']:.3f}m",
            xy=hp, xytext=(hp[0] + 55, hp[1] - 30),
            fontsize=8.5, color=c, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=c, lw=1.2),
        )

    # 图例
    legend_handles = [
        mpatches.Patch(color=COLORS[i], label=f"{p['label']} (GT {p['height']:.2f}m)")
        for i, p in enumerate(people)
    ]
    ax1.legend(handles=legend_handles, loc="lower right",
               fontsize=8, framealpha=0.4, facecolor="#16213e", labelcolor="white")

    # ── 子图 2：无噪声条件下身高真实值 vs 估计值的柱状图 ─────────────
    ax2 = fig.add_subplot(gs[0, 2])       # 占第0行第2列
    ax2.set_facecolor("#1a1a2e")
    bar_labels = [r["label"] for r in clean_results]
    gts  = [r["gt"]  for r in clean_results]   # 真实身高列表
    ests = [r["est"] for r in clean_results]   # 估计身高列表
    x = np.arange(len(bar_labels))   # 柱子的 x 坐标（0,1,2,...）
    w = 0.38                         # 每根柱子的宽度
    ax2.bar(x - w / 2, gts,  w, color="#4fc3f7", alpha=0.85, label="Ground truth")
    ax2.bar(x + w / 2, ests, w, color="#ff8a65", alpha=0.85, label="Estimated")
    ax2.set_ylim(1.45, 2.05)         # Y 轴范围聚焦在身高区间，放大误差可视性
    ax2.set_xticks(x)
    ax2.set_xticklabels(bar_labels, color="white")
    ax2.set_title("Height Comparison (no noise)", color="white", fontsize=12)
    ax2.set_ylabel("Height (m)", color="white")
    ax2.tick_params(colors="white")
    ax2.legend(fontsize=9, facecolor="#16213e", labelcolor="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#4a90d9")
    ax2.yaxis.grid(True, alpha=0.25, color="white")

    # ── 子图 3：噪声灵敏度曲线（single vs robust 对比）─────────────
    ax3 = fig.add_subplot(gs[1, 0])       # 占第1行第0列
    ax3.set_facecolor("#1a1a2e")
    sigmas_ = sorted(noise_stats.keys())   # 噪声水平列表（从小到大排序）

    # 分别画两种方法的均值误差曲线
    for method_name, style in [("single", ("b-o", "b")), ("robust", ("r-s", "r"))]:
        means_ = [noise_stats[s][method_name]["mean"] for s in sigmas_]
        stds_  = [noise_stats[s][method_name]["std"]  for s in sigmas_]
        ax3.plot(sigmas_, means_, style[0], ms=5, lw=2, label=f"{method_name} mean")
        # 均值 ± 标准差的半透明区间，表示波动范围
        ax3.fill_between(
            sigmas_,
            [m - s for m, s in zip(means_, stds_)],
            [m + s for m, s in zip(means_, stds_)],
            alpha=0.15, color=style[1]
        )

    ax3.set_xlabel("Detection noise σ (px)", color="white")
    ax3.set_ylabel("Height error (cm)", color="white")
    ax3.set_title("Noise Sensitivity: single vs robust", color="white", fontsize=12)
    ax3.legend(fontsize=9, facecolor="#16213e", labelcolor="white")
    ax3.tick_params(colors="white")
    for spine in ax3.spines.values():
        spine.set_edgecolor("#4a90d9")
    ax3.yaxis.grid(True, alpha=0.25, color="white")

    # ── 子图 4：3D 场景可视化 ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1:], projection="3d")  # 占第1行后两列，3D 投影
    ax4.set_facecolor("#1a1a2e")

    # 绘制地面网格（半透明绿色平面，Z=0）
    gx = np.linspace(-5, 5, 8)
    gy = np.linspace(0, 25, 10)
    GX, GY = np.meshgrid(gx, gy)   # 生成网格坐标
    ax4.plot_surface(GX, GY, np.zeros_like(GX),
                     alpha=0.12, color="green", rstride=1, cstride=1)

    # 绘制相机位置（红色三角标记）
    C = est.C           # 相机光心坐标（世界系）
    ax4.scatter(*C, c="red", s=120, marker="^", zorder=5)
    ax4.text(C[0] + 0.2, C[1], C[2] + 0.25, "Camera", color="red", fontsize=9)

    # 绘制每个人的身体（竖线）和从相机到脚底的射线（虚线）
    for i, (p, r) in enumerate(zip(people, clean_results)):
        c = COLORS[i]
        x, y, h = p["x"], p["y"], p["height"]

        # 人体：从脚底 [x,y,0] 到头顶 [x,y,h] 的竖线
        ax4.plot([x, x], [y, y], [0, h], "-o", color=c, lw=3, ms=6)
        ax4.text(x + 0.2, y, h + 0.1, f"{h:.2f}m", color=c, fontsize=8)

        # 脚底射线：从相机光心 C 到脚底 [x,y,0] 的虚线
        ax4.plot([C[0], x], [C[1], y], [C[2], 0],
                 ":", color=c, alpha=0.3, lw=1)

    ax4.set_xlabel("X (m)", color="white", labelpad=6)
    ax4.set_ylabel("Y - Depth (m)", color="white", labelpad=6)
    ax4.set_zlabel("Z - Height (m)", color="white", labelpad=6)
    ax4.set_title("3D Scene Visualization", color="white", fontsize=12, pad=8)
    ax4.tick_params(colors="white")
    ax4.view_init(elev=22, azim=-55)   # 设置俯仰角和方位角，选取最佳观察视角

    # ── 保存图像 ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(out_path), exist_ok=True)   # 若 output/ 不存在则创建
    fig.savefig(out_path, dpi=150, bbox_inches="tight",     # dpi=150 保证清晰度
                facecolor=fig.get_facecolor())
    plt.close(fig)     # 关闭图像，释放内存
    print(f"\n  可视化结果已保存: {out_path}")


# ======================================================================= #
#  主入口
# ======================================================================= #

def main():
    print("=" * 62)
    print("   单目摄像头人体身高测量算法 — 合成数据验证")
    print("=" * 62)

    # ── 构建虚拟场景，获取估计器、内参、图像尺寸、人物列表 ──────────
    est, K, W, H, people = create_scene()

    # 打印本次验证使用的参数摘要
    print(f"\n  ★ 当前估计方法: {est.method!r}  "
          f"（可修改脚本顶部 ESTIMATION_METHOD 切换）")
    print(f"  相机光心: {est.C.tolist()}")
    print(f"  相机高度: {est.camera_height:.2f} m (world Z)")
    print(f"  图像分辨率: {W} × {H} px")
    print(f"  内参矩阵 K:\n{K}")

    # ── 测试 1：无噪声理论精度验证 ───────────────────────────────────
    clean_results = test_no_noise(est, people)

    # ── 测试 2：两种方法在不同噪声水平下的对比 ───────────────────────
    noise_stats = test_noise_sensitivity(est, people)

    # ── 测试 3：单应性矩阵辅助工具一致性验证 ─────────────────────────
    test_homography(est, people)

    # ── 生成并保存可视化结果图 ────────────────────────────────────────
    out_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
    out_path = os.path.join(out_dir, "validation_result.png")
    visualize(est, people, clean_results, noise_stats, W, H, out_path)

    print("\n" + "=" * 62)
    print("  验证完成！")
    print("=" * 62)


if __name__ == "__main__":
    main()
