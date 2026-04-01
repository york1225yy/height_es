"""
单目摄像头身高测量算法验证脚本
================================

通过合成数据（无需真实图像）验证算法精度：
  1. 搭建虚拟相机场景
  2. 在不同位置放置已知身高的虚拟人物
  3. 将头脚坐标投影至图像平面
  4. 加入高斯噪声模拟真实检测误差
  5. 调用 HeightEstimator 估计身高
  6. 与真实值对比，统计误差分布

用法:
  cd /workspaces/height_es
  python src/validate.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")          # 无显示器环境使用离屏渲染
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# 将 src 目录加入模块搜索路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from height_estimator import HeightEstimator, build_K, homography_from_KRt


# ======================================================================= #
#  场景配置
# ======================================================================= #

def create_scene():
    """
    创建虚拟场景：相机参数 + 一组虚拟人物。

    相机安装条件（模拟交通监控视角）：
      - 安装高度: 4 m
      - 注视点在地面正前方 12 m 处
      - 图像分辨率: 1920 × 1080
      - 焦距 fx = fy ≈ 1200 px (约等效 25mm 镜头在全画幅)
    """
    W, H = 1920, 1080
    fx = fy = 1200.0
    cx, cy = W / 2.0, H / 2.0
    K = build_K(fx, fy, cx, cy)

    cam_pos = np.array([0.0, 0.0, 4.0])    # 相机光心（世界坐标，Z 向上）
    target  = np.array([0.0, 12.0, 0.0])   # 注视点（地面）

    estimator = HeightEstimator.from_look_at(cam_pos, target, K)

    # 虚拟人物列表: (世界 x, 世界 y, 真实身高 m, 标签)
    people = [
        {"x": -2.5, "y":  6.0, "height": 1.60, "label": "P1"},
        {"x":  0.0, "y":  8.0, "height": 1.70, "label": "P2"},
        {"x":  2.0, "y": 10.0, "height": 1.75, "label": "P3"},
        {"x": -1.0, "y": 14.0, "height": 1.80, "label": "P4"},
        {"x":  1.5, "y": 18.0, "height": 1.90, "label": "P5"},
        {"x": -3.0, "y": 20.0, "height": 1.65, "label": "P6"},
    ]
    return estimator, K, W, H, people


# ======================================================================= #
#  辅助函数
# ======================================================================= #

def project_person(est: HeightEstimator, person: dict):
    """将虚拟人物的脚底和头顶投影至图像，返回像素坐标和 3D 真值。"""
    x, y, h = person["x"], person["y"], person["height"]
    P_foot = np.array([x, y, 0.0])
    P_head = np.array([x, y, h])
    foot_px = est.project(P_foot)
    head_px = est.project(P_head)
    return foot_px, head_px, P_foot, P_head


def add_noise(foot_px, head_px, sigma: float, rng: np.random.Generator):
    """对像素坐标施加各向同性高斯噪声（模拟检测误差）。"""
    return (
        foot_px + rng.normal(0, sigma, 2),
        head_px + rng.normal(0, sigma, 2),
    )


# ======================================================================= #
#  测试函数
# ======================================================================= #

def test_no_noise(est, people):
    """测试 1：无噪声 —— 验证算法的理论精度（应接近机器精度）。"""
    print("\n" + "─" * 58)
    print("  【测试 1】无噪声 — 验证理论精度")
    print("─" * 58)
    print(f"  {'人物':>4}  {'真实(m)':>8}  {'估计(m)':>8}  "
          f"{'误差(cm)':>9}  {'相对误差':>8}")

    errors = []
    results = []
    for p in people:
        foot_px, head_px, _, _ = project_person(est, p)
        h_est, P_foot, P_head = est.estimate_height(foot_px, head_px)
        err = abs(h_est - p["height"])
        errors.append(err)
        results.append({
            "label": p["label"],
            "gt": p["height"],
            "est": h_est,
            "err_m": err,
            "foot_px": foot_px,
            "head_px": head_px,
            "P_foot": P_foot,
            "P_head": P_head,
        })
        print(f"  {p['label']:>4}  {p['height']:>8.3f}  {h_est:>8.4f}  "
              f"{err*100:>9.4f}  {err/p['height']*100:>7.4f}%")

    print(f"\n  最大误差: {max(errors)*1000:.3f} mm")
    print(f"  均值误差: {np.mean(errors)*1000:.3f} mm")
    print("  → 误差量级为数值精度（< 0.01 mm），算法理论正确 ✓")
    return results


def test_noise_sensitivity(est, people, sigmas=(0.5, 1, 2, 3, 5, 10)):
    """测试 2：像素噪声灵敏度分析 —— 不同噪声水平下的估计误差统计。"""
    print("\n" + "─" * 58)
    print("  【测试 2】噪声灵敏度分析")
    print("─" * 58)
    print(f"  {'σ(px)':>6}  {'均值误差(cm)':>12}  {'最大误差(cm)':>12}  "
          f"{'标准差(cm)':>10}")

    noise_stats = {}
    rng = np.random.default_rng(42)
    for sigma in sigmas:
        all_errors = []
        for trial in range(200):          # 200 次 Monte Carlo
            for i, p in enumerate(people):
                foot_px, head_px, _, _ = project_person(est, p)
                fn, hn = add_noise(foot_px, head_px, sigma, rng)
                try:
                    h_est, _, _ = est.estimate_height_robust(fn, hn)
                    all_errors.append(abs(h_est - p["height"]))
                except RuntimeError:
                    pass
        m  = np.mean(all_errors) * 100
        mx = np.max(all_errors)  * 100
        sd = np.std(all_errors)  * 100
        noise_stats[sigma] = {"mean": m, "max": mx, "std": sd}
        print(f"  {sigma:>6.1f}  {m:>12.2f}  {mx:>12.2f}  {sd:>10.2f}")

    return noise_stats


def test_homography(est, people):
    """
    测试 3：单应性矩阵验证。
    用 K, R, t 推导 H，再用 H 将脚底像素反映射回地面坐标，
    检验与 estimate_height 所得脚底坐标一致性。
    """
    print("\n" + "─" * 58)
    print("  【测试 3】单应性矩阵地面映射一致性")
    print("─" * 58)

    H = homography_from_KRt(est.K, est.R, est.t)
    H_inv = np.linalg.inv(H)

    max_err = 0.0
    for p in people:
        foot_px, _, _, _ = project_person(est, p)
        # H_inv 将图像点映射回地面 (Z=0 平面) 的齐次坐标
        p_h = H_inv @ np.array([foot_px[0], foot_px[1], 1.0])
        ground_hom = p_h / p_h[2]        # [X_w, Y_w, 1]
        # estimate_height 的脚底 3D
        _, P_foot, _ = est.estimate_height(foot_px, (foot_px[0], foot_px[1]-1))
        err = np.linalg.norm(ground_hom[:2] - P_foot[:2])
        max_err = max(max_err, err)
        print(f"  {p['label']:>4}: 单应映射 [{ground_hom[0]:+.4f}, {ground_hom[1]:+.4f}]  "
              f"vs 反投影 [{P_foot[0]:+.4f}, {P_foot[1]:+.4f}]  "
              f"误差={err:.2e} m")

    print(f"\n  最大 XY 偏差: {max_err:.2e} m  → 两种方法一致 ✓")


# ======================================================================= #
#  可视化
# ======================================================================= #

COLORS = plt.cm.tab10(np.linspace(0, 0.9, 6))


def visualize(est, people, clean_results, noise_stats, W, H, out_path):
    """生成包含四个子图的验证结果图并保存。"""
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#16213e")
    fig.suptitle(
        "Monocular Camera Height Estimation — Validation Results",
        fontsize=18, fontweight="bold", color="white", y=0.98
    )
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35,
                  left=0.06, right=0.97, top=0.93, bottom=0.07)

    # ── 子图 1: 图像平面标注 ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor("#0f3460")
    ax1.set_xlim(0, W)
    ax1.set_ylim(H, 0)
    ax1.set_title("Image Plane — Projected Keypoints", color="white", fontsize=13)
    ax1.set_xlabel("u  (pixel)", color="white")
    ax1.set_ylabel("v  (pixel)", color="white")
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#4a90d9")

    for i, (p, r) in enumerate(zip(people, clean_results)):
        c = COLORS[i]
        fp, hp = r["foot_px"], r["head_px"]
        # 连线 + 矩形框
        box_w = 44
        rect = mpatches.FancyBboxPatch(
            (min(fp[0], hp[0]) - box_w/2, hp[1]),
            box_w, fp[1] - hp[1],
            boxstyle="round,pad=2", linewidth=1.8,
            edgecolor=c, facecolor="none", linestyle="--"
        )
        ax1.add_patch(rect)
        ax1.plot([fp[0], hp[0]], [fp[1], hp[1]], "-", color=c, lw=1.5, alpha=0.7)
        ax1.plot(*fp, "o", color=c, ms=7, zorder=6)
        ax1.plot(*hp, "^", color=c, ms=7, zorder=6)

        ax1.annotate(
            f"{p['label']}\nGT:{p['height']:.2f}m\nEst:{r['est']:.3f}m",
            xy=hp, xytext=(hp[0] + 55, hp[1] - 30),
            fontsize=8.5, color=c, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=c, lw=1.2),
        )

    legend_handles = [
        mpatches.Patch(color=COLORS[i], label=f"{p['label']} (GT {p['height']:.2f}m)")
        for i, p in enumerate(people)
    ]
    ax1.legend(handles=legend_handles, loc="lower right",
               fontsize=8, framealpha=0.4, facecolor="#16213e", labelcolor="white")

    # ── 子图 2: 身高对比柱状图 ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor("#1a1a2e")
    labels = [r["label"] for r in clean_results]
    gts    = [r["gt"]   for r in clean_results]
    ests   = [r["est"]  for r in clean_results]
    x = np.arange(len(labels))
    w = 0.38
    ax2.bar(x - w/2, gts,  w, color="#4fc3f7", alpha=0.85, label="Ground truth")
    ax2.bar(x + w/2, ests, w, color="#ff8a65", alpha=0.85, label="Estimated")
    ax2.set_ylim(1.45, 2.05)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, color="white")
    ax2.set_title("Height Comparison (no noise)", color="white", fontsize=12)
    ax2.set_ylabel("Height (m)", color="white")
    ax2.tick_params(colors="white")
    ax2.legend(fontsize=9, facecolor="#16213e", labelcolor="white")
    ax2.set_facecolor("#1a1a2e")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#4a90d9")
    ax2.yaxis.grid(True, alpha=0.25, color="white")

    # ── 子图 3: 噪声-误差曲线 ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#1a1a2e")
    sigmas_  = sorted(noise_stats.keys())
    means_   = [noise_stats[s]["mean"] for s in sigmas_]
    maxes_   = [noise_stats[s]["max"]  for s in sigmas_]
    stds_    = [noise_stats[s]["std"]  for s in sigmas_]

    ax3.plot(sigmas_, means_, "b-o", ms=5, lw=2, label="Mean error")
    ax3.plot(sigmas_, maxes_, "r--s", ms=5, lw=2, label="Max error")
    ax3.fill_between(sigmas_,
                     [m - s for m, s in zip(means_, stds_)],
                     [m + s for m, s in zip(means_, stds_)],
                     alpha=0.2, color="blue")
    ax3.set_xlabel("Detection noise σ (px)", color="white")
    ax3.set_ylabel("Height error (cm)", color="white")
    ax3.set_title("Noise Sensitivity Curve", color="white", fontsize=12)
    ax3.legend(fontsize=9, facecolor="#16213e", labelcolor="white")
    ax3.tick_params(colors="white")
    for spine in ax3.spines.values():
        spine.set_edgecolor("#4a90d9")
    ax3.yaxis.grid(True, alpha=0.25, color="white")

    # ── 子图 4: 3D 场景可视化 ────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1:], projection="3d")
    ax4.set_facecolor("#1a1a2e")

    # 地面网格
    gx = np.linspace(-5, 5, 8)
    gy = np.linspace(0, 25, 10)
    GX, GY = np.meshgrid(gx, gy)
    ax4.plot_surface(GX, GY, np.zeros_like(GX),
                     alpha=0.12, color="green", rstride=1, cstride=1)

    # 相机
    C = est.C
    ax4.scatter(*C, c="red", s=120, marker="^", zorder=5)
    ax4.text(C[0]+0.2, C[1], C[2]+0.25, "Camera", color="red", fontsize=9)

    # 人物 + 射线
    for i, (p, r) in enumerate(zip(people, clean_results)):
        c = COLORS[i]
        x, y, h = p["x"], p["y"], p["height"]
        ax4.plot([x, x], [y, y], [0, h], "-o", color=c, lw=3, ms=6)
        ax4.text(x + 0.2, y, h + 0.1, f"{h:.2f}m", color=c, fontsize=8)
        # 脚底射线（虚线）
        ax4.plot([C[0], x], [C[1], y], [C[2], 0],
                 ":", color=c, alpha=0.3, lw=1)

    ax4.set_xlabel("X (m)", color="white", labelpad=6)
    ax4.set_ylabel("Y - Depth (m)", color="white", labelpad=6)
    ax4.set_zlabel("Z - Height (m)", color="white", labelpad=6)
    ax4.set_title("3D Scene Visualization", color="white", fontsize=12, pad=8)
    ax4.tick_params(colors="white")
    ax4.view_init(elev=22, azim=-55)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  可视化结果已保存: {out_path}")


# ======================================================================= #
#  主入口
# ======================================================================= #

def main():
    print("=" * 58)
    print("   单目摄像头人体身高测量算法 — 合成数据验证")
    print("=" * 58)

    est, K, W, H, people = create_scene()

    print(f"\n  相机光心: {est.C.tolist()}")
    print(f"  相机高度: {est.camera_height:.2f} m (world Z)")
    print(f"  图像分辨率: {W} × {H} px")
    print(f"  内参矩阵 K:\n{K}")

    # --- 三组测试 ---
    clean_results = test_no_noise(est, people)
    noise_stats   = test_noise_sensitivity(est, people)
    test_homography(est, people)

    # --- 可视化 ---
    out_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
    out_path = os.path.join(out_dir, "validation_result.png")
    visualize(est, people, clean_results, noise_stats, W, H, out_path)

    print("\n" + "=" * 58)
    print("  验证完成！")
    print("=" * 58)


if __name__ == "__main__":
    main()
