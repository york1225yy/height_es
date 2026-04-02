"""
直接身高估计接口
================
本脚本是面向"真实使用"的入口：你只需把上游已有的结果直接传进来：
  - 相机标定参数（内参矩阵 K、旋转矩阵 R、平移向量 t）
  - 或者更简单的相机几何描述（光心位置 cam_pos + 注视点 target）
  - 人体关键点检测结果（脚底像素坐标、头顶像素坐标）

不需要构造虚拟场景，不需要投影/反投影验证，直接输出估计身高。

【三种典型使用方式】

  方式 1 — 作为模块导入，在你的代码中调用 estimate_height():
    from src.estimate import estimate_height
    h = estimate_height(K=my_K, R=my_R, t=my_t,
                        foot_px=(u_f, v_f), head_px=(u_h, v_h))

  方式 2 — 命令行运行，传入参数（JSON 格式）:
    python src/estimate.py \\
        --K  "[[1200,0,960],[0,1200,540],[0,0,1]]" \\
        --R  "[[1,0,0],[0,0.707,-0.707],[0,0.707,0.707]]" \\
        --t  "[0, -2.828, 2.828]" \\
        --foot_px "[960, 850]" \\
        --head_px "[955, 460]" \\
        --method robust

  方式 3 — 直接运行脚本（使用内置默认参数演示）:
    python src/estimate.py

【坐标系约定（与 height_estimator.py 一致）】
  世界坐标系: X 向右, Y 向前(深度), Z 向上
  地面平面: Z = 0
  相机坐标系 (OpenCV): X 向右, Y 向下, Z 向前
"""

import argparse   # 命令行参数解析标准库
import json       # 用于解析命令行传入的 JSON 格式矩阵/坐标
import os
import sys

import numpy as np

# 将 src 目录加入模块搜索路径，使 import height_estimator 能找到文件
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from height_estimator import HeightEstimator, build_K


# =========================================================================== #
#  默认示例参数（与 validate.py 中的虚拟场景一致，方便对比验证）
#
#  场景描述：
#    相机安装高度 4m，镜头对准前方 12m 地面，图像分辨率 1920×1080，
#    焦距 fx=fy=1200px（约等效 25mm 全画幅镜头）。
#    示例人物：站在前方 8m，距中轴 0m，真实身高 1.70m。
# =========================================================================== #

# ── 内参矩阵 K（3×3）────────────────────────────────────────────────────────
DEFAULT_K = np.array([
    [1200.0,    0.0,  960.0],   # fx, skew, cx（主点在图像宽度一半处）
    [   0.0, 1200.0,  540.0],   # 0,  fy,  cy（主点在图像高度一半处）
    [   0.0,    0.0,    1.0],   # 齐次坐标固定行
], dtype=np.float64)

# ── 旋转矩阵 R 和平移向量 t（由 from_look_at 推导，此处给出等价的显式值）──
# 相机光心在 [0, 0, 4]，注视点在 [0, 12, 0]
# 通过 from_look_at 推导后直接用数值写出，方便理解"外参长什么样"
_tmp_est = HeightEstimator.from_look_at(
    cam_pos=[0.0, 0.0, 4.0],    # 相机安装位置（世界坐标，单位：m）
    target =[0.0, 12.0, 0.0],   # 注视点（地面位置）
    K=DEFAULT_K,
)
DEFAULT_R = _tmp_est.R.copy()   # 旋转矩阵，shape (3,3)
DEFAULT_T = _tmp_est.t.copy()   # 平移向量，shape (3,)
del _tmp_est                    # 临时对象用完即删

# ── 示例人物：位于 [0, 8, 0]，身高 1.70m（等同于 validate.py 中的 P2）──────
# 脚底和头顶像素坐标由 validate.py 的 project() 计算后复制过来作为示例输入
# 这两个值是"真实场景中你的检测器应该给出的坐标"
_tmp_est2 = HeightEstimator(DEFAULT_K, DEFAULT_R, DEFAULT_T)
DEFAULT_FOOT_PX = tuple(_tmp_est2.project([0.0, 8.0, 0.0]).tolist())   # 脚底像素
DEFAULT_HEAD_PX = tuple(_tmp_est2.project([0.0, 8.0, 1.70]).tolist())  # 头顶像素（1.70m处）
DEFAULT_TRUE_HEIGHT = 1.70   # 示例的真实身高（仅用于打印误差对比）
del _tmp_est2


# =========================================================================== #
#  核心接口函数
# =========================================================================== #

def estimate_height(
    K,
    R,
    t,
    foot_px,
    head_px,
    method: str = "robust",
):
    """
    直接从标定参数和检测结果估计人体身高。

    这是面向外部调用的主接口，把"创建估计器 + 调用 estimate()"封装为一步。

    Parameters
    ----------
    K : array-like (3, 3)
        相机内参矩阵。
        如果你只知道 fx, fy, cx, cy（来自标定工具输出），可先用
        ``build_K(fx, fy, cx, cy)`` 构造，再传入此函数。

    R : array-like (3, 3)
        旋转矩阵（世界坐标系 → 相机坐标系）。
        来源：OpenCV calibrateCamera / solvePnP 的 rvec 转换后的矩阵，
        或 Matlab Camera Calibrator 导出的 RotationMatrix。

    t : array-like (3,)
        平移向量（相机坐标系下世界原点的位置）。
        来源：OpenCV solvePnP 的 tvec（单位与标定时使用的世界坐标单位相同）。

    foot_px : (u, v)
        脚底关键点的图像像素坐标，格式 (列, 行) 即 (u, v)。
        来源：人体关键点检测器（如 YOLOv8-Pose、MMPose、OpenPose）的输出，
        通常对应脚踝/脚底关键点。

    head_px : (u, v)
        头顶关键点的图像像素坐标，格式 (列, 行) 即 (u, v)。
        来源：检测器输出，通常对应头顶/鼻子关键点（头顶精度更高）。

    method : str, 默认 "robust"
        估计算法选择：
          "robust" — 最小二乘双轴联合求解，有噪声时更稳（推荐）
          "single" — 单轴最稳定分量求解，更快

    Returns
    -------
    result : dict，包含以下键：
        "height"  : float         估计身高（与世界坐标同单位，通常为米）
        "P_foot"  : ndarray (3,)  脚底的 3D 世界坐标 [x, y, 0]
        "P_head"  : ndarray (3,)  头顶的 3D 世界坐标 [x, y, height]
        "method"  : str           使用的方法名称
        "foot_px" : tuple         传入的脚底像素坐标（原样返回，便于调试）
        "head_px" : tuple         传入的头顶像素坐标（原样返回，便于调试）

    Raises
    ------
    ValueError  : 参数格式错误或目标在相机后方
    RuntimeError: 几何求解失败（射线平行地面、目标在图像中心正下方等）

    Examples
    --------
    >>> import numpy as np
    >>> from src.estimate import estimate_height, DEFAULT_K, DEFAULT_R, DEFAULT_T
    >>> result = estimate_height(DEFAULT_K, DEFAULT_R, DEFAULT_T,
    ...                          foot_px=(960, 710), head_px=(960, 460))
    >>> print(f"身高: {result['height']:.3f} m")
    """
    # 构造估计器（仅传入参数，不构造虚拟场景）
    est = HeightEstimator(K, R, t, method=method)

    # 调用统一估计接口
    height, P_foot, P_head = est.estimate(foot_px, head_px)

    # 返回结构化结果字典，方便调用方按需取用
    return {
        "height":  height,
        "P_foot":  P_foot,
        "P_head":  P_head,
        "method":  method,
        "foot_px": tuple(foot_px),
        "head_px": tuple(head_px),
    }


def estimate_height_from_lookat(
    cam_pos,
    target,
    K,
    foot_px,
    head_px,
    method: str = "robust",
):
    """
    通过相机安装位置和注视点（而非 R/t）直接估计身高。

    当你知道"相机装在哪里、对准哪里"但还没有做完整标定时，
    可用此函数替代 estimate_height()，内部自动推导 R 和 t。

    Parameters
    ----------
    cam_pos : (x, y, z)
        相机光心在世界坐标系中的位置，例如 (0, 0, 4.0) 表示装在 4m 高处。
    target : (x, y, z)
        镜头对准的地面点世界坐标，例如 (0, 12, 0) 表示对准前方 12m 地面。
        ★ 此参数仅用于计算旋转矩阵，不参与身高计算。
    K : array-like (3, 3)
        相机内参矩阵。
    foot_px : (u, v)
        脚底像素坐标。
    head_px : (u, v)
        头顶像素坐标。
    method : str, 默认 "robust"
        估计方法，同 estimate_height()。

    Returns
    -------
    result : dict，同 estimate_height() 的返回值。
    """
    # 用 from_look_at 工厂方法构造估计器（自动推导 R、t）
    est = HeightEstimator.from_look_at(cam_pos, target, K, method=method)

    height, P_foot, P_head = est.estimate(foot_px, head_px)

    return {
        "height":   height,
        "P_foot":   P_foot,
        "P_head":   P_head,
        "method":   method,
        "cam_pos":  tuple(np.asarray(cam_pos).tolist()),
        "target":   tuple(np.asarray(target).tolist()),
        "foot_px":  tuple(foot_px),
        "head_px":  tuple(head_px),
    }


# =========================================================================== #
#  打印结果的辅助函数
# =========================================================================== #

def print_result(result: dict, true_height: float = None):
    """
    格式化打印估计结果，可选与真实值对比（便于调试和验证）。

    Parameters
    ----------
    result : dict
        estimate_height() 或 estimate_height_from_lookat() 的返回值。
    true_height : float, 可选
        已知的真实身高（米），若提供则额外打印误差。
    """
    print("\n┌─────────────────────────────────────────┐")
    print("│           身高估计结果                   │")
    print("├─────────────────────────────────────────┤")
    print(f"│  估计身高         : {result['height']:>8.4f} m           │")

    # 若提供真实值则打印误差
    if true_height is not None:
        err_cm = abs(result["height"] - true_height) * 100
        print(f"│  真实身高 (参考)  : {true_height:>8.4f} m           │")
        print(f"│  误差             : {err_cm:>8.4f} cm          │")

    # 打印 3D 定位结果
    pf = result["P_foot"]
    ph = result["P_head"]
    print(f"│  脚底 3D坐标      : [{pf[0]:+.3f}, {pf[1]:+.3f}, {pf[2]:+.3f}] m  │")
    print(f"│  头顶 3D坐标      : [{ph[0]:+.3f}, {ph[1]:+.3f}, {ph[2]:+.3f}] m  │")
    print(f"│  使用方法         : {result['method']:<10}              │")
    print(f"│  输入脚底像素     : ({result['foot_px'][0]:.1f}, {result['foot_px'][1]:.1f})              │")
    print(f"│  输入头顶像素     : ({result['head_px'][0]:.1f}, {result['head_px'][1]:.1f})              │")
    print("└─────────────────────────────────────────┘")


# =========================================================================== #
#  命令行解析
# =========================================================================== #

def _parse_matrix(s: str) -> np.ndarray:
    """把 JSON 格式的字符串解析成 numpy 数组（支持列表嵌套列表）。"""
    try:
        data = json.loads(s)           # 解析 JSON 字符串为 Python 列表
        return np.array(data, dtype=np.float64)
    except (json.JSONDecodeError, ValueError) as e:
        raise argparse.ArgumentTypeError(
            f"无法解析为数值数组: {s!r}\n  错误: {e}\n"
            "  请使用 JSON 格式，例如: \"[[1200,0,960],[0,1200,540],[0,0,1]]\""
        )


def _parse_pair(s: str) -> tuple:
    """把 JSON 格式的字符串解析成二元组，例如 \"[960, 540]\" → (960.0, 540.0)。"""
    arr = _parse_matrix(s)
    if arr.ndim != 1 or len(arr) != 2:
        raise argparse.ArgumentTypeError(f"期望长度为 2 的一维数组，收到: {s!r}")
    return (float(arr[0]), float(arr[1]))


def _build_arg_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器，定义所有支持的参数。"""
    parser = argparse.ArgumentParser(
        prog="python src/estimate.py",
        description=(
            "单目相机人体身高直接估计工具\n"
            "不传参数时使用内置示例值运行演示。\n\n"
            "参数均使用 JSON 格式传入，矩阵用 [[...],[...]] 表示，坐标用 [u,v] 表示。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "使用示例:\n"
            "  # 使用默认示例参数\n"
            "  python src/estimate.py\n\n"
            "  # 传入自己的参数\n"
            "  python src/estimate.py \\\n"
            "    --K  \"[[1200,0,960],[0,1200,540],[0,0,1]]\" \\\n"
            "    --foot_px \"[960, 850]\" \\\n"
            "    --head_px  \"[955, 460]\" \\\n"
            "    --cam_pos \"[0,0,4]\" \\\n"
            "    --target  \"[0,12,0]\" \\\n"
            "    --method robust\n\n"
            "  # 直接传 R 和 t（来自标定工具）\n"
            "  python src/estimate.py \\\n"
            "    --K \"[[1200,0,960],[0,1200,540],[0,0,1]]\" \\\n"
            "    --R \"[[1,0,0],[0,0.316,-0.949],[0,0.949,0.316]]\" \\\n"
            "    --t \"[0,-1.265,3.795]\" \\\n"
            "    --foot_px \"[960,850]\" \\\n"
            "    --head_px  \"[955,460]\""
        ),
    )

    # ── 相机参数（两种输入方式二选一） ─────────────────────────────────
    cam_group = parser.add_argument_group(
        "相机参数（二选一）",
        "方式A: 传 --cam_pos + --target（简便，会自动推导 R/t）\n"
        "方式B: 传 --R + --t（精确，来自标定工具的直接输出）\n"
        "若两种方式都传，优先使用方式B（--R/--t）。"
    )
    cam_group.add_argument(
        "--K",
        type=_parse_matrix,
        default=None,
        metavar="JSON",
        help=(
            "3×3 相机内参矩阵（JSON格式）。\n"
            "例: \"[[1200,0,960],[0,1200,540],[0,0,1]]\"\n"
            "默认: 内置示例值（fx=fy=1200px，cx=960，cy=540）"
        ),
    )
    cam_group.add_argument(
        "--R",
        type=_parse_matrix,
        default=None,
        metavar="JSON",
        help=(
            "3×3 旋转矩阵（JSON格式，世界→相机坐标系）。\n"
            "来源: OpenCV solvePnP 或 Rodrigues 转换后的矩阵。\n"
            "须与 --t 同时提供。"
        ),
    )
    cam_group.add_argument(
        "--t",
        type=_parse_matrix,
        default=None,
        metavar="JSON",
        help=(
            "长度为3的平移向量（JSON格式）。\n"
            "来源: OpenCV solvePnP 的 tvec（单位与世界坐标相同）。\n"
            "须与 --R 同时提供。"
        ),
    )
    cam_group.add_argument(
        "--cam_pos",
        type=_parse_matrix,
        default=None,
        metavar="JSON",
        help=(
            "相机光心在世界坐标系的位置 [x,y,z]（JSON格式）。\n"
            "例: \"[0,0,4]\" 表示装在地面正上方 4m。\n"
            "须与 --target 同时提供（方式A）。"
        ),
    )
    cam_group.add_argument(
        "--target",
        type=_parse_matrix,
        default=None,
        metavar="JSON",
        help=(
            "镜头注视点的世界坐标 [x,y,z]（JSON格式）。\n"
            "例: \"[0,12,0]\" 表示对准前方12m的地面。\n"
            "须与 --cam_pos 同时提供（方式A）。"
        ),
    )

    # ── 检测结果（关键点像素坐标） ──────────────────────────────────────
    det_group = parser.add_argument_group(
        "检测结果",
        "由上游检测器（如 YOLO-Pose、MMPose 等）提供的关键点像素坐标。"
    )
    det_group.add_argument(
        "--foot_px",
        type=_parse_pair,
        default=None,
        metavar="JSON",
        help=(
            "脚底关键点的像素坐标 [u,v]（JSON格式，u=列 v=行）。\n"
            "例: \"[960, 850]\"\n"
            "默认: 内置示例值（由示例场景正向投影计算）"
        ),
    )
    det_group.add_argument(
        "--head_px",
        type=_parse_pair,
        default=None,
        metavar="JSON",
        help=(
            "头顶关键点的像素坐标 [u,v]（JSON格式，u=列 v=行）。\n"
            "例: \"[955, 460]\"\n"
            "默认: 内置示例值（由示例场景正向投影计算）"
        ),
    )

    # ── 算法选择 ──────────────────────────────────────────────────────
    parser.add_argument(
        "--method",
        choices=("robust", "single"),
        default="robust",
        help="估计算法（默认: robust，推荐用于真实场景）",
    )

    # ── 与真实值对比（可选，调试用） ──────────────────────────────────
    parser.add_argument(
        "--true_height",
        type=float,
        default=None,
        metavar="FLOAT",
        help="已知的真实身高（米），提供后打印误差对比，不影响估计过程",
    )

    return parser


# =========================================================================== #
#  主入口
# =========================================================================== #

def main():
    """
    命令行入口。

    参数全部可选：不传参数时使用内置示例值，方便直接运行演示。
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    # ── 确定内参矩阵 K ──────────────────────────────────────────────
    # 若用户没有传 --K，使用内置默认值
    K = args.K if args.K is not None else DEFAULT_K

    # ── 确定脚底/头顶像素坐标 ───────────────────────────────────────
    foot_px = args.foot_px if args.foot_px is not None else DEFAULT_FOOT_PX
    head_px = args.head_px if args.head_px is not None else DEFAULT_HEAD_PX

    # ── 确定真实身高（用于误差打印，不影响估计） ────────────────────
    true_height = args.true_height if args.true_height is not None else DEFAULT_TRUE_HEIGHT

    # ── 判断使用哪种相机参数输入方式 ────────────────────────────────
    has_Rt       = (args.R is not None) and (args.t is not None)
    has_look_at  = (args.cam_pos is not None) and (args.target is not None)
    use_defaults = (not has_Rt) and (not has_look_at) and (args.K is None)

    if use_defaults:
        # ── 方式 0（默认演示）：全部使用内置示例参数 ──────────────────
        print("\n" + "=" * 50)
        print("  单目相机身高估计 — 默认示例参数演示")
        print("=" * 50)
        print("  （未传入任何参数，使用内置示例值运行）")
        print(f"\n  相机光心  : [0.0, 0.0, 4.0] m（地面上方4m）")
        print(f"  注视点    : [0.0, 12.0, 0.0] m（前方12m地面）")
        print(f"  脚底像素  : ({foot_px[0]:.1f}, {foot_px[1]:.1f})")
        print(f"  头顶像素  : ({head_px[0]:.1f}, {head_px[1]:.1f})")

        result = estimate_height(
            K=K, R=DEFAULT_R, t=DEFAULT_T,
            foot_px=foot_px, head_px=head_px,
            method=args.method,
        )

    elif has_Rt:
        # ── 方式 B：用户传入了精确的 R 和 t（来自标定工具） ─────────
        print("\n" + "=" * 50)
        print("  单目相机身高估计 — 使用 R/t 参数")
        print("=" * 50)

        # 若同时也传了 cam_pos/target，提示忽略
        if has_look_at:
            print("  ⚠  同时检测到 --R/--t 和 --cam_pos/--target，优先使用 --R/--t")

        result = estimate_height(
            K=K, R=args.R, t=args.t,
            foot_px=foot_px, head_px=head_px,
            method=args.method,
        )

    else:
        # ── 方式 A：用户传入了 cam_pos 和 target ───────────────────
        print("\n" + "=" * 50)
        print("  单目相机身高估计 — 使用 cam_pos/target 参数")
        print("=" * 50)
        print(f"  相机光心  : {args.cam_pos.tolist()} m")
        print(f"  注视点    : {args.target.tolist()} m")

        result = estimate_height_from_lookat(
            cam_pos=args.cam_pos, target=args.target, K=K,
            foot_px=foot_px, head_px=head_px,
            method=args.method,
        )

    # ── 打印结构化结果 ─────────────────────────────────────────────
    print_result(result, true_height=true_height)


if __name__ == "__main__":
    main()
