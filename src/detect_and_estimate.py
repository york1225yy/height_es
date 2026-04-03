"""
基于 YOLO11-Pose 的目标检测与身高估计
=====================================

流程：
  1. 使用 YOLO11-Pose 对图像/视频帧进行人体检测，同时获取骨骼关键点
  2. 以检测框推导头部坐标（bbox 最高点 v，中点 u）和脚部坐标（bbox 最低点 v，中点 u）
  3. 关键点后处理约束：
       - 必须检测到至少一个头部关键点（鼻子/眼/耳，索引 0-4）且置信度 ≥ 阈值
       - 必须检测到至少一个脚部关键点（左右脚踝，索引 15/16）且置信度 ≥ 阈值
       - 不满足上述条件的人物视为"不完整目标"，跳过身高估计
  4. 调用 estimate_height() 完成身高估计

COCO 17 关键点索引与语义：
  0  nose          1  left_eye       2  right_eye
  3  left_ear      4  right_ear      5  left_shoulder
  6  right_shoulder 7  left_elbow    8  right_elbow
  9  left_wrist    10 right_wrist   11  left_hip
  12 right_hip     13 left_knee     14  right_knee
  15 left_ankle    16 right_ankle

使用方式（命令行）：
  # 对单张图片处理，需提供相机参数（lookat 方式）
  python src/detect_and_estimate.py \\
    --source path/to/image.jpg \\
    --K "[[1200,0,960],[0,1200,540],[0,0,1]]" \\
    --cam_pos "[0,0,4]" \\
    --target "[0,12,0]"

  # 也可传入 R 和 t
  python src/detect_and_estimate.py \\
    --source path/to/video.mp4 \\
    --K "[[1200,0,960],[0,1200,540],[0,0,1]]" \\
    --R "[[1,0,0],[0,0.31622777,-0.9486833],[0,0.9486833,0.31622777]]" \\
    --t "[0,-1.26491106,3.79473319]"

  # 使用内置默认参数（方便快速验证）
  python src/detect_and_estimate.py

使用方式（Python 导入）：
  from src.detect_and_estimate import DetectAndEstimate
  pipeline = DetectAndEstimate(weights="weights/yolo11s-pose.pt", K=K, R=R, t=t)
  results = pipeline.run_image("frame.jpg")
  for r in results:
      print(r["person_id"], r["height"])
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from collections import defaultdict, deque

# 将 src 目录加入模块搜索路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimate import (
    estimate_height,
    estimate_height_from_lookat,
    DEFAULT_K,
    DEFAULT_R,
    DEFAULT_T,
)

try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError(
        "未找到 ultralytics 库，请先安装：pip install ultralytics"
    ) from e


# --------------------------------------------------------------------------- #
#  COCO 关键点索引定义
# --------------------------------------------------------------------------- #
# 头部关键点索引（鼻子、左眼、右眼、左耳、右耳）
HEAD_KP_INDICES = (0, 1, 2, 3, 4)

# 脚部关键点索引（左脚踝、右脚踝）
FOOT_KP_INDICES = (15, 16)

# 关键点索引 → 名称（用于日志输出）
KP_NAMES = {
    0: "nose", 1: "left_eye", 2: "right_eye",
    3: "left_ear", 4: "right_ear",
    5: "left_shoulder", 6: "right_shoulder",
    7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist",
    11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee",
    15: "left_ankle", 16: "right_ankle",
}

# 可视化颜色（BGR）
COLOR_COMPLETE = (0, 200, 0)      # 绿色：完整目标，已估计身高
COLOR_INCOMPLETE = (0, 100, 220)  # 橙色：关键点不完整，跳过
COLOR_TEXT = (255, 255, 255)      # 白色文字

# COCO-17 骨骼连线（关键点索引对）
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),   # 面部
    (5, 7), (7, 9),                    # 左臂
    (6, 8), (8, 10),                   # 右臂
    (5, 6),                            # 肩部横线
    (5, 11), (6, 12),                  # 躯干左右
    (11, 12),                          # 髋部横线
    (11, 13), (13, 15),                # 左腿
    (12, 14), (14, 16),                # 右腿
]

# 骨骼连线对应颜色（BGR），与 COCO_SKELETON 一一对应
SKELETON_COLORS = [
    (0, 255, 128), (0, 255, 128), (0, 255, 128), (0, 255, 128),  # 面部：绿
    (255, 165,   0), (255, 165,   0),                              # 左臂：橙
    (  0, 165, 255), (  0, 165, 255),                              # 右臂：蓝橙
    (  0, 255, 255),                                               # 肩：黄
    (255, 255,   0), (255, 255,   0),                              # 躯干：青
    (  0, 200, 255),                                               # 髋：浅蓝
    (128,   0, 128), (128,   0, 128),                              # 左腿：紫
    (  0, 128, 128), (  0, 128, 128),                              # 右腿：深青
]

# --------------------------------------------------------------------------- #
#  假设相机内外参（适配 960×720 视频）
#
#  推导场景：相机光心位于世界坐标 [0, 0, 2]（高 2m），
#           镜头朝向正前方 8m 处地面 [0, 8, 0]，
#           fx = fy = 800 px，主点 (cx, cy) = (480, 360)。
#  由 from_look_at 推导后显式写出，等价于：
#    HeightEstimator.from_look_at([0,0,2], [0,8,0], DEMO_K)
# --------------------------------------------------------------------------- #
DEMO_K = np.array([
    [800.0,   0.0, 480.0],
    [  0.0, 800.0, 360.0],
    [  0.0,   0.0,   1.0],
], dtype=np.float64)

# R 的行依次为相机坐标轴 X/Y/Z 在世界坐标系中的方向
DEMO_R = np.array([
    [ 1.0,        0.0,        0.0      ],
    [ 0.0,       -0.242536,  -0.970143 ],
    [ 0.0,        0.970143,  -0.242536 ],
], dtype=np.float64)

# t = -R @ cam_pos，即 [0, 1.940286, 0.485071]
DEMO_T = np.array([0.0, 1.940286, 0.485071], dtype=np.float64)


# --------------------------------------------------------------------------- #
#  核心类
# --------------------------------------------------------------------------- #
class DetectAndEstimate:
    """
    基于 YOLO11-Pose 的人体检测与单目身高估计流水线。

    Parameters
    ----------
    weights : str
        YOLO11-Pose 权重文件路径，例如 "weights/yolo11s-pose.pt"
    K : array-like (3,3)
        相机内参矩阵
    R : array-like (3,3)
        旋转矩阵（世界系 → 相机系）
    t : array-like (3,)
        平移向量
    kp_conf_threshold : float
        关键点置信度阈值，低于此值的关键点视为未检测到，默认 0.3
    det_conf_threshold : float
        目标检测置信度阈值，默认 0.3
    method : str
        身高估计方法，"robust"（默认）或 "single"
    """

    def __init__(
        self,
        weights: str,
        K,
        R,
        t,
        kp_conf_threshold: float = 0.3,
        det_conf_threshold: float = 0.3,
        method: str = "robust",
    ):
        self.model = YOLO(weights)
        self.K = np.array(K, dtype=np.float64)
        self.R = np.array(R, dtype=np.float64)
        self.t = np.array(t, dtype=np.float64).ravel()
        self.kp_conf_thr = kp_conf_threshold
        self.det_conf_thr = det_conf_threshold
        self.method = method
        # 每个人物的帧间身高缓冲（滑窗 60 帧，用于中位数平滑）
        self._height_buffers: dict = defaultdict(lambda: deque(maxlen=60))

    # ------------------------------------------------------------------ #
    #  公共接口
    # ------------------------------------------------------------------ #
    def run_image(self, image_path: str, save_vis: bool = True) -> list[dict]:
        """
        对单张图片执行检测与身高估计。

        Parameters
        ----------
        image_path : str
            输入图片路径
        save_vis : bool
            是否将可视化结果写入 output/ 目录

        Returns
        -------
        list[dict]
            每个检测到的人物对应一个结果字典（见 _process_person 文档）
        """
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"无法读取图片：{image_path}")

        results, vis_frame = self._process_frame(frame)

        if save_vis:
            out_dir = Path("output")
            out_dir.mkdir(exist_ok=True)
            stem = Path(image_path).stem
            out_path = out_dir / f"{stem}_height_est.jpg"
            cv2.imwrite(str(out_path), vis_frame)
            print(f"[可视化] 已保存至 {out_path}")

        return results

    def run_video(
        self,
        source,
        save_vis: bool = True,
        show: bool = False,
    ) -> None:
        """
        对视频文件或摄像头流执行逐帧检测与身高估计。

        Parameters
        ----------
        source : str | int
            视频文件路径，或摄像头设备号（如 0）
        save_vis : bool
            是否将结果写入 output/ 目录
        show : bool
            是否实时显示窗口（需要 GUI 环境）
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频源：{source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if save_vis:
            out_dir = Path("output")
            out_dir.mkdir(exist_ok=True)
            stem = Path(str(source)).stem if isinstance(source, str) else "camera"
            out_path = out_dir / f"{stem}_height_est.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
            print(f"[可视化] 视频将写入 {out_path}")

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_results, vis_frame = self._process_frame(frame, frame_idx)
                frame_idx += 1

                # 打印当次帧的估计结果
                for r in frame_results:
                    if r["valid"]:
                        print(
                            f"Frame {frame_idx:04d} | 人物 {r['person_id']:02d} | "
                            f"身高估计: {r['height']:.3f} m"
                        )

                if writer is not None:
                    writer.write(vis_frame)
                if show:
                    cv2.imshow("Height Estimation", vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if show:
                cv2.destroyAllWindows()

    # ------------------------------------------------------------------ #
    #  内部方法
    # ------------------------------------------------------------------ #
    def _process_frame(self, frame: np.ndarray, frame_idx: int = 0):
        """
        对单帧图像运行推理并估计所有可用人物的身高。

        Returns
        -------
        (results, vis_frame)
            results   : list[dict]
            vis_frame : BGR 可视化图像
        """
        vis_frame = frame.copy()

        # YOLO 推理
        yolo_results = self.model(
            frame,
            conf=self.det_conf_thr,
            verbose=False,
        )

        all_results = []

        # yolo_results 是列表，每个元素对应一张输入图片
        for yolo_res in yolo_results:
            boxes = yolo_res.boxes       # 检测框
            keypoints = yolo_res.keypoints  # 骨骼关键点

            if boxes is None or keypoints is None:
                continue

            n_persons = len(boxes)
            for person_idx in range(n_persons):
                result = self._process_person(
                    person_idx=person_idx,
                    box=boxes[person_idx],
                    kps=keypoints[person_idx],
                )
                # 在帧上绘制结果
                self._draw_result(vis_frame, result)
                all_results.append(result)

        return all_results, vis_frame

    def _process_person(
        self,
        person_idx: int,
        box,
        kps,
    ) -> dict:
        """
        处理单个人物的检测框与关键点，返回结果字典。

        关键点完整性约束：
          - 至少一个头部关键点（索引 0-4）置信度 ≥ kp_conf_threshold
          - 至少一个脚部关键点（索引 15/16）置信度 ≥ kp_conf_threshold

        bbox 坐标用于构造 head_px 和 foot_px：
          - head_px = (bbox_center_x, bbox_y_min)  # 框的顶边中点
          - foot_px = (bbox_center_x, bbox_y_max)  # 框的底边中点

        Returns
        -------
        dict 包含：
          person_id   : int    — 人物序号（从 0 开始）
          valid       : bool   — 是否满足关键点约束并完成了身高估计
          skip_reason : str    — 跳过原因（valid=False 时）
          bbox        : tuple  — (x1, y1, x2, y2) 像素坐标
          head_px     : tuple  — (u, v) 头部像素坐标
          foot_px     : tuple  — (u, v) 脚部像素坐标
          height      : float  — 估计身高（valid=True 时有效，否则 None）
          P_foot      : ndarray — 脚底 3D 坐标（valid=True 时有效）
          P_head      : ndarray — 头顶 3D 坐标（valid=True 时有效）
          det_conf    : float  — 目标检测置信度
        """
        # ── 解析检测框 ──────────────────────────────────────────────────
        # boxes.xyxy 形状 (1, 4)，取第一行
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
        det_conf = float(box.conf[0].cpu().numpy())

        # 从 bbox 推导 head_px 和 foot_px
        bbox_center_x = (x1 + x2) / 2.0
        head_px = (bbox_center_x, y1)   # 框顶边中点
        foot_px = (bbox_center_x, y2)   # 框底边中点

        base_result = {
            "person_id": person_idx,
            "valid": False,
            "skip_reason": "",
            "bbox": (x1, y1, x2, y2),
            "head_px": head_px,
            "foot_px": foot_px,
            "height": None,
            "P_foot": None,
            "P_head": None,
            "det_conf": det_conf,
            "kp_xy": None,    # 供绘制骨骼时使用
            "kp_conf": None,  # 供绘制骨骼时使用
        }

        # ── 解析关键点 ──────────────────────────────────────────────────
        # kps.xy  : (1, 17, 2)  像素坐标
        # kps.conf: (1, 17)     置信度
        kp_xy = kps.xy[0].cpu().numpy()       # shape (17, 2)
        kp_conf = kps.conf[0].cpu().numpy()   # shape (17,)
        base_result["kp_xy"] = kp_xy
        base_result["kp_conf"] = kp_conf

        # ── 关键点完整性约束 ────────────────────────────────────────────
        has_head = any(
            kp_conf[i] >= self.kp_conf_thr for i in HEAD_KP_INDICES
        )
        has_foot = any(
            kp_conf[i] >= self.kp_conf_thr for i in FOOT_KP_INDICES
        )

        if not has_head:
            base_result["skip_reason"] = "头部关键点未检测到（置信度不足）"
            return base_result
        if not has_foot:
            base_result["skip_reason"] = "脚部关键点未检测到（置信度不足）"
            return base_result

        # ── 调用身高估计 ────────────────────────────────────────────────
        try:
            est_result = estimate_height(
                K=self.K,
                R=self.R,
                t=self.t,
                foot_px=foot_px,
                head_px=head_px,
                method=self.method,
            )
        except Exception as exc:
            base_result["skip_reason"] = f"身高估计失败: {exc}"
            return base_result

        # 帧间平滑：60 帧滑动窗口取中位数，使显示身高稳定
        self._height_buffers[person_idx].append(est_result["height"])
        smoothed_height = float(np.median(list(self._height_buffers[person_idx])))

        base_result.update(
            {
                "valid": True,
                "height": smoothed_height,
                "P_foot": est_result["P_foot"],
                "P_head": est_result["P_head"],
            }
        )
        return base_result

    def _draw_result(self, frame: np.ndarray, result: dict) -> None:
        """在帧上绘制骨骼关键点、检测框和身高估计结果。"""
        x1, y1, x2, y2 = (int(v) for v in result["bbox"])
        color = COLOR_COMPLETE if result["valid"] else COLOR_INCOMPLETE

        # 先绘制骨骼（在检测框之下，避免遮挡文字）
        if result["kp_xy"] is not None:
            self._draw_skeleton(frame, result["kp_xy"], result["kp_conf"])

        # 绘制检测框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 绘制头部和脚部标记点
        hu, hv = int(result["head_px"][0]), int(result["head_px"][1])
        fu, fv = int(result["foot_px"][0]), int(result["foot_px"][1])
        cv2.circle(frame, (hu, hv), 5, (0, 255, 255), -1)   # 头部：黄色
        cv2.circle(frame, (fu, fv), 5, (255, 100, 0), -1)   # 脚部：蓝色

        # 文字标签
        if result["valid"]:
            label = f"#{result['person_id']} H={result['height']:.2f}m"
        else:
            label = f"#{result['person_id']} SKIP"

        # 背景矩形让文字更清晰
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = max(y1 - 6, th + 4)
        cv2.rectangle(
            frame,
            (x1, label_y - th - 4),
            (x1 + tw + 4, label_y + 2),
            color, -1,
        )
        cv2.putText(
            frame, label,
            (x1 + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2, cv2.LINE_AA,
        )

    def _draw_skeleton(self, frame: np.ndarray, kp_xy: np.ndarray, kp_conf: np.ndarray) -> None:
        """在帧上绘制 COCO-17 骨骼连接线和关键点圆圈。"""
        # 绘制肢体连接线
        for (i, j), color in zip(COCO_SKELETON, SKELETON_COLORS):
            if kp_conf[i] >= self.kp_conf_thr and kp_conf[j] >= self.kp_conf_thr:
                pt1 = (int(kp_xy[i, 0]), int(kp_xy[i, 1]))
                pt2 = (int(kp_xy[j, 0]), int(kp_xy[j, 1]))
                cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)
        # 绘制关键点圆圈
        for idx in range(len(kp_xy)):
            if kp_conf[idx] >= self.kp_conf_thr:
                cx_kp = int(kp_xy[idx, 0])
                cy_kp = int(kp_xy[idx, 1])
                cv2.circle(frame, (cx_kp, cy_kp), 3, (0, 255, 255), -1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
#  命令行入口
# --------------------------------------------------------------------------- #
def _parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO11-Pose 人体检测 + 单目身高估计"
    )
    parser.add_argument(
        "--weights", default="weights/yolo11s-pose.pt",
        help="YOLO11-Pose 权重文件路径（默认：weights/yolo11s-pose.pt）",
    )
    parser.add_argument(
        "--source", default=None,
        help="输入源：图片路径、视频路径或摄像头编号（不传则使用内置示例）",
    )
    parser.add_argument(
        "--K", default=None,
        help="相机内参矩阵，JSON 格式，如 '[[1200,0,960],[0,1200,540],[0,0,1]]'",
    )
    parser.add_argument("--R", default=None, help="旋转矩阵，JSON 格式 3×3")
    parser.add_argument("--t", default=None, help="平移向量，JSON 格式 [tx,ty,tz]")
    parser.add_argument(
        "--cam_pos", default=None,
        help="相机位置，JSON 格式 [x,y,z]（与 --target 配合使用，优先级低于 --R/--t）",
    )
    parser.add_argument(
        "--target", default=None,
        help="相机注视点，JSON 格式 [x,y,z]",
    )
    parser.add_argument(
        "--method", default="robust", choices=["robust", "single"],
        help="身高估计方法（默认：robust）",
    )
    parser.add_argument(
        "--kp_conf", type=float, default=0.3,
        help="关键点置信度阈值（默认：0.3）",
    )
    parser.add_argument(
        "--det_conf", type=float, default=0.3,
        help="检测置信度阈值（默认：0.3）",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="视频模式下实时显示窗口（需要 GUI 环境）",
    )
    parser.add_argument(
        "--no_save", action="store_true",
        help="不保存可视化输出",
    )
    return parser.parse_args()


def _build_camera_params(args):
    """从命令行参数解析相机内外参，未提供则使用默认值。"""
    K = np.array(json.loads(args.K), dtype=np.float64) if args.K else DEMO_K.copy()

    # 优先使用 R + t；其次使用 cam_pos + target；最后使用假定参数
    if args.R and args.t:
        R = np.array(json.loads(args.R), dtype=np.float64)
        t = np.array(json.loads(args.t), dtype=np.float64).ravel()
        print("[相机参数] 使用命令行传入的 R 和 t")
    elif args.cam_pos and args.target:
        cam_pos = json.loads(args.cam_pos)
        target = json.loads(args.target)
        # 通过 estimate_height_from_lookat 内部推导 R/t
        # 借用一个虚拟调用来获取 R/t（传一个合法像素坐标）
        from height_estimator import HeightEstimator
        _tmp = HeightEstimator.from_look_at(cam_pos=cam_pos, target=target, K=K)
        R, t = _tmp.R.copy(), _tmp.t.copy()
        del _tmp
        print(f"[相机参数] 由 cam_pos={cam_pos} target={target} 推导 R/t")
    else:
        R = DEMO_R.copy()
        t = DEMO_T.copy()
        print("[相机参数] 未传入参数，使用假定内外参（fx=fy=800，光心(480,360)，相机高 2m 俯视 8m 处地面）")

    return K, R, t


def main():
    args = _parse_args()

    K, R, t = _build_camera_params(args)

    pipeline = DetectAndEstimate(
        weights=args.weights,
        K=K, R=R, t=t,
        kp_conf_threshold=args.kp_conf,
        det_conf_threshold=args.det_conf,
        method=args.method,
    )

    save_vis = not args.no_save

    # ── 无 source：说明输出使用方式，不处理真实图像 ───────────────────
    if args.source is None:
        print("=" * 60)
        print("YOLO11-Pose 人体检测 + 单目身高估计")
        print("=" * 60)
        print("未传入 --source，以下为使用说明：\n")
        print("  图片示例：")
        print("    python src/detect_and_estimate.py \\")
        print("      --source path/to/image.jpg \\")
        print("      --K '[[1200,0,960],[0,1200,540],[0,0,1]]' \\")
        print("      --cam_pos '[0,0,4]' --target '[0,12,0]'\n")
        print("  视频示例：")
        print("    python src/detect_and_estimate.py \\")
        print("      --source path/to/video.mp4 \\")
        print("      --K '[[1200,0,960],[0,1200,540],[0,0,1]]' \\")
        print("      --R '[[1,0,0],[0,0.316,-0.949],[0,0.949,0.316]]' \\")
        print("      --t '[0,-1.265,3.795]'\n")
        print("  摄像头示例（设备号 0）：")
        print("    python src/detect_and_estimate.py --source 0 \\")
        print("      --K '[[1200,0,960],[0,1200,540],[0,0,1]]' \\")
        print("      --cam_pos '[0,0,4]' --target '[0,12,0]'\n")
        print("相机参数说明见 USAGE.md")
        return

    source = args.source
    # 如果 source 是纯数字字符串，转为整数（摄像头编号）
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    # ── 根据输入类型分发处理 ─────────────────────────────────────────
    if isinstance(source, int):
        # 摄像头
        pipeline.run_video(source, save_vis=save_vis, show=args.show)
    else:
        ext = Path(source).suffix.lower()
        if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}:
            results = pipeline.run_image(source, save_vis=save_vis)
            # 打印汇总
            print("\n" + "=" * 50)
            print(f"检测到 {len(results)} 人")
            for r in results:
                if r["valid"]:
                    print(
                        f"  人物 #{r['person_id']:02d} | 估计身高: {r['height']:.3f} m"
                        f" | 检测置信度: {r['det_conf']:.2f}"
                    )
                else:
                    print(
                        f"  人物 #{r['person_id']:02d} | 跳过：{r['skip_reason']}"
                        f" | 检测置信度: {r['det_conf']:.2f}"
                    )
        else:
            # 视频
            pipeline.run_video(source, save_vis=save_vis, show=args.show)


if __name__ == "__main__":
    main()
