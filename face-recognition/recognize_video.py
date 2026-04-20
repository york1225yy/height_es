# ============================================================
# recognize_video.py — 以视频文件作为输入的人脸识别程序
# 用法：python recognize_video.py --video path/to/video.mp4
#       python recognize_video.py --video 0        （摄像头，等同于 recognize.py）
# 说明：与 recognize.py 的双线程架构相同，仅将输入源从摄像头改为视频文件
#       增加了视频结果保存功能（output.mp4）
# ============================================================

import argparse    # 命令行参数解析
import threading   # 双线程并发
import time
import os

import cv2
import numpy as np
import torch
import yaml
from torchvision import transforms

from face_alignment.alignment import norm_crop
from face_detection.scrfd.detector import SCRFD
from face_detection.yolov5_face.detector import Yolov5Face
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import compare_encodings, read_features
from face_tracking.tracker.byte_tracker import BYTETracker
from face_tracking.tracker.visualize import plot_tracking

# ── 设备 ──────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 检测器（二选一） ────────────────────────────────────────
detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")
# detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5n-face.pt")

# ── ArcFace 识别模型 ─────────────────────────────────────
recognizer = iresnet_inference(
    model_name="r100", path="face_recognition/arcface/weights/arcface_r100.pth", device=device
)

# ── 人脸特征库 ───────────────────────────────────────────
images_names, images_embs = read_features(feature_path="./datasets/face_features/feature")

# ── 线程间共享数据结构 ────────────────────────────────────
id_face_mapping = {}  # {追踪ID: "姓名:置信度"} 由识别线程写入，追踪线程读取显示

data_mapping = {
    "raw_image": [],
    "tracking_ids": [],
    "detection_bboxes": [],
    "detection_landmarks": [],
    "tracking_bboxes": [],
    "end_of_video": False,  # 视频播放结束标志，通知识别线程退出
}


def load_config(file_name):
    with open(file_name, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def process_tracking(frame, detector, tracker, args, frame_id, fps):
    """检测 + 追踪 + 更新 data_mapping，与 recognize.py 相同。"""
    outputs, img_info, bboxes, landmarks = detector.detect_tracking(image=frame)

    tracking_tlwhs = []
    tracking_ids = []
    tracking_scores = []
    tracking_bboxes = []

    if outputs is not None:
        online_targets = tracker.update(
            outputs, [img_info["height"], img_info["width"]], (128, 128)
        )
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > args["aspect_ratio_thresh"]
            if tlwh[2] * tlwh[3] > args["min_box_area"] and not vertical:
                x1, y1, w, h = tlwh
                tracking_bboxes.append([x1, y1, x1 + w, y1 + h])
                tracking_tlwhs.append(tlwh)
                tracking_ids.append(tid)
                tracking_scores.append(t.score)

        tracking_image = plot_tracking(
            img_info["raw_img"],
            tracking_tlwhs,
            tracking_ids,
            names=id_face_mapping,
            frame_id=frame_id + 1,
            fps=fps,
        )
    else:
        tracking_image = img_info["raw_img"]

    data_mapping["raw_image"] = img_info["raw_img"]
    data_mapping["detection_bboxes"] = bboxes
    data_mapping["detection_landmarks"] = landmarks
    data_mapping["tracking_ids"] = tracking_ids
    data_mapping["tracking_bboxes"] = tracking_bboxes

    return tracking_image


@torch.no_grad()
def get_feature(face_image):
    face_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((112, 112)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_image = face_preprocess(face_image).unsqueeze(0).to(device)
    emb_img_face = recognizer(face_image).cpu().numpy()
    return emb_img_face / np.linalg.norm(emb_img_face)


def recognition(face_image):
    query_emb = get_feature(face_image)
    score, id_min = compare_encodings(query_emb, images_embs)
    return score[0], images_names[id_min]


def mapping_bbox(box1, box2):
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])
    intersection_area = max(0, x_max_inter - x_min_inter + 1) * max(0, y_max_inter - y_min_inter + 1)
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    return intersection_area / (area_box1 + area_box2 - intersection_area)


def tracking(video_source, args, output_path=None):
    """
    线程1：从视频文件（或摄像头）逐帧读取并执行检测追踪。

    Args:
        video_source: 视频文件路径（str）或摄像头索引（int，如 0）
        args: 追踪配置参数
        output_path: 结果视频保存路径（None 则不保存）
    """
    # ── 打开视频源 ────────────────────────────────────────
    # cv2.VideoCapture 统一接受：
    #   整数 → 摄像头索引（0 = 默认摄像头）
    #   字符串 → 视频文件路径（mp4/avi/mov 均支持）
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"[错误] 无法打开视频源: {video_source}")
        data_mapping["end_of_video"] = True
        return

    # 获取视频的宽、高和原始帧率
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[信息] 视频源: {video_source}")
    print(f"[信息] 分辨率: {frame_width}x{frame_height}, FPS: {video_fps:.1f}, 总帧数: {total_frames}")

    # ── 初始化结果视频写入器（可选）────────────────────────
    video_writer = None
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 编码器
        video_writer = cv2.VideoWriter(output_path, fourcc, video_fps, (frame_width, frame_height))
        print(f"[信息] 结果将保存到: {output_path}")

    # ── 帧率计算和追踪器初始化 ────────────────────────────
    start_time = time.time_ns()
    frame_count = 0
    fps = -1
    tracker = BYTETracker(args=args, frame_rate=int(video_fps) or 30)
    frame_id = 0

    while True:
        ret, img = cap.read()

        if not ret:
            # 视频读取完毕（摄像头断开 or 视频结尾）
            print("[信息] 视频播放结束")
            data_mapping["end_of_video"] = True  # 通知识别线程退出
            break

        # 检测 + 追踪 + 更新共享数据
        tracking_image = process_tracking(img, detector, tracker, args, frame_id, fps)

        # 帧率计算（每30帧更新一次）
        frame_count += 1
        if frame_count >= 30:
            fps = 1e9 * frame_count / (time.time_ns() - start_time)
            frame_count = 0
            start_time = time.time_ns()

        # 保存结果帧到视频文件
        if video_writer is not None:
            video_writer.write(tracking_image)

        # 显示结果窗口
        cv2.imshow("Face Recognition - Video", tracking_image)

        # 按 q / Esc 提前退出
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            data_mapping["end_of_video"] = True
            break

        frame_id += 1

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()


def recognize():
    """
    线程2：持续读取 data_mapping → 识别 → 写入 id_face_mapping。
    当 data_mapping["end_of_video"] 为 True 且当前帧无追踪目标时退出。
    """
    while True:
        # 检测到视频结束且无待处理目标时，退出识别线程
        if data_mapping["end_of_video"] and data_mapping["tracking_bboxes"] == []:
            break

        raw_image = data_mapping["raw_image"]
        detection_landmarks = data_mapping["detection_landmarks"]
        detection_bboxes = data_mapping["detection_bboxes"]
        tracking_ids = data_mapping["tracking_ids"]
        tracking_bboxes = data_mapping["tracking_bboxes"]

        for i in range(len(tracking_bboxes)):
            for j in range(len(detection_bboxes)):
                if mapping_bbox(tracking_bboxes[i], detection_bboxes[j]) > 0.9:
                    face_alignment = norm_crop(img=raw_image, landmark=detection_landmarks[j])
                    score, name = recognition(face_image=face_alignment)

                    if name is not None:
                        caption = "UN_KNOWN" if score < 0.25 else f"{name}:{score:.2f}"
                    id_face_mapping[tracking_ids[i]] = caption

                    detection_bboxes = np.delete(detection_bboxes, j, axis=0)
                    detection_landmarks = np.delete(detection_landmarks, j, axis=0)
                    break

        if tracking_bboxes == [] and not data_mapping["end_of_video"]:
            print("Waiting for a person...")


def main():
    # ── 命令行参数解析 ────────────────────────────────────
    parser = argparse.ArgumentParser(description="视频/摄像头人脸识别")
    parser.add_argument(
        "--video",
        type=str,
        default="0",
        help="视频文件路径（如 video.mp4）或摄像头索引（如 0）。默认: 0（摄像头）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/output.mp4",
        help="结果视频保存路径。默认: results/output.mp4（留空则不保存）",
    )
    args_cli = parser.parse_args()

    # 如果 --video 参数是纯数字，转换为整数（摄像头索引）
    video_source = int(args_cli.video) if args_cli.video.isdigit() else args_cli.video

    # 读取追踪配置
    config_tracking = load_config("./face_tracking/config/config_tracking.yaml")

    # 启动追踪线程
    thread_track = threading.Thread(
        target=tracking,
        args=(video_source, config_tracking, args_cli.output),
    )
    thread_track.start()

    # 启动识别线程
    thread_recognize = threading.Thread(target=recognize)
    thread_recognize.start()

    # 等待两个线程全部完成
    thread_track.join()
    thread_recognize.join()
    print("[完成] 程序退出")


if __name__ == "__main__":
    main()
