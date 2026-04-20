# ============================================================
# recognize.py — 实时人脸识别主程序
# 架构：双线程设计
#   线程1（tracking）：摄像头读帧 → 人脸检测 → ByteTrack追踪 → 绘制结果
#   线程2（recognize）：读取共享数据 → 人脸对齐 → ArcFace特征提取 → 余弦匹配
#   两线程通过全局字典 data_mapping / id_face_mapping 共享数据
# ============================================================

import threading  # 用于创建并发线程，实现检测追踪与识别并行运行
import time       # 用于计算帧率（FPS）

import cv2        # OpenCV：摄像头读取、图像处理、窗口显示
import numpy as np  # 数值计算，用于特征向量运算和数组操作
import torch      # PyTorch 框架，用于运行 ArcFace 神经网络
import yaml       # 解析 YAML 配置文件（追踪参数）
from torchvision import transforms  # 图像预处理流水线（Resize、Normalize 等）

from face_alignment.alignment import norm_crop          # 人脸对齐：基于关键点仿射变换裁剪标准人脸
from face_detection.scrfd.detector import SCRFD         # SCRFD 人脸检测器（ONNX推理）
from face_detection.yolov5_face.detector import Yolov5Face  # YOLOv5-Face 检测器（可选替代）
from face_recognition.arcface.model import iresnet_inference  # ArcFace 特征提取模型加载
from face_recognition.arcface.utils import compare_encodings, read_features  # 余弦相似度匹配 + 特征库读取
from face_tracking.tracker.byte_tracker import BYTETracker  # ByteTrack 多目标追踪器
from face_tracking.tracker.visualize import plot_tracking    # 追踪结果可视化（绘制追踪框和ID）

# ── 设备配置 ──────────────────────────────────────────────
# 自动检测是否有可用 GPU，有则用 CUDA 加速，否则回退到 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 人脸检测器初始化（二选一） ──────────────────────────────
# 默认使用 SCRFD（轻量 ONNX 模型，速度快）
detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")
# 可切换为 YOLOv5-Face（取消下行注释并注释上一行）
# detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5n-face.pt")

# ── ArcFace 人脸识别模型初始化 ────────────────────────────
# 加载 iResNet-100 权重，用于将人脸图像转换为 512 维特征向量
recognizer = iresnet_inference(
    model_name="r100", path="face_recognition/arcface/weights/arcface_r100.pth", device=device
)

# ── 加载预先提取好的人脸特征库 ───────────────────────────
# images_names: 人员姓名列表，如 ["张三", "李四", ...]
# images_embs:  对应的 512 维特征向量矩阵，shape = (N, 512)
# 该特征库由 add_persons.py 脚本生成并保存为 .npz 文件
images_names, images_embs = read_features(feature_path="./datasets/face_features/feature")

# ── 线程间共享数据结构 1：追踪ID → 人员姓名 映射表 ──────────
# 由识别线程写入（recognize()），由追踪线程读取（plot_tracking()绘制名字标签）
# key: ByteTrack 分配的追踪ID（整数），value: "姓名:置信度" 字符串
id_face_mapping = {}

# ── 线程间共享数据结构 2：当前帧数据快照 ────────────────────
# 追踪线程（线程1）在每帧处理完后将最新数据写入此字典
# 识别线程（线程2）从此字典读取数据，完成人脸识别后写回 id_face_mapping
data_mapping = {
    "raw_image": [],          # 当前原始帧图像（numpy数组）
    "tracking_ids": [],       # ByteTrack 分配的追踪ID列表
    "detection_bboxes": [],   # 检测框列表，格式 [x1,y1,x2,y2,score]
    "detection_landmarks": [], # 5点关键点列表，用于人脸对齐
    "tracking_bboxes": [],    # 追踪框列表，格式 [x1,y1,x2,y2]
}


def load_config(file_name):
    """
    读取 YAML 配置文件并返回字典。

    Args:
        file_name (str): YAML 文件路径（如 config_tracking.yaml）

    Returns:
        dict: 配置参数字典，包含 track_thresh、match_thresh 等追踪参数
    """
    with open(file_name, "r") as stream:
        try:
            return yaml.safe_load(stream)  # 安全加载 YAML，防止代码注入
        except yaml.YAMLError as exc:
            print(exc)


def process_tracking(frame, detector, tracker, args, frame_id, fps):
    """
    对单帧图像完成：人脸检测 → ByteTrack追踪 → 更新共享数据 → 返回可视化图像。

    Args:
        frame: 当前摄像头帧（BGR numpy数组）
        detector: 人脸检测器实例（SCRFD 或 Yolov5Face）
        tracker: BYTETracker 追踪器实例
        args (dict): 追踪配置参数
        frame_id (int): 当前帧编号（从0开始）
        fps (float): 当前计算出的帧率，用于显示

    Returns:
        numpy.ndarray: 叠加了追踪框和ID标签的可视化图像
    """
    # ── 步骤1：人脸检测 ──────────────────────────────────
    # detect_tracking() 同时返回：
    #   outputs: 检测结果张量（供 tracker.update 使用的格式）
    #   img_info: 包含原始图像和尺寸信息的字典
    #   bboxes:   检测框数组 [x1,y1,x2,y2,score]，shape=(N,5)
    #   landmarks: 5点关键点数组，shape=(N,5,2)
    outputs, img_info, bboxes, landmarks = detector.detect_tracking(image=frame)

    # 初始化本帧的追踪结果列表
    tracking_tlwhs = []   # 格式：[x,y,w,h]（左上角+宽高），供 plot_tracking 使用
    tracking_ids = []     # 每个追踪目标的唯一ID
    tracking_scores = []  # 每个追踪目标的置信度分数
    tracking_bboxes = []  # 格式：[x1,y1,x2,y2]，供识别线程的 IoU 映射使用

    if outputs is not None:
        # ── 步骤2：ByteTrack 更新追踪状态 ──────────────────
        # tracker.update() 接收检测结果，利用卡尔曼滤波预测并关联历史轨迹
        # 返回当前帧所有活跃追踪目标（STrack 对象列表）
        online_targets = tracker.update(
            outputs, [img_info["height"], img_info["width"]], (128, 128)
        )

        # 遍历每个追踪目标，过滤掉过小或比例异常的框
        for i in range(len(online_targets)):
            t = online_targets[i]
            tlwh = t.tlwh   # 目标边界框 [x, y, w, h]
            tid = t.track_id  # ByteTrack 分配的追踪ID（在整个视频生命周期内唯一）

            # 过滤条件1：宽高比 > 阈值则认为不是正常人脸框（太宽的框）
            vertical = tlwh[2] / tlwh[3] > args["aspect_ratio_thresh"]
            # 过滤条件2：面积 > 最小面积阈值 且 比例正常
            if tlwh[2] * tlwh[3] > args["min_box_area"] and not vertical:
                x1, y1, w, h = tlwh
                tracking_bboxes.append([x1, y1, x1 + w, y1 + h])  # 转换为 [x1,y1,x2,y2]
                tracking_tlwhs.append(tlwh)
                tracking_ids.append(tid)
                tracking_scores.append(t.score)

        # ── 步骤3：绘制追踪结果 ──────────────────────────
        # plot_tracking 在图像上绘制追踪框、追踪ID和人员姓名（从 id_face_mapping 读取）
        tracking_image = plot_tracking(
            img_info["raw_img"],   # 原始未加工的帧图像
            tracking_tlwhs,        # 各追踪目标的位置
            tracking_ids,          # 各追踪目标的ID
            names=id_face_mapping, # 识别线程写入的 {ID: 姓名} 映射，绘制姓名标签
            frame_id=frame_id + 1, # 显示帧编号
            fps=fps,               # 显示帧率
        )
    else:
        # 本帧未检测到任何人脸，直接返回原始图像
        tracking_image = img_info["raw_img"]

    # ── 步骤4：更新线程间共享数据 ──────────────────────────
    # 将本帧的检测和追踪结果写入全局 data_mapping
    # 识别线程（线程2）会从此读取数据进行人脸识别
    data_mapping["raw_image"] = img_info["raw_img"]       # 保存原始帧供识别线程裁剪人脸
    data_mapping["detection_bboxes"] = bboxes             # 检测框（含置信度分数）
    data_mapping["detection_landmarks"] = landmarks       # 关键点（对齐用）
    data_mapping["tracking_ids"] = tracking_ids           # 追踪ID列表
    data_mapping["tracking_bboxes"] = tracking_bboxes     # 追踪框列表（用于IoU映射）

    return tracking_image


@torch.no_grad()  # 禁用梯度计算，推理阶段不需要反向传播，节省内存和加速
def get_feature(face_image):
    """
    使用 ArcFace 模型从对齐后的人脸图像中提取 512 维特征向量。

    Args:
        face_image: 已对齐裁剪的人脸图像（112×112 BGR numpy数组）

    Returns:
        numpy.ndarray: 归一化后的 512 维特征向量（单位长度，用于余弦相似度计算）
    """
    # 定义图像预处理流水线（与训练时保持一致）
    face_preprocess = transforms.Compose(
        [
            transforms.ToTensor(),                                    # HWC numpy → CHW FloatTensor，像素值归一化到[0,1]
            transforms.Resize((112, 112)),                            # 统一缩放到 ArcFace 要求的 112×112
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 像素归一化到[-1,1]
        ]
    )

    # OpenCV 默认 BGR 格式，ArcFace 训练时使用 RGB，需转换
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    # 应用预处理并增加 batch 维度：(112,112,3) → (1,3,112,112)，移到指定设备
    face_image = face_preprocess(face_image).unsqueeze(0).to(device)

    # 通过 ArcFace 模型推理，得到 512 维特征向量（模型内部已做 L2 归一化）
    emb_img_face = recognizer(face_image).cpu().numpy()

    # 再次 L2 归一化（确保向量在单位球面上，cos相似度等价于点积）
    images_emb = emb_img_face / np.linalg.norm(emb_img_face)

    return images_emb


def recognition(face_image):
    """
    对单张人脸图像进行身份识别：提取特征 → 与特征库匹配 → 返回最佳匹配结果。

    Args:
        face_image: 对齐后的人脸图像（numpy数组）

    Returns:
        tuple: (score, name)
            - score (float): 余弦相似度分数，范围约 [0, 1]，越高越相似
            - name (str): 匹配到的人员姓名
    """
    # 提取当前人脸的 512 维特征向量
    query_emb = get_feature(face_image)

    # compare_encodings: 计算 query_emb 与特征库 images_embs 中所有人员的余弦相似度
    # 返回最高相似度分数 score 和对应的索引 id_min
    score, id_min = compare_encodings(query_emb, images_embs)

    # 根据索引从姓名列表中取出对应人员姓名
    name = images_names[id_min]
    score = score[0]  # score 是数组，取第一个元素

    return score, name


def mapping_bbox(box1, box2):
    """
    计算两个检测框之间的 IoU（交并比），用于将追踪框和检测框进行关联。

    原理：ByteTrack 输出追踪框，检测器输出检测框，两者坐标可能略有偏差。
    通过 IoU > 0.9 的阈值判断它们是否对应同一张人脸，从而将关键点（检测框附带）
    对应到正确的追踪ID，进而完成人脸识别身份绑定。

    Args:
        box1 (list): 追踪框 [x_min, y_min, x_max, y_max]
        box2 (list): 检测框 [x_min, y_min, x_max, y_max]

    Returns:
        float: IoU 值，范围 [0, 1]，越大表示重叠越多
    """
    # 计算两框交叉区域的左上角和右下角坐标
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])

    # 计算交叉面积（若两框不重叠则为 0）
    intersection_area = max(0, x_max_inter - x_min_inter + 1) * max(
        0, y_max_inter - y_min_inter + 1
    )

    # 分别计算两框各自的面积
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # 计算并集面积（总面积 - 重叠部分）
    union_area = area_box1 + area_box2 - intersection_area

    # IoU = 交集面积 / 并集面积
    iou = intersection_area / union_area

    return iou


def tracking(detector, args):
    """
    线程1：人脸检测 + ByteTrack追踪主循环。

    职责：
    - 持续从摄像头读取帧
    - 每帧调用 process_tracking() 进行检测和追踪
    - 将追踪结果（数据快照）写入全局 data_mapping（供线程2读取）
    - 从全局 id_face_mapping 读取识别结果（由线程2写入）并叠加显示姓名

    Args:
        detector: 人脸检测器实例
        args (dict): 追踪配置参数（来自 config_tracking.yaml）
    """
    # 帧率计算相关变量初始化
    start_time = time.time_ns()  # 记录起始时间（纳秒精度）
    frame_count = 0              # 已处理帧数计数器
    fps = -1                     # 当前帧率（-1 表示尚未计算）

    # 初始化 ByteTrack 追踪器（纯算法，无需预训练权重）
    tracker = BYTETracker(args=args, frame_rate=30)
    frame_id = 0  # 帧编号，从 0 开始递增

    # 打开摄像头（0 = 默认摄像头）
    cap = cv2.VideoCapture(0)

    while True:
        # 从摄像头读取一帧（_ 为返回值布尔，img 为帧图像）
        _, img = cap.read()

        # 核心处理：检测 + 追踪 + 更新 data_mapping + 生成可视化帧
        tracking_image = process_tracking(img, detector, tracker, args, frame_id, fps)

        # 帧率计算：每累计 30 帧计算一次平均帧率
        frame_count += 1
        if frame_count >= 30:
            fps = 1e9 * frame_count / (time.time_ns() - start_time)  # 帧率 = 帧数 / 耗时(秒)
            frame_count = 0
            start_time = time.time_ns()

        # 在窗口中显示带有追踪框和姓名的结果图像
        cv2.imshow("Face Recognition", tracking_image)

        # 检测按键：按 Esc(27)、q 或 Q 退出主循环
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

        frame_id += 1  # 帧编号递增


def recognize():
    """
    线程2：人脸识别主循环。

    职责：
    - 持续从全局 data_mapping 读取线程1写入的最新帧数据
    - 将追踪框与检测框通过 IoU 进行关联（找到对应关键点）
    - 对关联成功的人脸做对齐、特征提取、余弦匹配
    - 将识别结果（{追踪ID: 姓名}）写入全局 id_face_mapping（供线程1显示）

    注意：两线程共享内存，通过 Python 全局字典直接传递数据，无需队列或管道。
    由于 GIL（全局解释器锁），字典读写操作在 CPython 中是原子的，基本安全。
    """
    while True:
        # 从共享数据快照中读取最新帧的数据（线程1写入的）
        raw_image = data_mapping["raw_image"]               # 原始帧，用于裁剪对齐人脸
        detection_landmarks = data_mapping["detection_landmarks"]  # 5点关键点
        detection_bboxes = data_mapping["detection_bboxes"]        # 检测框
        tracking_ids = data_mapping["tracking_ids"]                # 追踪ID列表
        tracking_bboxes = data_mapping["tracking_bboxes"]          # 追踪框列表

        # 遍历每个追踪目标（线程1输出的已追踪人脸）
        for i in range(len(tracking_bboxes)):
            # 遍历所有检测框，寻找与当前追踪框 IoU > 0.9 的检测框
            # 目的：找到该追踪目标对应的关键点（关键点附属在检测框上，不在追踪框上）
            for j in range(len(detection_bboxes)):
                mapping_score = mapping_bbox(box1=tracking_bboxes[i], box2=detection_bboxes[j])
                if mapping_score > 0.9:
                    # ── 找到匹配的检测框，利用其关键点对齐人脸 ──────────
                    # norm_crop: 根据 5 点关键点做仿射变换，输出 112×112 标准正脸
                    face_alignment = norm_crop(img=raw_image, landmark=detection_landmarks[j])

                    # ── 识别对齐后的人脸 ─────────────────────────────
                    score, name = recognition(face_image=face_alignment)

                    if name is not None:
                        if score < 0.25:
                            # 相似度太低（< 0.25），认为是数据库中没有的陌生人
                            caption = "UN_KNOWN"
                        else:
                            # 相似度足够高，显示姓名和置信度（保留2位小数）
                            caption = f"{name}:{score:.2f}"

                    # ── 写入共享映射表（线程1从此读取并显示姓名标签）──────
                    id_face_mapping[tracking_ids[i]] = caption

                    # 该检测框已被匹配，从列表中删除，避免被其他追踪目标重复匹配
                    detection_bboxes = np.delete(detection_bboxes, j, axis=0)
                    detection_landmarks = np.delete(detection_landmarks, j, axis=0)

                    break  # 当前追踪目标已找到匹配，跳出内层循环处理下一个追踪目标

        # 若当前帧没有任何追踪目标（画面中无人），打印等待提示
        if tracking_bboxes == []:
            print("Waiting for a person...")


def main():
    """
    主函数：加载配置 → 启动双线程（追踪线程 + 识别线程）并行运行。

    双线程设计的原因：
    - 人脸检测+追踪 需要高帧率（实时性要求高）
    - ArcFace 特征提取 计算量大，耗时较长
    - 分离为两个线程后，追踪线程不需要等待识别完成，保证显示流畅
    - 识别结果通过 id_face_mapping 异步传递给追踪线程显示
    """
    # 读取追踪配置文件
    file_name = "./face_tracking/config/config_tracking.yaml"
    config_tracking = load_config(file_name)

    # 创建并启动追踪线程（线程1）
    # 负责：摄像头读帧 → 人脸检测 → ByteTrack追踪 → 写 data_mapping → 显示画面
    thread_track = threading.Thread(
        target=tracking,
        args=(
            detector,       # 传入已初始化的检测器
            config_tracking, # 传入追踪配置参数
        ),
    )
    thread_track.start()

    # 创建并启动识别线程（线程2）
    # 负责：读 data_mapping → 人脸对齐 → ArcFace特征提取 → 余弦匹配 → 写 id_face_mapping
    thread_recognize = threading.Thread(target=recognize)
    thread_recognize.start()


if __name__ == "__main__":
    main()
