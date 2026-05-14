# -*- coding: utf-8 -*-
"""
face_recognition 人脸识别演示程序
=====================================
功能：
    - 支持从摄像头或视频文件读取输入
    - 支持 CPU（HOG 模型）或 GPU（CNN 模型）进行人脸检测
    - 实时识别视频中的人脸，并与已知人脸数据库进行比对
    - 支持将识别结果保存为视频文件

依赖库：
    pip install face_recognition opencv-python numpy

使用方法：
    # 使用默认摄像头 + CPU 模式
    python demo.py

    # 使用摄像头 + GPU 模式
    python demo.py --device gpu

    # 读取视频文件 + CPU 模式
    python demo.py --input ./video/test.mp4

    # 读取视频文件 + GPU 模式，并保存结果
    python demo.py --input ./video/test.mp4 --device gpu --output result.avi

作者：GitHub Copilot
"""

import os          # 文件路径操作
import sys         # 系统模块，用于退出程序
import argparse    # 命令行参数解析
import time        # 时间模块，用于计算帧率

import cv2         # OpenCV，用于视频读写和图像绘制
import numpy as np # NumPy，用于数组运算

# 检测是否为无头（无显示器）环境
# 在服务器/容器中没有 DISPLAY 环境变量，无法使用 cv2.imshow
_HEADLESS = not bool(os.environ.get('DISPLAY', ''))
if _HEADLESS:
    # 使用不需要 GUI 的 OpenCV 后端
    os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '0')

# 导入 face_recognition 库
# 如果未安装，请执行：pip install face_recognition
try:
    import face_recognition
except ImportError:
    print("[错误] 未安装 face_recognition 库，请执行：pip install face_recognition")
    sys.exit(1)


# ============================================================
# 工具函数
# ============================================================

def check_gpu_available():
    """
    检查当前 dlib 是否支持 CUDA（GPU 加速）。
    
    face_recognition 底层使用 dlib，只有当 dlib 以 CUDA 支持编译时，
    GPU 模式才会真正加速。否则即使使用 CNN 模型，依然在 CPU 上运行。
    
    返回：
        bool: True 表示支持 CUDA，False 表示不支持
    """
    try:
        import dlib
        # dlib.DLIB_USE_CUDA 是一个布尔值，表示 dlib 是否使用了 CUDA 编译
        cuda_available = dlib.DLIB_USE_CUDA
        if cuda_available:
            print("[信息] 检测到 CUDA 支持，将使用 GPU（CNN 模型）进行推理")
        else:
            print("[警告] dlib 未使用 CUDA 编译，GPU 加速不可用，将回退到 CPU 模式")
        return cuda_available
    except AttributeError:
        # 旧版 dlib 可能没有 DLIB_USE_CUDA 属性
        print("[警告] 无法检测 CUDA 状态（dlib 版本较旧），将使用 CPU 模式")
        return False


def load_known_faces(known_faces_dir):
    """
    从指定目录加载已知人脸图片，并提取人脸特征编码。
    
    目录结构要求：
        known_faces/
        ├── 张三.jpg     <- 文件名（不含扩展名）即为人名
        ├── 李四.png
        └── obama.jpg
    
    参数：
        known_faces_dir (str): 已知人脸图片所在目录路径
    
    返回：
        known_encodings (list): 人脸特征编码列表（每个编码是 128 维向量）
        known_names (list): 对应的人名列表
    """
    known_encodings = []  # 存储所有已知人脸的特征编码
    known_names = []      # 存储对应的人名

    # 检查目录是否存在
    if not os.path.isdir(known_faces_dir):
        print(f"[警告] 已知人脸目录不存在：{known_faces_dir}")
        print("[提示] 请创建该目录并放入人脸图片，文件名即为人名")
        return known_encodings, known_names

    # 支持的图片格式
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

    print(f"[信息] 正在从 '{known_faces_dir}' 目录加载已知人脸...")

    # 遍历目录中的所有图片文件
    for filename in os.listdir(known_faces_dir):
        # 检查文件扩展名是否为支持的图片格式
        if not filename.lower().endswith(supported_formats):
            continue

        # 提取人名（文件名去掉扩展名）
        name = os.path.splitext(filename)[0]
        image_path = os.path.join(known_faces_dir, filename)

        try:
            # 使用 face_recognition 加载图片（返回 RGB numpy 数组）
            image = face_recognition.load_image_file(image_path)

            # 提取图片中的人脸特征编码
            # face_encodings() 返回图片中所有人脸的编码列表
            # 每个编码是一个 128 维的特征向量
            encodings = face_recognition.face_encodings(image)

            if len(encodings) == 0:
                # 图片中没有检测到人脸
                print(f"  [跳过] {filename}：未检测到人脸")
                continue
            elif len(encodings) > 1:
                # 图片中有多张人脸，只取第一张
                print(f"  [警告] {filename}：检测到 {len(encodings)} 张人脸，仅使用第一张")

            # 将第一张人脸的编码和对应人名加入列表
            known_encodings.append(encodings[0])
            known_names.append(name)
            print(f"  [成功] 加载：{name}")

        except Exception as e:
            print(f"  [错误] 加载 {filename} 时出错：{e}")

    print(f"[信息] 成功加载 {len(known_names)} 个已知人脸：{known_names}")
    return known_encodings, known_names


def open_video_source(input_source, camera_id):
    """
    打开视频源（摄像头或视频文件）。
    
    参数：
        input_source (str): 'camera' 表示摄像头，否则为视频文件路径
        camera_id (int): 摄像头编号（仅在 input_source='camera' 时使用）
    
    返回：
        cap (cv2.VideoCapture): 视频捕获对象
        is_camera (bool): 是否为摄像头输入
    """
    if input_source == 'camera':
        # 打开摄像头，camera_id=0 表示默认摄像头
        print(f"[信息] 正在打开摄像头（编号：{camera_id}）...")
        cap = cv2.VideoCapture(camera_id)
        is_camera = True

        if not cap.isOpened():
            print(f"[错误] 无法打开摄像头（编号：{camera_id}）")
            print("[提示] 请检查摄像头是否连接，或尝试其他摄像头编号（--camera-id 1）")
            sys.exit(1)

        # 设置摄像头分辨率（可选）
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f"[信息] 摄像头已打开，分辨率：{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    else:
        # 打开视频文件
        if not os.path.isfile(input_source):
            print(f"[错误] 视频文件不存在：{input_source}")
            sys.exit(1)

        print(f"[信息] 正在打开视频文件：{input_source}")
        cap = cv2.VideoCapture(input_source)
        is_camera = False

        if not cap.isOpened():
            print(f"[错误] 无法打开视频文件：{input_source}")
            sys.exit(1)

        # 获取并显示视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[信息] 视频信息：{width}x{height}，{fps:.1f} FPS，共 {total_frames} 帧")

    return cap, is_camera


def create_video_writer(output_path, cap):
    """
    创建视频写入器，用于保存处理后的视频。
    
    参数：
        output_path (str): 输出视频文件路径
        cap (cv2.VideoCapture): 视频捕获对象（用于获取原视频参数）
    
    返回：
        writer (cv2.VideoWriter): 视频写入器对象
    """
    # 获取原视频的宽高和帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0  # 默认帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 使用 XVID 编码器（兼容性好，输出 .avi 格式）
    # 也可以使用 'mp4v' 编码器输出 .mp4 格式
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        print(f"[警告] 无法创建输出视频文件：{output_path}")
        return None

    print(f"[信息] 输出视频将保存至：{output_path}（{width}x{height}，{fps:.1f} FPS）")
    return writer


# ============================================================
# 核心处理函数
# ============================================================

def process_frame(frame, known_encodings, known_names, detection_model, scale, tolerance):
    """
    对单帧视频进行人脸检测和识别。
    
    处理流程：
        1. 缩小图像以加快处理速度
        2. 转换颜色空间（BGR -> RGB）
        3. 检测人脸位置（HOG/CNN 模型）
        4. 提取人脸特征编码
        5. 与已知人脸库进行比对
        6. 返回识别结果
    
    参数：
        frame (numpy.ndarray): 原始视频帧（BGR 格式，OpenCV 默认）
        known_encodings (list): 已知人脸特征编码列表
        known_names (list): 已知人名列表
        detection_model (str): 检测模型，'hog'（CPU）或 'cnn'（GPU）
        scale (float): 图像缩放比例，用于加快处理速度
        tolerance (float): 人脸匹配容差，越小越严格（默认 0.6）
    
    返回：
        results (list): 每个元素为 (位置, 人名, 相似度) 的元组
                        位置格式：(top, right, bottom, left)（原始尺寸坐标）
    """
    # ----------------------------------------------------------
    # 步骤 1：缩小图像以加快处理速度
    # 例如 scale=0.25 表示缩小到原来的 1/4，处理速度提升约 16 倍
    # ----------------------------------------------------------
    if scale != 1.0:
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    else:
        small_frame = frame

    # ----------------------------------------------------------
    # 步骤 2：颜色空间转换 BGR -> RGB
    # OpenCV 默认使用 BGR 格式，而 face_recognition 需要 RGB 格式
    # 使用 [:, :, ::-1] 翻转颜色通道顺序
    # 注意：需要 np.ascontiguousarray 确保内存连续，否则 dlib 会报类型错误
    # ----------------------------------------------------------
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

    # ----------------------------------------------------------
    # 步骤 3：检测人脸位置
    # detection_model='hog'：使用 HOG 方向梯度直方图检测，速度快，适合 CPU
    # detection_model='cnn'：使用 CNN 卷积神经网络检测，精度高，适合 GPU
    # 返回的 face_locations 格式：[(top, right, bottom, left), ...]
    # ----------------------------------------------------------
    face_locations = face_recognition.face_locations(
        rgb_small_frame,
        model=detection_model  # 'hog' 或 'cnn'
    )

    # 如果没有检测到人脸，直接返回空结果
    if not face_locations:
        return []

    # ----------------------------------------------------------
    # 步骤 4：提取人脸特征编码
    # face_encodings() 将每张检测到的人脸转换为 128 维特征向量
    # 这个特征向量代表了人脸的独特特征，可用于比对不同人脸
    # ----------------------------------------------------------
    face_encodings_in_frame = face_recognition.face_encodings(
        rgb_small_frame,
        face_locations  # 指定人脸位置，避免重复检测
    )

    # ----------------------------------------------------------
    # 步骤 5：与已知人脸库比对，识别每张人脸
    # ----------------------------------------------------------
    results = []
    for face_encoding, face_location in zip(face_encodings_in_frame, face_locations):
        name = "Unknown"         # 默认为未知人物
        similarity_percent = 0.0 # 相似度百分比

        if known_encodings:
            # compare_faces()：将当前人脸与所有已知人脸比对
            # 返回布尔列表，True 表示匹配，False 表示不匹配
            # tolerance 是容差阈值，欧式距离小于该值视为匹配
            matches = face_recognition.compare_faces(
                known_encodings,
                face_encoding,
                tolerance=tolerance
            )

            # face_distance()：计算当前人脸与每个已知人脸的欧式距离
            # 距离越小，表示越相似（距离为 0 表示完全相同）
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            if len(face_distances) > 0:
                # 找到距离最小（最相似）的已知人脸的索引
                best_match_index = np.argmin(face_distances)

                # 如果最相似的人脸通过了匹配阈值，则认为识别成功
                if matches[best_match_index]:
                    name = known_names[best_match_index]

                    # 将欧式距离转换为相似度百分比（仅供参考）
                    # 距离 0 -> 100%，距离 0.6 -> 0%，线性映射
                    distance = face_distances[best_match_index]
                    similarity_percent = max(0.0, (1.0 - distance / tolerance) * 100)

        # ----------------------------------------------------------
        # 步骤 6：将坐标还原为原始图像尺寸
        # 因为我们在缩小的图像上检测，需要将坐标放大回原始尺寸
        # ----------------------------------------------------------
        scale_inv = 1.0 / scale  # 缩放系数的倒数
        top    = int(face_location[0] * scale_inv)
        right  = int(face_location[1] * scale_inv)
        bottom = int(face_location[2] * scale_inv)
        left   = int(face_location[3] * scale_inv)

        results.append(((top, right, bottom, left), name, similarity_percent))

    return results


def draw_results(frame, results):
    """
    在视频帧上绘制人脸识别结果（矩形框 + 标签）。
    
    参数：
        frame (numpy.ndarray): 视频帧（将被原地修改）
        results (list): process_frame() 返回的识别结果
    
    返回：
        frame (numpy.ndarray): 绘制了标注的视频帧
    """
    for (top, right, bottom, left), name, similarity in results:
        # ----------------------------------------------------------
        # 根据识别结果选择颜色
        # 已知人物：绿色 (0, 200, 0)
        # 未知人物：红色 (0, 0, 220)
        # ----------------------------------------------------------
        if name == "Unknown":
            box_color = (0, 0, 220)    # BGR 格式：红色
            text_color = (255, 255, 255)  # 白色文字
        else:
            box_color = (0, 200, 0)    # BGR 格式：绿色
            text_color = (255, 255, 255)  # 白色文字

        # ----------------------------------------------------------
        # 绘制人脸边界框（矩形）
        # cv2.rectangle(图像, 左上角, 右下角, 颜色, 线宽)
        # ----------------------------------------------------------
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, thickness=2)

        # ----------------------------------------------------------
        # 绘制标签背景框（填充矩形，用于提高文字可读性）
        # 位置在人脸框的下方
        # ----------------------------------------------------------
        label_height = 30  # 标签区域高度（像素）
        cv2.rectangle(
            frame,
            (left, bottom),                        # 左上角（人脸框底部）
            (right, bottom + label_height),        # 右下角
            box_color,
            cv2.FILLED  # 填充模式
        )

        # ----------------------------------------------------------
        # 绘制人名和相似度文字
        # 已识别人物显示：姓名 (相似度%)
        # 未知人物显示：Unknown
        # ----------------------------------------------------------
        if name != "Unknown":
            label_text = f"{name} ({similarity:.0f}%)"
        else:
            label_text = "Unknown"

        # cv2.putText(图像, 文字, 位置, 字体, 字号, 颜色, 线宽, 抗锯齿)
        cv2.putText(
            frame,
            label_text,
            (left + 5, bottom + label_height - 8),  # 文字位置（左下对齐）
            cv2.FONT_HERSHEY_SIMPLEX,  # 字体
            0.6,            # 字号
            text_color,     # 颜色
            1,              # 线宽
            cv2.LINE_AA     # 抗锯齿
        )

    return frame


def draw_info_overlay(frame, fps, frame_count, detection_model, device_label, total_faces):
    """
    在视频帧左上角绘制运行状态信息（帧率、设备、人脸数量等）。
    
    参数：
        frame (numpy.ndarray): 视频帧
        fps (float): 当前帧率
        frame_count (int): 已处理帧数
        detection_model (str): 检测模型名称（'hog' 或 'cnn'）
        device_label (str): 设备标签（'CPU' 或 'GPU'）
        total_faces (int): 当前帧检测到的人脸数量
    """
    # 准备要显示的信息行
    info_lines = [
        f"FPS: {fps:.1f}",                                    # 实时帧率
        f"Device: {device_label} ({detection_model.upper()})", # 推理设备和模型
        f"Faces: {total_faces}",                               # 检测到的人脸数
        f"Frame: {frame_count}",                               # 当前帧数
        "Press 'q' to quit | 's' to screenshot",               # 操作提示
    ]

    # 绘制半透明背景（提高文字可读性）
    overlay_width = 340
    overlay_height = len(info_lines) * 22 + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (overlay_width, overlay_height), (0, 0, 0), cv2.FILLED)
    # alpha 混合：将黑色背景以 60% 透明度叠加到原图上
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # 逐行绘制信息文字
    for i, line in enumerate(info_lines):
        y_position = 22 + i * 22  # 每行 22 像素的行高
        cv2.putText(
            frame,
            line,
            (10, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,           # 字号
            (0, 255, 0),    # 绿色文字
            1,
            cv2.LINE_AA
        )


# ============================================================
# 命令行参数定义
# ============================================================

def parse_arguments():
    """
    解析命令行参数，支持灵活配置程序行为。
    
    返回：
        args (argparse.Namespace): 解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description="face_recognition 人脸识别演示程序",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  # 使用摄像头 + CPU 模式（默认）
  python demo.py

  # 使用摄像头 + GPU 模式
  python demo.py --device gpu

  # 读取视频文件 + CPU 模式
  python demo.py --input ./video/test.mp4

  # 读取视频文件 + GPU 模式，并保存结果
  python demo.py --input ./video/test.mp4 --device gpu --output result.avi

  # 提高速度：降低处理分辨率 + 隔帧处理
  python demo.py --scale 0.25 --skip-frames 2
        """
    )

    # ----------------------------------------------------------
    # 输入源参数
    # ----------------------------------------------------------
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='camera',
        help="输入源：'camera' 使用摄像头（默认），或指定视频文件路径（如 ./test.mp4）"
    )

    parser.add_argument(
        '--camera-id', '-c',
        type=int,
        default=0,
        dest='camera_id',
        help="摄像头编号（默认：0，即第一个摄像头），仅在 --input camera 时有效"
    )

    # ----------------------------------------------------------
    # 推理设备参数
    # ----------------------------------------------------------
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='cpu',
        choices=['cpu', 'gpu'],
        help="推理设备：'cpu' 使用 HOG 模型（默认），'gpu' 使用 CNN 模型（需要 CUDA）"
    )

    # ----------------------------------------------------------
    # 已知人脸目录参数
    # ----------------------------------------------------------
    parser.add_argument(
        '--known-faces', '-k',
        type=str,
        default='known_faces',
        dest='known_faces',
        help="已知人脸图片所在目录路径（默认：./known_faces）"
    )

    # ----------------------------------------------------------
    # 性能调优参数
    # ----------------------------------------------------------
    parser.add_argument(
        '--scale', '-s',
        type=float,
        default=0.25,
        help="视频帧缩放比例，越小处理越快（默认：0.25，即缩小为原来的 1/4）"
    )

    parser.add_argument(
        '--skip-frames', '-f',
        type=int,
        default=1,
        dest='skip_frames',
        help="每隔几帧处理一次（默认：1，即每帧都处理；设为 2 表示每隔一帧处理一次）"
    )

    # ----------------------------------------------------------
    # 识别精度参数
    # ----------------------------------------------------------
    parser.add_argument(
        '--tolerance', '-t',
        type=float,
        default=0.6,
        help="人脸匹配容差（默认：0.6；越小越严格，建议范围 0.4~0.7）"
    )

    # ----------------------------------------------------------
    # 输出参数
    # ----------------------------------------------------------
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help="结果视频保存路径（可选，如 output.avi）；不指定则不保存"
    )

    parser.add_argument(
        '--no-display',
        action='store_true',
        default=False,
        dest='no_display',
        help="无头模式：不显示视频窗口（在无显示器的服务器上自动启用）"
    )

    return parser.parse_args()


# ============================================================
# 主函数
# ============================================================

def main():
    """
    程序主入口：
        1. 解析命令行参数
        2. 检测 GPU 可用性并确定推理模型
        3. 加载已知人脸数据库
        4. 打开视频源（摄像头或视频文件）
        5. 逐帧处理并显示结果
        6. 响应用户按键（退出、截图）
        7. 释放资源
    """
    # ----------------------------------------------------------
    # 步骤 1：解析命令行参数
    # ----------------------------------------------------------
    args = parse_arguments()

    print("=" * 55)
    print("      face_recognition 人脸识别演示程序")
    print("=" * 55)
    print(f"  输入源：{'摄像头 #' + str(args.camera_id) if args.input == 'camera' else args.input}")
    print(f"  推理设备：{args.device.upper()}")
    print(f"  帧缩放比例：{args.scale}")
    print(f"  跳帧设置：每 {args.skip_frames} 帧处理一次")
    print(f"  匹配容差：{args.tolerance}")
    print(f"  已知人脸目录：{args.known_faces}")
    if args.output:
        print(f"  输出视频：{args.output}")
    print("=" * 55)

    # 若检测到无头环境，强制启用 no_display
    if _HEADLESS and not args.no_display:
        args.no_display = True
        print("[信息] 检测到无显示器环境，自动启用无头模式（--no-display）")

    # ----------------------------------------------------------
    # 步骤 2：确定推理模型
    # ----------------------------------------------------------
    if args.device == 'gpu':
        # 用户指定 GPU 模式，检查是否真正支持 CUDA
        gpu_available = check_gpu_available()
        if gpu_available:
            detection_model = 'cnn'    # CNN 模型，支持 GPU 加速
            device_label = 'GPU'
        else:
            # CUDA 不可用，回退到 CPU 模式
            print("[信息] 回退到 CPU 模式（HOG 模型）")
            detection_model = 'hog'    # HOG 模型，CPU 运行
            device_label = 'CPU (fallback)'
    else:
        # CPU 模式：使用 HOG 模型，速度快，无需 CUDA
        detection_model = 'hog'
        device_label = 'CPU'
        print(f"[信息] 使用 CPU 模式（HOG 模型）")

    # ----------------------------------------------------------
    # 步骤 3：加载已知人脸数据库
    # ----------------------------------------------------------
    known_encodings, known_names = load_known_faces(args.known_faces)

    if not known_names:
        print("[提示] 未加载任何已知人脸，所有检测到的人脸将标记为 'Unknown'")
        print(f"[提示] 请在 '{args.known_faces}/' 目录中放置人脸图片后重新运行")

    # ----------------------------------------------------------
    # 步骤 4：打开视频源
    # ----------------------------------------------------------
    cap, is_camera = open_video_source(args.input, args.camera_id)

    # ----------------------------------------------------------
    # 步骤 5：可选，创建视频写入器
    # ----------------------------------------------------------
    video_writer = None
    if args.output:
        video_writer = create_video_writer(args.output, cap)

    # ----------------------------------------------------------
    # 步骤 6：初始化运行状态变量
    # ----------------------------------------------------------
    frame_count = 0          # 总帧计数
    process_count = 0        # 已处理帧计数
    last_results = []        # 上一帧的识别结果（跳帧时复用）
    screenshot_count = 0     # 截图计数

    # FPS 计算相关变量
    fps_start_time = time.time()  # FPS 计算起始时间
    fps_frame_count = 0           # FPS 计算帧计数
    current_fps = 0.0             # 当前帧率

    print("\n[信息] 开始处理视频，按 'q' 键退出，按 's' 键截图")
    print("-" * 55)

    # ----------------------------------------------------------
    # 步骤 7：主循环，逐帧处理视频
    # ----------------------------------------------------------
    while True:
        # ——— 读取一帧视频 ———
        ret, frame = cap.read()

        # 如果读取失败（视频结束或摄像头断开），退出循环
        if not ret:
            if is_camera:
                print("[警告] 摄像头读取失败，可能已断开")
            else:
                print("[信息] 视频文件处理完毕")
            break

        frame_count += 1
        fps_frame_count += 1

        # ——— 计算实时 FPS ———
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:  # 每秒更新一次 FPS
            current_fps = fps_frame_count / elapsed
            fps_frame_count = 0
            fps_start_time = time.time()

        # ——— 决定是否对当前帧进行人脸识别处理 ———
        # skip_frames=1 表示每帧都处理
        # skip_frames=2 表示每隔一帧处理一次（提高速度）
        should_process = (frame_count % args.skip_frames == 0)

        if should_process:
            # 对当前帧进行人脸检测和识别
            last_results = process_frame(
                frame,
                known_encodings,
                known_names,
                detection_model,
                args.scale,
                args.tolerance
            )
            process_count += 1

        # ——— 在帧上绘制识别结果 ———
        # 即使本帧跳过处理，也使用上一帧的结果继续显示（避免闪烁）
        draw_results(frame, last_results)

        # ——— 绘制左上角状态信息 ———
        draw_info_overlay(
            frame,
            current_fps,
            frame_count,
            detection_model,
            device_label,
            len(last_results)
        )

        # ——— 将处理后的帧写入输出视频（如果开启了输出）———
        if video_writer is not None:
            video_writer.write(frame)

        # ——— 显示视频窗口 ———
        if not args.no_display:
            window_title = "人脸识别演示 - face_recognition"
            cv2.imshow(window_title, frame)

        # ——— 无头模式：每 50 帧打印一次进度 ———
        if args.no_display and frame_count % 50 == 0:
            face_info = ", ".join(
                f"{name}({sim:.0f}%)" if name != "Unknown" else "Unknown"
                for (_, name, sim) in last_results
            ) or "无人脸"
            print(f"  [帧 {frame_count:04d}] FPS={current_fps:.1f}  检测到 {len(last_results)} 张人脸：{face_info}")

        # ——— 处理键盘输入 ———
        # cv2.waitKey(1) 等待 1ms，返回按键的 ASCII 码
        # & 0xFF 确保在 64 位系统上正常工作
        if not args.no_display:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                # 按 'q' 退出程序
                print("[信息] 用户按下 'q' 键，退出程序")
                break

            elif key == ord('s'):
                # 按 's' 保存当前帧截图
                screenshot_count += 1
                screenshot_path = f"screenshot_{screenshot_count:04d}.jpg"
                cv2.imwrite(screenshot_path, frame)
                print(f"[截图] 已保存截图：{screenshot_path}")

    # ----------------------------------------------------------
    # 步骤 8：释放资源
    # ----------------------------------------------------------
    cap.release()  # 释放摄像头/视频文件资源

    if video_writer is not None:
        video_writer.release()  # 关闭视频写入器，确保文件正确保存
        print(f"[信息] 输出视频已保存至：{args.output}")

    cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口（有头模式下）

    # ——— 打印运行统计信息 ———
    print("\n" + "=" * 55)
    print("              运行统计")
    print("=" * 55)
    print(f"  总帧数：{frame_count}")
    print(f"  已处理帧数：{process_count}")
    print(f"  截图数量：{screenshot_count}")
    if args.output and video_writer is not None:
        print(f"  输出视频：{args.output}")
    print("=" * 55)
    print("程序已正常退出。")


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    """
    当直接运行此脚本时（而非作为模块导入），执行 main() 函数。
    
    运行方式：
        python demo.py [参数]
    
    查看所有参数说明：
        python demo.py --help
    """
    main()
