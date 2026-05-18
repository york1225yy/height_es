# -*- coding: utf-8 -*-
"""
face_recognition Jetson 演示程序
=====================================
功能：
    - 支持 Jetson Orin Nano 上的 USB / CSI 摄像头
    - 支持 CPU（HOG）或 GPU（CNN）人脸检测
    - 支持 GStreamer / V4L2 摄像头后端切换
    - 支持将识别结果保存为视频文件

使用方法：
    # Jetson 摄像头自动探测 + GPU 模式
    python demojetson.py --device gpu

    # Jetson CSI 摄像头
    python demojetson.py --device gpu --camera-backend gstreamer

    # Jetson USB 摄像头
    python demojetson.py --device gpu --camera-backend v4l2 --camera-id 0
"""

import os
import sys
import argparse
import time

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VENDORED_FACE_RECOGNITION_DIR = os.path.join(PROJECT_DIR, 'face_recognition')
SYSTEM_DIST_PACKAGES = '/usr/lib/python3/dist-packages'

if os.path.isdir(SYSTEM_DIST_PACKAGES):
    if SYSTEM_DIST_PACKAGES in sys.path:
        sys.path.remove(SYSTEM_DIST_PACKAGES)
    sys.path.insert(0, SYSTEM_DIST_PACKAGES)

import cv2
import numpy as np


if os.path.isdir(VENDORED_FACE_RECOGNITION_DIR) and VENDORED_FACE_RECOGNITION_DIR not in sys.path:
    sys.path.insert(0, VENDORED_FACE_RECOGNITION_DIR)


_HEADLESS = not bool(os.environ.get('DISPLAY', ''))
if _HEADLESS:
    os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '0')

try:
    import face_recognition
except ImportError:
    print("[错误] 未安装 face_recognition 库，请先配置 Jetson Python 环境")
    sys.exit(1)


def check_gpu_available():
    """检查 dlib 是否启用 CUDA。"""
    try:
        import dlib

        cuda_available = bool(getattr(dlib, 'DLIB_USE_CUDA', False))
        if cuda_available:
            print("[信息] 检测到 CUDA 支持，将使用 GPU（CNN 模型）进行推理")
        else:
            print("[警告] dlib 未启用 CUDA，GPU 模式将回退到 CPU")
        return cuda_available
    except ImportError:
        print("[错误] 未安装 dlib，无法启用 GPU 模式")
        return False
    except AttributeError:
        print("[警告] 当前 dlib 无法确认 CUDA 状态，将回退到 CPU")
        return False


def load_known_faces(known_faces_dir):
    """加载已知人脸编码。"""
    known_encodings = []
    known_names = []
    known_faces_dir = os.path.abspath(known_faces_dir)

    if not os.path.isdir(known_faces_dir):
        print(f"[警告] 已知人脸目录不存在：{known_faces_dir}")
        return known_encodings, known_names

    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    print(f"[信息] 正在从 '{known_faces_dir}' 目录加载已知人脸...")

    for filename in os.listdir(known_faces_dir):
        if not filename.lower().endswith(supported_formats):
            continue

        name = os.path.splitext(filename)[0]
        image_path = os.path.join(known_faces_dir, filename)

        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) == 0:
                print(f"  [跳过] {filename}：未检测到人脸")
                continue
            if len(encodings) > 1:
                print(f"  [警告] {filename}：检测到 {len(encodings)} 张人脸，仅使用第一张")

            known_encodings.append(encodings[0])
            known_names.append(name)
            print(f"  [成功] 加载：{name}")
        except Exception as exc:
            print(f"  [错误] 加载 {filename} 时出错：{exc}")

    print(f"[信息] 成功加载 {len(known_names)} 个已知人脸：{known_names}")
    return known_encodings, known_names


def build_jetson_camera_pipeline(camera_id, width, height, fps):
    """构造适用于 Jetson CSI 摄像头的 GStreamer 管道。"""
    return (
        f"nvarguscamerasrc sensor-id={camera_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"framerate=(fraction){fps}/1 ! "
        "nvvidconv ! video/x-raw, format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=true sync=false"
    )


def open_camera(camera_id, camera_backend, camera_width, camera_height, camera_fps, camera_pipeline):
    """按优先级打开 Jetson 摄像头。"""
    attempts = []

    if camera_pipeline:
        attempts.append(("gstreamer-custom", camera_pipeline, cv2.CAP_GSTREAMER))
    elif camera_backend == 'gstreamer':
        attempts.append((
            "gstreamer-jetson",
            build_jetson_camera_pipeline(camera_id, camera_width, camera_height, camera_fps),
            cv2.CAP_GSTREAMER,
        ))
    elif camera_backend == 'v4l2':
        attempts.append(("v4l2", camera_id, cv2.CAP_V4L2))
    else:
        attempts.extend([
            (
                "gstreamer-jetson",
                build_jetson_camera_pipeline(camera_id, camera_width, camera_height, camera_fps),
                cv2.CAP_GSTREAMER,
            ),
            ("v4l2", camera_id, cv2.CAP_V4L2),
            ("default", camera_id, cv2.CAP_ANY),
        ])

    for backend_name, source, api_preference in attempts:
        print(f"[信息] 尝试打开摄像头后端：{backend_name}")
        cap = cv2.VideoCapture(source, api_preference)
        if not cap.isOpened():
            cap.release()
            continue

        if backend_name in ('v4l2', 'default'):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
            cap.set(cv2.CAP_PROP_FPS, camera_fps)

        return cap, backend_name

    return None, None


def open_video_source(input_source, camera_id, camera_backend='auto', camera_width=640, camera_height=480, camera_fps=30, camera_pipeline=None):
    """打开摄像头或视频文件。"""
    if input_source == 'camera':
        print(f"[信息] 正在打开摄像头（编号：{camera_id}，后端：{camera_backend}）...")
        cap, selected_backend = open_camera(
            camera_id,
            camera_backend,
            camera_width,
            camera_height,
            camera_fps,
            camera_pipeline,
        )
        is_camera = True

        if cap is None or not cap.isOpened():
            print(f"[错误] 无法打开摄像头（编号：{camera_id}）")
            print("[提示] USB 摄像头可尝试：--camera-backend v4l2")
            print("[提示] CSI 摄像头可尝试：--camera-backend gstreamer")
            sys.exit(1)

        print(
            f"[信息] 摄像头已打开（后端：{selected_backend}），分辨率："
            f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
        )
    else:
        if not os.path.isfile(input_source):
            print(f"[错误] 视频文件不存在：{input_source}")
            sys.exit(1)

        print(f"[信息] 正在打开视频文件：{input_source}")
        cap = cv2.VideoCapture(input_source)
        is_camera = False

        if not cap.isOpened():
            print(f"[错误] 无法打开视频文件：{input_source}")
            sys.exit(1)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[信息] 视频信息：{width}x{height}，{fps:.1f} FPS，共 {total_frames} 帧")

    return cap, is_camera


def create_video_writer(output_path, cap):
    """创建输出视频写入器。"""
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        print(f"[警告] 无法创建输出视频文件：{output_path}")
        return None

    print(f"[信息] 输出视频将保存至：{output_path}（{width}x{height}，{fps:.1f} FPS）")
    return writer


def process_frame(frame, known_encodings, known_names, detection_model, scale, tolerance):
    """检测并识别人脸。"""
    if scale != 1.0:
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    else:
        small_frame = frame

    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
    face_locations = face_recognition.face_locations(rgb_small_frame, model=detection_model)
    if not face_locations:
        return []

    face_encodings_in_frame = face_recognition.face_encodings(rgb_small_frame, face_locations)
    results = []

    for face_encoding, face_location in zip(face_encodings_in_frame, face_locations):
        name = 'Unknown'
        similarity_percent = 0.0

        if known_encodings:
            matches = face_recognition.compare_faces(
                known_encodings,
                face_encoding,
                tolerance=tolerance,
            )
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    distance = face_distances[best_match_index]
                    similarity_percent = max(0.0, (1.0 - distance / tolerance) * 100)

        scale_inv = 1.0 / scale
        top = int(face_location[0] * scale_inv)
        right = int(face_location[1] * scale_inv)
        bottom = int(face_location[2] * scale_inv)
        left = int(face_location[3] * scale_inv)
        results.append(((top, right, bottom, left), name, similarity_percent))

    return results


def draw_results(frame, results):
    """绘制识别框与标签。"""
    for (top, right, bottom, left), name, similarity in results:
        if name == 'Unknown':
            box_color = (0, 0, 220)
            text_color = (255, 255, 255)
        else:
            box_color = (0, 200, 0)
            text_color = (255, 255, 255)

        cv2.rectangle(frame, (left, top), (right, bottom), box_color, thickness=2)

        label_height = 30
        cv2.rectangle(frame, (left, bottom), (right, bottom + label_height), box_color, cv2.FILLED)

        label_text = f"{name} ({similarity:.0f}%)" if name != 'Unknown' else 'Unknown'
        cv2.putText(
            frame,
            label_text,
            (left + 5, bottom + label_height - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            1,
            cv2.LINE_AA,
        )

    return frame


def draw_info_overlay(frame, fps, frame_count, detection_model, device_label, total_faces, camera_backend):
    """绘制运行状态信息。"""
    info_lines = [
        f"FPS: {fps:.1f}",
        f"Device: {device_label} ({detection_model.upper()})",
        f"Camera: {camera_backend}",
        f"Faces: {total_faces}",
        f"Frame: {frame_count}",
        "Press 'q' to quit | 's' to screenshot",
    ]

    overlay_width = 360
    overlay_height = len(info_lines) * 22 + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (overlay_width, overlay_height), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for index, line in enumerate(info_lines):
        y_position = 22 + index * 22
        cv2.putText(frame, line, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)


def parse_arguments():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description='face_recognition Jetson 演示程序',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  python demojetson.py --device gpu
  python demojetson.py --device gpu --camera-backend gstreamer
  python demojetson.py --device gpu --camera-backend v4l2 --camera-id 0
  python demojetson.py --input ./video/test.mp4 --device gpu --no-display
        """,
    )

    parser.add_argument('--input', '-i', type=str, default='camera', help="输入源：'camera' 或视频文件路径")
    parser.add_argument('--camera-id', '-c', type=int, default=0, dest='camera_id', help='摄像头编号（默认：0）')
    parser.add_argument(
        '--camera-backend',
        type=str,
        default='auto',
        choices=['auto', 'v4l2', 'gstreamer'],
        dest='camera_backend',
        help='摄像头后端：auto 自动探测，v4l2 适合 USB，gstreamer 适合 CSI',
    )
    parser.add_argument('--camera-width', type=int, default=640, dest='camera_width', help='摄像头宽度（默认：640）')
    parser.add_argument('--camera-height', type=int, default=480, dest='camera_height', help='摄像头高度（默认：480）')
    parser.add_argument('--camera-fps', type=int, default=30, dest='camera_fps', help='摄像头帧率（默认：30）')
    parser.add_argument('--camera-pipeline', type=str, default=None, dest='camera_pipeline', help='自定义 GStreamer 管道')
    parser.add_argument('--device', '-d', type=str, default='cpu', choices=['cpu', 'gpu'], help="推理设备：'cpu' 或 'gpu'")
    parser.add_argument(
        '--known-faces',
        '-k',
        type=str,
        default=os.path.join(PROJECT_DIR, 'known_faces'),
        dest='known_faces',
        help='已知人脸目录路径',
    )
    parser.add_argument('--scale', '-s', type=float, default=0.25, help='视频帧缩放比例（默认：0.25）')
    parser.add_argument('--skip-frames', '-f', type=int, default=1, dest='skip_frames', help='每隔几帧处理一次（默认：1）')
    parser.add_argument('--tolerance', '-t', type=float, default=0.6, help='人脸匹配容差（默认：0.6）')
    parser.add_argument('--output', '-o', type=str, default=None, help='结果视频保存路径')
    parser.add_argument('--no-display', action='store_true', default=False, dest='no_display', help='无头模式：不显示视频窗口')
    return parser.parse_args()


def main():
    """程序主入口。"""
    args = parse_arguments()

    print('=' * 60)
    print('      face_recognition Jetson 演示程序')
    print('=' * 60)
    print(f"  输入源：{'摄像头 #' + str(args.camera_id) if args.input == 'camera' else args.input}")
    print(f"  推理设备：{args.device.upper()}")
    if args.input == 'camera':
        print(f"  摄像头后端：{args.camera_backend}")
        print(f"  摄像头参数：{args.camera_width}x{args.camera_height} @ {args.camera_fps} FPS")
    print(f"  帧缩放比例：{args.scale}")
    print(f"  跳帧设置：每 {args.skip_frames} 帧处理一次")
    print(f"  匹配容差：{args.tolerance}")
    print(f"  已知人脸目录：{args.known_faces}")
    if args.output:
        print(f"  输出视频：{args.output}")
    print('=' * 60)

    if _HEADLESS and not args.no_display:
        args.no_display = True
        print('[信息] 检测到无显示器环境，自动启用无头模式（--no-display）')

    if args.device == 'gpu':
        gpu_available = check_gpu_available()
        if gpu_available:
            detection_model = 'cnn'
            device_label = 'GPU'
        else:
            print('[信息] 回退到 CPU 模式（HOG 模型）')
            detection_model = 'hog'
            device_label = 'CPU (fallback)'
    else:
        detection_model = 'hog'
        device_label = 'CPU'
        print('[信息] 使用 CPU 模式（HOG 模型）')

    known_encodings, known_names = load_known_faces(args.known_faces)
    if not known_names:
        print("[提示] 未加载任何已知人脸，所有检测到的人脸将标记为 'Unknown'")

    cap, is_camera = open_video_source(
        args.input,
        args.camera_id,
        camera_backend=args.camera_backend,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
        camera_pipeline=args.camera_pipeline,
    )

    video_writer = create_video_writer(args.output, cap) if args.output else None

    frame_count = 0
    process_count = 0
    last_results = []
    screenshot_count = 0
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0.0

    print('\n[信息] 开始处理视频，按 q 退出，按 s 截图')
    print('-' * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            if is_camera:
                print('[警告] 摄像头读取失败，可能已断开')
            else:
                print('[信息] 视频文件处理完毕')
            break

        frame_count += 1
        fps_frame_count += 1

        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            current_fps = fps_frame_count / elapsed
            fps_frame_count = 0
            fps_start_time = time.time()

        should_process = (frame_count % args.skip_frames == 0)
        if should_process:
            last_results = process_frame(
                frame,
                known_encodings,
                known_names,
                detection_model,
                args.scale,
                args.tolerance,
            )
            process_count += 1

        draw_results(frame, last_results)
        draw_info_overlay(
            frame,
            current_fps,
            frame_count,
            detection_model,
            device_label,
            len(last_results),
            args.camera_backend,
        )

        if video_writer is not None:
            video_writer.write(frame)

        if not args.no_display:
            cv2.imshow('Jetson Face Recognition Demo', frame)

        if args.no_display and frame_count % 50 == 0:
            face_info = ', '.join(
                f"{name}({sim:.0f}%)" if name != 'Unknown' else 'Unknown'
                for (_, name, sim) in last_results
            ) or '无人脸'
            print(f"  [帧 {frame_count:04d}] FPS={current_fps:.1f} 检测到 {len(last_results)} 张人脸：{face_info}")

        if not args.no_display:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print('[信息] 用户按下 q 键，退出程序')
                break
            if key == ord('s'):
                screenshot_count += 1
                screenshot_path = f'screenshot_{screenshot_count:04d}.jpg'
                cv2.imwrite(screenshot_path, frame)
                print(f"[截图] 已保存截图：{screenshot_path}")

    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"[信息] 输出视频已保存至：{args.output}")

    cv2.destroyAllWindows()

    print('\n' + '=' * 60)
    print('                      运行统计')
    print('=' * 60)
    print(f"  总帧数：{frame_count}")
    print(f"  已处理帧数：{process_count}")
    print(f"  截图数量：{screenshot_count}")
    if args.output and video_writer is not None:
        print(f"  输出视频：{args.output}")
    print('=' * 60)
    print('程序已正常退出。')


if __name__ == '__main__':
    main()