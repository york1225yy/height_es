# Jetson 版本使用说明

## 1. 适用对象

本文档适用于 NVIDIA Jetson Orin Nano 板卡上的 `demojetson.py`。

当前已在本机验证通过的能力包括：
- CUDA 加速的 `dlib 19.24.6`
- `face_recognition_models 0.3.0`
- USB 摄像头的 `v4l2` 模式
- USB 摄像头经 GStreamer 管道输入到 `demojetson.py`
- GPU 模式下的人脸识别

## 2. 当前设备实测环境

本机实测环境如下：
- 板卡：Jetson Orin Nano
- 系统：Ubuntu 22.04.5 LTS
- L4T：36.4.3
- JetPack：6.1
- CUDA：12.6
- GStreamer：1.20.3
- Python：3.10.12
- dlib：19.24.6
- `DLIB_USE_CUDA`：`True`
- face_recognition_models：0.3.0
- 当前连接摄像头：Logitech C270 HD WEBCAM
- 当前传统视频节点：`/dev/video0`
- 当前摄像头类型：USB UVC 摄像头，不是 CSI 摄像头

## 3. Jetson CSI 相机路径是什么意思

Jetson 上常说的 CSI 相机路径，本质上不是指一个固定的 Linux 文件路径，而是指一条通过 Jetson 相机栈访问 MIPI CSI 摄像头的采集链路。

对于 Jetson CSI 摄像头，应用通常不会直接只靠 `/dev/video0` 打开，而是通过 NVIDIA 的 Argus/GStreamer 栈访问，例如：

```bash
nvarguscamerasrc sensor-id=0 ! \
video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! \
nvvidconv ! video/x-raw, format=BGRx ! \
videoconvert ! appsink
```

这里的关键点是：
- `nvarguscamerasrc` 代表 Jetson 的 CSI 摄像头采集源
- 它走的是 NVIDIA 的相机驱动和 ISP/内存通路
- 常见于树莓派风格的 MIPI CSI 摄像头、IMX 系列 CSI 模组等

## 4. 当前相机路径是什么

当前机器上实际在用的相机不是 CSI，而是 USB 摄像头。

本机当前相机路径是：
- `/dev/video0`

这表示它是通过 Linux V4L2 设备节点暴露给用户空间的传统摄像头设备。当前自动模式最终也是通过：
- `v4l2`

成功打开摄像头，而不是通过 `nvarguscamerasrc`。

## 5. 和传统 Ubuntu 相机路径有什么区别

### 5.1 传统 Ubuntu USB 摄像头

传统 Ubuntu 上最常见的是 USB 摄像头，通常表现为：
- `/dev/video0`
- `/dev/video1`
- 驱动一般是 `uvcvideo`
- 采集方式通常是 V4L2

典型打开方式：

```python
cv2.VideoCapture(0)
```

或者：

```bash
v4l2src device=/dev/video0 ! ...
```

### 5.2 Jetson CSI 摄像头

Jetson CSI 摄像头虽然有时也会伴随视频节点，但工程上更常用的是 NVIDIA 提供的 Argus 管线，而不是把它当成普通 USB 摄像头来处理。

典型特点：
- 更常见的入口是 `nvarguscamerasrc`
- 可以直接使用 Jetson 的硬件图像链路
- 更适合 CSI 摄像头
- 常与 `nvvidconv`、`memory:NVMM` 一起使用

### 5.3 当前这台机器的实际情况

当前这台 Orin Nano 上：
- 系统支持 CSI 相机栈，`nvarguscamerasrc` 插件存在
- 但你现在接入并实际使用的是 USB 摄像头 `C270 HD WEBCAM`
- 所以当前真实可用路径是 `/dev/video0`
- 自动模式下最终命中的是 `v4l2`
- 如果要强制走 GStreamer，需要为 USB 摄像头传入 `v4l2src` 管道，而不是 CSI 的 `nvarguscamerasrc`

## 6. demojetson.py 的关键行为

`demojetson.py` 是 Jetson 专用入口，和 `demo.py` 分离。

当前脚本的关键处理逻辑：
- 优先使用系统 OpenCV（Jetson 系统自带版本），因为它启用了 GStreamer
- 保留 Python 用户环境里的 `dlib`、`face_recognition_models` 等包
- 支持 `auto`、`v4l2`、`gstreamer` 三种后端
- 支持自定义 `--camera-pipeline`
- 已解决工作区内 `face_recognition` 目录与 Python 包重名导致的导入冲突问题

## 7. 环境准备命令

### 7.1 安装 CUDA 版 dlib

```bash
git clone --depth 1 --branch v19.24.6 https://github.com/davisking/dlib.git /tmp/dlib-src
export PATH=/usr/local/cuda/bin:$PATH
export CMAKE_ARGS="-DDLIB_USE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda"
cd /tmp/dlib-src
python3 -m pip install --no-cache-dir .
```

验证：

```bash
python3 - <<'PY'
import dlib
print(dlib.__version__)
print(dlib.DLIB_USE_CUDA)
PY
```

期望输出中包含：

```text
19.24.6
True
```

### 7.2 安装 face_recognition_models

```bash
python3 -m pip install --user --no-cache-dir git+https://github.com/ageitgey/face_recognition_models
```

### 7.3 安装摄像头工具

```bash
sudo apt-get update
sudo apt-get install -y v4l-utils
```

## 8. 启动命令

### 8.1 USB 摄像头 + GPU + 自动后端

```bash
python3 /home/byd/workstation/height_es/demojetson.py \
  --device gpu \
  --camera-backend auto
```

说明：
- 当前设备实测会自动回退到 `v4l2`
- 适合当前这只 Logitech C270

### 8.2 USB 摄像头 + GPU + V4L2 后端

```bash
python3 /home/byd/workstation/height_es/demojetson.py \
  --device gpu \
  --camera-backend v4l2 \
  --camera-id 0
```

### 8.3 USB 摄像头 + GPU + GStreamer 后端

这是当前机器上已经验证通过的 GStreamer 启动方式：

```bash
DISPLAY=:1 python3 /home/byd/workstation/height_es/demojetson.py \
  --device gpu \
  --camera-backend gstreamer \
  --camera-pipeline "v4l2src device=/dev/video0 ! image/jpeg,width=640,height=480,framerate=30/1 ! jpegdec ! videoconvert ! appsink"
```

说明：
- 这条命令适用于当前 USB 摄像头
- 这是可视化模式，不要加 `--no-display`
- 当前系统中 `DISPLAY=:1` 已验证可用

### 8.4 USB 摄像头 + GPU + GStreamer 无头模式

```bash
python3 /home/byd/workstation/height_es/demojetson.py \
  --device gpu \
  --camera-backend gstreamer \
  --camera-pipeline "v4l2src device=/dev/video0 ! image/jpeg,width=640,height=480,framerate=30/1 ! jpegdec ! videoconvert ! appsink" \
  --no-display
```

### 8.5 Jetson CSI 摄像头 + GPU

如果后续换成真正的 CSI 摄像头，可使用：

```bash
python3 /home/byd/workstation/height_es/demojetson.py \
  --device gpu \
  --camera-backend gstreamer \
  --camera-id 0
```

说明：
- 这个用法会走脚本内置的 `nvarguscamerasrc` 管道
- 只适合 CSI 摄像头，不适合当前的 USB C270

## 9. 已验证的运行结果

### 9.1 自动模式 GPU 摄像头

已验证命令：

```bash
python3 /home/byd/workstation/height_es/demojetson.py --device gpu --camera-backend auto --no-display
```

实测结果：
- 成功启用 CUDA
- 成功打开 `/dev/video0`
- 自动命中 `v4l2`
- 已知人脸 `yy` 加载成功
- 持续识别正常
- 实测帧率约 `24 FPS`

### 9.2 GStreamer 可视化模式

已验证命令：

```bash
DISPLAY=:1 python3 /home/byd/workstation/height_es/demojetson.py \
  --device gpu \
  --camera-backend gstreamer \
  --camera-pipeline "v4l2src device=/dev/video0 ! image/jpeg,width=640,height=480,framerate=30/1 ! jpegdec ! videoconvert ! appsink"
```

实测结果：
- 成功命中系统 OpenCV 的 GStreamer 支持
- 成功打开 `gstreamer-custom`
- 成功进入窗口显示模式
- 成功开始处理视频帧

## 10. 常见问题

### 10.1 为什么 `--camera-backend gstreamer` 一开始打不开

因为最开始导入的是 `pip` 安装的 `opencv-python`，该版本未启用 GStreamer。

当前已经在 `demojetson.py` 中改为优先使用 Jetson 系统自带的 OpenCV，因此 GStreamer 模式可以正常工作。

### 10.2 为什么 `auto` 模式没有走 CSI

因为当前接入的是 USB 摄像头，不是 CSI 摄像头。

`auto` 模式会尝试：
- Jetson CSI GStreamer 管道
- V4L2
- 默认 OpenCV 后端

当前实际成功的是 `v4l2`。

### 10.3 如果我换成 CSI 摄像头该怎么做

1. 保持 `--camera-backend gstreamer`
2. 不要传 USB 的 `v4l2src` 自定义管道
3. 直接使用：

```bash
python3 /home/byd/workstation/height_es/demojetson.py --device gpu --camera-backend gstreamer
```

4. 如有多个 CSI 传感器，再通过 `--camera-id` 切换 `sensor-id`

### 10.4 如果窗口不显示

请检查：
- 是否有图形环境
- `DISPLAY` 是否正确
- 当前会话是否能弹出 OpenCV 窗口

本机当前已验证：

```bash
echo $DISPLAY
```

输出为：

```text
:1
```

## 11. 推荐命令汇总

当前这台机器最推荐的两条命令是：

GPU 自动模式：

```bash
python3 /home/byd/workstation/height_es/demojetson.py --device gpu --camera-backend auto
```

GPU GStreamer 可视化模式：

```bash
DISPLAY=:1 python3 /home/byd/workstation/height_es/demojetson.py \
  --device gpu \
  --camera-backend gstreamer \
  --camera-pipeline "v4l2src device=/dev/video0 ! image/jpeg,width=640,height=480,framerate=30/1 ! jpegdec ! videoconvert ! appsink"
```
