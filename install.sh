#!/bin/bash
# =============================================================================
#  face_recognition 人脸识别环境一键安装脚本
#  适用系统：Ubuntu 20.04 / 22.04 / 24.04（AutoDL、本地、云服务器均可）
#  Python  ：通过 uv 自动安装 Python 3.8.20 并创建独立虚拟环境
#  作者    ：york1225yy
#  版本    ：1.0.0
# =============================================================================

set -e  # 遇到错误立即退出

# -------------------------------------------------------
# 颜色输出辅助
# -------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERR]${NC}  $*"; exit 1; }

# -------------------------------------------------------
# 安装目标目录（默认当前目录）
# -------------------------------------------------------
INSTALL_DIR="${1:-$(pwd)}"
VENV_DIR="$INSTALL_DIR/.venv"

echo ""
echo "============================================================"
echo "   face_recognition 人脸识别环境一键安装"
echo "============================================================"
echo "  安装目录：$INSTALL_DIR"
echo "  虚拟环境：$VENV_DIR"
echo "============================================================"
echo ""

# -------------------------------------------------------
# 步骤 1：更新 apt 并安装系统依赖
# -------------------------------------------------------
info "步骤 1/6：安装系统依赖包（cmake、dlib 编译依赖等）..."
sudo apt-get update -qq
sudo apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    python3-dev \
    git \
    curl \
    2>&1 | tail -5
success "系统依赖安装完成"

# -------------------------------------------------------
# 步骤 2：安装 uv（快速 Python 包管理器）
# -------------------------------------------------------
info "步骤 2/6：安装 uv 包管理器..."
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    success "uv 安装完成：$(uv --version)"
else
    export PATH="$HOME/.local/bin:$PATH"
    success "uv 已存在：$(uv --version)"
fi

# -------------------------------------------------------
# 步骤 3：安装 Python 3.8 并创建虚拟环境
# -------------------------------------------------------
info "步骤 3/6：安装 Python 3.8.20 并创建虚拟环境..."
uv python install 3.8
uv venv "$VENV_DIR" --python 3.8
success "Python 虚拟环境创建完成：$VENV_DIR"
"$VENV_DIR/bin/python" --version

# -------------------------------------------------------
# 步骤 4：安装基础包（wheel / setuptools / numpy）
# -------------------------------------------------------
info "步骤 4/6：安装基础依赖（wheel、setuptools、numpy）..."
uv pip install --python "$VENV_DIR/bin/python" \
    "setuptools==75.3.4" \
    "wheel==0.45.1" \
    "numpy==1.24.4"
success "基础依赖安装完成"

# -------------------------------------------------------
# 步骤 5：编译并安装 dlib 19.24.6
#   注意：dlib 需要从源码编译，耗时约 10~15 分钟
#   dlib 20.x 与 face_recognition 1.3.0 API 不兼容，须使用 19.24.6
# -------------------------------------------------------
info "步骤 5/6：编译安装 dlib==19.24.6（约需 10~15 分钟，请耐心等待）..."
uv pip install --python "$VENV_DIR/bin/python" "dlib==19.24.6"
success "dlib 19.24.6 安装完成"

# -------------------------------------------------------
# 步骤 6：安装 face_recognition 及其模型和其他依赖
# -------------------------------------------------------
info "步骤 6/6：安装 face_recognition、face_recognition_models、opencv、pillow..."

# face_recognition_models 须从 GitHub 安装（包含 dlib 预训练模型文件）
uv pip install --python "$VENV_DIR/bin/python" \
    "git+https://github.com/ageitgey/face_recognition_models"

uv pip install --python "$VENV_DIR/bin/python" \
    "face-recognition==1.3.0" \
    "opencv-python==4.13.0.92" \
    "pillow==10.4.0" \
    "click==8.1.8"

success "所有 Python 依赖安装完成"

# -------------------------------------------------------
# 验证安装
# -------------------------------------------------------
echo ""
info "验证安装结果..."
"$VENV_DIR/bin/python" - <<'EOF'
import face_recognition, cv2, numpy, dlib, PIL
print(f"  dlib              : {dlib.__version__}")
print(f"  face_recognition  : {face_recognition.__version__}")
print(f"  opencv-python     : {cv2.__version__}")
print(f"  numpy             : {numpy.__version__}")
print(f"  pillow            : {PIL.__version__}")
EOF

# -------------------------------------------------------
# 创建便捷激活脚本
# -------------------------------------------------------
cat > "$INSTALL_DIR/activate_env.sh" <<ACTIVATE_EOF
#!/bin/bash
# 激活 face_recognition 虚拟环境
source "$VENV_DIR/bin/activate"
echo "已激活虚拟环境：$VENV_DIR"
echo "Python: \$(python --version)"
ACTIVATE_EOF
chmod +x "$INSTALL_DIR/activate_env.sh"

# -------------------------------------------------------
# 完成
# -------------------------------------------------------
echo ""
echo "============================================================"
echo -e "${GREEN}  环境安装完成！${NC}"
echo "============================================================"
echo ""
echo "  使用方法："
echo "  1. 激活环境：source activate_env.sh"
echo "     或直接：source $VENV_DIR/bin/activate"
echo ""
echo "  2. 运行 demo（CPU 模式，视频文件）："
echo "     python demo.py --input ./video/test.mp4 --device cpu \\"
echo "                    --output ./output/result.mp4 --no-display"
echo ""
echo "  3. 运行 demo（GPU 模式）："
echo "     python demo.py --input ./video/test.mp4 --device gpu \\"
echo "                    --output ./output/result.mp4 --no-display"
echo ""
echo "  4. 摄像头实时识别（需要显示器）："
echo "     python demo.py --input camera --device cpu"
echo ""
echo "  详细说明请查看：使用说明.md"
echo "============================================================"
