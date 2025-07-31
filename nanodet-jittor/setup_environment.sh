#!/bin/bash

# NanoDet-Jittor 环境配置脚本
# 适用于Ubuntu 18.04/20.04/22.04 + CUDA 10.2+

set -e

echo "=========================================="
echo "NanoDet-Jittor Environment Setup"
echo "=========================================="

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Python version: $python_version"

if [[ $(echo "$python_version >= 3.7" | bc -l) -eq 0 ]]; then
    echo "Error: Python 3.7+ is required, but found $python_version"
    exit 1
fi

# 检查CUDA是否可用
if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d, -f1 | cut -dV -f2)
    echo "CUDA version: $cuda_version"
else
    echo "Warning: CUDA not found. CPU-only mode will be used."
fi

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "Warning: nvidia-smi not found. Please check GPU drivers."
fi

echo ""
echo "Step 1: Creating conda environment..."
# 创建conda环境
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# 检查环境是否已存在
if conda env list | grep -q "nanodet-jittor"; then
    echo "Environment 'nanodet-jittor' already exists. Activating..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate nanodet-jittor
else
    echo "Creating new environment 'nanodet-jittor'..."
    conda create -n nanodet-jittor python=3.8 -y
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate nanodet-jittor
fi

echo ""
echo "Step 2: Installing Jittor..."
# 安装Jittor
pip install jittor -i https://pypi.tuna.tsinghua.edu.cn/simple

# 验证Jittor安装
echo "Testing Jittor installation..."
python -c "import jittor as jt; print('Jittor version:', jt.__version__)"

# 测试CUDA支持
if command -v nvcc &> /dev/null; then
    echo "Testing CUDA support..."
    python -c "import jittor as jt; jt.flags.use_cuda=1; print('CUDA available:', jt.flags.use_cuda)"
fi

echo ""
echo "Step 3: Installing dependencies..."
# 安装项目依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo ""
echo "Step 4: Installing project..."
# 安装项目
python setup.py develop

echo ""
echo "Step 5: Verifying installation..."
# 验证安装
python -c "
import jittor as jt
from nanodet.model.backbone import build_backbone

print('✓ Jittor imported successfully')
print('✓ NanoDet modules imported successfully')

# 测试backbone
cfg = {'name': 'ShuffleNetV2', 'model_size': '1.0x'}
backbone = build_backbone(cfg)
print('✓ ShuffleNetV2 backbone created successfully')

# 测试前向传播
x = jt.randn(1, 3, 320, 320)
with jt.no_grad():
    outputs = backbone(x)
print(f'✓ Forward pass successful, output shapes: {[o.shape for o in outputs]}')
"

echo ""
echo "=========================================="
echo "Environment setup completed successfully!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate nanodet-jittor"
echo ""
echo "To test the installation, run:"
echo "  python -c \"import jittor as jt; print('Jittor version:', jt.__version__)\""
echo ""
echo "Next steps:"
echo "1. Download COCO dataset or create mini dataset:"
echo "   python tools/create_mini_dataset.py --src-dir /path/to/coco --dst-dir data/coco_mini --train-samples 100 --val-samples 50"
echo ""
echo "2. Start training:"
echo "   python tools/train.py config/nanodet-plus-m_320_mini.yml"
echo ""

# 保存环境信息
echo "Saving environment information..."
cat > environment_info.txt << EOF
NanoDet-Jittor Environment Information
=====================================
Date: $(date)
Python: $(python --version)
Jittor: $(python -c "import jittor as jt; print(jt.__version__)")
CUDA: $(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | cut -d, -f1 | cut -dV -f2 || echo "Not available")
GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "Not available")
Conda Environment: nanodet-jittor
EOF

echo "Environment information saved to environment_info.txt"
