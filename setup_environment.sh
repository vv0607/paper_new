#!/bin/bash
# 环境安装脚本
# 位置: setup_environment.sh
# 理由: 自动化环境配置，确保所有依赖正确安装

set -e  # 遇到错误立即退出

echo "=========================================="
echo "3D Detection Fusion Model Setup Script"
echo "=========================================="

# 检查CUDA版本
echo "Checking CUDA version..."
nvcc --version

# 创建conda环境
echo "Creating conda environment..."
conda create -n fusion3d python=3.8 -y
conda activate fusion3d

# 安装PyTorch (根据CUDA版本调整)
echo "Installing PyTorch..."
conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 -c pytorch -y

# 克隆主项目
echo "Cloning main repository..."
if [ ! -d "MPCF--3d_object_detection" ]; then
    git clone https://github.com/ELOESZHANG/MPCF--3d_object_detection.git
fi
cd MPCF--3d_object_detection

# 安装OpenPCDet依赖
echo "Installing OpenPCDet dependencies..."
pip install -r requirements.txt

# 安装spconv-plus (包含FocalsConv)
echo "Installing spconv-plus with FocalsConv..."
pip install pccm cumm wheel

# 设置环境变量
export SPCONV_DISABLE_JIT="1"

# 克隆并安装spconv-plus
cd ..
if [ ! -d "spconv-plus" ]; then
    git clone https://github.com/dvlab-research/spconv-plus.git
fi
cd spconv-plus
python setup.py bdist_wheel
pip install dist/*.whl
cd ..

# 回到主项目
cd MPCF--3d_object_detection

# 安装额外依赖
echo "Installing additional dependencies..."
pip install tensorboardX
pip install opencv-python
pip install scikit-learn
pip install matplotlib
pip install pandas
pip install seaborn
pip install open3d
pip install mayavi  # 可选，用于3D可视化

# 编译自定义CUDA操作
echo "Compiling CUDA operators..."
cd pcdet/ops/iou3d_nms
python setup.py build_ext --inplace
cd ../../..

cd pcdet/ops/roiaware_pool3d
python setup.py build_ext --inplace
cd ../../..

cd pcdet/ops/pointnet2/pointnet2_batch
python setup.py build_ext --inplace
cd ../../../..

cd pcdet/ops/pointnet2/pointnet2_stack
python setup.py build_ext --inplace
cd ../../../..

# 创建必要的目录结构
echo "Creating directory structure..."
mkdir -p data/kitti
mkdir -p output
mkdir -p experiments
mkdir -p tools/cfgs/kitti_models

# 下载预训练模型（可选）
echo "Downloading pretrained models (optional)..."
mkdir -p pretrained_models
cd pretrained_models

# 下载MonoDepth2预训练模型
if [ ! -f "monodepth2_mono+stereo_640x192.pth" ]; then
    wget https://storage.googleapis.com/niantic-lon-static/research/monodepth2/monodepth2_mono+stereo_640x192.zip
    unzip monodepth2_mono+stereo_640x192.zip
    rm monodepth2_mono+stereo_640x192.zip
fi

cd ..

# 验证安装
echo "Verifying installation..."
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import spconv.pytorch as spconv; print('spconv installed successfully')"
python -c "import pcdet; print('pcdet installed successfully')"

# 创建数据软链接（根据实际数据路径调整）
echo "Setting up KITTI dataset..."
echo "Please create a symbolic link to your KITTI dataset:"
echo "ln -s /path/to/your/kitti/dataset data/kitti"

# 下载KITTI数据集（如果需要）
read -p "Do you want to download KITTI dataset? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd data/kitti
    # 下载KITTI 3D目标检测数据集
    echo "Downloading KITTI dataset..."
    wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
    wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip
    wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip
    wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip
    
    # 解压
    unzip data_object_image_2.zip
    unzip data_object_velodyne.zip
    unzip data_object_calib.zip
    unzip data_object_label_2.zip
    
    # 清理
    rm *.zip
    cd ../..
fi

# 生成数据信息文件
echo "Generating dataset info..."
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo "To activate the environment, run:"
echo "conda activate fusion3d"
echo ""
echo "Next steps:"
echo "1. Ensure KITTI dataset is properly linked in data/kitti"
echo "2. Copy the provided code files to their respective locations"
echo "3. Run visualization test: python tools/visualize_fusion.py --mode stats"
echo "4. Start training: python tools/train_fusion.py --cfg_file tools/cfgs/kitti_models/voxel_rcnn_fusion_focals.yaml"