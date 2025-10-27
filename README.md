
基于去噪伪点云与高级 RoI 融合的 VoxelRCNN 3D 检测

本项目是基于 VoxelRCNN 的一个高级 3D 目标检测实现，参考了 MPCF 伪点云项目 的思想，并进行了深度改进和优化。
GitHub 仓库: https://github.com/vv0607/paper_new/tree/main

1. 核心挑战与贡献

在早期的实验中，我们发现直接引入 MPCF 伪点云会导致灾难性的性能下降，尤其是对于行人 (Pedestrian) 类别（3D AP @ Moderate 从 ~62% 暴跌至 ~48%）。
我们的分析定位到根源在于伪点云中存在大量噪声（例如，CONFIDENCE_THRESHOLD: 0.7 的配置会过滤掉 99% 以上的点）。
因此，本项目的贡献分为两个阶段：
1.[Stage 1] 伪点云去噪：实现了一个自定义的伪点云去噪模块，以解决性能崩溃问题，建立一个强大的、干净的 3D 基线。
2.[Stage 2] 高级 RoI 融合：在干净的 3D 基线上，实现了一个复杂的多模态融合模块，在 RoI 阶段精确地引入 2D 图像特征。

2. 关键模块与文件


A. 3D 特征提取 (Stage 1 & 2)

●FocalSparseConv: 在 3D 稀疏主干网络中集成了 FocalConv，以增强点云特征提取。
●伪点云去噪 (Denoising):
○在 pcdet/datasets/kitti_dataset_custom.py 和 pcdet/datasets/processor/ 中实现了自定义去噪逻辑。
○通过置信度、深度范围和采样限制，在数据加载时严格清洗伪点云，这是建立高性能基线的关键。
○解决了调试过程中遇到的 NoneType、空数组和 reshape（7D -> 9D）等一系列问题。

B. 高级 RoI 融合 (Stage 2)

我们的多模态融合不在主干网络 (Backbone) 中进行（以避免早期噪声污染），而是在**第二阶段 (RoI Head) **精确实现。这套系统由以下文件协同工作：
1.image_backbone.py:
○作用：作为 2D 图像特征提取器。
○实现：通常基于 ResNet，并集成了 CBAM (Convolutional Block Attention Module) 注意力模块，以提炼图像的通道和空间特征。
2.epnet_ported_fusion.py:
○作用：核心的跨模态融合模块之一。
○实现：可能包含 EPNet 所需的点云与图像特征对齐、投影和融合逻辑。
3.li_fusion_module.py:
○作用：核心的跨模态融合模块之二。
○实现：实现了 LI-Fusion 的思想，定义了如何将 2D 特征与 3D RoI 特征进行拼接或门控融合。
4.voxel_rcnn_fusion.py (修改):
○作用：可能是 VoxelRCNN 检测器（Detector）的顶层文件。
○修改：修改 forward 函数，以接纳和传递 image_backbone.py 提取的 2D 图像特征图 (batch_dict['image_features'])。
5.voxelrcnn_head.py (修改):
○作用：VoxelRCNN 的 RoI 头部，融合的最终执行者。
○修改：
■在 __init__ 中初始化 epnet_ported_fusion.py 和 li_fusion_module.py 中定义的融合模块。
■在 forward 函数中，在 3D RoI 特征池化 (pooling) 之后，调用融合模块，将对齐后的 2D 图像特征注入，最终生成用于分类和回归的融合特征。

3. 消融实验配置


实验一：[Stage 1] 去噪伪点云基线

●配置文件: voxel_rcnn_fusion_focals_denoised.yaml
●目的: 验证去噪模块的有效性。
●关键配置:
○USE_PSEUDO_LABEL: True
○PSEUDO_POINT_DENOISER: (启用自定义去噪配置)
■CONFIDENCE_THRESHOLD: 0.3 (或一个经过调试的合理值)
■MAX_DEPTH: 70.0
■USE_DOWNSAMPLING: False
○USE_IMAGES: False (完全不使用图像)
●预期结果: 行人 (Pedestrian) 3D AP @ Moderate 恢复到 ~55% 以上。

实验二：[Stage 2] 高级 RoI 多模态融合

●配置文件: voxel_rcnn_fusion_focals_roi.yaml
●目的: 在 Stage 1 的干净基线上，验证高级 RoI 融合模块的效果。
●关键配置:
○USE_PSEUDO_LABEL: True
○PSEUDO_POINT_DENOISER: (与 Stage 1 保持一致)
○USE_IMAGES: True (启用图像)
○BACKBONE_3D.USE_IMG: False (关键：不在主干网络融合)
○ROI_HEAD.USE_LI_FUSION: True (关键：在 RoI 头部融合)
○ROI_HEAD.FUSION_MODULE: (配置 epnet_ported_fusion 和 li_fusion_module 的相关参数)
●预期结果: 车辆 (Car) 性能显著提升，行人 (Pedestrian) 性能保持稳定（不下降）。

4. 环境与数据

●环境: (请在此处补充您的 Conda, PyTorch 1.10+, spconv 2.x, pcdet 0.5+ 等配置)
●数据:
○标准 KITTI 数据集。
○9 维伪点云（x, y, z, intensity, r, g, b, u, v），由 MPCF 项目 生成。

5. 训练流程


阶段一：训练去噪基线


Bash


# 训练 Stage 1 模型
python tools/train.py \
    --cfg_file cfgs/kitti_models/voxel_rcnn_fusion_focals_denoised.yaml \
    --batch_size [Your_Batch_Size] \
    --epochs 60 \
    --extra_tag stage1_denoised_baseline

评估: 必须检查 Pedestrian AP 是否 > 55%。

阶段二：训练 RoI 融合模型


Bash


# 评估 Stage 1 成功后，加载其权重进行 Stage 2 训练
python tools/train.py \
    --cfg_file cfgs/kitti_models/voxel_rcnn_fusion_focals_roi.yaml \
    --batch_size [Your_Batch_Size] \
    --epochs 80 \
    --pretrained_model [Path_To_Stage1_Checkpoint.pth] \
    --extra_tag stage2_roi_fusion_final


## 数据准备

###  数据结构

需要以下数据结构：

```
data/kitti_pseudo/
├── ImageSets/
├── training/
│   ├── calib/                          # 标定文件
│   ├── velodyne/                       # 原始LiDAR点云
│   ├── label_2/                        # 标注文件
│   ├── image_2/                        # RGB图像
│   ├── depth_dense_twise/              # 稠密深度图
│   └── depth_pseudo_rgbseguv_twise/    # 伪点云（7维）
├── testing/
│   └── ...（同training结构）
├── gt_database/                        # 原始GT数据库
├── gt_database_pseudo_seguv/           # 伪点云GT数据库
└── *.pkl                               # 数据信息文件
```

###  生成数据信息

```bash
# 生成MPCF格式的数据信息
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos \
    tools/cfgs/dataset_configs/kitti_dataset.yaml \
    --data_path data/kitti_pseudo
```

