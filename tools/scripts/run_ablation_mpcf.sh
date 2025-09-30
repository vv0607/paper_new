#!/bin/bash
# 基于MPCF项目的消融实验脚本
# 位置: tools/scripts/run_ablation_mpcf.sh

# 环境设置
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 配置路径
BASE_DIR="output/ablation_study"
DATA_PATH="../data/kitti_pseudo"  # MPCF伪点云数据路径
EPOCHS=80
BATCH_SIZE=4  # MPCF默认批次大小

# 创建输出目录
mkdir -p ${BASE_DIR}
mkdir -p ${BASE_DIR}/logs

# 日志文件
LOG_FILE="${BASE_DIR}/ablation_$(date +%Y%m%d_%H%M%S).log"

echo "========================================" | tee ${LOG_FILE}
echo "MPCF Ablation Study with FocalsConv and Fusion" | tee -a ${LOG_FILE}
echo "Date: $(date)" | tee -a ${LOG_FILE}
echo "========================================" | tee -a ${LOG_FILE}

# 函数：运行实验
run_experiment() {
    local exp_name=$1
    local config_file=$2
    local extra_args=$3
    
    echo "" | tee -a ${LOG_FILE}
    echo "-----------------------------------" | tee -a ${LOG_FILE}
    echo "Experiment: ${exp_name}" | tee -a ${LOG_FILE}
    echo "Config: ${config_file}" | tee -a ${LOG_FILE}
    echo "Start: $(date)" | tee -a ${LOG_FILE}
    echo "-----------------------------------" | tee -a ${LOG_FILE}
    
    # 运行训练
    python tools/train.py \
        --cfg_file ${config_file} \
        --extra_tag ${exp_name} \
        --epochs ${EPOCHS} \
        --batch_size_per_gpu ${BATCH_SIZE} \
        ${extra_args} \
        2>&1 | tee -a ${LOG_FILE}
    
    # 运行测试
    echo "Testing ${exp_name}..." | tee -a ${LOG_FILE}
    python tools/test.py \
        --cfg_file ${config_file} \
        --extra_tag ${exp_name} \
        --ckpt output/${exp_name}/ckpt/checkpoint_epoch_80.pth \
        --batch_size ${BATCH_SIZE} \
        2>&1 | tee -a ${LOG_FILE}
        
    echo "End: $(date)" | tee -a ${LOG_FILE}
}

# E1: 基线 - MPCF原始VoxelRCNN（使用伪点云但不用FocalsConv和融合）
echo "====== E1: Baseline (MPCF VoxelRCNN with Pseudo Points) ======" | tee -a ${LOG_FILE}
cat > ${BASE_DIR}/e1_baseline.yaml << EOF
CLASS_NAMES: ['Car', 'Pedestrian', 'Cyclist']

DATA_CONFIG:
    _BASE_CONFIG_: cfgs/dataset_configs/kitti_dataset.yaml
    DATA_PATH: '${DATA_PATH}'
    USE_PSEUDO_LABEL: True
    PSEUDO_LABEL_DIR: 'depth_pseudo_rgbseguv_twise'
    POINT_CLOUD_RANGE: [0, -39.68, -3, 69.12, 39.68, 1]

MODEL:
    NAME: VoxelRCNN
    
    VFE:
        NAME: MeanVFE
        
    BACKBONE_3D:
        NAME: VoxelBackBone8x  # 标准spconv
        
    MAP_TO_BEV:
        NAME: HeightCompression
        NUM_BEV_FEATURES: 256
        
    BACKBONE_2D:
        NAME: BaseBEVBackbone
        LAYER_NUMS: [5, 5]
        LAYER_STRIDES: [1, 2]
        NUM_FILTERS: [128, 256]
        UPSAMPLE_STRIDES: [1, 2]
        NUM_UPSAMPLE_FILTERS: [256, 256]
        
    # ... 其他配置同主配置文件
EOF

run_experiment "e1_baseline" "${BASE_DIR}/e1_baseline.yaml" ""

# E2: 基线 + FocalsConv（无融合）
echo "====== E2: Baseline + FocalsConv (No Fusion) ======" | tee -a ${LOG_FILE}
cat > ${BASE_DIR}/e2_focals.yaml << EOF
CLASS_NAMES: ['Car', 'Pedestrian', 'Cyclist']

DATA_CONFIG:
    _BASE_CONFIG_: cfgs/dataset_configs/kitti_dataset.yaml
    DATA_PATH: '${DATA_PATH}'
    USE_PSEUDO_LABEL: True
    PSEUDO_LABEL_DIR: 'depth_pseudo_rgbseguv_twise'

MODEL:
    NAME: VoxelRCNN
    
    BACKBONE_3D:
        NAME: VoxelBackBone8xFocal  # 使用FocalsConv
        USE_FOCALS: True
        FOCALS_CONFIG:
            fuse_sum: True
            fuse_max: True
            focal_dynamic: True
            kernel_threshold: 0.5
            
    # ... 其他配置同E1
EOF

run_experiment "e2_focals" "${BASE_DIR}/e2_focals.yaml" ""

# E3: 基线 + FocalsConv + 单阶段融合（conv3）
echo "====== E3: Baseline + FocalsConv + Single-stage Fusion (conv3) ======" | tee -a ${LOG_FILE}
run_experiment "e3_fusion_conv3" \
    "tools/cfgs/kitti_models/voxel_rcnn_focal_fusion.yaml" \
    "--set MODEL.FUSION_MODULES.FUSION_STAGES ['x_conv3']"

# E4: 基线 + FocalsConv + 多阶段融合
echo "====== E4: Baseline + FocalsConv + Multi-stage Fusion ======" | tee -a ${LOG_FILE}
run_experiment "e4_fusion_multi" \
    "tools/cfgs/kitti_models/voxel_rcnn_focal_fusion.yaml" \
    "--set MODEL.FUSION_MODULES.FUSION_STAGES ['x_conv2','x_conv3','x_conv4']"

# E5: 完整模型 + CB融合
echo "====== E5: Complete Model with CB Fusion ======" | tee -a ${LOG_FILE}
run_experiment "e5_complete" \
    "tools/cfgs/kitti_models/voxel_rcnn_focal_fusion.yaml" \
    "--set MODEL.FUSION_MODULES.USE_CB_FUSION True"

# E6: 不使用伪点云（仅原始LiDAR）作为对比
echo "====== E6: No Pseudo Points (Original LiDAR Only) ======" | tee -a ${LOG_FILE}
run_experiment "e6_no_pseudo" \
    "tools/cfgs/kitti_models/voxel_rcnn_focal_fusion.yaml" \
    "--set DATA_CONFIG.USE_PSEUDO_LABEL False"

# 汇总结果
echo "" | tee -a ${LOG_FILE}
echo "========================================" | tee -a ${LOG_FILE}
echo "Summarizing Results..." | tee -a ${LOG_FILE}

# 创建结果汇总表
python tools/scripts/summarize_results.py \
    --exp_dir ${BASE_DIR} \
    --output ${BASE_DIR}/summary.csv \
    2>&1 | tee -a ${LOG_FILE}

echo "Ablation study complete!" | tee -a ${LOG_FILE}
echo "Results saved to: ${BASE_DIR}" | tee -a ${LOG_FILE}