#!/bin/bash
# 消融实验执行脚本
# 位置: experiments/run_ablation_study.sh
# 理由: 自动化执行所有消融实验，便于系统评估每个模块的贡献

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 配置路径
CONFIG_FILE="tools/cfgs/kitti_models/voxel_rcnn_fusion_focals.yaml"
BASE_DIR="output/ablation_study"
EPOCHS=80
BATCH_SIZE=8

# 创建实验目录
mkdir -p ${BASE_DIR}
mkdir -p ${BASE_DIR}/logs

# 实验日志
LOG_FILE="${BASE_DIR}/ablation_experiments_$(date +%Y%m%d_%H%M%S).log"

echo "==========================================" | tee -a ${LOG_FILE}
echo "Starting Ablation Study Experiments" | tee -a ${LOG_FILE}
echo "Date: $(date)" | tee -a ${LOG_FILE}
echo "==========================================" | tee -a ${LOG_FILE}

# 函数: 运行单个实验
run_experiment() {
    local exp_name=$1
    local ablation_mode=$2
    local extra_args=$3
    
    echo "" | tee -a ${LOG_FILE}
    echo "-------------------------------------------" | tee -a ${LOG_FILE}
    echo "Experiment: ${exp_name}" | tee -a ${LOG_FILE}
    echo "Ablation Mode: ${ablation_mode}" | tee -a ${LOG_FILE}
    echo "Start Time: $(date)" | tee -a ${LOG_FILE}
    echo "-------------------------------------------" | tee -a ${LOG_FILE}
    
    # 创建实验特定目录
    exp_dir="${BASE_DIR}/${exp_name}"
    mkdir -p ${exp_dir}
    
    # 运行训练
    if [ "${ablation_mode}" = "none" ]; then
        python tools/train_fusion.py \
            --cfg_file ${CONFIG_FILE} \
            --extra_tag ${exp_name} \
            --epochs ${EPOCHS} \
            --batch_size ${BATCH_SIZE} \
            ${extra_args} \
            2>&1 | tee -a ${LOG_FILE}
    else
        python tools/train_fusion.py \
            --cfg_file ${CONFIG_FILE} \
            --extra_tag ${exp_name} \
            --ablation_mode ${ablation_mode} \
            --epochs ${EPOCHS} \
            --batch_size ${BATCH_SIZE} \
            ${extra_args} \
            2>&1 | tee -a ${LOG_FILE}
    fi
    
    # 运行测试
    echo "Testing ${exp_name}..." | tee -a ${LOG_FILE}
    python tools/test.py \
        --cfg_file ${CONFIG_FILE} \
        --extra_tag ${exp_name} \
        --ckpt ${exp_dir}/ckpt/best_model.pth \
        --batch_size ${BATCH_SIZE} \
        2>&1 | tee -a ${LOG_FILE}
    
    echo "End Time: $(date)" | tee -a ${LOG_FILE}
    echo "-------------------------------------------" | tee -a ${LOG_FILE}
}

# E1: 基线模型 (VoxelRCNN原始)
echo "====== E1: Baseline (Original VoxelRCNN) ======" | tee -a ${LOG_FILE}
run_experiment "E1_baseline" "none" \
    "--set DATA_CONFIG.PSEUDO_CLOUD_CONFIG.ENABLED False \
     MODEL.BACKBONE_3D.USE_FOCALS False \
     MODEL.FUSION_CONFIG.FUSION_STAGES []"

# E2: 基线 + 伪点云
echo "====== E2: Baseline + Pseudo Point Cloud ======" | tee -a ${LOG_FILE}
run_experiment "E2_pseudo_cloud" "none" \
    "--set DATA_CONFIG.PSEUDO_CLOUD_CONFIG.ENABLED True \
     MODEL.BACKBONE_3D.USE_FOCALS False \
     MODEL.FUSION_CONFIG.FUSION_STAGES []"

# E3: 基线 + 伪点云 + FocalsConv
echo "====== E3: Baseline + Pseudo + FocalsConv ======" | tee -a ${LOG_FILE}
run_experiment "E3_focals" "none" \
    "--set DATA_CONFIG.PSEUDO_CLOUD_CONFIG.ENABLED True \
     MODEL.BACKBONE_3D.USE_FOCALS True \
     MODEL.FUSION_CONFIG.FUSION_STAGES []"

# E4: 基线 + 伪点云 + FocalsConv + 融合
echo "====== E4: Baseline + Pseudo + FocalsConv + Fusion ======" | tee -a ${LOG_FILE}
run_experiment "E4_fusion" "none" \
    "--set DATA_CONFIG.PSEUDO_CLOUD_CONFIG.ENABLED True \
     MODEL.BACKBONE_3D.USE_FOCALS True \
     MODEL.FUSION_CONFIG.FUSION_STAGES ['conv3']"

# E5: 完整模型 (所有组件)
echo "====== E5: Complete Model (All Components) ======" | tee -a ${LOG_FILE}
run_experiment "E5_complete" "none" ""

# 额外实验: 不同融合阶段
echo "====== Extra: Different Fusion Stages ======" | tee -a ${LOG_FILE}

# 早期融合
run_experiment "E6_early_fusion" "none" \
    "--set MODEL.FUSION_CONFIG.FUSION_STAGES ['conv1']"

# 晚期融合
run_experiment "E7_late_fusion" "none" \
    "--set MODEL.FUSION_CONFIG.FUSION_STAGES ['conv4']"

# 多阶段融合
run_experiment "E8_multi_fusion" "none" \
    "--set MODEL.FUSION_CONFIG.FUSION_STAGES ['conv2','conv3','conv4']"

# 生成结果汇总
echo "" | tee -a ${LOG_FILE}
echo "==========================================" | tee -a ${LOG_FILE}
echo "Generating Results Summary..." | tee -a ${LOG_FILE}
python experiments/summarize_ablation_results.py \
    --exp_dir ${BASE_DIR} \
    --output ${BASE_DIR}/ablation_summary.csv \
    2>&1 | tee -a ${LOG_FILE}

echo "Ablation Study Complete!" | tee -a ${LOG_FILE}
echo "Results saved to: ${BASE_DIR}" | tee -a ${LOG_FILE}
echo "==========================================" | tee -a ${LOG_FILE}