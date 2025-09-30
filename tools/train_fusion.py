"""
多模态融合模型训练脚本
位置: tools/train_fusion.py
理由: 专门用于训练融合模型的脚本，支持分阶段训练和消融实验
"""

import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

import _init_path
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    
    # 基础训练参数
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    
    # 融合训练特定参数
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained model path')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='fix random seed for reproducibility')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--start_epoch', type=int, default=0, help='starting epoch')
    
    # 分阶段训练参数
    parser.add_argument('--train_stage', type=str, default='full', 
                       choices=['backbone_only', 'fusion_only', 'full', 'finetune'],
                       help='training stage for progressive training')
    parser.add_argument('--freeze_modules', type=str, nargs='+', default=None,
                       help='modules to freeze during training')
    
    # 消融实验参数
    parser.add_argument('--ablation_mode', type=str, default=None,
                       choices=['no_pseudo', 'no_focals', 'no_fusion', 'no_augment'],
                       help='ablation study mode')
    
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
        
    # 应用消融实验设置
    if args.ablation_mode:
        apply_ablation_settings(cfg, args.ablation_mode)

    return args, cfg


def apply_ablation_settings(cfg, ablation_mode):
    """
    应用消融实验设置
    Args:
        cfg: 配置对象
        ablation_mode: 消融模式
    """
    if ablation_mode == 'no_pseudo':
        # 禁用伪点云
        cfg.DATA_CONFIG.PSEUDO_CLOUD_CONFIG.ENABLED = False
        print("Ablation: Disabled pseudo point cloud generation")
        
    elif ablation_mode == 'no_focals':
        # 使用普通稀疏卷积
        cfg.MODEL.BACKBONE_3D.USE_FOCALS = False
        cfg.MODEL.BACKBONE_3D.NAME = 'VoxelBackBone8x'
        print("Ablation: Using standard sparse convolution instead of FocalsConv")
        
    elif ablation_mode == 'no_fusion':
        # 禁用特征融合
        cfg.MODEL.FUSION_CONFIG.FUSION_STAGES = []
        print("Ablation: Disabled feature fusion")
        
    elif ablation_mode == 'no_augment':
        # 禁用数据增强
        cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST = []
        cfg.DATA_CONFIG.IMAGE_CONFIG.IMAGE_AUGMENTOR = {}
        print("Ablation: Disabled all data augmentation")


def freeze_model_parts(model, freeze_modules):
    """
    冻结模型的特定部分
    Args:
        model: 模型对象
        freeze_modules: 要冻结的模块名称列表
    """
    for module_name in freeze_modules:
        if hasattr(model, module_name):
            module = getattr(model, module_name)
            for param in module.parameters():
                param.requires_grad = False
            print(f"Frozen module: {module_name}")
        else:
            print(f"Warning: Module {module_name} not found in model")


def setup_training_stage(model, args, cfg):
    """
    设置分阶段训练
    Args:
        model: 模型对象
        args: 命令行参数
        cfg: 配置对象
    """
    if args.train_stage == 'backbone_only':
        # 只训练3D主干网络
        freeze_modules = ['image_backbone', 'image_fpn', 'fusion_modules']
        freeze_model_parts(model, freeze_modules)
        # 调整学习率
        cfg.OPTIMIZATION.LR *= 0.5
        print("Training stage: Backbone only")
        
    elif args.train_stage == 'fusion_only':
        # 只训练融合模块
        freeze_modules = ['module_list.backbone_3d', 'module_list.backbone_2d']
        freeze_model_parts(model, freeze_modules)
        # 调整学习率
        cfg.OPTIMIZATION.LR *= 0.1
        print("Training stage: Fusion modules only")
        
    elif args.train_stage == 'finetune':
        # 微调整个网络
        cfg.OPTIMIZATION.LR *= 0.01
        print("Training stage: Finetuning entire network")
        
    elif args.train_stage == 'full':
        # 完整训练
        print("Training stage: Full training")
        
    # 处理额外的冻结模块
    if args.freeze_modules:
        freeze_model_parts(model, args.freeze_modules)


def main():
    args, cfg = parse_config()
    
    # 设置分布式训练
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    # 设置批次大小
    if args.batch_size is not None:
        cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU = args.batch_size
        
    cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU // total_gpus
    
    # 设置训练轮数
    if args.epochs is not None:
        cfg.OPTIMIZATION.NUM_EPOCHS = args.epochs

    # 固定随机种子
    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    # 创建输出目录
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 日志文件
    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # 记录配置
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('Training in distributed mode : total_batch_size: %d' % (total_gpus * cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU))
    else:
        logger.info('Training with a single process')

    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    # TensorBoard
    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # 创建数据加载器
    logger.info('----------- Create dataloader & network & optimizer -----------')
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU,
        dist=dist_train,
        workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=False,
        total_epochs=cfg.OPTIMIZATION.NUM_EPOCHS,
        seed=666 if args.fix_random_seed else None
    )

    # 构建网络
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    
    # 设置分阶段训练
    setup_training_stage(model, args, cfg)
    
    # 加载预训练模型
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)
        
    # 加载检查点
    if args.ckpt is not None:
        model.load_params_from_file(filename=args.ckpt, to_cpu=dist_train, logger=logger)

    # 同步批归一化
    if dist_train:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
    model.cuda()

    # 构建优化器
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # 加载优化器状态
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None and cfg.get('RESUME_TRAINING', False):
        ckpt_list = glob.glob(str(ckpt_dir / 'checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            checkpoint = torch.load(ckpt_list[-1])
            
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint['epoch']
            it = checkpoint.get('it', 0)
            last_epoch = start_epoch - 1
            logger.info('Resumed from checkpoint: %s' % ckpt_list[-1])

    # 用于分布式训练
    model.train()
    if dist_train:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[cfg.LOCAL_RANK],
            broadcast_buffers=False,
            find_unused_parameters=True
        )
    
    # 学习率调度器
    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader),
        total_epochs=cfg.OPTIMIZATION.NUM_EPOCHS,
        last_epoch=last_epoch,
        optim_cfg=cfg.OPTIMIZATION
    )

    # 训练统计
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    # 训练循环
    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=cfg.OPTIMIZATION.NUM_EPOCHS,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=1,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=False,
        logger=logger,
        logger_iter_interval=50,
        ckpt_save_time_interval=300,
        show_gpu_stat=False
    )

    # 记录最终模型信息
    if hasattr(model, 'module'):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
        
    # 保存最终模型
    final_output = ckpt_dir / 'final_model.pth'
    torch.save(model_state, final_output)
    logger.info('Final model saved to: %s' % final_output)

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()