"""
VoxelRCNN Fusion检测器 - 支持图像融合和CBAM注意力
位置: pcdet/models/detectors/voxel_rcnn_fusion.py
"""

import torch
import torch.nn as nn
from .detector3d_template import Detector3DTemplate
from ..fusion_modules.epnet_ported_fusion import LI_FusionModule_ClassAware

class VoxelRCNNFusion(Detector3DTemplate):
    """
    VoxelRCNN融合检测器
    集成了:
    1. FocalConv 3D骨干网络 (VoxelBackBone8xFocal)
    2. 图像分支 + CBAM注意力机制
    3. LI-Fusion多模态融合模块
    """
    
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        
        # 配置标志
        self.use_image_branch = model_cfg.get('USE_IMAGE_BRANCH', False)
        self.use_fusion = model_cfg.get('USE_FUSION', False)
        
        # 构建网络模块
        self.module_list = self.build_networks()
        
        # 如果使用图像分支,构建图像主干网络
        if self.use_image_branch:
            self.build_image_branch()
            
        # 如果使用融合,构建融合模块
        if self.use_fusion and self.use_image_branch:
            self.build_fusion_modules()

    def build_image_branch(self):
        """构建图像特征提取分支(带CBAM)"""
        from ..backbones_2d.cbam_attention import ImageBackboneWithCBAM
        
        # 图像主干网络配置
        image_cfg = self.model_cfg.get('IMAGE_BACKBONE', {
            'IN_CHANNELS': 3,
            'BASE_CHANNELS': 64,
            'NUM_STAGES': 4,
            'OUT_CHANNELS': 256,
            'USE_CBAM': True,
            'CBAM_REDUCTION': 16
        })
        
        self.image_backbone = ImageBackboneWithCBAM(image_cfg)
        print(f"[INFO] 图像分支已启用,使用CBAM注意力机制")
        
    def build_fusion_modules(self):
        """构建融合模块"""
        # from ..fusion_modules.li_fusion import LI_FusionModule
        
        # 融合模块配置
        fusion_cfg = self.model_cfg.get('FUSION_CONFIG', {
            'IN_CHANNELS_3D': 64,
            'IN_CHANNELS_2D': 256,
            'OUT_CHANNELS': 64,
            'MID_CHANNELS': 128,
            'FUSION_STAGES': ['x_conv3']  # 在哪些阶段进行融合
        })
        
        self.fusion_stages = fusion_cfg.get('FUSION_STAGES', ['x_conv3'])
        
        # 为每个融合阶段创建融合模块
        self.fusion_modules = nn.ModuleDict()
        for stage in self.fusion_stages:
            self.fusion_modules[stage] = LI_FusionModule_ClassAware(fusion_cfg)
            
        print(f"[INFO] 融合模块已启用,融合阶段: {self.fusion_stages}")

    def forward(self, batch_dict):
        """
        前向传播
        Args:
            batch_dict: 批次数据字典
                - voxels: 体素特征
                - voxel_coords: 体素坐标
                - images: RGB图像 (B, 3, H, W) [如果使用图像分支]
                - calib: 相机标定信息
        Returns:
            训练时: (loss_dict, tb_dict, disp_dict)
            测试时: (pred_dicts, recall_dicts)
        """
        # 1. 图像特征提取(如果启用)
        if self.use_image_branch:
            batch_dict = self.image_backbone(batch_dict)
        
        # 2. 3D体素特征提取和融合
        for i, cur_module in enumerate(self.module_list):
            batch_dict = cur_module(batch_dict)
            
            # 在指定阶段进行特征融合
            if self.use_fusion and self.use_image_branch:
                # 检查当前模块是否在融合阶段
                if hasattr(cur_module, '__class__'):
                    module_name = cur_module.__class__.__name__
                    
                    # 在3D backbone之后进行融合
                    if 'VoxelBackBone8xFocal' in module_name or 'BACKBONE_3D' in str(self.module_list[i]):
                        self._apply_fusion(batch_dict)

        # 3. 计算损失或后处理
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            
            ret_dict = {
                'loss': loss,
                'loss_rpn': tb_dict.get('loss_rpn', loss),
                'loss_rcnn': tb_dict.get('loss_rcnn', loss * 0)
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def _apply_fusion(self, batch_dict):
        """
        应用多模态融合
        在指定的3D特征层与图像特征进行融合
        """
        if batch_dict.get('image_features') is None:
            return
        
        image_features = batch_dict['image_features']
        
        # 对指定阶段的特征进行融合
        for stage in self.fusion_stages:
            if stage in batch_dict:
                # 获取3D稀疏特征
                sparse_features = batch_dict[stage]
                
                if hasattr(sparse_features, 'features') and hasattr(sparse_features, 'indices'):
                    # SpConv格式
                    voxel_features = sparse_features.features
                    voxel_coords = sparse_features.indices
                    
                    # 应用融合
                    fused_features = self.fusion_modules[stage](
                        voxel_features=voxel_features,
                        voxel_coords=voxel_coords,
                        image_features=image_features,
                        batch_dict=batch_dict
                    )
                    
                    # 更新特征
                    sparse_features.features = fused_features
                    batch_dict[stage] = sparse_features
                    
                    print(f"[DEBUG] 已在 {stage} 阶段应用融合")

    def get_training_loss(self):
        """计算训练损失"""
        disp_dict = {}
        loss = 0
        
        # RPN损失
        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict['loss_rpn'] = loss_rpn
        loss = loss + loss_rpn

        # Point Head损失(如果有)
        if self.point_head is not None:
            loss_point, tb_dict = self.point_head.get_loss(tb_dict)
            loss = loss + loss_point

        # RCNN损失
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
        tb_dict['loss_rcnn'] = loss_rcnn
        loss = loss + loss_rcnn

        return loss, tb_dict, disp_dict
