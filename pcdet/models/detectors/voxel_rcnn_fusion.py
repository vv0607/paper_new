"""
基于MPCF的VoxelRCNN，添加多模态融合功能
位置: pcdet/models/detectors/voxel_rcnn_fusion.py
理由: 扩展MPCF现有的VoxelRCNN，添加图像分支和融合模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .detector3d_template import Detector3DTemplate
from ..model_utils import model_nms_utils


class VoxelRCNNFusion(Detector3DTemplate):
    """
    VoxelRCNN with image branch and fusion modules
    基于MPCF的VoxelRCNN扩展
    """
    
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg, num_class, dataset)
        
        # 构建模块（继承自Detector3DTemplate）
        self.module_list = self.build_networks()
        
        # 添加图像分支
        if model_cfg.get('USE_IMAGE_BRANCH', False):
            self.image_backbone = self.build_image_backbone(model_cfg.IMAGE_BACKBONE)
            self.image_fpn = self.build_image_fpn(model_cfg.IMAGE_FPN)
            
        # 添加融合模块
        if model_cfg.get('USE_FUSION', False):
            self.fusion_modules = self.build_fusion_modules(model_cfg.FUSION_MODULES)
            
    def build_image_backbone(self, backbone_cfg):
        """构建图像特征提取主干"""
        backbone_type = backbone_cfg.get('NAME', 'resnet50')
        pretrained = backbone_cfg.get('PRETRAINED', True)
        
        if 'resnet' in backbone_type:
            # 使用预训练的ResNet
            if backbone_type == 'resnet50':
                backbone = torchvision.models.resnet50(pretrained=pretrained)
            elif backbone_type == 'resnet101':
                backbone = torchvision.models.resnet101(pretrained=pretrained)
            else:
                backbone = torchvision.models.resnet34(pretrained=pretrained)
                
            # 移除全连接层
            modules = list(backbone.children())[:-2]
            backbone = nn.Sequential(*modules)
            
            # 输出通道数
            if backbone_type == 'resnet50' or backbone_type == 'resnet101':
                self.image_channels = [256, 512, 1024, 2048]
            else:
                self.image_channels = [64, 128, 256, 512]
                
        return backbone
        
    def build_image_fpn(self, fpn_cfg):
        """构建FPN"""
        in_channels = fpn_cfg.get('IN_CHANNELS', self.image_channels)
        out_channels = fpn_cfg.get('OUT_CHANNELS', 256)
        
        # 横向连接
        lateral_convs = nn.ModuleList()
        for in_ch in in_channels:
            lateral_convs.append(
                nn.Conv2d(in_ch, out_channels, 1)
            )
            
        # FPN输出
        fpn_convs = nn.ModuleList()
        for _ in range(len(in_channels)):
            fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            
        fpn = nn.ModuleDict({
            'lateral': lateral_convs,
            'fpn': fpn_convs
        })
        
        return fpn
        
    def build_fusion_modules(self, fusion_cfg):
        """构建融合模块"""
        from ..fusion_modules.li_fusion import LI_FusionModule, CB_FusionModule
        
        fusion_modules = nn.ModuleDict()
        
        # LI-Fusion模块
        if fusion_cfg.get('USE_LI_FUSION', True):
            for stage in fusion_cfg.get('FUSION_STAGES', ['x_conv3']):
                # 获取该阶段的通道数
                stage_channels = {
                    'x_conv1': 16,
                    'x_conv2': 32,
                    'x_conv3': 64,
                    'x_conv4': 64
                }.get(stage, 64)
                
                li_config = fusion_cfg.LI_FUSION_CONFIG.copy()
                li_config['IN_CHANNELS_3D'] = stage_channels
                
                fusion_modules[f'li_fusion_{stage}'] = LI_FusionModule(li_config)
                
        # CB-Fusion模块
        if fusion_cfg.get('USE_CB_FUSION', False):
            fusion_modules['cb_fusion'] = CB_FusionModule(fusion_cfg.CB_FUSION_CONFIG)
            
        return fusion_modules
        
    def extract_image_features(self, images):
        """提取图像特征"""
        # 通过主干网络
        features = []
        x = images
        
        # ResNet的各个阶段
        for i, module in enumerate(self.image_backbone):
            x = module(x)
            if i >= 4:  # 从layer1开始
                features.append(x)
                
        # FPN处理
        fpn_features = []
        
        # 横向连接
        for i, feat in enumerate(features):
            lateral = self.image_fpn['lateral'][i](feat)
            fpn_features.append(lateral)
            
        # 自顶向下
        for i in range(len(fpn_features) - 2, -1, -1):
            # 上采样
            upsampled = F.interpolate(
                fpn_features[i+1],
                size=fpn_features[i].shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            fpn_features[i] = fpn_features[i] + upsampled
            
        # FPN输出
        for i in range(len(fpn_features)):
            fpn_features[i] = self.image_fpn['fpn'][i](fpn_features[i])
            
        return fpn_features
        
    def forward(self, batch_dict):
        """
        前向传播
        Args:
            batch_dict: MPCF格式的数据字典
        """
        # 如果有图像输入，提取图像特征
        if hasattr(self, 'image_backbone') and 'images' in batch_dict:
            image_features = self.extract_image_features(batch_dict['images'])
            batch_dict['image_features'] = image_features
            
        # 逐个模块前向传播
        for module in self.module_list:
            # 在3D主干网络阶段进行融合
            if module.__class__.__name__ == 'VoxelBackBone8xFocal':
                batch_dict = self.forward_with_fusion(module, batch_dict)
            else:
                batch_dict = module(batch_dict)
                
        # 后处理
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts
            
    def forward_with_fusion(self, backbone_3d, batch_dict):
        """带融合的3D主干网络前向传播"""
        # 获取初始稀疏张量
        from spconv.pytorch import SparseConvTensor
        
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        
        input_sp_tensor = SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=backbone_3d.sparse_shape,
            batch_size=batch_size
        )
        
        # 前向传播并在指定阶段融合
        x = backbone_3d.conv_input(input_sp_tensor)
        x_conv1 = backbone_3d.conv1(x)
        
        # 在conv1后融合
        if hasattr(self, 'fusion_modules') and 'li_fusion_x_conv1' in self.fusion_modules:
            x_conv1 = self.apply_fusion(x_conv1, 'x_conv1', batch_dict)
            
        x_conv2 = backbone_3d.conv2(x_conv1)
        
        # 在conv2后融合
        if hasattr(self, 'fusion_modules') and 'li_fusion_x_conv2' in self.fusion_modules:
            x_conv2 = self.apply_fusion(x_conv2, 'x_conv2', batch_dict)
            
        x_conv3 = backbone_3d.conv3(x_conv2)
        
        # 在conv3后融合（默认融合点）
        if hasattr(self, 'fusion_modules') and 'li_fusion_x_conv3' in self.fusion_modules:
            x_conv3 = self.apply_fusion(x_conv3, 'x_conv3', batch_dict)
            
        x_conv4 = backbone_3d.conv4(x_conv3)
        
        # 在conv4后融合
        if hasattr(self, 'fusion_modules') and 'li_fusion_x_conv4' in self.fusion_modules:
            x_conv4 = self.apply_fusion(x_conv4, 'x_conv4', batch_dict)
            
        # 输出
        out = backbone_3d.conv_out(x_conv4)
        
        # 更新batch_dict
        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8,
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4
            }
        })
        
        # 转换为稠密特征
        batch_dict['spatial_features'] = out.dense()
        N, C, D, H, W = batch_dict['spatial_features'].shape
        batch_dict['spatial_features'] = batch_dict['spatial_features'].view(N, C * D, H, W)
        
        return batch_dict
        
    def apply_fusion(self, sparse_tensor, stage, batch_dict):
        """应用融合模块"""
        fusion_module = self.fusion_modules[f'li_fusion_{stage}']
        
        # 获取对应分辨率的图像特征
        stage_to_fpn = {
            'x_conv1': 0,  # 最高分辨率
            'x_conv2': 1,
            'x_conv3': 2,
            'x_conv4': 3   # 最低分辨率
        }
        
        if 'image_features' in batch_dict:
            fpn_idx = stage_to_fpn.get(stage, 2)
            image_features = batch_dict['image_features'][fpn_idx]
            
            # 应用融合
            fused_features = fusion_module(
                sparse_tensor.features,
                sparse_tensor.indices,
                image_features,
                batch_dict
            )
            
            # 替换特征
            sparse_tensor = sparse_tensor.replace_feature(fused_features)
            
        return sparse_tensor
        
    def get_training_loss(self):
        """获取训练损失"""
        disp_dict = {}
        
        # 调用各模块的损失函数
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
        
        loss = loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict