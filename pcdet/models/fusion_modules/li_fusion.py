"""
LI-Fusion模块，适配MPCF项目结构
位置: pcdet/models/fusion_modules/li_fusion.py
理由: 在MPCF框架内实现EPNet++的融合功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LI_FusionModule(nn.Module):
    """
    Location-aware Image fusion module
    适配MPCF项目的LI-Fusion实现
    """
    
    def __init__(self, model_cfg):
        """
        初始化LI融合模块
        Args:
            model_cfg: 模型配置字典
        """
        super().__init__()
        
        # 配置参数
        self.in_channels_3d = model_cfg.get('IN_CHANNELS_3D', 64)
        self.in_channels_2d = model_cfg.get('IN_CHANNELS_2D', 256)
        self.out_channels = model_cfg.get('OUT_CHANNELS', 64)
        self.mid_channels = model_cfg.get('MID_CHANNELS', 128)
        
        # 3D特征变换
        self.voxel_transform = nn.Sequential(
            nn.Linear(self.in_channels_3d, self.mid_channels),
            nn.BatchNorm1d(self.mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.mid_channels, self.mid_channels)
        )
        
        # 2D特征变换
        self.image_transform = nn.Sequential(
            nn.Conv2d(self.in_channels_2d, self.mid_channels, 1),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(self.mid_channels * 2, self.mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.mid_channels, 1),
            nn.Sigmoid()
        )
        
        # 融合输出
        self.fusion_conv = nn.Sequential(
            nn.Linear(self.mid_channels * 2, self.out_channels),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, voxel_features, voxel_coords, image_features, batch_dict):
        """
        前向传播
        Args:
            voxel_features: (N, C) 体素特征
            voxel_coords: (N, 4) 体素坐标 [batch, z, y, x]
            image_features: (B, C, H, W) 图像特征
            batch_dict: MPCF的批次字典
        Returns:
            fused_features: (N, out_channels) 融合后的特征
        """
        batch_size = batch_dict['batch_size']
        
        # 变换3D特征
        voxel_feats_transformed = self.voxel_transform(voxel_features)
        
        # 采样对应的2D特征
        image_feats_sampled = self.sample_image_features(
            voxel_coords, image_features, batch_dict
        )
        
        # 计算注意力权重
        combined_feats = torch.cat([
            voxel_feats_transformed, 
            image_feats_sampled
        ], dim=1)
        
        attention_weights = self.attention(combined_feats)
        
        # 加权融合
        weighted_voxel = voxel_feats_transformed * attention_weights
        weighted_image = image_feats_sampled * (1 - attention_weights)
        
        # 最终融合
        fused = torch.cat([weighted_voxel, weighted_image], dim=1)
        fused_features = self.fusion_conv(fused)
        
        return fused_features
        
    def sample_image_features(self, voxel_coords, image_features, batch_dict):
        """
        根据体素坐标采样图像特征
        Args:
            voxel_coords: (N, 4) [batch, z, y, x]
            image_features: (B, C, H, W)
            batch_dict: 包含标定信息
        Returns:
            sampled_features: (N, C)
        """
        N = voxel_coords.shape[0]
        B, C, H, W = image_features.shape
        device = voxel_coords.device
        
        # 获取体素大小和范围（MPCF配置）
        voxel_size = batch_dict.get('voxel_size', [0.16, 0.16, 4])  # MPCF默认
        pc_range = batch_dict.get('point_cloud_range', [0, -39.68, -3, 69.12, 39.68, 1])
        
        # 转换体素坐标到3D点
        voxel_centers = voxel_coords[:, [3, 2, 1]].float()  # [x, y, z]
        voxel_centers[:, 0] = voxel_centers[:, 0] * voxel_size[0] + pc_range[0] + voxel_size[0] / 2
        voxel_centers[:, 1] = voxel_centers[:, 1] * voxel_size[1] + pc_range[1] + voxel_size[1] / 2
        voxel_centers[:, 2] = voxel_centers[:, 2] * voxel_size[2] + pc_range[2] + voxel_size[2] / 2
        
        # 投影到图像平面
        sampled_features = torch.zeros(N, C, device=device)
        
        for b in range(B):
            # 获取当前批次的体素
            batch_mask = voxel_coords[:, 0] == b
            if not batch_mask.any():
                continue
                
            batch_voxel_centers = voxel_centers[batch_mask]
            
            # 获取标定信息
            calib = batch_dict.get('calib', None)
            if calib is not None and len(calib) > b:
                # 投影到图像
                points_2d = self.project_to_image(batch_voxel_centers, calib[b])
            else:
                # 简单映射（用于测试）
                points_2d = batch_voxel_centers[:, :2] * 10 + torch.tensor([W/2, H/2], device=device)
                
            # 归一化坐标到[-1, 1]
            points_2d[:, 0] = (points_2d[:, 0] / W) * 2 - 1
            points_2d[:, 1] = (points_2d[:, 1] / H) * 2 - 1
            points_2d = points_2d.clamp(-1, 1)
            
            # 使用grid_sample采样
            grid = points_2d.unsqueeze(0).unsqueeze(1)  # (1, 1, N_b, 2)
            sampled = F.grid_sample(
                image_features[b:b+1],
                grid,
                mode='bilinear',
                align_corners=False
            )  # (1, C, 1, N_b)
            
            sampled = sampled.squeeze(0).squeeze(1).transpose(0, 1)  # (N_b, C)
            sampled_features[batch_mask] = sampled
            
        # 变换采样的特征
        sampled_features = self.image_transform(
            sampled_features.unsqueeze(2).unsqueeze(3)
        ).squeeze(3).squeeze(2)
        
        return sampled_features
        
    def project_to_image(self, points_3d, calib):
        """
        将3D点投影到图像平面（KITTI格式）
        Args:
            points_3d: (N, 3) 激光雷达坐标系下的3D点
            calib: 标定对象
        Returns:
            points_2d: (N, 2) 图像坐标
        """
        # KITTI标定投影
        if hasattr(calib, 'P2') and hasattr(calib, 'V2C'):
            # Velodyne到相机坐标系
            pts_3d_homo = torch.cat([
                points_3d, 
                torch.ones((points_3d.shape[0], 1), device=points_3d.device)
            ], dim=1)
            
            V2C = torch.from_numpy(calib.V2C).float().to(points_3d.device)
            pts_cam = torch.matmul(pts_3d_homo, V2C.T)
            
            # 相机坐标系到图像平面
            P2 = torch.from_numpy(calib.P2).float().to(points_3d.device)
            pts_img = torch.matmul(pts_cam, P2.T)
            pts_img[:, :2] /= pts_img[:, 2:3]
            
            return pts_img[:, :2]
        else:
            # 默认投影（用于测试）
            return points_3d[:, :2] * 10 + 400


class CB_FusionModule(nn.Module):
    """
    Cross-level Bidirectional Fusion Module
    跨层双向融合模块
    """
    
    def __init__(self, model_cfg):
        """
        初始化CB融合模块
        Args:
            model_cfg: 模型配置
        """
        super().__init__()
        
        # 获取各层通道数
        self.channels = model_cfg.get('CHANNELS', [16, 32, 64, 64])
        self.out_channels = model_cfg.get('OUT_CHANNELS', 64)
        
        # 横向连接
        self.lateral_convs = nn.ModuleList()
        for ch in self.channels:
            self.lateral_convs.append(
                nn.Conv1d(ch, self.out_channels, 1)
            )
            
        # 自顶向下路径
        self.td_convs = nn.ModuleList()
        for i in range(len(self.channels) - 1):
            self.td_convs.append(
                nn.Conv1d(self.out_channels, self.out_channels, 1)
            )
            
        # 自底向上路径  
        self.bu_convs = nn.ModuleList()
        for i in range(len(self.channels) - 1):
            self.bu_convs.append(
                nn.Conv1d(self.out_channels, self.out_channels, 1)
            )
            
        # 输出层
        self.output_convs = nn.ModuleList()
        for _ in self.channels:
            self.output_convs.append(
                nn.Sequential(
                    nn.Conv1d(self.out_channels * 2, self.out_channels, 1),
                    nn.BatchNorm1d(self.out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            
    def forward(self, multi_scale_features):
        """
        前向传播
        Args:
            multi_scale_features: 多尺度特征字典
        Returns:
            fused_features: 融合后的特征
        """
        # 提取各层特征
        features = []
        for key in ['x_conv1', 'x_conv2', 'x_conv3', 'x_conv4']:
            if key in multi_scale_features:
                feat = multi_scale_features[key].features  # 稀疏特征
                features.append(feat)
                
        # 横向连接
        lateral_features = []
        for i, feat in enumerate(features):
            lateral = self.lateral_convs[i](feat.unsqueeze(0).transpose(1, 2))
            lateral_features.append(lateral.transpose(1, 2).squeeze(0))
            
        # 自顶向下
        td_features = [lateral_features[-1]]
        for i in range(len(lateral_features) - 2, -1, -1):
            # 上采样并融合
            td = self.td_convs[i](td_features[0].unsqueeze(0).transpose(1, 2))
            td = td.transpose(1, 2).squeeze(0)
            td_features.insert(0, td + lateral_features[i])
            
        # 自底向上
        bu_features = [lateral_features[0]]
        for i in range(1, len(lateral_features)):
            # 下采样并融合
            bu = self.bu_convs[i-1](bu_features[-1].unsqueeze(0).transpose(1, 2))
            bu = bu.transpose(1, 2).squeeze(0)
            bu_features.append(bu + lateral_features[i])
            
        # 融合输出
        output_features = []
        for i in range(len(features)):
            concat = torch.cat([td_features[i], bu_features[i]], dim=1)
            out = self.output_convs[i](concat.unsqueeze(0).transpose(1, 2))
            output_features.append(out.transpose(1, 2).squeeze(0))
            
        return output_features[-1]  # 返回最后一层特征