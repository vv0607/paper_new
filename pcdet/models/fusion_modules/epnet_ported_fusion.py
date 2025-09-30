"""
EPNetV2融合模块移植与适配
位置: pcdet/models/fusion_layers/epnet_ported_fusion.py
理由: 将EPNetV2的LI-Fusion和CB-Fusion模块适配到OpenPCDet框架
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple


class PointNetSetAbstraction(nn.Module):
    """PointNet++ Set Abstraction层"""
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
            
    def forward(self, xyz, points):
        """
        Args:
            xyz: (B, N, 3) 坐标
            points: (B, N, C) 特征
        Returns:
            new_xyz: (B, npoint, 3) 采样点坐标
            new_points: (B, npoint, mlp[-1]) 采样点特征
        """
        B, N, C = xyz.shape
        
        if self.group_all:
            new_xyz = xyz.mean(dim=1, keepdim=True)
            new_points = torch.cat([xyz, points], dim=-1).transpose(1, 2).unsqueeze(2)
        else:
            # 简化的采样实现
            fps_idx = torch.randint(0, N, (B, self.npoint), device=xyz.device)
            new_xyz = torch.gather(xyz, 1, fps_idx.unsqueeze(-1).expand(-1, -1, 3))
            
            # 简化的分组实现
            grouped_points = []
            for b in range(B):
                dists = torch.cdist(new_xyz[b], xyz[b])
                idx = dists.argsort(dim=-1)[:, :self.nsample]
                grouped = torch.cat([
                    xyz[b][idx],
                    points[b][idx] if points is not None else xyz[b][idx]
                ], dim=-1)
                grouped_points.append(grouped)
            new_points = torch.stack(grouped_points).permute(0, 3, 2, 1)
        
        # 通过MLP
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
            
        new_points = torch.max(new_points, 2)[0]
        new_points = new_points.transpose(1, 2)
        
        return new_xyz, new_points


class LI_FusionModule(nn.Module):
    """
    Location-aware Image fusion module (LI-Fusion)
    从EPNetV2移植的位置感知图像融合模块
    """
    def __init__(self, in_channels_3d, in_channels_2d, out_channels,
                 image_scale=1, mid_channels=128):
        """
        初始化LI融合模块
        Args:
            in_channels_3d: 3D特征输入通道数
            in_channels_2d: 2D特征输入通道数
            out_channels: 输出通道数
            image_scale: 图像特征的缩放因子
            mid_channels: 中间层通道数
        """
        super().__init__()
        self.in_channels_3d = in_channels_3d
        self.in_channels_2d = in_channels_2d
        self.out_channels = out_channels
        self.image_scale = image_scale
        
        # 3D特征变换
        self.voxel_transform = nn.Sequential(
            nn.Linear(in_channels_3d, mid_channels),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, mid_channels)
        )
        
        # 2D特征变换
        self.image_transform = nn.Sequential(
            nn.Conv2d(in_channels_2d, mid_channels, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, 1)
        )
        
        # 融合层
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(mid_channels * 2, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(mid_channels * 2, mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, voxel_features, voxel_coords, image_features, calib_dict):
        """
        前向传播
        Args:
            voxel_features: (N, C) 体素特征
            voxel_coords: (N, 4) 体素坐标 [batch_idx, z, y, x]
            image_features: (B, C, H, W) 图像特征
            calib_dict: 标定信息字典
        Returns:
            fused_features: (N, out_channels) 融合后的特征
        """
        batch_size = image_features.shape[0]
        
        # 变换3D特征
        voxel_feats_transformed = self.voxel_transform(voxel_features)
        
        # 获取每个体素对应的图像特征
        image_feats_sampled = self.sample_image_features(
            voxel_coords, image_features, calib_dict, batch_size
        )
        
        # 变换采样的2D特征
        image_feats_transformed = image_feats_sampled
        
        # 计算注意力权重
        combined_feats = torch.cat([voxel_feats_transformed, image_feats_transformed], dim=1)
        attention_weights = self.attention(combined_feats)
        
        # 加权融合
        weighted_voxel = voxel_feats_transformed * attention_weights
        weighted_image = image_feats_transformed * (1 - attention_weights)
        
        # 最终融合
        fused = torch.cat([weighted_voxel, weighted_image], dim=1)
        fused = fused.unsqueeze(0).transpose(1, 2)  # (1, C*2, N)
        fused_features = self.fusion_conv(fused)
        fused_features = fused_features.squeeze(0).transpose(0, 1)  # (N, C)
        
        return fused_features
        
    def sample_image_features(self, voxel_coords, image_features, calib_dict, batch_size):
        """
        根据体素坐标采样图像特征
        Args:
            voxel_coords: (N, 4) [batch_idx, z, y, x]
            image_features: (B, C, H, W)
            calib_dict: 标定信息
            batch_size: 批次大小
        Returns:
            sampled_features: (N, C) 采样的图像特征
        """
        N = voxel_coords.shape[0]
        C = image_features.shape[1]
        device = voxel_coords.device
        
        sampled_features = []
        
        for b in range(batch_size):
            # 获取当前批次的体素
            batch_mask = voxel_coords[:, 0] == b
            if not batch_mask.any():
                continue
                
            batch_voxel_coords = voxel_coords[batch_mask][:, 1:]  # (N_b, 3) [z, y, x]
            
            # 体素坐标转换到3D点 (简化版本，实际需要考虑体素大小和原点)
            voxel_size = calib_dict.get('voxel_size', [0.05, 0.05, 0.1])
            pc_range = calib_dict.get('point_cloud_range', [0, -40, -3, 70.4, 40, 1])
            
            points_3d = batch_voxel_coords.float() * torch.tensor(voxel_size, device=device)
            points_3d[:, 0] += pc_range[0]  # x
            points_3d[:, 1] += pc_range[1]  # y  
            points_3d[:, 2] += pc_range[2]  # z
            
            # 3D到2D投影
            points_2d = self.project_to_image(points_3d, calib_dict, b)
            
            # 归一化到[-1, 1]用于grid_sample
            H, W = image_features.shape[2:]
            points_2d[:, 0] = (points_2d[:, 0] / W) * 2 - 1
            points_2d[:, 1] = (points_2d[:, 1] / H) * 2 - 1
            points_2d = points_2d.clamp(-1, 1)
            
            # grid_sample采样
            grid = points_2d.unsqueeze(0).unsqueeze(1)  # (1, 1, N_b, 2)
            sampled = F.grid_sample(
                image_features[b:b+1],
                grid,
                mode='bilinear',
                align_corners=False
            )  # (1, C, 1, N_b)
            
            sampled = sampled.squeeze(0).squeeze(1).transpose(0, 1)  # (N_b, C)
            sampled_features.append(sampled)
            
        # 合并所有批次
        if sampled_features:
            sampled_features = torch.cat(sampled_features, dim=0)
        else:
            sampled_features = torch.zeros((N, C), device=device)
            
        return sampled_features
        
    def project_to_image(self, points_3d, calib_dict, batch_idx):
        """
        将3D点投影到图像平面
        Args:
            points_3d: (N, 3) 3D点坐标
            calib_dict: 标定信息
            batch_idx: 批次索引
        Returns:
            points_2d: (N, 2) 2D图像坐标
        """
        # 简化的投影实现，实际应使用标定矩阵
        # 这里假设有P2矩阵 (3x4投影矩阵)
        if 'P2' in calib_dict:
            P2 = calib_dict['P2'][batch_idx]  # (3, 4)
            
            # 添加齐次坐标
            points_homo = torch.cat([
                points_3d,
                torch.ones((points_3d.shape[0], 1), device=points_3d.device)
            ], dim=1)  # (N, 4)
            
            # 投影
            points_cam = torch.matmul(points_homo, P2.T)  # (N, 3)
            points_2d = points_cam[:, :2] / (points_cam[:, 2:3] + 1e-8)
        else:
            # 如果没有标定矩阵，使用默认投影
            points_2d = points_3d[:, :2] * 10 + 200  # 简单的缩放和偏移
            
        return points_2d


class CB_FusionModule(nn.Module):
    """
    Cross-level bidirectional fusion module (CB-Fusion)
    跨层双向融合模块
    """
    def __init__(self, channels_list, scale_factors):
        """
        初始化CB融合模块
        Args:
            channels_list: 各层通道数列表
            scale_factors: 各层缩放因子
        """
        super().__init__()
        self.channels_list = channels_list
        self.scale_factors = scale_factors
        
        # 上采样路径
        self.upsample_layers = nn.ModuleList()
        for i in range(len(channels_list) - 1):
            self.upsample_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(channels_list[i+1], channels_list[i], 3, 
                                      stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(channels_list[i]),
                    nn.ReLU()
                )
            )
            
        # 下采样路径
        self.downsample_layers = nn.ModuleList()
        for i in range(len(channels_list) - 1):
            self.downsample_layers.append(
                nn.Sequential(
                    nn.Conv2d(channels_list[i], channels_list[i+1], 3,
                             stride=2, padding=1),
                    nn.BatchNorm2d(channels_list[i+1]),
                    nn.ReLU()
                )
            )
            
        # 融合卷积
        self.fusion_convs = nn.ModuleList()
        for channels in channels_list:
            self.fusion_convs.append(
                nn.Sequential(
                    nn.Conv2d(channels * 2, channels, 1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU()
                )
            )
            
    def forward(self, features_list):
        """
        前向传播
        Args:
            features_list: 多尺度特征列表 [feat1, feat2, ...]
        Returns:
            fused_features_list: 融合后的特征列表
        """
        # 自顶向下路径
        top_down_features = [features_list[-1]]
        for i in range(len(features_list) - 2, -1, -1):
            upsampled = self.upsample_layers[i](top_down_features[0])
            top_down_features.insert(0, upsampled)
            
        # 自底向上路径
        bottom_up_features = [features_list[0]]
        for i in range(len(features_list) - 1):
            downsampled = self.downsample_layers[i](bottom_up_features[-1])
            bottom_up_features.append(downsampled)
            
        # 融合
        fused_features = []
        for i in range(len(features_list)):
            concat = torch.cat([top_down_features[i], bottom_up_features[i]], dim=1)
            fused = self.fusion_convs[i](concat)
            fused_features.append(fused)
            
        return fused_features


class LIFusionAdapter(nn.Module):
    """
    LI-Fusion适配器，用于OpenPCDet框架
    将OpenPCDet的数据格式转换为EPNetV2融合模块所需的格式
    """
    def __init__(self, model_cfg):
        """
        初始化适配器
        Args:
            model_cfg: 模型配置
        """
        super().__init__()
        self.model_cfg = model_cfg
        
        # 3D和2D特征通道配置
        self.in_channels_3d = model_cfg.get('IN_CHANNELS_3D', 64)
        self.in_channels_2d = model_cfg.get('IN_CHANNELS_2D', 256)
        self.out_channels = model_cfg.get('OUT_CHANNELS', 64)
        
        # 创建LI融合模块
        self.li_fusion = LI_FusionModule(
            in_channels_3d=self.in_channels_3d,
            in_channels_2d=self.in_channels_2d,
            out_channels=self.out_channels,
            image_scale=model_cfg.get('IMAGE_SCALE', 1),
            mid_channels=model_cfg.get('MID_CHANNELS', 128)
        )
        
        # 可选：CB融合模块
        if model_cfg.get('USE_CB_FUSION', False):
            channels_list = model_cfg.get('CB_CHANNELS', [64, 128, 256])
            scale_factors = model_cfg.get('CB_SCALES', [1, 2, 4])
            self.cb_fusion = CB_FusionModule(channels_list, scale_factors)
        else:
            self.cb_fusion = None
            
    def forward(self, voxel_sparse_tensor, image_features, batch_dict):
        """
        前向传播，适配OpenPCDet数据格式
        Args:
            voxel_sparse_tensor: spconv.SparseConvTensor 稀疏张量
            image_features: (B, C, H, W) 图像特征
            batch_dict: 包含标定等信息的批次字典
        Returns:
            fused_sparse_tensor: 融合后的稀疏张量
        """
        # 提取体素特征和坐标
        voxel_features = voxel_sparse_tensor.features  # (N, C)
        voxel_coords = voxel_sparse_tensor.indices  # (N, 4) [batch, z, y, x]
        
        # 准备标定信息
        calib_dict = self.prepare_calib_dict(batch_dict)
        
        # 执行融合
        fused_features = self.li_fusion(
            voxel_features, voxel_coords, 
            image_features, calib_dict
        )
        
        # 创建新的稀疏张量
        fused_sparse_tensor = voxel_sparse_tensor.replace_feature(fused_features)
        
        return fused_sparse_tensor
        
    def prepare_calib_dict(self, batch_dict):
        """
        准备标定信息字典
        Args:
            batch_dict: 批次字典
        Returns:
            calib_dict: 标定信息字典
        """
        calib_dict = {}
        
        # 提取标定矩阵
        if 'calib' in batch_dict:
            calib_dict['P2'] = batch_dict['calib'].P2
            calib_dict['R0'] = batch_dict['calib'].R0
            calib_dict['V2C'] = batch_dict['calib'].V2C
            
        # 提取体素和点云范围信息
        if 'voxel_size' in batch_dict:
            calib_dict['voxel_size'] = batch_dict['voxel_size']
        else:
            calib_dict['voxel_size'] = [0.05, 0.05, 0.1]  # 默认值
            
        if 'point_cloud_range' in batch_dict:
            calib_dict['point_cloud_range'] = batch_dict['point_cloud_range']
        else:
            calib_dict['point_cloud_range'] = [0, -40, -3, 70.4, 40, 1]  # 默认KITTI范围
            
        return calib_dict