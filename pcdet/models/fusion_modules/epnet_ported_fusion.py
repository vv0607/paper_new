"""
类别感知的LI-Fusion模块
根据消融实验结果，针对不同类别使用不同的融合权重

位置: pcdet/models/fusion_modules/epnet_ported_fusion.py

关键改进:
1. Car: 85% 3D + 15% 图像 (图像帮助不大)
2. Pedestrian: 95% 3D + 5% 图像 (图像有负面影响,尽量少用)
3. Cyclist: 60% 3D + 40% 图像 (图像有明显帮助,多用)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LI_FusionModule_ClassAware(nn.Module):
    """
    类别感知的LI-Fusion模块
    """
    
    def __init__(self, model_cfg):
        super().__init__()
        
        self.in_channels_3d = model_cfg.get('IN_CHANNELS_3D', 64)
        self.in_channels_2d = model_cfg.get('IN_CHANNELS_2D', 256)
        self.out_channels = model_cfg.get('OUT_CHANNELS', 64)
        self.mid_channels = model_cfg.get('MID_CHANNELS', 128)
        
        # ⭐ 类别感知配置
        self.use_class_aware = model_cfg.get('USE_CLASS_AWARE_FUSION', True)
        self.class_fusion_weights = model_cfg.get('CLASS_FUSION_WEIGHTS', {
            'Car': [0.85, 0.15],
            'Pedestrian': [0.95, 0.05],
            'Cyclist': [0.60, 0.40]
        })
        self.default_weight = model_cfg.get('DEFAULT_FUSION_WEIGHT', [0.80, 0.20])
        
        # 类别名称到索引的映射 (KITTI标准)
        self.class_names = ['Car', 'Pedestrian', 'Cyclist']
        self.class_name_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # 3D特征变换
        self.voxel_transform = nn.Sequential(
            nn.Linear(self.in_channels_3d, self.mid_channels),
            nn.BatchNorm1d(self.mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # 2D特征变换
        self.image_transform = nn.Sequential(
            nn.Conv2d(self.in_channels_2d, self.mid_channels, 1),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # 融合输出
        self.fusion_conv = nn.Sequential(
            nn.Linear(self.mid_channels * 2, self.out_channels),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 预计算类别权重tensor (加速)
        self._prepare_class_weights()
        
    def _prepare_class_weights(self):
        """预计算类别权重tensor"""
        num_classes = len(self.class_names)
        self.class_weight_3d = torch.zeros(num_classes)
        self.class_weight_2d = torch.zeros(num_classes)
        
        for class_name, idx in self.class_name_to_idx.items():
            if class_name in self.class_fusion_weights:
                weights = self.class_fusion_weights[class_name]
                self.class_weight_3d[idx] = weights[0]
                self.class_weight_2d[idx] = weights[1]
            else:
                self.class_weight_3d[idx] = self.default_weight[0]
                self.class_weight_2d[idx] = self.default_weight[1]
    
    def forward(self, voxel_features, voxel_coords, image_features, batch_dict):
        """
        前向传播
        Args:
            voxel_features: (N, C) 体素特征
            voxel_coords: (N, 4) 体素坐标 [batch, z, y, x]
            image_features: (B, C, H, W) 图像特征
            batch_dict: 批次字典
        Returns:
            fused_features: (N, out_channels) 融合后的特征
        """
        # 变换3D特征
        voxel_feats_transformed = self.voxel_transform(voxel_features)
        
        # 采样对应的2D特征
        image_feats_sampled = self.sample_image_features(
            voxel_coords, image_features, batch_dict
        )
        
        # ⭐ 核心: 根据类别计算融合权重
        if self.use_class_aware and 'gt_boxes' in batch_dict:
            fusion_weights_3d = self.compute_class_aware_weights(
                voxel_coords, batch_dict
            )
        else:
            # 使用默认权重
            fusion_weights_3d = torch.ones(
                voxel_features.shape[0], 1, 
                device=voxel_features.device
            ) * self.default_weight[0]
        
        # 加权融合
        weighted_voxel = voxel_feats_transformed * fusion_weights_3d
        weighted_image = image_feats_sampled * (1 - fusion_weights_3d)
        
        # 最终融合
        fused = torch.cat([weighted_voxel, weighted_image], dim=1)
        fused_features = self.fusion_conv(fused)
        
        return fused_features
    
    def compute_class_aware_weights(self, voxel_coords, batch_dict):
        """
        根据体素所属的物体类别计算融合权重
        
        Args:
            voxel_coords: (N, 4) [batch, z, y, x]
            batch_dict: 包含gt_boxes和gt_names
        Returns:
            weights_3d: (N, 1) 3D特征权重
        """
        N = voxel_coords.shape[0]
        device = voxel_coords.device
        
        # 初始化为默认权重
        weights_3d = torch.ones(N, 1, device=device) * self.default_weight[0]
        
        if 'gt_boxes' not in batch_dict or 'gt_names' not in batch_dict:
            return weights_3d
        
        gt_boxes = batch_dict['gt_boxes']  # (B, M, 7+)
        gt_names = batch_dict.get('gt_names', None)
        
        if gt_names is None:
            return weights_3d
        
        # 将类别权重移到正确的设备
        if self.class_weight_3d.device != device:
            self.class_weight_3d = self.class_weight_3d.to(device)
            self.class_weight_2d = self.class_weight_2d.to(device)
        
        voxel_size = batch_dict.get('voxel_size', [0.05, 0.05, 0.1])
        pc_range = batch_dict.get('point_cloud_range', [0, -40, -3, 70.4, 40, 1])
        
        # 转换体素坐标到实际坐标
        voxel_centers = voxel_coords[:, [3, 2, 1]].float()  # [x, y, z]
        voxel_centers[:, 0] = voxel_centers[:, 0] * voxel_size[0] + pc_range[0]
        voxel_centers[:, 1] = voxel_centers[:, 1] * voxel_size[1] + pc_range[1]
        voxel_centers[:, 2] = voxel_centers[:, 2] * voxel_size[2] + pc_range[2]
        
        # 对每个batch处理
        for b in range(batch_dict['batch_size']):
            batch_mask = voxel_coords[:, 0] == b
            if not batch_mask.any():
                continue
            
            batch_voxels = voxel_centers[batch_mask]
            batch_gt = gt_boxes[b]
            
            # 获取该batch的类别名称
            if isinstance(gt_names, list):
                batch_names = gt_names[b]
            else:
                batch_names = gt_names[b].cpu().numpy() if hasattr(gt_names[b], 'cpu') else gt_names[b]
            
            # 过滤无效boxes
            valid_mask = batch_gt[:, 3] > 0  # 长度>0的box
            batch_gt = batch_gt[valid_mask]
            if isinstance(batch_names, np.ndarray):
                batch_names = batch_names[valid_mask.cpu().numpy()]
            else:
                batch_names = [batch_names[i] for i in range(len(batch_names)) if valid_mask[i]]
            
            if len(batch_gt) == 0:
                continue
            
            # 为每个box分配voxel
            for box, class_name in zip(batch_gt, batch_names):
                # 获取类别对应的融合权重
                if isinstance(class_name, bytes):
                    class_name = class_name.decode('utf-8')
                elif isinstance(class_name, np.ndarray):
                    class_name = str(class_name)
                
                if class_name in self.class_name_to_idx:
                    class_idx = self.class_name_to_idx[class_name]
                    weight_3d = self.class_weight_3d[class_idx]
                else:
                    weight_3d = self.default_weight[0]
                
                # 找到box内的voxels
                center = box[:3]
                dims = box[3:6]
                
                # 简化的box内判断
                in_box = (
                    (torch.abs(batch_voxels[:, 0] - center[0]) < dims[0] / 2) &
                    (torch.abs(batch_voxels[:, 1] - center[1]) < dims[1] / 2) &
                    (torch.abs(batch_voxels[:, 2] - center[2]) < dims[2] / 2)
                )
                
                # 更新权重
                if in_box.any():
                    global_indices = torch.where(batch_mask)[0][in_box]
                    weights_3d[global_indices] = weight_3d
        
        return weights_3d
    
    def sample_image_features(self, voxel_coords, image_features, batch_dict):
        """
        根据体素坐标采样图像特征
        """
        N = voxel_coords.shape[0]
        B, C, H, W = image_features.shape
        device = voxel_coords.device
        
        voxel_size = batch_dict.get('voxel_size', [0.05, 0.05, 0.1])
        pc_range = batch_dict.get('point_cloud_range', [0, -40, -3, 70.4, 40, 1])
        
        # 转换体素坐标到3D点
        voxel_centers = voxel_coords[:, [3, 2, 1]].float()  # [x, y, z]
        voxel_centers[:, 0] = voxel_centers[:, 0] * voxel_size[0] + pc_range[0] + voxel_size[0] / 2
        voxel_centers[:, 1] = voxel_centers[:, 1] * voxel_size[1] + pc_range[1] + voxel_size[1] / 2
        voxel_centers[:, 2] = voxel_centers[:, 2] * voxel_size[2] + pc_range[2] + voxel_size[2] / 2
        
        # 投影到图像平面
        sampled_features = torch.zeros(N, self.mid_channels, device=device)
        
        for b in range(B):
            batch_mask = voxel_coords[:, 0] == b
            if not batch_mask.any():
                continue
                
            batch_voxel_centers = voxel_centers[batch_mask]
            
            # 获取标定信息
            calib = batch_dict.get('calib', None)
            if calib is not None and len(calib) > b:
                points_2d = self.project_to_image(batch_voxel_centers, calib[b])
            else:
                # 简单映射 (fallback)
                points_2d = batch_voxel_centers[:, :2] * 10 + torch.tensor([W/2, H/2], device=device)
            
            # 归一化坐标到[-1, 1]
            points_2d[:, 0] = (points_2d[:, 0] / W) * 2 - 1
            points_2d[:, 1] = (points_2d[:, 1] / H) * 2 - 1
            points_2d = points_2d.clamp(-1, 1)
            
            # 使用grid_sample采样
            grid = points_2d.unsqueeze(0).unsqueeze(1)  # (1, 1, N_b, 2)
            
            # 变换图像特征
            img_feats = self.image_transform(image_features[b:b+1])  # (1, mid_ch, H, W)
            
            sampled = F.grid_sample(
                img_feats,
                grid,
                mode='bilinear',
                align_corners=False
            )  # (1, mid_ch, 1, N_b)
            
            sampled = sampled.squeeze(0).squeeze(1).transpose(0, 1)  # (N_b, mid_ch)
            sampled_features[batch_mask] = sampled
        
        return sampled_features
    
    def project_to_image(self, points_3d, calib):
        """将3D点投影到图像平面"""
        if hasattr(calib, 'P2') and hasattr(calib, 'V2C'):
            pts_3d_homo = torch.cat([
                points_3d, 
                torch.ones((points_3d.shape[0], 1), device=points_3d.device)
            ], dim=1)
            
            V2C = torch.from_numpy(calib.V2C).float().to(points_3d.device)
            pts_cam = torch.matmul(pts_3d_homo, V2C.T)
            
            P2 = torch.from_numpy(calib.P2).float().to(points_3d.device)
            pts_img = torch.matmul(pts_cam, P2.T)
            pts_img[:, :2] /= pts_img[:, 2:3].clamp(min=1e-5)
            
            return pts_img[:, :2]
        else:
            # Fallback
            return points_3d[:, :2] * 10 + 400
