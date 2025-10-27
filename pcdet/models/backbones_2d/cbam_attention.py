"""
CBAM (Convolutional Block Attention Module)
用于图像分支的注意力机制
位置: pcdet/models/backbones_2d/cbam_attention.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """通道注意力模块"""
    
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 平均池化和最大池化
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        # 融合并激活
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 通道维度的平均和最大
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """
    CBAM注意力模块
    结合通道注意力和空间注意力
    """
    
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        """
        Args:
            in_channels: 输入特征通道数
            reduction: 通道注意力的降维比例
            kernel_size: 空间注意力的卷积核大小
        """
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        # 先应用通道注意力
        x = x * self.channel_attention(x)
        # 再应用空间注意力
        x = x * self.spatial_attention(x)
        return x


class ImageBackboneWithCBAM(nn.Module):
    """
    带CBAM注意力的图像特征提取主干网络
    用于从RGB图像提取特征
    """
    
    def __init__(self, model_cfg):
        """
        Args:
            model_cfg: 模型配置
        """
        super().__init__()
        
        self.model_cfg = model_cfg
        
        # 配置参数
        in_channels = model_cfg.get('IN_CHANNELS', 3)  # RGB图像
        base_channels = model_cfg.get('BASE_CHANNELS', 64)
        num_stages = model_cfg.get('NUM_STAGES', 4)
        out_channels = model_cfg.get('OUT_CHANNELS', 256)
        use_cbam = model_cfg.get('USE_CBAM', True)
        cbam_reduction = model_cfg.get('CBAM_REDUCTION', 16)
        
        # 构建多阶段特征提取网络
        self.stages = nn.ModuleList()
        current_channels = in_channels
        
        for i in range(num_stages):
            stage_out_channels = base_channels * (2 ** i)
            
            # 卷积块
            conv_block = nn.Sequential(
                nn.Conv2d(current_channels, stage_out_channels, 
                         kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(stage_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(stage_out_channels, stage_out_channels,
                         kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(stage_out_channels),
                nn.ReLU(inplace=True)
            )
            
            # 添加CBAM注意力
            if use_cbam:
                cbam = CBAM(stage_out_channels, reduction=cbam_reduction)
                stage = nn.Sequential(conv_block, cbam)
            else:
                stage = conv_block
                
            self.stages.append(stage)
            current_channels = stage_out_channels
        
        # 输出投影层
        self.output_proj = nn.Sequential(
            nn.Conv2d(current_channels, out_channels, 
                     kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.num_bev_features = out_channels
        
    def forward(self, batch_dict):
        """
        前向传播
        Args:
            batch_dict: 包含图像数据的字典
        Returns:
            batch_dict: 添加了图像特征的字典
        """
        # 获取图像数据
        if 'images' in batch_dict:
            images = batch_dict['images']  # (B, 3, H, W)
        else:
            # 如果没有图像,返回None特征
            B = batch_dict['batch_size']
            device = batch_dict['voxel_features'].device
            batch_dict['image_features'] = None
            return batch_dict
        
        # 逐阶段提取特征
        x = images
        multi_scale_features = []
        
        for i, stage in enumerate(self.stages):
            x = stage(x)
            multi_scale_features.append(x)
        
        # 输出投影
        image_features = self.output_proj(x)
        
        # 保存到batch_dict
        batch_dict['image_features'] = image_features
        batch_dict['image_multi_scale_features'] = multi_scale_features
        
        return batch_dict
"""
CBAM (Convolutional Block Attention Module)
用于图像分支的注意力机制
位置: pcdet/models/backbones_2d/cbam_attention.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """通道注意力模块"""
    
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 平均池化和最大池化
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        # 融合并激活
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 通道维度的平均和最大
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """
    CBAM注意力模块
    结合通道注意力和空间注意力
    """
    
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        """
        Args:
            in_channels: 输入特征通道数
            reduction: 通道注意力的降维比例
            kernel_size: 空间注意力的卷积核大小
        """
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        # 先应用通道注意力
        x = x * self.channel_attention(x)
        # 再应用空间注意力
        x = x * self.spatial_attention(x)
        return x


class ImageBackboneWithCBAM(nn.Module):
    """
    带CBAM注意力的图像特征提取主干网络
    用于从RGB图像提取特征
    """
    
    def __init__(self, model_cfg):
        """
        Args:
            model_cfg: 模型配置
        """
        super().__init__()
        
        self.model_cfg = model_cfg
        
        # 配置参数
        in_channels = model_cfg.get('IN_CHANNELS', 3)  # RGB图像
        base_channels = model_cfg.get('BASE_CHANNELS', 64)
        num_stages = model_cfg.get('NUM_STAGES', 4)
        out_channels = model_cfg.get('OUT_CHANNELS', 256)
        use_cbam = model_cfg.get('USE_CBAM', True)
        cbam_reduction = model_cfg.get('CBAM_REDUCTION', 16)
        
        # 构建多阶段特征提取网络
        self.stages = nn.ModuleList()
        current_channels = in_channels
        
        for i in range(num_stages):
            stage_out_channels = base_channels * (2 ** i)
            
            # 卷积块
            conv_block = nn.Sequential(
                nn.Conv2d(current_channels, stage_out_channels, 
                         kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(stage_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(stage_out_channels, stage_out_channels,
                         kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(stage_out_channels),
                nn.ReLU(inplace=True)
            )
            
            # 添加CBAM注意力
            if use_cbam:
                cbam = CBAM(stage_out_channels, reduction=cbam_reduction)
                stage = nn.Sequential(conv_block, cbam)
            else:
                stage = conv_block
                
            self.stages.append(stage)
            current_channels = stage_out_channels
        
        # 输出投影层
        self.output_proj = nn.Sequential(
            nn.Conv2d(current_channels, out_channels, 
                     kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.num_bev_features = out_channels
        
    def forward(self, batch_dict):
        """
        前向传播
        Args:
            batch_dict: 包含图像数据的字典
        Returns:
            batch_dict: 添加了图像特征的字典
        """
        # 获取图像数据
        if 'images' in batch_dict:
            images = batch_dict['images']  # (B, 3, H, W)
        else:
            # 如果没有图像,返回None特征
            B = batch_dict['batch_size']
            device = batch_dict['voxel_features'].device
            batch_dict['image_features'] = None
            return batch_dict
        
        # 逐阶段提取特征
        x = images
        multi_scale_features = []
        
        for i, stage in enumerate(self.stages):
            x = stage(x)
            multi_scale_features.append(x)
        
        # 输出投影
        image_features = self.output_proj(x)
        
        # 保存到batch_dict
        batch_dict['image_features'] = image_features
        batch_dict['image_multi_scale_features'] = multi_scale_features
        
        return batch_dict

