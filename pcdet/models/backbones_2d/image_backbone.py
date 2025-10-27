import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50Backbone(nn.Module):
    """ResNet50图像特征提取"""
    
    def __init__(self, pretrained=True, frozen_stages=1):
        super().__init__()
        
        # 加载预训练ResNet50
        resnet = models.resnet50(pretrained=pretrained)
        
        # 提取各个stage
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 输出: 256通道
        self.layer2 = resnet.layer2  # 输出: 512通道
        self.layer3 = resnet.layer3  # 输出: 1024通道
        self.layer4 = resnet.layer4  # 输出: 2048通道
        
        # 冻结前几层
        self._freeze_stages(frozen_stages)
        
    def _freeze_stages(self, frozen_stages):
        """冻结前N个stage"""
        if frozen_stages >= 0:
            self.conv1.eval()
            self.bn1.eval()
            for param in [self.conv1.parameters(), self.bn1.parameters()]:
                for p in param:
                    p.requires_grad = False
        
        for i in range(1, frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] 图像
        Returns:
            dict: 多尺度特征
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # 提取多尺度特征
        c2 = self.layer1(x)   # [B, 256, H/4, W/4]
        c3 = self.layer2(c2)  # [B, 512, H/8, W/8]
        c4 = self.layer3(c3)  # [B, 1024, H/16, W/16]
        c5 = self.layer4(c4)  # [B, 2048, H/32, W/32]
        
        return {
            'c2': c2,
            'c3': c3,
            'c4': c4,
            'c5': c5
        }


class ImageFeaturePyramid(nn.Module):
    """特征金字塔网络"""
    
    def __init__(self, in_channels=[256, 512, 1024, 2048], out_channel=256):
        super().__init__()
        
        # 横向连接（降维）
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channel, 1) 
            for in_ch in in_channels
        ])
        
        # 平滑层
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channel, out_channel, 3, padding=1)
            for _ in range(len(in_channels))
        ])
    
    def forward(self, features):
        """
        Args:
            features: dict with keys ['c2', 'c3', 'c4', 'c5']
        Returns:
            dict: 统一维度的多尺度特征
        """
        # 从高层到低层
        feat_list = [features['c5'], features['c4'], 
                     features['c3'], features['c2']]
        
        # Top-down pathway
        laterals = [lateral(feat) for lateral, feat 
                    in zip(self.lateral_convs, feat_list)]
        
        # 从上到下融合
        for i in range(len(laterals) - 1):
            # 上采样高层特征
            upsampled = nn.functional.interpolate(
                laterals[i], 
                size=laterals[i+1].shape[2:],
                mode='bilinear', 
                align_corners=False
            )
            # 相加
            laterals[i+1] = laterals[i+1] + upsampled
        
        # 平滑
        outputs = [smooth(lateral) for smooth, lateral 
                   in zip(self.smooth_convs, laterals)]
        
        return {
            'p5': outputs[0],  # [B, 256, H/32, W/32]
            'p4': outputs[1],  # [B, 256, H/16, W/16]
            'p3': outputs[2],  # [B, 256, H/8, W/8]
            'p2': outputs[3],  # [B, 256, H/4, W/4]
        }
