import torch
import torch.nn as nn


class LIFusionModule(nn.Module):
    """
    LI-Fusion: 类别感知的多模态融合模块
    """
    
    def __init__(self, point_dim=128, image_dim=256, fusion_dim=128,
                 class_names=['Car', 'Pedestrian', 'Cyclist'],
                 class_fusion_weights=None):
        super().__init__()
        
        self.point_dim = point_dim
        self.image_dim = image_dim
        self.fusion_dim = fusion_dim
        self.class_names = class_names
        
        # 默认融合权重
        if class_fusion_weights is None:
            class_fusion_weights = {
                'Car': [0.75, 0.25],
                'Pedestrian': [1.0, 0.0],
                'Cyclist': [1.0, 0.0]
            }
        self.class_fusion_weights = class_fusion_weights
        
        # 特征投影层
        self.point_proj = nn.Sequential(
            nn.Linear(point_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # 可选：注意力机制
        self.use_attention = True
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.ReLU(inplace=True),
                nn.Linear(fusion_dim, 2),  # 输出2个权重
                nn.Softmax(dim=-1)
            )
        
        print(f"[LIFusionModule] Initialized with class weights:")
        for cls_name, weights in class_fusion_weights.items():
            print(f"  {cls_name}: 3D={weights[0]:.2f}, Image={weights[1]:.2f}")
    
    def forward(self, point_features, image_features, roi_labels=None):
        """
        Args:
            point_features: [N, point_dim] 3D ROI特征
            image_features: [N, image_dim] 2D图像特征
            roi_labels: [N] ROI的类别标签 (0: Car, 1: Ped, 2: Cyc)
                        如果为None，使用默认权重
        Returns:
            [N, fusion_dim] 融合后的特征
        """
        # 1. 投影到统一维度
        point_feat = self.point_proj(point_features)  # [N, fusion_dim]
        image_feat = self.image_proj(image_features)  # [N, fusion_dim]
        
        # 2. 类别感知融合
        if roi_labels is not None:
            fused_feat = self.class_aware_fusion(
                point_feat, image_feat, roi_labels
            )
        else:
            # 默认融合（如果没有标签）
            fused_feat = 0.8 * point_feat + 0.2 * image_feat
        
        return fused_feat
    
    def class_aware_fusion(self, point_feat, image_feat, labels):
        """
        根据类别使用不同的融合权重
        
        Args:
            point_feat: [N, fusion_dim]
            image_feat: [N, fusion_dim]
            labels: [N] 类别ID
        Returns:
            [N, fusion_dim]
        """
        N = point_feat.shape[0]
        fused_feat = torch.zeros_like(point_feat)
        
        # 对每个类别分别处理
        for class_id, class_name in enumerate(self.class_names):
            # 找到属于这个类别的ROI
            mask = (labels == class_id)
            if mask.sum() == 0:
                continue
            
            # 获取这个类别的融合权重
            weights = self.class_fusion_weights.get(
                class_name, [0.8, 0.2]
            )
            w_3d, w_2d = weights
            
            # 如果使用注意力机制
            if self.use_attention and w_2d > 0:
                # 拼接特征
                concat_feat = torch.cat([
                    point_feat[mask], 
                    image_feat[mask]
                ], dim=1)  # [N_cls, fusion_dim*2]
                
                # 学习权重
                attn_weights = self.attention(concat_feat)  # [N_cls, 2]
                
                # 加权融合
                fused_feat[mask] = (
                    attn_weights[:, 0:1] * point_feat[mask] +
                    attn_weights[:, 1:2] * image_feat[mask]
                )
            else:
                # 固定权重融合
                fused_feat[mask] = (
                    w_3d * point_feat[mask] + 
                    w_2d * image_feat[mask]
                )
        
        return fused_feat
    
    def get_fusion_weights(self, class_name):
        """获取指定类别的融合权重"""
        return self.class_fusion_weights.get(
            class_name, [0.8, 0.2]
        )
