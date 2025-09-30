"""
多模态融合可视化调试工具
位置: tools/visualize_fusion.py
理由: 验证坐标系对齐和融合质量的关键工具
"""

import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import cv2

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils


class FusionVisualizer:
    """融合可视化器"""
    
    def __init__(self, cfg_file, data_path):
        """
        初始化可视化器
        Args:
            cfg_file: 配置文件路径
            data_path: 数据路径
        """
        self.cfg = cfg_from_yaml_file(cfg_file, cfg)
        self.data_path = Path(data_path)
        
        # 创建数据加载器
        self.dataset, self.dataloader, _ = build_dataloader(
            dataset_cfg=self.cfg.DATA_CONFIG,
            class_names=self.cfg.CLASS_NAMES,
            batch_size=1,
            dist=False,
            training=False
        )
        
        # 可视化参数
        self.point_size = 2
        self.point_alpha = 0.5
        self.colors = {
            'lidar': 'blue',
            'pseudo': 'green', 
            'fused': 'red',
            'voxel': 'yellow'
        }
        
    def visualize_projection(self, idx=0, save_path=None):
        """
        可视化点云到图像的投影
        Args:
            idx: 数据索引
            save_path: 保存路径
        """
        # 获取数据
        data_dict = self.dataset[idx]
        
        # 获取图像和点云
        image = data_dict.get('images', None)
        points = data_dict.get('points', None)
        calib = data_dict.get('calib', None)
        
        if image is None or points is None or calib is None:
            print("Missing required data for visualization")
            return
            
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 原始图像
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 2. 原始LiDAR投影
        self.project_points_to_image(
            points[points[:, 4] == 0],  # 原始LiDAR点
            image, calib, axes[0, 1], 
            color=self.colors['lidar'],
            title='LiDAR Points Projection'
        )
        
        # 3. 伪点云投影
        if 'pseudo_points' in data_dict:
            self.project_points_to_image(
                data_dict['pseudo_points'],
                image, calib, axes[1, 0],
                color=self.colors['pseudo'],
                title='Pseudo Points Projection'
            )
        else:
            axes[1, 0].imshow(image)
            axes[1, 0].set_title('No Pseudo Points')
            axes[1, 0].axis('off')
            
        # 4. 融合点云投影
        self.project_points_to_image(
            points,  # 所有点
            image, calib, axes[1, 1],
            color=self.colors['fused'],
            title='Fused Points Projection'
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def project_points_to_image(self, points, image, calib, ax, 
                                color='blue', title=''):
        """
        将点云投影到图像上
        Args:
            points: (N, 3+) 点云
            image: 图像数组
            calib: 标定对象
            ax: matplotlib轴对象
            color: 点的颜色
            title: 标题
        """
        # 显示图像
        ax.imshow(image)
        
        # 投影点云
        pts_3d = points[:, :3]
        pts_img, pts_depth = calib.lidar_to_img(pts_3d)
        
        # 过滤有效点
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < image.shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < image.shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid = pts_img[val_flag_merge]
        
        # 绘制投影点
        ax.scatter(pts_valid[:, 0], pts_valid[:, 1], 
                  c=color, s=self.point_size, alpha=self.point_alpha)
        ax.set_title(title)
        ax.axis('off')
        
    def visualize_voxel_projection(self, idx=0, save_path=None):
        """
        可视化体素中心到图像的投影
        Args:
            idx: 数据索引
            save_path: 保存路径
        """
        # 获取数据并进行体素化
        data_dict = self.dataset[idx]
        
        # 执行数据处理获取体素
        from pcdet.datasets.processor.data_processor import DataProcessor
        data_processor = DataProcessor(
            self.cfg.DATA_CONFIG.DATA_PROCESSOR,
            point_cloud_range=self.cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
            training=False
        )
        data_dict = data_processor.forward(data_dict)
        
        if 'voxels' not in data_dict:
            print("No voxels generated")
            return
            
        # 获取体素坐标
        voxel_coords = data_dict['voxel_coords']
        voxel_size = self.cfg.DATA_CONFIG.DATA_PROCESSOR[-1]['VOXEL_SIZE']
        pc_range = self.cfg.DATA_CONFIG.POINT_CLOUD_RANGE
        
        # 计算体素中心的3D坐标
        voxel_centers = voxel_coords[:, [3, 2, 1]].float() * torch.tensor(voxel_size)
        voxel_centers[:, 0] += pc_range[0] + voxel_size[0] / 2
        voxel_centers[:, 1] += pc_range[1] + voxel_size[1] / 2
        voxel_centers[:, 2] += pc_range[2] + voxel_size[2] / 2
        
        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 原始图像
        image = data_dict.get('images', np.zeros((375, 1242, 3)))
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 体素投影
        calib = data_dict.get('calib')
        if calib:
            self.project_points_to_image(
                voxel_centers.numpy(),
                image, calib, axes[1],
                color=self.colors['voxel'],
                title='Voxel Centers Projection'
            )
        else:
            axes[1].text(0.5, 0.5, 'No calibration available', 
                        ha='center', va='center')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def visualize_depth_map(self, idx=0, save_path=None):
        """
        可视化深度图生成
        Args:
            idx: 数据索引  
            save_path: 保存路径
        """
        # 获取数据
        data_dict = self.dataset[idx]
        
        points = data_dict.get('points')
        image = data_dict.get('images', np.zeros((375, 1242, 3)))
        calib = data_dict.get('calib')
        
        if points is None or calib is None:
            print("Missing required data")
            return
            
        # 生成深度图
        h, w = image.shape[:2]
        depth_map = np.full((h, w), np.inf)
        
        # 投影点云
        pts_3d = points[:, :3]
        pts_img, pts_depth = calib.lidar_to_img(pts_3d)
        pts_img = np.round(pts_img).astype(int)
        
        # 填充深度图
        for i in range(len(pts_img)):
            x, y = pts_img[i]
            if 0 <= x < w and 0 <= y < h:
                depth_map[y, x] = min(depth_map[y, x], pts_depth[i])
                
        # 处理无穷值
        depth_map[depth_map == np.inf] = 0
        
        # 可视化
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 原始图像
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 深度图
        depth_vis = axes[0, 1].imshow(depth_map, cmap='jet')
        axes[0, 1].set_title('Depth Map from LiDAR')
        axes[0, 1].axis('off')
        plt.colorbar(depth_vis, ax=axes[0, 1])
        
        # 如果有伪深度图
        if 'pseudo_depth' in data_dict:
            pseudo_depth = data_dict['pseudo_depth']
            pseudo_vis = axes[1, 0].imshow(pseudo_depth, cmap='jet')
            axes[1, 0].set_title('Pseudo Depth Map')
            axes[1, 0].axis('off')
            plt.colorbar(pseudo_vis, ax=axes[1, 0])
            
            # 深度差异图
            diff = np.abs(depth_map - pseudo_depth)
            diff_vis = axes[1, 1].imshow(diff, cmap='hot')
            axes[1, 1].set_title('Depth Difference')
            axes[1, 1].axis('off')
            plt.colorbar(diff_vis, ax=axes[1, 1])
        else:
            axes[1, 0].text(0.5, 0.5, 'No pseudo depth available',
                          ha='center', va='center')
            axes[1, 1].text(0.5, 0.5, 'No depth difference',
                          ha='center', va='center')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def check_alignment_statistics(self, num_samples=10):
        """
        检查投影对齐的统计信息
        Args:
            num_samples: 要检查的样本数
        """
        stats = {
            'total_points': [],
            'valid_projections': [],
            'projection_ratio': [],
            'depth_range': []
        }
        
        for idx in range(min(num_samples, len(self.dataset))):
            data_dict = self.dataset[idx]
            
            points = data_dict.get('points')
            calib = data_dict.get('calib')
            image = data_dict.get('images', np.zeros((375, 1242, 3)))
            
            if points is None or calib is None:
                continue
                
            # 计算投影统计
            pts_3d = points[:, :3]
            pts_img, pts_depth = calib.lidar_to_img(pts_3d)
            
            # 统计有效投影
            h, w = image.shape[:2]
            val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < w)
            val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < h)
            val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
            
            total = len(pts_3d)
            valid = val_flag_merge.sum()
            
            stats['total_points'].append(total)
            stats['valid_projections'].append(valid)
            stats['projection_ratio'].append(valid / total if total > 0 else 0)
            stats['depth_range'].append([pts_depth.min(), pts_depth.max()])
            
        # 打印统计信息
        print("\n=== Projection Alignment Statistics ===")
        print(f"Samples checked: {len(stats['total_points'])}")
        print(f"Average points per frame: {np.mean(stats['total_points']):.1f}")
        print(f"Average valid projections: {np.mean(stats['valid_projections']):.1f}")
        print(f"Average projection ratio: {np.mean(stats['projection_ratio']):.2%}")
        print(f"Depth range: [{np.min([d[0] for d in stats['depth_range']]):.1f}, "
              f"{np.max([d[1] for d in stats['depth_range']]):.1f}]")
        
        return stats


def main():
    parser = argparse.ArgumentParser(description='Visualize fusion results')
    parser.add_argument('--cfg_file', type=str, 
                       default='cfgs/kitti_models/voxel_rcnn_fusion_focals.yaml',
                       help='Config file path')
    parser.add_argument('--data_path', type=str,
                       default='../data/kitti',
                       help='Data path')
    parser.add_argument('--idx', type=int, default=0,
                       help='Data index to visualize')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['projection', 'voxel', 'depth', 'stats', 'all'],
                       help='Visualization mode')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = FusionVisualizer(args.cfg_file, args.data_path)
    
    # 创建保存目录
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None
        
    # 执行可视化
    if args.mode == 'projection' or args.mode == 'all':
        save_path = save_dir / f'projection_{args.idx}.png' if save_dir else None
        visualizer.visualize_projection(args.idx, save_path)
        
    if args.mode == 'voxel' or args.mode == 'all':
        save_path = save_dir / f'voxel_{args.idx}.png' if save_dir else None
        visualizer.visualize_voxel_projection(args.idx, save_path)
        
    if args.mode == 'depth' or args.mode == 'all':
        save_path = save_dir / f'depth_{args.idx}.png' if save_dir else None
        visualizer.visualize_depth_map(args.idx, save_path)
        
    if args.mode == 'stats' or args.mode == 'all':
        visualizer.check_alignment_statistics(num_samples=10)


if __name__ == '__main__':
    main()