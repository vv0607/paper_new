"""
深度感知的地面真值增强模块
位置: pcdet/datasets/augmentor/depth_aware_gt_augmentor.py
理由: 避免GT采样时的遮挡问题，提高数据增强质量
"""

import numpy as np
import pickle
from pathlib import Path
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils, box_utils


class DepthAwareDataBaseSampler(object):
    """深度感知的数据库采样器"""
    
    def __init__(self, root_path, sampler_cfg, class_names, logger=None):
        """
        初始化深度感知采样器
        Args:
            root_path: 数据集根路径
            sampler_cfg: 采样器配置
            class_names: 类别名称列表
            logger: 日志记录器
        """
        self.root_path = root_path
        self.class_names = class_names
        self.sampler_cfg = sampler_cfg
        self.logger = logger
        
        # 数据库路径
        self.db_infos = {}
        for class_name in class_names:
            self.db_infos[class_name] = []
            
        # 加载数据库
        db_info_path = self.root_path / sampler_cfg.DB_INFO_PATH[0]
        if db_info_path.exists():
            with open(str(db_info_path), 'rb') as f:
                infos = pickle.load(f)
                [self.db_infos[cur_class].extend(infos[cur_class]) 
                 for cur_class in class_names if cur_class in infos]
                 
        # 采样配置
        self.sample_groups = {}
        self.sample_class_num = {}
        self.limit_whole_scene = sampler_cfg.get('LIMIT_WHOLE_SCENE', False)
        
        for x in sampler_cfg.SAMPLE_GROUPS:
            class_name = x['NAME']
            if class_name not in class_names:
                continue
            self.sample_groups[class_name] = {
                'sample_num': x['NUM'],
                'pointer': len(self.db_infos[class_name]),
                'indices': np.arange(len(self.db_infos[class_name]))
            }
            self.sample_class_num[class_name] = x['NUM']
            
        # 深度感知配置
        self.use_depth_aware = sampler_cfg.get('USE_DEPTH_AWARE', True)
        self.depth_threshold = sampler_cfg.get('DEPTH_THRESHOLD', 0.5)  # 深度差阈值
        self.min_visible_ratio = sampler_cfg.get('MIN_VISIBLE_RATIO', 0.5)  # 最小可见比例
        
    def filter_by_difficulty(self, db_infos, difficulty):
        """根据难度过滤数据库信息"""
        new_db_infos = []
        for info in db_infos:
            if info['difficulty'] not in difficulty:
                continue
            new_db_infos.append(info)
        return new_db_infos
        
    def filter_by_min_points(self, db_infos, min_gt_points_list):
        """根据最小点数过滤"""
        new_db_infos = []
        for info in db_infos:
            if info['num_points_in_gt'] >= min_gt_points_list[info['difficulty']]:
                new_db_infos.append(info)
        return new_db_infos
        
    def sample_with_fixed_number(self, class_name, sample_group):
        """固定数量采样"""
        sample_num, pointer, indices = int(sample_group['sample_num']), \
                                       sample_group['pointer'], \
                                       sample_group['indices']
        if pointer >= len(self.db_infos[class_name]):
            indices = np.random.permutation(len(self.db_infos[class_name]))
            pointer = 0
            
        sampled_dict = [self.db_infos[class_name][idx] 
                       for idx in indices[pointer: pointer + sample_num]]
        pointer += sample_num
        sample_group['pointer'] = pointer
        sample_group['indices'] = indices
        
        return sampled_dict
        
    def depth_aware_filtering(self, sampled_points, sampled_box, 
                             existing_points, existing_depth_map, calib):
        """
        深度感知过滤，移除被遮挡的采样点
        Args:
            sampled_points: 采样的点云 (N, 3+C)
            sampled_box: 采样的3D框 (7,)
            existing_points: 场景中现有的点云
            existing_depth_map: 现有场景的深度图
            calib: 标定信息
        Returns:
            filtered_points: 过滤后的点云
            is_valid: 是否有效（可见点比例是否满足要求）
        """
        if not self.use_depth_aware:
            return sampled_points, True
            
        # 将采样点云投影到图像平面
        sampled_pts_3d = sampled_points[:, :3]
        sampled_pts_img, sampled_pts_depth = calib.lidar_to_img(sampled_pts_3d)
        
        # 获取投影位置的深度值
        h, w = existing_depth_map.shape
        sampled_pts_img = np.round(sampled_pts_img).astype(np.int32)
        
        # 确保索引在有效范围内
        valid_mask = (sampled_pts_img[:, 0] >= 0) & (sampled_pts_img[:, 0] < w) & \
                    (sampled_pts_img[:, 1] >= 0) & (sampled_pts_img[:, 1] < h)
        
        # 初始化可见性掩码
        visible_mask = np.ones(len(sampled_points), dtype=bool)
        
        # 检查每个有效投影点的深度
        for i, is_valid in enumerate(valid_mask):
            if is_valid:
                img_x, img_y = sampled_pts_img[i]
                scene_depth = existing_depth_map[img_y, img_x]
                
                # 如果采样点的深度大于场景深度+阈值，则被遮挡
                if sampled_pts_depth[i] > scene_depth + self.depth_threshold:
                    visible_mask[i] = False
                    
        # 计算可见点的比例
        visible_ratio = visible_mask.sum() / len(sampled_points)
        
        # 如果可见比例太低，则丢弃这个采样
        if visible_ratio < self.min_visible_ratio:
            return sampled_points[visible_mask], False
            
        # 返回可见的点
        return sampled_points[visible_mask], True
        
    def generate_depth_map(self, points, calib, image_shape):
        """
        从点云生成深度图
        Args:
            points: 点云数据 (N, 3+C)
            calib: 标定信息
            image_shape: 图像尺寸 (H, W)
        Returns:
            depth_map: 深度图
        """
        h, w = image_shape
        depth_map = np.full((h, w), np.inf)
        
        # 投影点云到图像
        pts_3d = points[:, :3]
        pts_img, pts_depth = calib.lidar_to_img(pts_3d)
        pts_img = np.round(pts_img).astype(np.int32)
        
        # 更新深度图
        for i in range(len(pts_img)):
            x, y = pts_img[i]
            if 0 <= x < w and 0 <= y < h:
                # 保留最近的深度值
                depth_map[y, x] = min(depth_map[y, x], pts_depth[i])
                
        return depth_map
        
    def __call__(self, data_dict):
        """
        执行深度感知的GT采样
        Args:
            data_dict: 数据字典，包含points, gt_boxes, gt_names等
        Returns:
            augmented data_dict
        """
        gt_boxes = data_dict['gt_boxes'].copy()
        gt_names = data_dict['gt_names'].copy()
        points = data_dict['points'].copy()
        
        # 如果存在深度图或标定信息，生成深度图
        depth_map = None
        if self.use_depth_aware and 'calib' in data_dict:
            if 'depth_map' in data_dict:
                depth_map = data_dict['depth_map']
            else:
                # 从当前点云生成深度图
                image_shape = data_dict.get('image_shape', (375, 1242))
                depth_map = self.generate_depth_map(points, data_dict['calib'], image_shape)
                
        # 对每个类别进行采样
        sampled_boxes = []
        sampled_names = []
        sampled_points_list = []
        
        for class_name, sample_group in self.sample_groups.items():
            if self.limit_whole_scene:
                # 计算场景中已有的该类别数量
                num_existing = (gt_names == class_name).sum()
                sample_group['sample_num'] = max(0, 
                    sample_group['sample_num'] - num_existing)
                    
            if sample_group['sample_num'] > 0:
                # 采样
                sampled_dict = self.sample_with_fixed_number(class_name, sample_group)
                
                for sample_info in sampled_dict:
                    # 加载采样点云
                    file_path = self.root_path / sample_info['path']
                    obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(-1, 4)
                    
                    # 变换到当前场景坐标系
                    obj_box = sample_info['box3d_lidar'].copy()
                    
                    # 深度感知过滤
                    if depth_map is not None and 'calib' in data_dict:
                        filtered_points, is_valid = self.depth_aware_filtering(
                            obj_points, obj_box, points, depth_map, data_dict['calib']
                        )
                        
                        if not is_valid:
                            continue  # 跳过被严重遮挡的对象
                            
                        obj_points = filtered_points
                        
                    # 添加到列表
                    sampled_boxes.append(obj_box)
                    sampled_names.append(class_name)
                    sampled_points_list.append(obj_points)
                    
        # 合并采样结果
        if len(sampled_boxes) > 0:
            sampled_boxes = np.array(sampled_boxes)
            sampled_names = np.array(sampled_names)
            
            # 移除碰撞的采样
            total_valid_sampled_dict = self.remove_collisions(
                sampled_boxes, sampled_names, sampled_points_list, 
                gt_boxes, gt_names, points
            )
            
            # 更新数据
            data_dict = self.update_data_dict(
                data_dict, total_valid_sampled_dict
            )
            
        return data_dict
        
    def remove_collisions(self, sampled_boxes, sampled_names, sampled_points_list,
                         gt_boxes, gt_names, points):
        """移除与现有物体碰撞的采样"""
        # 计算IoU
        iou_matrix = box_utils.boxes_bev_iou_cpu(sampled_boxes[:, :7], gt_boxes[:, :7])
        
        # 找出没有碰撞的采样
        valid_mask = np.max(iou_matrix, axis=1) < 0.05
        
        valid_sampled_dict = {
            'gt_boxes': sampled_boxes[valid_mask],
            'gt_names': sampled_names[valid_mask],
            'points': [sampled_points_list[i] for i in range(len(valid_mask)) if valid_mask[i]]
        }
        
        return valid_sampled_dict
        
    def update_data_dict(self, data_dict, sampled_dict):
        """更新数据字典"""
        # 合并框和名称
        data_dict['gt_boxes'] = np.concatenate([data_dict['gt_boxes'], 
                                                sampled_dict['gt_boxes']], axis=0)
        data_dict['gt_names'] = np.concatenate([data_dict['gt_names'], 
                                                sampled_dict['gt_names']], axis=0)
        
        # 合并点云
        sampled_points = np.concatenate(sampled_dict['points'], axis=0)
        data_dict['points'] = np.concatenate([data_dict['points'], 
                                              sampled_points], axis=0)
        
        return data_dict