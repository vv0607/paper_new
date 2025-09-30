"""
完整的MPCF KittiDataset类，包含融合功能
位置: pcdet/datasets/kitti/kitti_dataset.py
基于MPCF原有功能，添加FocalsConv和多模态融合支持
"""

import copy
import pickle
import os
from pathlib import Path

import numpy as np
from skimage import io
import torch

from ..dataset import DatasetTemplate
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from .kitti_object_eval_python import kitti_utils


class KittiDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        初始化KITTI数据集
        Args:
            dataset_cfg: 数据集配置
            class_names: 类别名称
            training: 是否训练模式
            root_path: 数据根路径
            logger: 日志记录器
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, 
            root_path=root_path, logger=logger
        )
        
        # 数据路径设置
        self.split = self.dataset_cfg.DATA_SPLIT['train' if training else 'test']
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')
        
        # 数据文件列表
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None
        
        # MPCF伪点云配置
        self.use_pseudo_label = dataset_cfg.get('USE_PSEUDO_LABEL', False)
        self.pseudo_label_dir = dataset_cfg.get('PSEUDO_LABEL_DIR', 'depth_pseudo_rgbseguv_twise')
        
        # 新增：融合相关配置
        self.merge_pseudo_points = dataset_cfg.get('MERGE_PSEUDO_POINTS', False)
        self.merge_mode = dataset_cfg.get('MERGE_MODE', 'concat')  # 'concat' or 'smart'
        self.pseudo_point_features = dataset_cfg.get('PSEUDO_POINT_FEATURES', 7)  # [x,y,z,i,r,g,b]
        self.use_image_features = dataset_cfg.get('USE_IMAGE_FEATURES', False)
        self.image_size = dataset_cfg.get('IMAGE_SIZE', [375, 1242])
        
        # 新增：图像增强配置
        self.image_augmentor = None
        if dataset_cfg.get('IMAGE_AUGMENTOR', None) is not None and training:
            from ..augmentor.image_augmentor import ImageAugmentor
            self.image_augmentor = ImageAugmentor(dataset_cfg.IMAGE_AUGMENTOR)
            
        # 加载数据信息
        self.kitti_infos = []
        self.include_kitti_data(self.mode)
        
        # 设置数据增强器
        if self.training:
            self.set_augmentor()
            
        self.logger.info('Total samples for KITTI dataset: %d' % (len(self.kitti_infos)))
        
    def include_kitti_data(self, mode):
        """加载KITTI数据信息"""
        if self.logger is not None:
            self.logger.info('Loading KITTI dataset')
            
        kitti_infos = []
        
        if mode == 'train':
            info_paths = self.dataset_cfg.INFO_PATH['train']
        elif mode == 'test':
            info_paths = self.dataset_cfg.INFO_PATH['test']
        else:
            info_paths = []
            
        for info_path in info_paths:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                kitti_infos.extend(infos)
                
        self.kitti_infos.extend(kitti_infos)
        
        if self.logger is not None:
            self.logger.info('Total samples for KITTI dataset: %d' % (len(kitti_infos)))
            
    def set_split(self, split):
        """设置数据集划分"""
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')
        
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None
        
    def get_lidar(self, idx):
        """
        获取原始激光雷达点云
        Args:
            idx: 样本索引
        Returns:
            points: (N, 4) [x, y, z, intensity]
        """
        lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        assert lidar_file.exists(), f'Lidar file not found: {lidar_file}'
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
        
    def get_lidar_pseudo(self, idx):
        """
        获取伪点云数据（MPCF格式）
        Args:
            idx: 样本索引
        Returns:
            pseudo_points: (N, 7) [x, y, z, intensity, r, g, b]
        """
        # 从depth_pseudo_rgbseguv_twise目录读取
        pseudo_file = self.root_split_path / self.pseudo_label_dir / ('%s.bin' % idx)
        
        if pseudo_file.exists():
            # MPCF的伪点云格式：[x, y, z, intensity, r, g, b]
            pseudo_points = np.fromfile(str(pseudo_file), dtype=np.float32)
            
            # 确保是7维特征
            if len(pseudo_points) % 7 == 0:
                pseudo_points = pseudo_points.reshape(-1, 7)
            else:
                # 如果维度不对，尝试其他格式
                num_points = len(pseudo_points) // self.pseudo_point_features
                pseudo_points = pseudo_points[:num_points * self.pseudo_point_features]
                pseudo_points = pseudo_points.reshape(num_points, -1)
                
                # 如果不足7维，补充默认值
                if pseudo_points.shape[1] < 7:
                    padding = np.ones((num_points, 7 - pseudo_points.shape[1])) * 0.5
                    pseudo_points = np.hstack([pseudo_points, padding])
                    
            return pseudo_points
        else:
            if self.logger is not None:
                self.logger.warning(f'Pseudo point cloud not found: {pseudo_file}')
            return None
            
    def get_image(self, idx):
        """
        获取RGB图像
        Args:
            idx: 样本索引
        Returns:
            image: (H, W, 3) RGB图像
        """
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists(), f'Image file not found: {img_file}'
        
        import cv2
        image = cv2.imread(str(img_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 调整图像大小（如果需要）
        if self.image_size is not None:
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
            
        return image
        
    def get_calib(self, idx):
        """
        获取标定信息
        Args:
            idx: 样本索引
        Returns:
            calib: 标定对象
        """
        calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
        assert calib_file.exists(), f'Calibration file not found: {calib_file}'
        return calibration_kitti.Calibration(calib_file)
        
    def get_road_plane(self, idx):
        """
        获取地面平面（如果有）
        Args:
            idx: 样本索引
        Returns:
            plane: 地面平面参数
        """
        plane_file = self.root_split_path / 'planes' / ('%s.txt' % idx)
        if not plane_file.exists():
            return None
            
        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)
        
        # 转换到激光雷达坐标系
        if plane.shape[0] == 4:
            return plane.reshape(4,)
        else:
            return None
            
    def merge_point_clouds(self, original_points, pseudo_points):
        """
        合并原始点云和伪点云
        Args:
            original_points: (N1, 4) [x, y, z, intensity]
            pseudo_points: (N2, 7) [x, y, z, intensity, r, g, b]
        Returns:
            merged_points: (N1+N2, 7) 合并后的点云
        """
        if pseudo_points is None or len(pseudo_points) == 0:
            # 没有伪点云，为原始点云添加RGB通道
            rgb_default = np.ones((len(original_points), 3)) * 0.5
            merged_points = np.hstack([original_points, rgb_default])
            return merged_points
            
        # 为原始点云添加RGB通道（默认值）
        original_expanded = np.hstack([
            original_points,
            np.ones((len(original_points), 3)) * 0.5  # 默认RGB
        ])
        
        # 合并策略
        if self.merge_mode == 'concat':
            # 简单拼接
            merged_points = np.vstack([original_expanded, pseudo_points])
            
        elif self.merge_mode == 'smart':
            # 智能合并：基于距离的策略
            merged_points = self.smart_merge(original_expanded, pseudo_points)
            
        elif self.merge_mode == 'replace_far':
            # 远距离替换策略
            original_dist = np.linalg.norm(original_points[:, :2], axis=1)
            near_mask = original_dist < 30.0  # 30米内保留原始点
            
            pseudo_dist = np.linalg.norm(pseudo_points[:, :2], axis=1)
            far_mask = pseudo_dist >= 25.0  # 25米外使用伪点云
            
            merged_points = np.vstack([
                original_expanded[near_mask],
                pseudo_points[far_mask]
            ])
        else:
            merged_points = original_expanded
            
        return merged_points
        
    def smart_merge(self, original_points, pseudo_points):
        """
        智能合并策略：根据点云密度和距离动态合并
        Args:
            original_points: (N1, 7) 扩展后的原始点云
            pseudo_points: (N2, 7) 伪点云
        Returns:
            merged_points: 合并后的点云
        """
        # 计算原始点云的密度
        original_dist = np.linalg.norm(original_points[:, :2], axis=1)
        
        # 定义距离区间
        dist_ranges = [(0, 15), (15, 30), (30, 50), (50, 100)]
        merged_list = []
        
        for min_dist, max_dist in dist_ranges:
            # 原始点云在该区间的点
            orig_mask = (original_dist >= min_dist) & (original_dist < max_dist)
            orig_in_range = original_points[orig_mask]
            
            # 伪点云在该区间的点
            pseudo_dist = np.linalg.norm(pseudo_points[:, :2], axis=1)
            pseudo_mask = (pseudo_dist >= min_dist) & (pseudo_dist < max_dist)
            pseudo_in_range = pseudo_points[pseudo_mask]
            
            # 根据距离决定混合比例
            if max_dist <= 20:
                # 近距离：主要使用原始点云
                ratio = 0.9
            elif max_dist <= 40:
                # 中距离：平衡使用
                ratio = 0.5
            else:
                # 远距离：主要使用伪点云
                ratio = 0.1
                
            # 采样
            n_orig = int(len(orig_in_range) * ratio)
            n_pseudo = int(len(pseudo_in_range) * (1 - ratio))
            
            if n_orig > 0 and len(orig_in_range) > 0:
                indices = np.random.choice(len(orig_in_range), min(n_orig, len(orig_in_range)), replace=False)
                merged_list.append(orig_in_range[indices])
                
            if n_pseudo > 0 and len(pseudo_in_range) > 0:
                indices = np.random.choice(len(pseudo_in_range), min(n_pseudo, len(pseudo_in_range)), replace=False)
                merged_list.append(pseudo_in_range[indices])
                
        if merged_list:
            merged_points = np.vstack(merged_list)
        else:
            merged_points = original_points
            
        return merged_points
        
    def get_label(self, idx):
        """
        获取标注信息
        Args:
            idx: 样本索引
        Returns:
            obj_list: 物体列表
        """
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists()
        return object3d_kitti.get_objects_from_label(label_file)
        
    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """生成预测结果字典"""
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict
            
        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            
            if pred_scores.shape[0] == 0:
                return pred_dict
                
            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )
            
            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            
            return pred_dict
            
        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]
            
            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)
            
            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl
                    
                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)
                                 
        return annos
        
    def evaluation(self, det_annos, class_names, **kwargs):
        """评估函数"""
        if 'annos' not in self.kitti_infos[0].keys():
            return None, {}
            
        from .kitti_object_eval_python import eval as kitti_eval
        
        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)
        
        return ap_result_str, ap_dict
        
    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs
            
        return len(self.kitti_infos)
        
    def __getitem__(self, index):
        """
        获取一个数据样本
        Args:
            index: 索引
        Returns:
            data_dict: 数据字典
        """
        # 处理合并epoch的情况
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)
            
        info = copy.deepcopy(self.kitti_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']
        
        # 获取原始点云
        points = self.get_lidar(sample_idx)
        
        # 获取并合并伪点云（如果启用）
        if self.use_pseudo_label:
            pseudo_points = self.get_lidar_pseudo(sample_idx)
            
            if self.merge_pseudo_points and pseudo_points is not None:
                # 保存原始点云数量（用于调试）
                info['original_points_num'] = len(points)
                info['pseudo_points_num'] = len(pseudo_points)
                
                # 合并点云
                points = self.merge_point_clouds(points, pseudo_points)
                info['merged_points_num'] = len(points)
                
            elif not self.merge_pseudo_points and pseudo_points is not None:
                # 仅使用伪点云（MPCF原始模式）
                points = pseudo_points
                
        # 构建输入字典
        input_dict = {
            'points': points,
            'frame_id': sample_idx,
        }
        
        # 获取标定信息
        calib = self.get_calib(sample_idx)
        input_dict['calib'] = calib
        
        # 加载图像（用于融合）
        if self.use_image_features or self.dataset_cfg.get('USE_IMAGE_BRANCH', False):
            image = self.get_image(sample_idx)
            
            # 应用图像增强
            if self.image_augmentor is not None and self.training:
                image = self.image_augmentor.forward(image)
                
            input_dict['images'] = image
            
        # 获取路面信息（如果有）
        road_plane = self.get_road_plane(sample_idx)
        if road_plane is not None:
            input_dict['road_plane'] = road_plane
            
        # 获取标注信息（训练时）
        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)
            
            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': gt_boxes_lidar
            })
            
            if "gt_boxes2d" in self.dataset_cfg.get('USE_2D_BOXES', False):
                input_dict['gt_boxes2d'] = annos["bbox"]
                
        # 数据增强（训练时）
        if self.training and len(input_dict['gt_boxes']) > 0:
            # 如果有深度图，用于深度感知增强
            if self.dataset_cfg.get('USE_DEPTH_AWARE_AUG', False) and 'images' in input_dict:
                input_dict['depth_map'] = self.generate_depth_map_from_points(
                    points, calib, image.shape[:2]
                )
                
            # 应用数据增强
            input_dict = self.data_augmentor.forward(
                data_dict=input_dict
            )
            
        # 准备数据（体素化等）
        data_dict = self.prepare_data(data_dict=input_dict)
        
        # 添加图像形状信息
        if 'images' in input_dict:
            data_dict['images'] = input_dict['images']
            data_dict['image_shape'] = np.array(input_dict['images'].shape[:2], dtype=np.int32)
            
        # 添加元信息
        data_dict['metadata'] = info.get('metadata', {})
        
        return data_dict
        
    def generate_depth_map_from_points(self, points, calib, image_shape):
        """
        从点云生成深度图（用于深度感知增强）
        Args:
            points: 点云
            calib: 标定
            image_shape: (H, W)
        Returns:
            depth_map: 深度图
        """
        h, w = image_shape
        depth_map = np.full((h, w), np.inf)
        
        # 投影点云到图像
        pts_3d = points[:, :3]
        pts_rect = calib.lidar_to_rect(pts_3d)
        pts_img, pts_depth = calib.rect_to_img(pts_rect)
        pts_img = pts_img.astype(np.int32)
        
        # 填充深度图
        mask = (pts_img[:, 0] >= 0) & (pts_img[:, 0] < w) & \
               (pts_img[:, 1] >= 0) & (pts_img[:, 1] < h) & \
               (pts_depth > 0)
               
        pts_img = pts_img[mask]
        pts_depth = pts_depth[mask]
        
        for i in range(len(pts_img)):
            x, y = pts_img[i]
            depth_map[y, x] = min(depth_map[y, x], pts_depth[i])
            
        # 处理无穷值
        depth_map[depth_map == np.inf] = 0
        
        return depth_map
        
    @staticmethod
    def collate_batch(batch_list, _unused=False):
        """
        整理批次数据
        Args:
            batch_list: 批次列表
        Returns:
            batch_dict: 批次字典
        """
        data_dict = {}
        for key in batch_list[0].keys():
            if key in ['points', 'voxels']:
                # 处理可变长度的数据
                data_list = []
                for i, data in enumerate(batch_list):
                    data_item = data[key]
                    if key == 'points':
                        # 添加批次索引
                        batch_idx = np.ones((data_item.shape[0], 1)) * i
                        data_item = np.concatenate([batch_idx, data_item], axis=1)
                    data_list.append(data_item)
                    
                data_dict[key] = np.concatenate(data_list, axis=0)
                
            elif key in ['gt_boxes']:
                # 处理GT框
                max_gt = max([len(x[key]) for x in batch_list])
                batch_size = len(batch_list)
                
                batch_gt_boxes = np.zeros((batch_size, max_gt, batch_list[0][key].shape[-1]), dtype=np.float32)
                for i in range(batch_size):
                    batch_gt_boxes[i, :batch_list[i][key].__len__(), :] = batch_list[i][key]
                    
                data_dict[key] = batch_gt_boxes
                
            elif key in ['gt_names']:
                # 处理GT名称
                batch_size = len(batch_list)
                max_gt = max([len(x[key]) for x in batch_list])
                
                batch_gt_names = []
                for i in range(batch_size):
                    batch_gt_names_i = [''] * max_gt
                    batch_gt_names_i[:len(batch_list[i][key])] = batch_list[i][key]
                    batch_gt_names.append(batch_gt_names_i)
                    
                data_dict[key] = batch_gt_names
                
            elif key in ['images']:
                # 处理图像
                data_dict[key] = np.stack([x[key] for x in batch_list], axis=0)
                
            elif key in ['calib']:
                # 保持标定列表
                data_dict[key] = [x[key] for x in batch_list]
                
            else:
                # 默认堆叠
                try:
                    data_dict[key] = np.stack([x[key] for x in batch_list], axis=0)
                except:
                    data_dict[key] = [x[key] for x in batch_list]
                    
        batch_size = len(batch_list)
        data_dict['batch_size'] = batch_size
        
        return data_dict
        
    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        """
        创建包含伪点云的GT数据库
        Args:
            info_path: 信息文件路径
            used_classes: 使用的类别
            split: 数据集划分
        """
        import torch
        
        database_save_path = Path(self.root_path) / (
            'gt_database_pseudo_seguv' if self.use_pseudo_label else 'gt_database'
        )
        db_info_save_path = Path(self.root_path) / (
            'kitti_dbinfos_train_custom_seguv.pkl' if self.use_pseudo_label else 'kitti_dbinfos_train.pkl'
        )
        
        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}
        
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
            
        for k in range(len(infos)):
            if k % 100 == 0:
                print('gt_database sample: %d/%d' % (k + 1, len(infos)))
                
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            
            # 获取点云（原始或伪点云）
            if self.use_pseudo_label:
                points = self.get_lidar_pseudo(sample_idx)
                if points is None:
                    points = self.get_lidar(sample_idx)
                    # 扩展到7维
                    points = np.hstack([points, np.ones((len(points), 3)) * 0.5])
            else:
                points = self.get_lidar(sample_idx)
                
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            
            # 获取GT框
            calib = self.get_calib(sample_idx)
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)
            
            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)
            
            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]
                
                # 标准化到物体坐标系
                gt_points[:, :3] -= gt_boxes[i, :3]
                
                # 保存点云
                with open(filepath, 'wb') as f:
                    gt_points.tofile(f)
                    
                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))
                    db_info = {
                        'name': names[i],
                        'path': db_path,
                        'image_idx': sample_idx,
                        'gt_idx': i,
                        'box3d_lidar': gt_boxes[i],
                        'num_points_in_gt': gt_points.shape[0],
                        'difficulty': difficulty[i],
                        'bbox': bbox[i],
                        'score': annos['score'][i] if 'score' in annos else -1
                    }
                    
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
                        
        # 保存数据库信息
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))
            
        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)
            
        print(f'GT database saved to {database_save_path}')
        print(f'GT database info saved to {db_info_save_path}')
        
    @staticmethod
    def create_kitti_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
        """创建KITTI数据集信息文件"""
        dataset = KittiDataset(
            dataset_cfg=dataset_cfg, class_names=class_names, 
            root_path=data_path, training=False
        )
        train_split, val_split = 'train', 'val'
        
        train_filename = save_path / ('kitti_infos_%s.pkl' % train_split)
        val_filename = save_path / ('kitti_infos_%s.pkl' % val_split)
        trainval_filename = save_path / 'kitti_infos_trainval.pkl'
        test_filename = save_path / 'kitti_infos_test.pkl'
        
        print('---------------Start to generate data infos---------------')
        
        dataset.set_split(train_split)
        kitti_infos_train = dataset.get_infos(
            num_workers=workers, has_label=True, 
            count_inside_pts=True
        )
        with open(train_filename, 'wb') as f:
            pickle.dump(kitti_infos_train, f)
        print('Kitti info train file is saved to %s' % train_filename)
        
        dataset.set_split(val_split)
        kitti_infos_val = dataset.get_infos(
            num_workers=workers, has_label=True, 
            count_inside_pts=True
        )
        with open(val_filename, 'wb') as f:
            pickle.dump(kitti_infos_val, f)
        print('Kitti info val file is saved to %s' % val_filename)
        
        with open(trainval_filename, 'wb') as f:
            pickle.dump(kitti_infos_train + kitti_infos_val, f)
        print('Kitti info trainval file is saved to %s' % trainval_filename)
        
        dataset.set_split('test')
        kitti_infos_test = dataset.get_infos(
            num_workers=workers, has_label=False, 
            count_inside_pts=False
        )
        with open(test_filename, 'wb') as f:
            pickle.dump(kitti_infos_test, f)
        print('Kitti info test file is saved to %s' % test_filename)
        
        if dataset_cfg.get('CREATE_GT_DATABASE', False):
            print('---------------Start create groundtruth database for data augmentation---------------')
            dataset.set_split(train_split)
            dataset.create_groundtruth_database(
                info_path=train_filename,
                used_classes=class_names,
                split=train_split
            )
            print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_kitti_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        
        create_kitti_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'kitti_pseudo',
            save_path=ROOT_DIR / 'data' / 'kitti_pseudo',
            workers=4
        )