"""
位置: pcdet/datasets/kitti/kitti_dataset_custom.py
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
# from .kitti_object_eval_python import kitti_utils
# 🔥 新增：导入去噪模块
from ..processor.pseudo_point_denoiser import PseudoPointDenoiser


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
        
        # 如果没有提供logger，创建一个基本的logger
        if self.logger is None:
            import logging
            self.logger = logging.getLogger(__name__)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        
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
        self.use_images = dataset_cfg.get('USE_IMAGES', False)
        
        # 新增：图像增强配置
        self.image_augmentor = None
        if dataset_cfg.get('IMAGE_AUGMENTOR', None) is not None and training:
            from ..augmentor.image_augmentor import ImageAugmentor
            self.image_augmentor = ImageAugmentor(dataset_cfg.IMAGE_AUGMENTOR)
        # 🔥 新增：初始化伪点云去噪器
        if self.use_pseudo_label:
            denoiser_config = dataset_cfg.get('PSEUDO_POINT_DENOISER', {
                'USE_CONFIDENCE_FILTER': True,
                'CONFIDENCE_THRESHOLD': 0.7,
                'USE_DEPTH_FILTER': True,
                'MIN_DEPTH': 2.0,
                'MAX_DEPTH': 50.0,
                'USE_SOR': False,
                'USE_DOWNSAMPLING': True,
                'MAX_PSEUDO_POINTS': 40000
            })
            self.pseudo_denoiser = PseudoPointDenoiser(denoiser_config)
            self.logger.info(f'✓ Initialized pseudo point denoiser with config: {denoiser_config}')
        else:
            self.pseudo_denoiser = None
            
        # 加载数据信息
        self.kitti_infos = []
        mode = 'train' if training else 'test' 
        self.include_kitti_data(self.mode)
        
        # 设置数据增强器
        # if self.training:
        #     self.set_augmentor()
            
        # 只在数据已加载时输出日志
        if len(self.kitti_infos) > 0:
            self.logger.info('Total samples for KITTI dataset: %d' % (len(self.kitti_infos)))
        # ============ 🔥 新增：初始化双路径特征编码器 ============
        from ..processor import point_feature_encoder
        
        # 检查是否使用伪点云，如果使用则启用双路径编码
        if self.use_pseudo_label:
            self.logger.info('Using DualAbsoluteCoordinatesEncoding for pseudo labels')
            self.point_feature_encoder = point_feature_encoder.PointFeatureEncoder(
                config=self.dataset_cfg.POINT_FEATURE_ENCODING,
                point_cloud_range=self.dataset_cfg.POINT_CLOUD_RANGE
            )
        else:
            # 使用标准编码器
            encoding_type = self.dataset_cfg.POINT_FEATURE_ENCODING.get('encoding_type', 'absolute_coordinates_encoding')
            self.point_feature_encoder = getattr(point_feature_encoder, encoding_type)(
                config=self.dataset_cfg.POINT_FEATURE_ENCODING,
                point_cloud_range=self.dataset_cfg.POINT_CLOUD_RANGE
            )
    def include_kitti_data(self, mode):
        """加载KITTI数据信息"""
        if self.logger is not None:
            self.logger.info('Loading KITTI dataset')
        
        kitti_infos = []
        
        # 根据模式选择info文件
        if mode == 'train':
            info_paths = self.dataset_cfg.INFO_PATH.get('train', [])
        elif mode == 'test':
            info_paths = self.dataset_cfg.INFO_PATH.get('test', [])
        else:
            raise ValueError(f'Unknown mode: {mode}')
        
        # 加载所有info文件
        for info_path in info_paths:
            info_path = self.root_path / info_path
            if not info_path.exists():
                if self.logger is not None:
                    self.logger.warning(f'Info file not found: {info_path}')
                continue
                
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                kitti_infos.extend(infos)
        
        self.kitti_infos.extend(kitti_infos)
        
        if self.logger is not None:
            self.logger.info(f'Total samples for KITTI dataset: {len(self.kitti_infos)}')
        
    def set_augmentor(self):
        """设置数据增强器"""
        if self.dataset_cfg.get('DATA_AUGMENTOR', None) is not None:
            from ..augmentor.data_augmentor import DataAugmentor
            self.data_augmentor = DataAugmentor(
                self.root_path, 
                self.dataset_cfg.DATA_AUGMENTOR, 
                self.class_names, 
                logger=self.logger
            )
        else:
            self.data_augmentor = None
            
    def set_split(self, split):
        """设置数据集划分并加载数据"""
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')
        
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None
        
        # 加载对应split的数据信息
        self.kitti_infos = []
        mode = 'train' if split in ['train', 'val', 'trainval'] else 'test'
        
        if mode == 'train':
            info_paths = self.dataset_cfg.INFO_PATH.get('train', [])
        else:
            info_paths = self.dataset_cfg.INFO_PATH.get('test', [])
        
        for info_path in info_paths:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                self.kitti_infos.extend(infos)
        
        if self.logger is not None:
            self.logger.info(f'Loaded {len(self.kitti_infos)} samples for {split} split')
        
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
        获取伪点云数据 + 自适应降采样
        确保每个样本都有足够的点
        """
        pseudo_file = self.root_split_path / self.pseudo_label_dir / ('%s.bin' % idx)
        
        if not pseudo_file.exists():
            if self.logger is not None:
                self.logger.warning(f'Pseudo point cloud not found: {pseudo_file}')
            return None
        
        try:
            # 加载原始伪点云（9维）
            point_pseudo = np.fromfile(str(pseudo_file), dtype=np.float32).reshape(-1, 9)
            
            # ========== 自适应处理 ==========
            min_points = 5000      # 最少保留点数
            target_points = 30000  # 目标点数
            max_points = 80000     # 最多点数
            
            # 基础范围过滤（高度过滤）
            range_mask = (point_pseudo[:, 2] < 1.7) & (point_pseudo[:, 2] > -1.7)
            point_pseudo = point_pseudo[range_mask]
            
            num_points = len(point_pseudo)
            
            # 检查是否为空
            if num_points == 0:
                if self.logger is not None:
                    self.logger.warning(f'All points filtered by range for {idx}')
                return None
            
            # 🔥 自适应降采样
            if num_points <= target_points:
                # 点数合适，直接返回
                pass
                
            elif num_points <= max_points:
                # 点数略多，随机降采样
                indices = np.random.choice(num_points, target_points, replace=False)
                point_pseudo = point_pseudo[indices]
                
            else:
                # 点数太多，体素降采样 + 随机采样
                voxel_size = 0.1  # 10cm
                points_int = (point_pseudo[:, :3] / voxel_size).astype(np.int32)
                _, unique_indices = np.unique(points_int, axis=0, return_index=True)
                point_pseudo = point_pseudo[unique_indices]
                
                # 如果还是太多，随机采样
                if len(point_pseudo) > max_points:
                    indices = np.random.choice(len(point_pseudo), max_points, replace=False)
                    point_pseudo = point_pseudo[indices]
            
            # 检查是否太少
            if len(point_pseudo) < min_points:
                if self.logger is not None and np.random.rand() < 0.01:  # 1% 概率打印
                    self.logger.warning(
                        f'Sample {idx}: only {len(point_pseudo)} points (< {min_points})'
                    )
            
            return point_pseudo
            
        except Exception as e:
            if self.logger is not None:
                self.logger.error(f'Error loading pseudo points for {idx}: {e}')
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
            # image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()
            # ⭐ 修复：兼容 tensor 和 numpy 两种类型
            image_shape = batch_dict['image_shape'][batch_index]
            if hasattr(image_shape, 'cpu'):
                # 如果是 tensor，转换为 numpy
                image_shape = image_shape.cpu().numpy()
            elif isinstance(image_shape, np.ndarray):
                # 如果已经是 numpy，直接使用
                pass
            else:
                # 其他类型，尝试转换
                image_shape = np.array(image_shape, dtype=np.int32)
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

        # 加载点云
        points = self.get_lidar(sample_idx)  # (N, 4) - 原始LiDAR点云
        points_pseudo = self.get_lidar_pseudo(sample_idx)  # (M, 9) - 伪点云
        
        calib = self.get_calib(sample_idx)

        # FOV过滤（如果需要）
        img_shape = info['image']['image_shape']
        if self.dataset_cfg.FOV_POINTS_ONLY:
            pts_rect = calib.lidar_to_rect(points[:, 0:3])
            fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
            points = points[fov_flag]

        # 构建输入字典
        input_dict = {
            'points': points,              # (N, 4) - 保持原样
            'points_pseudo': points_pseudo, # (M, 9) - 保持原样
            'frame_id': sample_idx,
            'calib': calib,
        }

        # 获取标注信息
        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            
            # 处理Van类别（如果配置了）
            if (self.dataset_cfg.get('USE_VAN', None) is True) and (self.training is True):
                gt_names = np.array(['Car' if gt_names[i]=='Van' else gt_names[i] for i in range(len(gt_names))])

            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            
            # 添加地面平面信息
            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        # 数据准备（包含数据增强和体素化）
        data_dict = self.prepare_data(data_dict=input_dict)
        # 🔥 prepare_data 会丢弃某些字段（如calib），需要重新加回来
        data_dict['calib'] = calib
        
        # ========== ⭐ 新增: 加载图像数据 ==========
        if self.use_images:
            try:
                # 加载RGB图像
                image = self.get_image(sample_idx)  # (H, W, 3) numpy array
                
                # 转换为tensor并归一化
                import torch
                import torchvision.transforms as transforms
                
                # 定义图像预处理
                image_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),  # 转换为 (3, H, W) 并归一化到 [0, 1]
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],  # ImageNet标准化
                        std=[0.229, 0.224, 0.225]
                    )
                ])
                
                # 应用变换
                image_tensor = image_transform(image)  # (3, H, W)
                data_dict['images'] = image_tensor
                
                if self.logger is not None and index % 1000 == 0:  # 每1000个样本打印一次
                    self.logger.info(f'[Image Loading] Sample {sample_idx}: image shape {image_tensor.shape}')
                    
            except Exception as e:
                if self.logger is not None:
                    self.logger.warning(f'Failed to load image for {sample_idx}: {e}')
                data_dict['images'] = None
        else:
            data_dict['images'] = None
        # ========================================
        
        # 添加图像形状信息
        data_dict['image_shape'] = img_shape
        
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
        # 🔥 修复：首先收集所有样本的数据
        from collections import defaultdict
        data_dict = defaultdict(list)
        
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                if key == 'valid_noise':
                    continue
                data_dict[key].append(val)
        
        batch_size = len(batch_list)
        ret = {}

        # 现在处理收集到的数据
        for key, val in data_dict.items():
            try:
                # 🔑 支持双体素：voxels/voxels_pseudo
                if key in ['voxels', 'voxel_num_points', 'voxels_pseudo', 'voxel_num_points_pseudo']:
                    ret[key] = np.concatenate(val, axis=0)
                    
                # 🔑 支持双点云：points/points_pseudo
                elif key in ['points', 'voxel_coords', 'points_pseudo', 'voxel_coords_pseudo']:
                    coors = []
                    # for i, coor in enumerate(val):
                    #     coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                    #     coors.append(coor_pad)
                    for i, coor in enumerate(data_dict[key]):
                        # 🔥 修复：处理 None 和空数组
                        if coor is None:
                            continue  # 跳过 None
                        if coor.shape[0] == 0:
                            continue  # 跳过空数组
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)

                    # 如果所有样本都被过滤了，创建一个空数组
                    if len(coors) == 0:
                        # 需要知道维度，使用默认值（通常是 4: batch_id, z, y, x）
                        ret[key] = np.zeros((0, 4), dtype=np.int32)
                    else:
                        ret[key] = np.concatenate(coors, axis=0)
                    # ret[key] = np.concatenate(coors, axis=0)
                    
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                    
                elif key in ['gt_names']:
                    # GT names 需要 pad
                    max_gt = max([len(x) for x in val])
                    batch_gt_names = []
                    for i in range(batch_size):
                        batch_gt_names_i = [''] * max_gt
                        batch_gt_names_i[:len(val[i])] = val[i]
                        batch_gt_names.append(batch_gt_names_i)
                    ret[key] = batch_gt_names
                    
                elif key in ['calib']:
                    # 标定信息保持列表
                    ret[key] = val
                    
                else:
                    ret[key] = np.stack(val, axis=0)
                    
            except Exception as e:
                print(f'Error in collate_batch: key={key}, error={e}')
                raise TypeError

        ret['batch_size'] = batch_size
        return ret
    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        """
        获取数据集信息
        Args:
            num_workers: 工作进程数
            has_label: 是否有标注
            count_inside_pts: 是否统计GT框内的点数
            sample_id_list: 样本ID列表
        Returns:
            kitti_infos: 数据集信息列表
        """
        import concurrent.futures as futures
        
        def process_single_scene(sample_idx):
            """处理单个场景"""
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info
            
            # 图像信息
            image_info = {
                'image_idx': sample_idx,
                'image_shape': self.get_image(sample_idx).shape[:2]
            }
            info['image'] = image_info
            
            # 标定信息
            calib = self.get_calib(sample_idx)
            P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = calib.R0
            V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}
            
            info['calib'] = calib_info
            
            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)
                
                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)
                
                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                loc_lidar = calib.rect_to_lidar(loc)
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                loc_lidar[:, 2] += h[:, 0] / 2
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar
                
                info['annos'] = annotations
                
                if count_inside_pts:
                    # 获取点云
                    if self.use_pseudo_label:
                        points = self.get_lidar_pseudo(sample_idx)
                        if points is None:
                            points = self.get_lidar(sample_idx)
                    else:
                        points = self.get_lidar(sample_idx)
                        
                    calib = self.get_calib(sample_idx)
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])
                    
                    fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                    pts_fov = points[fov_flag]
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)
                    
                    for k in range(num_objects):
                        flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt
            
            return info
        
        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        
        # 使用多进程处理
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
            
        return list(infos)
    
    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        获取在视野内的点的标记
        Args:
            pts_rect: 矩形坐标系中的点
            img_shape: 图像形状
            calib: 标定信息
        Returns:
            fov_flag: 视野标记
        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
        
        return pts_valid_flag

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        """
        创建包含真实点云和伪点云的GT数据库（MPCF格式）
        """
        import torch
        
        # 【关键】创建两个数据库目录
        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else f'gt_database_{split}')
        database_save_path_pseudo = Path(self.root_path) / ('gt_database_pseudo_seguv' if split == 'train' else f'gt_database_{split}_pseudo_seguv')
        
        db_info_save_path = Path(self.root_path) / f'kitti_dbinfos_{split}_custom_seguv.pkl'
        
        database_save_path.mkdir(parents=True, exist_ok=True)
        database_save_path_pseudo.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}
        
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
            
        for k in range(len(infos)):
            if k % 100 == 0:
                print('gt_database sample: %d/%d' % (k + 1, len(infos)))
                
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            
            # 【关键】同时获取真实点云和伪点云
            points = self.get_lidar(sample_idx)
            points_pseudo = self.get_lidar_pseudo(sample_idx)
            
            if points_pseudo is None:
                if self.logger is not None:
                    self.logger.warning(f'Skipping {sample_idx}: no pseudo points')
                continue
            
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
            
            # 【关键】分别计算真实点云和伪点云在GT框内的点
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()
            
            point_indices_pseudo = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points_pseudo[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()
            
            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                
                filepath = database_save_path / filename
                filepath_pseudo = database_save_path_pseudo / filename
                
                # 提取GT框内的点并归一化
                gt_points = points[point_indices[i] > 0]
                gt_points[:, :3] -= gt_boxes[i, :3]
                
                gt_points_pseudo = points_pseudo[point_indices_pseudo[i] > 0]
                gt_points_pseudo[:, :3] -= gt_boxes[i, :3]
                
                # 【关键】分别保存两个数据库
                with open(filepath, 'wb') as f:
                    gt_points.tofile(f)
                    
                with open(filepath_pseudo, 'wb') as f:
                    gt_points_pseudo.tofile(f)
                
                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))
                    db_path_pseudo = str(filepath_pseudo.relative_to(self.root_path))
                    
                    # 【关键】保存两个路径
                    db_info = {
                        'name': names[i],
                        'path': db_path,
                        'path_pseudo': db_path_pseudo,
                        'image_idx': sample_idx,
                        'gt_idx': i,
                        'box3d_lidar': gt_boxes[i],
                        'num_points_in_gt': gt_points.shape[0],
                        'num_points_in_gt_pseudo': gt_points_pseudo.shape[0],
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
        print(f'GT database (pseudo) saved to {database_save_path_pseudo}')
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
        ROOT_DIR = Path('data/kitti_pseudo')  # 简化路径
        
        KittiDataset.create_kitti_infos(  # ← 改为调用类的静态方法
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR,
            save_path=ROOT_DIR,
            workers=4
        )
