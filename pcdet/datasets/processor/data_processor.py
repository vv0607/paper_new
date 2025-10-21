from functools import partial

import numpy as np
from skimage import transform

from ...utils import box_utils, common_utils

tv = None
try:
    import cumm.tensorview as tv
except:
    pass


class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training, num_point_features):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.num_point_features = num_point_features
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []

        self.voxel_generator = None

        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)

        if data_dict.get('points', None) is not None:
            mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
            data_dict['points'] = data_dict['points'][mask]

        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points

        return data_dict

    def transform_points_to_voxels(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

            points = data_dict['points']
            voxel_output = self.voxel_generator.generate(points)
            voxels, coordinates, num_points = voxel_output

            if not data_dict['use_lead_xyz']:
                voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

            data_dict['voxels'] = voxels
            data_dict['voxel_coords'] = coordinates
            data_dict['voxel_num_points'] = num_points
        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else: 
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def mask_points_and_boxes_outside_range_custom(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range_custom, config=config)

            
        mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
        data_dict['points'] = data_dict['points'][mask]

        mask = common_utils.mask_points_by_range(data_dict['points_pseudo'], self.point_cloud_range)
        data_dict['points_pseudo'] = data_dict['points_pseudo'][mask]


        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def shuffle_points_custom(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points_custom, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points

            points_pseudo = data_dict['points_pseudo']
            shuffle_idx = np.random.permutation(points_pseudo.shape[0])
            points_pseudo = points_pseudo[shuffle_idx]
            data_dict['points_pseudo'] = points_pseudo
            
        if config.get('USE_RAW_FEATURES', False):
            data_dict['points_valid'] = data_dict['points']
            data_dict['points'] = data_dict['points'][:,:4]

        return data_dict

    def transform_points_to_voxels_valid(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            self.max_voxels = config.MAX_NUMBER_OF_VOXELS[self.mode]
            self.skip_voxel_generator = config.get('SKIP_VOXEL_GENERATOR', False)
            return partial(self.transform_points_to_voxels_valid, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,  # 4维
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        if self.skip_voxel_generator:
            points = data_dict['points']
            pc_range_min = np.array(self.point_cloud_range[:3]).reshape(-1, 3)
            voxel_size_array = np.array(self.voxel_size).reshape(-1, 3)
            keep = common_utils.mask_points_by_range_hard(points, self.point_cloud_range)
            chosen_points = points[keep]
            chosen_points = chosen_points[:self.max_voxels, :]
            coords = (chosen_points[:, :3] - pc_range_min) // voxel_size_array
            coords = coords.astype(int)
            num_points = np.ones(chosen_points.shape[0])
            data_dict['voxels'] = chosen_points
            data_dict['voxel_coords'] = coords[:, [2, 1, 0]]
            data_dict['voxel_num_points'] = num_points
        else:
            points = data_dict['points']
            voxel_output = self.voxel_generator.generate(points)
            voxels, coordinates, num_points = voxel_output

            # 🔥 关键修复：强制移除xyz，只保留intensity
            # 输入: (N, max_points, 4) [x,y,z,intensity]
            # 输出: (N, max_points, 1) [intensity]
            voxels = voxels[..., 3:]  # 强制移除前3维
            
            print(f"[DEBUG] Original voxels shape after removing xyz: {voxels.shape}")  # 调试信息

            data_dict['voxels'] = voxels
            data_dict['voxel_coords'] = coordinates
            data_dict['voxel_num_points'] = num_points
        return data_dict
    # ============ 🔥 新增：伪点云体素化方法 ============
    def transform_points_to_voxels_pseudo(self, data_dict=None, config=None):
        """伪点云体素化"""
        if data_dict is None:
            # 初始化时创建伪点云体素生成器
            return partial(self.transform_points_to_voxels_pseudo, config=config)
        
        # 检查是否存在伪点云
        if 'points_pseudo' not in data_dict:
            return data_dict
        
        # 🔥 关键：为伪点云创建单独的体素生成器
        if not hasattr(self, 'voxel_generator_pseudo') or self.voxel_generator_pseudo is None:
            # 获取伪点云的特征维度
            num_features_pseudo = config.get('NUM_POINT_FEATURES_PSEUDO', 9)
            
            self.voxel_generator_pseudo = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=num_features_pseudo,  # 9维
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )
        
        # 生成伪点云体素
        points_pseudo = data_dict['points_pseudo']
        voxel_output_pseudo = self.voxel_generator_pseudo.generate(points_pseudo)
        voxels_pseudo, coordinates_pseudo, num_points_pseudo = voxel_output_pseudo
        
        # 🔥 关键修复：强制移除xyz，保留i+rgb+uv (6维)
        # 输入: (M, max_points, 9) [x,y,z,i,r,g,b,u,v]
        # 输出: (M, max_points, 6) [i,r,g,b,u,v]
        voxels_pseudo = voxels_pseudo[..., 3:]  # 强制移除前3维
        
        print(f"[DEBUG] Pseudo voxels shape after removing xyz: {voxels_pseudo.shape}")  # 调试信息
        
        # 保存到data_dict
        data_dict['voxels_pseudo'] = voxels_pseudo
        data_dict['voxel_coords_pseudo'] = coordinates_pseudo
        data_dict['voxel_num_points_pseudo'] = num_points_pseudo
        
        return data_dict
    def grid_sample_points_pseudo(self, data_dict=None, config=None):
        """伪点云网格采样(降采样)"""
        if data_dict is None:
            return partial(self.grid_sample_points_pseudo, config=config)

        if 'points_pseudo' not in data_dict:
            return data_dict
        
        max_distance = config.MAX_DISTANCE
        points = data_dict['points_pseudo']
        
        # 🔥 关键修复:检查实际维度
        num_features = points.shape[1]
        
        # 距离mask
        dist_mask = points[:, 0] < max_distance
        
        # 🔥 根据实际维度决定使用哪些列
        if num_features >= 9:
            # 原始9维: [x, y, z, i, r, g, b, u, v]
            col_mask = (points[:, 7] % 2 == 0) & dist_mask  # 使用u通道(索引7)
            row_mask = (points[:, 8] % 2 == 0) & dist_mask  # 使用v通道(索引8)
        elif num_features >= 7:
            # 编码后7维: [x, y, z, i, r, g, b]
            col_mask = (points[:, 5] % 2 == 0) & dist_mask  # 使用g通道
            row_mask = (points[:, 6] % 2 == 0) & dist_mask  # 使用b通道
        else:
            # 维度不足,只用距离mask
            print(f"WARNING: points_pseudo has only {num_features} features, using distance mask only")
            sample_mask = dist_mask
            data_dict['points_pseudo'] = points[sample_mask]
            return data_dict
        
        ignore_mask = col_mask | row_mask
        sample_mask = ~ignore_mask
        data_dict['points_pseudo'] = points[sample_mask]
        
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
