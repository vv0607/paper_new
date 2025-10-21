import pickle

import numpy as np

from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils import box_utils


class DataBaseSampler(object):
    def __init__(self, root_path, sampler_cfg, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.sampler_cfg = sampler_cfg
        self.logger = logger
        self.db_infos = {}
        for class_name in class_names:
            self.db_infos[class_name] = []

        for db_info_path in sampler_cfg.DB_INFO_PATH:
            db_info_path = self.root_path.resolve() / db_info_path
            with open(str(db_info_path), 'rb') as f:
                infos = pickle.load(f)
                [self.db_infos[cur_class].extend(infos[cur_class]) for cur_class in class_names]

        for func_name, val in sampler_cfg.PREPARE.items():
            self.db_infos = getattr(self, func_name)(self.db_infos, val)

        self.sample_groups = {}
        self.sample_class_num = {}
        self.limit_whole_scene = sampler_cfg.get('LIMIT_WHOLE_SCENE', False)
        # for x in sampler_cfg.SAMPLE_GROUPS:
        #     class_name, sample_num = x.split(':')
        raw_sample_groups = sampler_cfg.SAMPLE_GROUPS
        if isinstance(raw_sample_groups, dict):
            sample_groups_map = {
                str(name).strip(): int(num)
                for name, num in raw_sample_groups.items()
            }
        else:
            sample_groups_map = {}
            for item in raw_sample_groups:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    class_name, sample_num = item
                elif isinstance(item, str) and ':' in item:
                    class_name, sample_num = item.split(':', 1)
                else:
                    if self.logger is not None:
                        self.logger.warning(f'Invalid sample group entry: {item}')
                    continue

                sample_groups_map[str(class_name).strip()] = int(sample_num)

        for class_name, sample_num in sample_groups_map.items():
            if class_name not in class_names:
                continue
            self.sample_class_num[class_name] = sample_num
            self.sample_groups[class_name] = {
                'sample_num': sample_num,
                'pointer': len(self.db_infos[class_name]),
                'indices': np.arange(len(self.db_infos[class_name]))
            }

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def filter_by_difficulty(self, db_infos, removed_difficulty):
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            pre_len = len(dinfos)
            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty
            ]
            if self.logger is not None:
                self.logger.info('Database filter by difficulty %s: %d => %d' % (key, pre_len, len(new_db_infos[key])))
        return new_db_infos

    def filter_by_min_points(self, db_infos, min_gt_points_list):
        # for name_num in min_gt_points_list:
        #     name, min_num = name_num.split(':')
        #     min_num = int(min_num)
        if isinstance(min_gt_points_list, dict):
            min_points_map = {
                str(class_name).strip(): int(min_points_list)
                for class_name, min_points_list in min_gt_points_list.items()
            }
        else:
            min_points_map = {}
            for item in min_gt_points_list:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    name, min_num = item
                elif isinstance(item, str) and ':' in item:
                    name, min_num = item.split(':', 1)
                else:
                    if self.logger is not None:
                        self.logger.warning(f'Invalid min_gt_points entry: {item}')
                    continue
                min_points_map[str(name).strip()] = int(min_num)

        for name, min_num in min_points_map.items():
            if min_num > 0 and name in db_infos.keys():
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)

                if self.logger is not None:
                    self.logger.info('Database filter by min points %s: %d => %d' %
                                     (name, len(db_infos[name]), len(filtered_infos)))
                db_infos[name] = filtered_infos

        return db_infos

    def sample_with_fixed_number(self, class_name, sample_group):
        """
        Args:
            class_name:
            sample_group:
        Returns:

        """
        sample_num, pointer, indices = int(sample_group['sample_num']), sample_group['pointer'], sample_group['indices']
        if pointer >= len(self.db_infos[class_name]):
            indices = np.random.permutation(len(self.db_infos[class_name]))
            pointer = 0

        sampled_dict = [self.db_infos[class_name][idx] for idx in indices[pointer: pointer + sample_num]]
        pointer += sample_num
        sample_group['pointer'] = pointer
        sample_group['indices'] = indices
        return sampled_dict

    @staticmethod
    def put_boxes_on_road_planes(gt_boxes, road_planes, calib):
        """
        Only validate in KITTIDataset
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            road_planes: [a, b, c, d]
            calib:

        Returns:
        """
        a, b, c, d = road_planes
        center_cam = calib.lidar_to_rect(gt_boxes[:, 0:3])
        cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b
        center_cam[:, 1] = cur_height_cam
        cur_lidar_height = calib.rect_to_lidar(center_cam)[:, 2]
        mv_height = gt_boxes[:, 2] - gt_boxes[:, 5] / 2 - cur_lidar_height
        gt_boxes[:, 2] -= mv_height  # lidar view
        return gt_boxes, mv_height

    def add_sampled_boxes_to_scene(self, data_dict, sampled_gt_boxes, total_valid_sampled_dict):
        gt_boxes_mask = data_dict['gt_boxes_mask']
        gt_boxes = data_dict['gt_boxes'][gt_boxes_mask]
        gt_names = data_dict['gt_names'][gt_boxes_mask]
        points = data_dict['points']
        mv_height = None
        
        if self.sampler_cfg.get('USE_ROAD_PLANE', False):
            sampled_gt_boxes, mv_height = self.put_boxes_on_road_planes(
                sampled_gt_boxes, data_dict['road_plane'], data_dict['calib']
            )
            data_dict.pop('calib')
            data_dict.pop('road_plane')

        obj_points_list = []
        valid_sampled_indices = []
        num_point_features = int(self.sampler_cfg.NUM_POINT_FEATURES)
        
        # 🔥 新增：获取场景点云的实际维度
        scene_point_dim = points.shape[1]
        
        for idx, info in enumerate(total_valid_sampled_dict):
            file_path = self.root_path / info['path']
            obj_points_raw = np.fromfile(str(file_path), dtype=np.float32)
            
            # 🔥 修改：先尝试用配置的维度，如果不行就尝试常见维度
            possible_dims = [num_point_features, 4, 7, 9]  # 尝试多种维度
            obj_points = None
            
            for dim in possible_dims:
                if obj_points_raw.size % dim == 0:
                    obj_points = obj_points_raw.reshape([-1, dim])
                    break
            
            if obj_points is None:
                if self.logger is not None:
                    self.logger.warning(
                        'Skip database sample %s with %d values (cannot reshape)',
                        info.get('path', '<unknown>'),
                        obj_points_raw.size
                    )
                continue
            
            # 🔥 关键：维度对齐到场景点云
            if obj_points.shape[1] != scene_point_dim:
                if obj_points.shape[1] < scene_point_dim:
                    # GT样本维度小（比如4维），需要补充到场景维度（比如7维）
                    padding = np.ones((obj_points.shape[0], scene_point_dim - obj_points.shape[1]), dtype=np.float32) * 0.5
                    obj_points = np.concatenate([obj_points, padding], axis=1)
                    if self.logger is not None and idx == 0:  # 只打印一次
                        self.logger.info(f'Auto-padding GT samples from {obj_points.shape[1]} to {scene_point_dim} dims')
                else:
                    # GT样本维度大，截取前面的维度
                    obj_points = obj_points[:, :scene_point_dim]
                    if self.logger is not None and idx == 0:
                        self.logger.info(f'Auto-truncating GT samples from {obj_points.shape[1]} to {scene_point_dim} dims')
            
            obj_points[:, :3] += info['box3d_lidar'][:3]

            if self.sampler_cfg.get('USE_ROAD_PLANE', False):
                obj_points[:, 2] -= mv_height[idx]

            obj_points_list.append(obj_points)
            valid_sampled_indices.append(idx)

        if len(obj_points_list) == 0:
            if self.logger is not None:
                self.logger.warning('No valid sampled objects to add to scene')
            return data_dict

        valid_sampled_gt_boxes = sampled_gt_boxes[valid_sampled_indices]
        valid_total_sampled_dict = [total_valid_sampled_dict[i] for i in valid_sampled_indices]

        obj_points = np.concatenate(obj_points_list, axis=0)
        sampled_gt_names = np.array([x['name'] for x in valid_total_sampled_dict])

        large_sampled_gt_boxes = box_utils.enlarge_box3d(
            valid_sampled_gt_boxes[:, 0:7], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH
        )
        points = box_utils.remove_points_in_boxes3d(points, large_sampled_gt_boxes)
        
        # 🔥 最终拼接 - 现在维度一定匹配了
        points = np.concatenate([obj_points, points], axis=0)
        gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
        gt_boxes = np.concatenate([gt_boxes, valid_sampled_gt_boxes], axis=0)
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_names'] = gt_names
        data_dict['points'] = points
        return data_dict

    def __call__(self, data_dict):
        """
        Args:
            data_dict:
                gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        gt_boxes_mask = data_dict.get('gt_boxes_mask')
        gt_boxes = data_dict['gt_boxes']
        
        if gt_boxes_mask is None:
            gt_boxes_mask = np.ones(gt_boxes.shape[0], dtype=bool)
        else:
            gt_boxes_mask = np.asarray(gt_boxes_mask).astype(bool)
            if gt_boxes_mask.shape[0] != gt_boxes.shape[0]:
                if self.logger is not None:
                    self.logger.warning(
                        'Mismatched gt_boxes_mask length (%s) for %s boxes; '
                        'falling back to all-True mask.',
                        gt_boxes_mask.shape[0],
                        gt_boxes.shape[0]
                    )
                gt_boxes_mask = np.ones(gt_boxes.shape[0], dtype=bool)

        data_dict['gt_boxes_mask'] = gt_boxes_mask

        gt_names = data_dict['gt_names'].astype(str)
        # existed_boxes = gt_boxes
        existed_boxes = gt_boxes[gt_boxes_mask]
        total_valid_sampled_dict = []
        for class_name, sample_group in self.sample_groups.items():
            if self.limit_whole_scene:
                # num_gt = np.sum(class_name == gt_names)
                num_gt = np.sum(class_name == gt_names[gt_boxes_mask])
                sample_group['sample_num'] = str(int(self.sample_class_num[class_name]) - num_gt)
            if int(sample_group['sample_num']) > 0:
                sampled_dict = self.sample_with_fixed_number(class_name, sample_group)

                sampled_boxes = np.stack([x['box3d_lidar'] for x in sampled_dict], axis=0).astype(np.float32)

                if self.sampler_cfg.get('DATABASE_WITH_FAKELIDAR', False):
                    sampled_boxes = box_utils.boxes3d_kitti_fakelidar_to_lidar(sampled_boxes)

                iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], existed_boxes[:, 0:7])
                iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], sampled_boxes[:, 0:7])
                iou2[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0
                iou1 = iou1 if iou1.shape[1] > 0 else iou2
                valid_mask = ((iou1.max(axis=1) + iou2.max(axis=1)) == 0).nonzero()[0]
                valid_sampled_dict = [sampled_dict[x] for x in valid_mask]
                valid_sampled_boxes = sampled_boxes[valid_mask]

                existed_boxes = np.concatenate((existed_boxes, valid_sampled_boxes), axis=0)
                total_valid_sampled_dict.extend(valid_sampled_dict)

        # sampled_gt_boxes = existed_boxes[gt_boxes.shape[0]:, :]
        sampled_gt_boxes = existed_boxes[gt_boxes[gt_boxes_mask].shape[0]:, :]
        if total_valid_sampled_dict.__len__() > 0:
            data_dict = self.add_sampled_boxes_to_scene(data_dict, sampled_gt_boxes, total_valid_sampled_dict)

        # data_dict.pop('gt_boxes_mask')
        data_dict.pop('gt_boxes_mask', None)
        return data_dict
