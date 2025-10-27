# pcdet/datasets/processor/point_feature_encoder.py

import numpy as np

class PointFeatureEncoder(object):
    def __init__(self, config, point_cloud_range=None):
        super().__init__()
        self.point_encoding_config = config
        
        # 确保源特征列表的前三个是 x, y, z
        assert list(self.point_encoding_config.src_feature_list[0:3]) == ['x', 'y', 'z']
        
        # 从配置中读取原始点云的特征列表
        self.used_feature_list = self.point_encoding_config.used_feature_list
        self.src_feature_list = self.point_encoding_config.src_feature_list
        self.point_cloud_range = point_cloud_range

        # 从配置中读取伪点云的特征列表（如果存在）
        if self.point_encoding_config.get('used_feature_list_pseudo', False):
            self.used_feature_list_pseudo = self.point_encoding_config.used_feature_list_pseudo
            self.src_feature_list_pseudo = self.point_encoding_config.src_feature_list_pseudo
            print(f"[DEBUG] Pseudo features: {len(self.used_feature_list_pseudo)}D - {self.used_feature_list_pseudo}")
        else:
            # 如果配置中没有,使用默认的9维
            self.used_feature_list_pseudo = ['x', 'y', 'z', 'intensity', 'r', 'g', 'b', 'u', 'v']
            self.src_feature_list_pseudo = ['x', 'y', 'z', 'intensity', 'r', 'g', 'b', 'u', 'v']
            print(f"[DEBUG] Using default 9D pseudo features")

    @property
    def num_point_features(self):
        # 这个属性会根据配置文件中的 encoding_type 调用对应的方法
        # 例如，如果 type 是 'dual_absolute_coordinates_encoding', 它就会调用下面的同名方法
        return getattr(self, self.point_encoding_config.encoding_type)(points=None)

    @property
    def num_point_features_pseudo(self):
        # 同上，处理伪点云的特征维度
        return getattr(self, self.point_encoding_config.encoding_type_pseudo)(points=None)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                ...
        Returns:
            data_dict:
                points: (N, 3 + C_out),
                use_lead_xyz: whether to use xyz as point-wise features
                ...
        """
        # 根据配置调用正确的编码函数来处理原始点云
        data_dict['points'], use_lead_xyz = getattr(self, self.point_encoding_config.encoding_type)(
            data_dict['points']
        )
        data_dict['use_lead_xyz'] = use_lead_xyz

        # 根据配置调用正确的编码函数来处理伪点云
        # if self.point_encoding_config.get('used_feature_list_pseudo', False) and 'points_pseudo' in data_dict:
        if (self.point_encoding_config.get('used_feature_list_pseudo', False) and 
            'points_pseudo' in data_dict and 
            data_dict['points_pseudo'] is not None):  # ← 加这个检查！
            data_dict['points_pseudo'], use_lead_xyz_pseudo = getattr(self, self.point_encoding_config.encoding_type_pseudo)(
                data_dict['points_pseudo'], is_pseudo=True
            )
            data_dict['use_lead_xyz_pseudo'] = use_lead_xyz_pseudo

        return data_dict

    def absolute_coordinates_encoding(self, points=None, is_pseudo=False):
        # 这是一个框架自带的或你已有的方法，保持不变
        if points is None:
            num_output_features = len(self.used_feature_list)
            return num_output_features

        point_feature_list = [points[:, 0:3]]
        for x in self.used_feature_list:
            if x in ['x', 'y', 'z']:
                continue
            idx = self.src_feature_list.index(x)
            point_feature_list.append(points[:, idx:idx+1])
        point_features = np.concatenate(point_feature_list, axis=1)
        return point_features, True

    def absolute_coordinates_encoding_pseudo(self, points=None, is_pseudo=False):
        # 这是一个框架自带的或你已有的方法，保持不变
        if points is None:
            num_output_features = len(self.used_feature_list_pseudo)
            return num_output_features

        point_feature_list = [points[:, 0:3]]
        for x in self.used_feature_list_pseudo:
            if x in ['x', 'y', 'z']:
                continue
            idx = self.src_feature_list_pseudo.index(x)
            point_feature_list.append(points[:, idx:idx+1])
        point_features = np.concatenate(point_feature_list, axis=1)
        return point_features, True


    # ==============================================================================
    # =============== 在这里添加新的方法来解决 `AttributeError` =======================
    # ==============================================================================
    
    def dual_absolute_coordinates_encoding(self, points=None, is_pseudo=False):
        """
        为原始点云自定义的特征编码方法。
        逻辑从你之前创建的 DualAbsoluteCoordinatesEncoding 类中迁移而来。
        """
        # Part 1: 当初始化数据集时，points为None，此时必须返回特征维度
        if points is None:
            # self.used_feature_list 是在 __init__ 中从配置文件加载的
            return len(self.used_feature_list)

        # Part 2: 当处理真实点云数据时，提取所需的特征列
        point_feature_list = []
        for feature_name in self.used_feature_list:
            if feature_name in self.src_feature_list:
                idx = self.src_feature_list.index(feature_name)
                point_feature_list.append(points[:, idx:idx+1])
        
        point_features = np.concatenate(point_feature_list, axis=1)
        # return point_features, True
        return point_features, False


    def dual_absolute_coordinates_encoding_pseudo(self, points=None, is_pseudo=True):
        """
        为伪点云自定义的特征编码方法。
        逻辑从你之前创建的 DualAbsoluteCoordinatesEncoding 类中迁移而来。
        
        注意：假设你的配置文件中 `encoding_type_pseudo` 设置为 `dual_absolute_coordinates_encoding_pseudo`
        """
        # Part 1: 初始化时返回特征维度
        if points is None:
            # self.used_feature_list_pseudo 是在 __init__ 中从配置文件加载的
            return len(self.used_feature_list_pseudo)

        # Part 2: 处理真实伪点云数据，提取所需的特征列
        point_feature_list_pseudo = []
        for feature_name in self.used_feature_list_pseudo:
            if feature_name in self.src_feature_list_pseudo:
                idx = self.src_feature_list_pseudo.index(feature_name)
                point_feature_list_pseudo.append(points[:, idx:idx+1])
        
        point_features = np.concatenate(point_feature_list_pseudo, axis=1)
        return point_features, False
