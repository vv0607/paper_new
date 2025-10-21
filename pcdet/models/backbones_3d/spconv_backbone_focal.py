from functools import partial # 从functools模块导入partial，用于创建偏函数

import torch # 导入PyTorch库
import spconv.pytorch as spconv # 导入spconv库，用于稀疏卷积
import torch.nn as nn # 导入PyTorch的神经网络模块

from .focal_sparse_conv.focal_sparse_conv import FocalSparseConv # 从本地模块导入FocalSparseConv类
from .SemanticSeg.pyramid_ffn import PyramidFeat2D # 从本地模块导入PyramidFeat2D类


class objDict: # 定义一个名为objDict的类
    @staticmethod # 定义一个静态方法
    def to_object(obj: object, **data): # 定义to_object方法，将字典数据更新到对象的__dict__中
        obj.__dict__.update(data) # 更新对象的属性

class ConfigDict: # 定义一个名为ConfigDict的类
    def __init__(self, name): # 定义构造函数
        self.name = name # 初始化name属性
    def __getitem__(self, item): # 定义__getitem__方法，使得可以像字典一样通过键获取属性
        return getattr(self, item) # 返回对象的属性值


class SparseSequentialBatchdict(spconv.SparseSequential): # 定义一个继承自spconv.SparseSequential的类
    def __init__(self, *args, **kwargs): # 定义构造函数
        super(SparseSequentialBatchdict, self).__init__(*args, **kwargs) # 调用父类的构造函数

    def forward(self, input, batch_dict=None): # 定义前向传播方法
        for k, module in self._modules.items(): # 遍历所有子模块
            if module is None: # 如果模块为空
                continue # 则跳过
            if isinstance(module, (FocalSparseConv,)): # 如果模块是FocalSparseConv的实例
                input, batch_dict = module(input, batch_dict) # 模块同时返回输出和更新后的batch_dict
            else: # 否则
                input = module(input) # 模块只返回输出
        return input, batch_dict # 返回输出和batch_dict


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0, # 定义一个函数，用于创建后激活块
                   conv_type='subm', norm_fn=None): # 函数参数包括输入/输出通道、卷积核大小、indice_key、步长、填充、卷积类型和归一化函数

    if conv_type == 'subm': # 如果卷积类型是'subm'（子流形稀疏卷积）
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key) # 创建SubMConv3d实例
    elif conv_type == 'spconv': # 如果卷积类型是'spconv'（稀疏卷积）
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, # 创建SparseConv3d实例
                                   bias=False, indice_key=indice_key) # 参数包括步长和填充
    elif conv_type == 'inverseconv': # 如果卷积类型是'inverseconv'（稀疏逆卷积）
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False) # 创建SparseInverseConv3d实例
    else: # 否则
        raise NotImplementedError # 抛出未实现错误

    m = spconv.SparseSequential( # 创建一个稀疏序列模块
        conv, # 包含卷积层
        norm_fn(out_channels), # 归一化层
        nn.ReLU(True), # ReLU激活函数
    )

    return m # 返回创建的模块


class SparseBasicBlock(spconv.SparseModule): # 定义一个继承自spconv.SparseModule的类
    expansion = 1 # 定义扩展因子为1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None): # 定义构造函数
        super(SparseBasicBlock, self).__init__() # 调用父类的构造函数

        assert norm_fn is not None # 断言归一化函数不为空
        bias = norm_fn is not None # 根据归一化函数是否存在来决定是否使用偏置
        self.conv1 = spconv.SubMConv3d( # 定义第一个子流形卷积层
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes) # 定义第一个批归一化层
        self.relu = nn.ReLU(True) # 定义ReLU激活函数
        self.conv2 = spconv.SubMConv3d( # 定义第二个子流形卷积层
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes) # 定义第二个批归一化层
        self.downsample = downsample # 定义下采样层
        self.stride = stride # 保存步长

    def forward(self, x): # 定义前向传播方法
        identity = x # 保存输入作为残差连接的identity

        out = self.conv1(x) # 第一个卷积层
        out = out.replace_feature(self.bn1(out.features)) # 第一个批归一化
        out = out.replace_feature(self.relu(out.features)) # ReLU激活

        out = self.conv2(out) # 第二个卷积层
        out = out.replace_feature(self.bn2(out.features)) # 第二个批归一化

        if self.downsample is not None: # 如果存在下采样层
            identity = self.downsample(x) # 对输入进行下采样

        out = out.replace_feature(out.features + identity.features) # 将输出与identity相加（残差连接）
        out = out.replace_feature(self.relu(out.features)) # ReLU激活

        return out # 返回输出


class VoxelBackBone8xFocal(nn.Module): # 定义一个继承自nn.Module的类
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs): # 定义构造函数
        super().__init__() # 调用父类的构造函数
        self.model_cfg = model_cfg # 保存模型配置

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01) # 创建一个偏函数，用于批归一化
        print(f"[DEBUG] Backbone input_channels: {input_channels}")
        self.sparse_shape = grid_size[::-1] + [1, 0, 0] # 定义稀疏张量的形状

        self.conv_input = spconv.SparseSequential( # 定义输入卷积模块
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'), # 子流形卷积
            norm_fn(16), # 批归一化
            nn.ReLU(True), # ReLU激活
        )

        block = post_act_block # 将post_act_block函数赋值给block变量

        use_img = model_cfg.get('USE_IMG', False) # 从模型配置中获取是否使用图像，默认为False
        topk = model_cfg.get('TOPK', True) # 从模型配置中获取是否使用topk，默认为True
        threshold = model_cfg.get('THRESHOLD', 0.5) # 从模型配置中获取阈值，默认为0.5
        kernel_size = model_cfg.get('KERNEL_SIZE', 3) # 从模型配置中获取卷积核大小，默认为3
        mask_multi = model_cfg.get('MASK_MULTI', False) # 从模型配置中获取是否使用多掩码，默认为False
        skip_mask_kernel = model_cfg.get('SKIP_MASK_KERNEL', False) # 从模型配置中获取是否跳过掩码核，默认为False
        skip_mask_kernel_image =  model_cfg.get('SKIP_MASK_KERNEL_IMG', False) # 从模型配置中获取是否跳过图像掩码核，默认为False
        enlarge_voxel_channels = model_cfg.get('ENLARGE_VOXEL_CHANNELS', -1) # 从模型配置中获取扩大的体素通道数，默认为-1
        img_pretrain = model_cfg.get('IMG_PRETRAIN', "checkpoints/deeplabv3_resnet50_coco-cd0a2569.pth") # 获取图像预训练模型路径
        use_stages = model_cfg.get('USE_STAGES', [1, 2, 3]) # 获取使用的阶段，默认为[1, 2, 3]
        
        if use_img: # 如果使用图像
            model_cfg_seg=dict( # 定义语义分割模型的配置字典
                name='SemDeepLabV3', # 模型名称
                backbone='ResNet50', # 主干网络
                num_class=21, # 类别数（在COCO上预训练）
                args={"feat_extract_layer": ["layer1"], # 提取特征的层
                    "pretrained_path": img_pretrain}, # 预训练模型路径
                channel_reduce={ # 通道缩减配置
                    "in_channels": [256], # 输入通道
                    "out_channels": [16], # 输出通道
                    "kernel_size": [1], # 卷积核大小
                    "stride": [1], # 步长
                    "bias": [False] # 是否使用偏置
                }
            )
            cfg_dict = ConfigDict('SemDeepLabV3') # 创建ConfigDict实例
            objDict.to_object(cfg_dict, **model_cfg_seg) # 将配置字典转换为对象属性
            self.semseg = PyramidFeat2D(optimize=True, model_cfg=cfg_dict) # 创建PyramidFeat2D实例

            self.conv_focal_multimodal = FocalSparseConv(16, 16, image_channel=model_cfg_seg['channel_reduce']['out_channels'][0], # 创建多模态焦点稀疏卷积实例
                                        topk=topk, threshold=threshold, use_img=True, skip_mask_kernel=skip_mask_kernel_image, # 传入相关参数
                                        voxel_stride=1, norm_fn=norm_fn, indice_key='spconv_focal_multimodal') # 传入体素步长、归一化函数和indice_key

        special_spconv_fn = partial(FocalSparseConv, mask_multi=mask_multi, enlarge_voxel_channels=enlarge_voxel_channels, # 创建一个用于FocalSparseConv的偏函数
                                    topk=topk, threshold=threshold, kernel_size=kernel_size, padding=kernel_size//2, # 传入相关参数
                                    skip_mask_kernel=skip_mask_kernel) # 传入是否跳过掩码核
        self.use_img = use_img # 保存是否使用图像的标志

        self.conv1 = SparseSequentialBatchdict( # 定义第一个卷积块
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'), # 一个基本的后激活块
            special_spconv_fn(16, 16, voxel_stride=1, norm_fn=norm_fn, indice_key='focal1') if 1 in use_stages else None, # 如果使用阶段1，则添加一个焦点稀疏卷积
        )

        self.conv2 =SparseSequentialBatchdict( # 定义第二个卷积块
            # [1600, 1408, 41] <- [800, 704, 21] # 注释：输入输出的稀疏张量形状变化
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'), # 一个带步长的卷积块，用于下采样
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'), # 两个基本的后激活块
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            special_spconv_fn(32, 32, voxel_stride=2, norm_fn=norm_fn, indice_key='focal2') if 2 in use_stages else None, # 如果使用阶段2，则添加一个焦点稀疏卷积
        )

        self.conv3 = SparseSequentialBatchdict( # 定义第三个卷积块
            # [800, 704, 21] <- [400, 352, 11] # 注释：输入输出的稀疏张量形状变化
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'), # 一个带步长的卷积块，用于下采样
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'), # 两个基本的后激活块
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            special_spconv_fn(64, 64, voxel_stride=4, norm_fn=norm_fn, indice_key='focal3') if 3 in use_stages else None, # 如果使用阶段3，则添加一个焦点稀疏卷积
        )

        self.conv4 = SparseSequentialBatchdict( # 定义第四个卷积块
            # [400, 352, 11] <- [200, 176, 5] # 注释：输入输出的稀疏张量形状变化
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'), # 一个带步长的卷积块，用于下采样
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'), # 两个基本的后激活块
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0 # 初始化最后的填充为0
        last_pad = self.model_cfg.get('last_pad', last_pad) # 从模型配置中获取最后的填充
        self.conv_out = spconv.SparseSequential( # 定义输出卷积模块
            # [200, 150, 5] -> [200, 150, 2] # 注释：输入输出的稀疏张量形状变化
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad, # 一个稀疏卷积层，用于最终的下采样
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128), # 批归一化
            nn.ReLU(True), # ReLU激活
        )
        self.num_point_features = 128 # 定义点特征的数量
        self.backbone_channels = { # 定义主干网络各层的输出通道数
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }
    def forward(self, batch_dict):
        """
        现在 batch_dict['voxel_features'] 已经是拼接后的 (N+M, 7)
        batch_dict['voxel_coords'] 已经是拼接后的 (N+M, 4)
        """
        voxel_features = batch_dict['voxel_features']  # (N+M, 7)
        voxel_coords = batch_dict['voxel_coords']      # (N+M, 4)
        batch_size = batch_dict['batch_size']
        
        print(f"[DEBUG Backbone] voxel_features shape: {voxel_features.shape}")
        print(f"[DEBUG Backbone] voxel_coords shape: {voxel_coords.shape}")
        
        # 确保坐标是整数类型
        if not isinstance(voxel_coords, torch.Tensor):
            voxel_coords = torch.from_numpy(voxel_coords)
        voxel_coords = voxel_coords.int().contiguous()
        
        # 验证
        assert voxel_coords.ndim == 2, f"coords must be 2D, got {voxel_coords.shape}"
        assert voxel_coords.shape[1] == 4, f"coords must be (N, 4), got {voxel_coords.shape}"
        assert voxel_features.shape[1] == 7, f"features must be (N, 7), got {voxel_features.shape}"
        
        # 创建稀疏张量
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords,
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        
        batch_dict['loss_box_of_pts'] = 0

        x = self.conv_input(input_sp_tensor)
        x_conv1, batch_dict = self.conv1(x, batch_dict)

        if self.use_img:
            x_image = self.semseg(batch_dict['images'])['layer1_feat2d']
            x_conv1, batch_dict = self.conv_focal_multimodal(x_conv1, batch_dict, x_image)

        x_conv2, batch_dict = self.conv2(x_conv1, batch_dict)
        x_conv3, batch_dict = self.conv3(x_conv2, batch_dict)
        x_conv4, batch_dict = self.conv4(x_conv3, batch_dict)

        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        
        # 转换为 BEV 特征
        spatial_features = out.dense()
        N, C, D, H, W = spatial_features.shape
        
        if D > 1:
            spatial_features = spatial_features.mean(dim=2)
        else:
            spatial_features = spatial_features.squeeze(2)
        
        batch_dict['spatial_features'] = spatial_features

        return batch_dict
