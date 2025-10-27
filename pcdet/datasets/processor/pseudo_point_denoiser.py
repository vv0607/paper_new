"""
伪点云去噪模块
使用多种策略去除伪点云中的噪声
"""

import numpy as np
from scipy.spatial import KDTree


class PseudoPointDenoiser:
    """伪点云去噪器"""
    
    def __init__(self, config):
        """
        Args:
            config: 字典，包含去噪参数
                - USE_CONFIDENCE_FILTER: bool, 是否使用置信度过滤
                - CONFIDENCE_THRESHOLD: float, 置信度阈值 (0.5-0.9)
                - USE_DEPTH_FILTER: bool, 是否使用深度过滤
                - MIN_DEPTH: float, 最小深度 (米)
                - MAX_DEPTH: float, 最大深度 (米)
                - USE_SOR: bool, 是否使用统计离群点去除
                - SOR_K: int, SOR的邻居数量
                - SOR_STD: float, SOR的标准差倍数
                - USE_DOWNSAMPLING: bool, 是否下采样
                - MAX_PSEUDO_POINTS: int, 最大伪点云数量
        """
        self.config = config
        self.stats = {
            'original': 0,
            'after_confidence': 0,
            'after_depth': 0,
            'after_sor': 0,
            'final': 0
        }
    
    def __call__(self, pseudo_points):
        """执行完整去噪pipeline"""
        return self.denoise(pseudo_points)
    
    def denoise(self, pseudo_points):
        """
        完整去噪流程
        
        Args:
            pseudo_points: numpy array, shape (N, C)
                前3列: x, y, z
                第4列: 反射强度 (可选)
                第7列: 置信度 (如果有)
        
        Returns:
            denoised_points: numpy array, 去噪后的点云
        """
        if len(pseudo_points) == 0:
            return pseudo_points
        
        self.stats['original'] = len(pseudo_points)
        
        # 1. 置信度过滤（最关键）
        if self.config.get('USE_CONFIDENCE_FILTER', True):
            pseudo_points = self.confidence_filter(pseudo_points)
            self.stats['after_confidence'] = len(pseudo_points)
        
        # 2. 深度范围过滤
        if self.config.get('USE_DEPTH_FILTER', True):
            pseudo_points = self.depth_filter(pseudo_points)
            self.stats['after_depth'] = len(pseudo_points)
        
        # 3. 统计离群点去除（可选，较慢）
        if self.config.get('USE_SOR', False) and len(pseudo_points) > 0:
            pseudo_points = self.statistical_outlier_removal(pseudo_points)
            self.stats['after_sor'] = len(pseudo_points)
        
        # 4. 随机下采样
        if self.config.get('USE_DOWNSAMPLING', True):
            pseudo_points = self.downsample(pseudo_points)
        
        self.stats['final'] = len(pseudo_points)
        
        # 打印统计信息（每1000个样本打印一次）
        if np.random.rand() < 0.001:
            self.print_stats()
        
        return pseudo_points
    
    def confidence_filter(self, points):
        """
        置信度过滤 - 最重要的去噪步骤
        
        只保留高置信度的伪点云
        """
        if points.shape[1] < 7:
            print("Warning: No confidence column found, skipping confidence filter")
            return points
        
        threshold = self.config.get('CONFIDENCE_THRESHOLD', 0.7)
        confidence = points[:, 6]  # 假设第7列是置信度
        
        mask = confidence > threshold
        filtered_points = points[mask]
        
        return filtered_points
    
    def depth_filter(self, points):
        """
        深度范围过滤
        
        远距离的伪点云深度估计非常不准确，直接去除
        """
        # 计算每个点的深度（距离传感器的距离）
        depth = np.linalg.norm(points[:, :3], axis=1)
        
        min_depth = self.config.get('MIN_DEPTH', 2.0)
        max_depth = self.config.get('MAX_DEPTH', 50.0)
        
        mask = (depth > min_depth) & (depth < max_depth)
        filtered_points = points[mask]
        
        return filtered_points
    
    def statistical_outlier_removal(self, points):
        """
        统计离群点去除 (Statistical Outlier Removal, SOR)
        
        计算每个点到其邻居的平均距离，去除异常远的点
        注意: 这个方法比较慢，适合离线处理或小规模点云
        """
        if len(points) < 50:
            return points
        
        k = self.config.get('SOR_K', 20)
        std_multiplier = self.config.get('SOR_STD', 2.0)
        
        # 构建KD树
        tree = KDTree(points[:, :3])
        
        # 查询每个点的k个最近邻
        k_actual = min(k + 1, len(points))
        distances, _ = tree.query(points[:, :3], k=k_actual)
        
        # 计算到邻居的平均距离（跳过自己）
        mean_distances = distances[:, 1:].mean(axis=1)
        
        # 计算全局统计量
        global_mean = mean_distances.mean()
        global_std = mean_distances.std()
        
        # 过滤离群点
        threshold = global_mean + std_multiplier * global_std
        mask = mean_distances < threshold
        
        filtered_points = points[mask]
        
        return filtered_points
    
    def downsample(self, points):
        """
        随机下采样
        
        如果伪点云太多，随机选择一部分
        """
        max_points = self.config.get('MAX_PSEUDO_POINTS', 50000)
        
        if len(points) <= max_points:
            return points
        
        # 随机选择
        indices = np.random.choice(len(points), max_points, replace=False)
        downsampled_points = points[indices]
        
        return downsampled_points
    
    def print_stats(self):
        """打印去噪统计信息"""
        original = self.stats['original']
        if original == 0:
            return
        
        print("\n" + "="*60)
        print("Pseudo Point Denoising Statistics:")
        print("="*60)
        print(f"Original points:          {original:8d} (100.0%)")
        
        if self.stats['after_confidence'] > 0:
            removed = original - self.stats['after_confidence']
            ratio = 100 * removed / original
            print(f"After confidence filter:  {self.stats['after_confidence']:8d} "
                  f"({100 - ratio:.1f}%, removed {removed} points)")
        
        if self.stats['after_depth'] > 0:
            removed = self.stats['after_confidence'] - self.stats['after_depth']
            ratio = 100 * removed / original
            print(f"After depth filter:       {self.stats['after_depth']:8d} "
                  f"(removed {removed} points, {ratio:.1f}%)")
        
        if self.stats['after_sor'] > 0:
            removed = self.stats['after_depth'] - self.stats['after_sor']
            ratio = 100 * removed / original
            print(f"After SOR:                {self.stats['after_sor']:8d} "
                  f"(removed {removed} points, {ratio:.1f}%)")
        
        final = self.stats['final']
        total_removed = original - final
        total_ratio = 100 * total_removed / original
        print(f"Final points:             {final:8d} "
              f"({100 - total_ratio:.1f}%, total removed {total_removed} points)")
        print("="*60 + "\n")


def merge_real_and_pseudo_points(real_points, pseudo_points, 
                                  pseudo_weight=0.5):
    """
    融合真实点云和伪点云
    
    Args:
        real_points: numpy array, 真实点云 (N1, C1)
        pseudo_points: numpy array, 伪点云 (N2, C2)
        pseudo_weight: float, 伪点云权重 (0-1)
    
    Returns:
        merged_points: numpy array, 融合后的点云
    """
    if len(pseudo_points) == 0:
        return real_points
    
    # 特征对齐
    real_dim = real_points.shape[1]
    pseudo_dim = pseudo_points.shape[1]
    
    if pseudo_dim < real_dim:
        # 伪点云特征数少，补零
        padding = np.zeros((len(pseudo_points), real_dim - pseudo_dim))
        pseudo_points = np.concatenate([pseudo_points, padding], axis=1)
    elif pseudo_dim > real_dim:
        # 伪点云特征数多，截断
        pseudo_points = pseudo_points[:, :real_dim]
    
    # 可选: 给伪点云添加权重标记
    if pseudo_weight < 1.0:
        # 假设最后一列是强度或其他特征
        if real_dim >= 4:
            pseudo_points[:, 3] *= pseudo_weight
    
    # 合并
    merged_points = np.vstack([real_points, pseudo_points])
    
    return merged_points


# 便捷函数
def create_denoiser(confidence_threshold=0.7, 
                   max_depth=50.0,
                   min_depth=2.0,
                   use_sor=False):
    """
    快速创建去噪器
    
    Args:
        confidence_threshold: 置信度阈值，推荐0.6-0.8
        max_depth: 最大深度(米)，推荐40-60
        min_depth: 最小深度(米)，推荐1-3
        use_sor: 是否使用统计离群点去除（较慢）
    
    Returns:
        denoiser: PseudoPointDenoiser实例
    """
    config = {
        'USE_CONFIDENCE_FILTER': True,
        'CONFIDENCE_THRESHOLD': confidence_threshold,
        'USE_DEPTH_FILTER': True,
        'MIN_DEPTH': min_depth,
        'MAX_DEPTH': max_depth,
        'USE_SOR': use_sor,
        'SOR_K': 20,
        'SOR_STD': 2.0,
        'USE_DOWNSAMPLING': True,
        'MAX_PSEUDO_POINTS': 40000
    }
    return PseudoPointDenoiser(config)


if __name__ == '__main__':
    # 测试代码
    print("Testing PseudoPointDenoiser...")
    
    # 创建模拟伪点云
    np.random.seed(42)
    n_points = 100000
    
    # xyz坐标
    points = np.random.randn(n_points, 3) * 10
    
    # 反射强度
    intensity = np.random.rand(n_points, 1)
    
    # 其他特征
    features = np.random.rand(n_points, 2)
    
    # 置信度（第7列）
    confidence = np.random.rand(n_points, 1)
    
    pseudo_points = np.hstack([points, intensity, features, confidence])
    
    print(f"Original pseudo points shape: {pseudo_points.shape}")
    
    # 创建去噪器
    denoiser = create_denoiser(
        confidence_threshold=0.7,
        max_depth=50.0,
        use_sor=False
    )
    
    # 去噪
    denoised = denoiser(pseudo_points)
    
    print(f"Denoised pseudo points shape: {denoised.shape}")
    print(f"Removed {len(pseudo_points) - len(denoised)} points "
          f"({100*(1 - len(denoised)/len(pseudo_points)):.1f}%)")
