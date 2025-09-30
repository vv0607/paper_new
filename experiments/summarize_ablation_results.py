"""
消融实验结果汇总脚本
位置: experiments/summarize_ablation_results.py
理由: 自动汇总所有消融实验结果，生成性能对比表格
"""

import argparse
import json
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def parse_test_results(result_file):
    """
    解析测试结果文件
    Args:
        result_file: 结果文件路径
    Returns:
        metrics: 性能指标字典
    """
    metrics = {
        'Car_3D_AP40_Easy': 0.0,
        'Car_3D_AP40_Moderate': 0.0,
        'Car_3D_AP40_Hard': 0.0,
        'Car_BEV_AP40_Easy': 0.0,
        'Car_BEV_AP40_Moderate': 0.0,
        'Car_BEV_AP40_Hard': 0.0,
        'Pedestrian_3D_AP40_Moderate': 0.0,
        'Cyclist_3D_AP40_Moderate': 0.0,
        'mAP_3D': 0.0,
        'mAP_BEV': 0.0
    }
    
    if not result_file.exists():
        return metrics
        
    try:
        with open(result_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            # 解析AP值
            if 'Car AP40@' in line:
                parts = line.split()
                if '3d' in line.lower():
                    if 'Easy' in line:
                        metrics['Car_3D_AP40_Easy'] = float(parts[-3])
                    elif 'Moderate' in line:
                        metrics['Car_3D_AP40_Moderate'] = float(parts[-2])
                    elif 'Hard' in line:
                        metrics['Car_3D_AP40_Hard'] = float(parts[-1])
                elif 'bev' in line.lower():
                    if 'Easy' in line:
                        metrics['Car_BEV_AP40_Easy'] = float(parts[-3])
                    elif 'Moderate' in line:
                        metrics['Car_BEV_AP40_Moderate'] = float(parts[-2])
                    elif 'Hard' in line:
                        metrics['Car_BEV_AP40_Hard'] = float(parts[-1])
                        
            # 类似地解析Pedestrian和Cyclist
            elif 'Pedestrian AP40@' in line and 'Moderate' in line:
                parts = line.split()
                metrics['Pedestrian_3D_AP40_Moderate'] = float(parts[-2])
                
            elif 'Cyclist AP40@' in line and 'Moderate' in line:
                parts = line.split()
                metrics['Cyclist_3D_AP40_Moderate'] = float(parts[-2])
                
    except Exception as e:
        print(f"Error parsing {result_file}: {e}")
        
    # 计算平均值
    metrics['mAP_3D'] = np.mean([
        metrics['Car_3D_AP40_Moderate'],
        metrics['Pedestrian_3D_AP40_Moderate'],
        metrics['Cyclist_3D_AP40_Moderate']
    ])
    
    metrics['mAP_BEV'] = np.mean([
        metrics['Car_BEV_AP40_Easy'],
        metrics['Car_BEV_AP40_Moderate'],
        metrics['Car_BEV_AP40_Hard']
    ])
    
    return metrics


def get_model_info(exp_dir):
    """
    获取模型信息
    Args:
        exp_dir: 实验目录
    Returns:
        info: 模型信息字典
    """
    info = {
        'num_params': 0,
        'flops': 0,
        'inference_time': 0,
        'gpu_memory': 0,
        'training_time': 0
    }
    
    # 读取训练日志获取信息
    log_file = exp_dir / 'log_train.txt'
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                if 'Total params:' in line:
                    info['num_params'] = float(line.split(':')[-1].strip().split()[0])
                elif 'Inference time:' in line:
                    info['inference_time'] = float(line.split(':')[-1].strip().split()[0])
                elif 'GPU memory:' in line:
                    info['gpu_memory'] = float(line.split(':')[-1].strip().split()[0])
                elif 'Total training time:' in line:
                    info['training_time'] = float(line.split(':')[-1].strip().split()[0])
                    
        except Exception as e:
            print(f"Error reading log from {exp_dir}: {e}")
            
    return info


def create_summary_table(exp_dir):
    """
    创建汇总表格
    Args:
        exp_dir: 实验根目录
    Returns:
        df: 汇总DataFrame
    """
    exp_dir = Path(exp_dir)
    
    experiments = [
        ('E1_baseline', 'Baseline (VoxelRCNN)', 
         {'pseudo_cloud': False, 'focals': False, 'fusion': False}),
        ('E2_pseudo_cloud', 'Baseline + Pseudo Cloud',
         {'pseudo_cloud': True, 'focals': False, 'fusion': False}),
        ('E3_focals', 'Baseline + Pseudo + FocalsConv',
         {'pseudo_cloud': True, 'focals': True, 'fusion': False}),
        ('E4_fusion', 'Baseline + Pseudo + FocalsConv + Fusion',
         {'pseudo_cloud': True, 'focals': True, 'fusion': True}),
        ('E5_complete', 'Complete Model',
         {'pseudo_cloud': True, 'focals': True, 'fusion': True}),
        ('E6_early_fusion', 'Early Fusion (conv1)',
         {'pseudo_cloud': True, 'focals': True, 'fusion': True}),
        ('E7_late_fusion', 'Late Fusion (conv4)',
         {'pseudo_cloud': True, 'focals': True, 'fusion': True}),
        ('E8_multi_fusion', 'Multi-stage Fusion',
         {'pseudo_cloud': True, 'focals': True, 'fusion': True}),
    ]
    
    results = []
    
    for exp_name, description, components in experiments:
        exp_path = exp_dir / exp_name
        
        if not exp_path.exists():
            print(f"Experiment {exp_name} not found, skipping...")
            continue
            
        # 获取测试结果
        result_file = exp_path / 'result.txt'
        metrics = parse_test_results(result_file)
        
        # 获取模型信息
        model_info = get_model_info(exp_path)
        
        # 组合结果
        row = {
            'Experiment': exp_name,
            'Description': description,
            **components,
            **metrics,
            **model_info
        }
        
        results.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 计算相对提升
    if len(df) > 0 and 'E1_baseline' in df['Experiment'].values:
        baseline_idx = df[df['Experiment'] == 'E1_baseline'].index[0]
        baseline_map = df.loc[baseline_idx, 'mAP_3D']
        
        df['Improvement (%)'] = ((df['mAP_3D'] - baseline_map) / baseline_map * 100).round(2)
    
    return df


def create_performance_plots(df, output_dir):
    """
    创建性能对比图表
    Args:
        df: 结果DataFrame
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置绘图风格
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # 1. mAP对比条形图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 3D mAP
    df_sorted = df.sort_values('mAP_3D')
    ax1.barh(df_sorted['Description'], df_sorted['mAP_3D'], 
             color='steelblue', edgecolor='black')
    ax1.set_xlabel('3D mAP (%)')
    ax1.set_title('3D Detection Performance')
    ax1.grid(axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, (desc, val) in enumerate(zip(df_sorted['Description'], df_sorted['mAP_3D'])):
        ax1.text(val + 0.5, i, f'{val:.2f}%', va='center')
    
    # BEV mAP
    ax2.barh(df_sorted['Description'], df_sorted['mAP_BEV'],
             color='darkgreen', edgecolor='black')
    ax2.set_xlabel('BEV mAP (%)')
    ax2.set_title('BEV Detection Performance')
    ax2.grid(axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, (desc, val) in enumerate(zip(df_sorted['Description'], df_sorted['mAP_BEV'])):
        ax2.text(val + 0.5, i, f'{val:.2f}%', va='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 组件贡献分析
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 选择消融实验 E1-E5
    ablation_df = df[df['Experiment'].str.contains('E[1-5]')].copy()
    
    if len(ablation_df) > 0:
        x = range(len(ablation_df))
        width = 0.25
        
        # Car性能
        ax.bar([i - width for i in x], ablation_df['Car_3D_AP40_Moderate'],
               width, label='Car', color='blue', alpha=0.8)
        # Pedestrian性能
        ax.bar(x, ablation_df['Pedestrian_3D_AP40_Moderate'],
               width, label='Pedestrian', color='green', alpha=0.8)
        # Cyclist性能
        ax.bar([i + width for i in x], ablation_df['Cyclist_3D_AP40_Moderate'],
               width, label='Cyclist', color='red', alpha=0.8)
        
        ax.set_xlabel('Experiment')
        ax.set_ylabel('AP40 Moderate (%)')
        ax.set_title('Per-class Performance Across Ablation Study')
        ax.set_xticks(x)
        ax.set_xticklabels(ablation_df['Experiment'].values)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'ablation_per_class.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. 效率分析
    if 'inference_time' in df.columns and df['inference_time'].sum() > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 推理时间 vs mAP
        ax1.scatter(df['inference_time'], df['mAP_3D'], s=100, alpha=0.7)
        for i, exp in enumerate(df['Experiment']):
            ax1.annotate(exp, (df.iloc[i]['inference_time'], df.iloc[i]['mAP_3D']),
                        fontsize=8)
        ax1.set_xlabel('Inference Time (ms)')
        ax1.set_ylabel('3D mAP (%)')
        ax1.set_title('Accuracy vs Speed Trade-off')
        ax1.grid(alpha=0.3)
        
        # 参数量 vs mAP
        if 'num_params' in df.columns and df['num_params'].sum() > 0:
            ax2.scatter(df['num_params'], df['mAP_3D'], s=100, alpha=0.7)
            for i, exp in enumerate(df['Experiment']):
                ax2.annotate(exp, (df.iloc[i]['num_params'], df.iloc[i]['mAP_3D']),
                           fontsize=8)
            ax2.set_xlabel('Parameters (M)')
            ax2.set_ylabel('3D mAP (%)')
            ax2.set_title('Accuracy vs Model Size Trade-off')
            ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'efficiency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Plots saved to {output_dir}")


def generate_latex_table(df, output_file):
    """
    生成LaTeX表格
    Args:
        df: 结果DataFrame
        output_file: 输出文件路径
    """
    # 选择关键列
    columns = ['Description', 'Car_3D_AP40_Moderate', 'Pedestrian_3D_AP40_Moderate',
               'Cyclist_3D_AP40_Moderate', 'mAP_3D', 'Improvement (%)']
    
    df_latex = df[columns].copy()
    
    # 格式化数值
    for col in columns[1:-1]:
        df_latex[col] = df_latex[col].round(2)
    
    # 生成LaTeX
    latex_str = df_latex.to_latex(index=False, escape=False, column_format='l' + 'c'*(len(columns)-1))
    
    # 保存
    with open(output_file, 'w') as f:
        f.write(latex_str)
    
    print(f"LaTeX table saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Summarize ablation study results')
    parser.add_argument('--exp_dir', type=str, required=True,
                       help='Experiment directory')
    parser.add_argument('--output', type=str, default='ablation_summary.csv',
                       help='Output CSV file')
    parser.add_argument('--plot_dir', type=str, default=None,
                       help='Directory to save plots')
    parser.add_argument('--latex', type=str, default=None,
                       help='Output LaTeX table file')
    
    args = parser.parse_args()
    
    # 创建汇总表
    print("Collecting experiment results...")
    df = create_summary_table(args.exp_dir)
    
    if len(df) == 0:
        print("No results found!")
        return
    
    # 保存CSV
    df.to_csv(args.output, index=False)
    print(f"\nResults summary saved to: {args.output}")
    
    # 打印表格
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    print(df[['Description', 'mAP_3D', 'Car_3D_AP40_Moderate', 
              'Pedestrian_3D_AP40_Moderate', 'Cyclist_3D_AP40_Moderate', 
              'Improvement (%)']].to_string(index=False))
    
    # 创建图表
    if args.plot_dir:
        print("\nGenerating plots...")
        create_performance_plots(df, args.plot_dir)
    
    # 生成LaTeX表格
    if args.latex:
        generate_latex_table(df, args.latex)


if __name__ == '__main__':
    main()