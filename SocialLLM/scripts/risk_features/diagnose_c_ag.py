#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断C_ag值较小的原因
分析phi_i（总贡献绝对值）和sigma_i（时间标准差）的分布和相关性
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_shapley_data(result_dir: Path):
    """加载Shapley数据"""
    shapley_file = result_dir / "shapley" / "shapley_attribution_timeseries.csv"
    
    if not shapley_file.exists():
        raise FileNotFoundError(f"Shapley CSV file not found: {shapley_file}")
    
    df_shapley = pd.read_csv(shapley_file)
    
    # 获取agent_ids和timesteps
    agent_ids = sorted(df_shapley['agent_id'].unique().tolist())
    timesteps = sorted(df_shapley['timestep'].unique().tolist())
    
    # 创建agent_id和timestep到索引的映射
    agent_id_to_idx = {aid: idx for idx, aid in enumerate(agent_ids)}
    timestep_to_idx = {t: idx for idx, t in enumerate(timesteps)}
    
    # 构建Shapley矩阵 (num_agents × num_timesteps)
    shapley_matrix = np.zeros((len(agent_ids), len(timesteps)), dtype=np.float64)
    
    for _, row in df_shapley.iterrows():
        agent_id = int(row['agent_id'])
        timestep = int(row['timestep'])
        shapley_value = float(row['shapley_value'])
        
        if agent_id in agent_id_to_idx and timestep in timestep_to_idx:
            agent_idx = agent_id_to_idx[agent_id]
            timestep_idx = timestep_to_idx[timestep]
            shapley_matrix[agent_idx, timestep_idx] = shapley_value
    
    return shapley_matrix, agent_ids, timesteps


def calculate_c_ag_detailed(shapley_matrix: np.ndarray):
    """
    详细计算C_ag，返回中间结果
    """
    # 计算每个agent的总Shapley值，取绝对值
    phi_i = np.abs(np.sum(shapley_matrix, axis=1))
    
    # 计算每个agent的标准差（不稳定性）
    sigma_i = np.std(shapley_matrix, axis=1, ddof=0)
    
    # 计算Pearson相关系数
    if np.std(phi_i) == 0 or np.std(sigma_i) == 0:
        correlation = 0.0
    else:
        correlation = np.corrcoef(phi_i, sigma_i)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
    
    return {
        'phi_i': phi_i,
        'sigma_i': sigma_i,
        'correlation': correlation,
        'phi_i_mean': np.mean(phi_i),
        'phi_i_std': np.std(phi_i),
        'phi_i_min': np.min(phi_i),
        'phi_i_max': np.max(phi_i),
        'sigma_i_mean': np.mean(sigma_i),
        'sigma_i_std': np.std(sigma_i),
        'sigma_i_min': np.min(sigma_i),
        'sigma_i_max': np.max(sigma_i),
    }


def main():
    parser = argparse.ArgumentParser(description='诊断C_ag值')
    parser.add_argument('--result_dir', type=str, required=True, help='结果目录路径')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir)
    if not result_dir.is_absolute():
        result_dir = project_root / result_dir
    result_dir = result_dir.resolve()
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = result_dir / "risk_feature" / "diagnosis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("C_ag诊断分析")
    print("=" * 60)
    print(f"结果目录: {result_dir}")
    print()
    
    # 加载数据
    print("加载Shapley数据...")
    shapley_matrix, agent_ids, timesteps = load_shapley_data(result_dir)
    print(f"  Shapley矩阵形状: {shapley_matrix.shape}")
    print(f"  Agents: {len(agent_ids)}, Timesteps: {len(timesteps)}")
    print()
    
    # 计算C_ag详细信息
    print("计算C_ag详细信息...")
    details = calculate_c_ag_detailed(shapley_matrix)
    
    print(f"C_ag = {details['correlation']:.6f}")
    print()
    print("phi_i (总贡献绝对值) 统计:")
    print(f"  均值: {details['phi_i_mean']:.6f}")
    print(f"  标准差: {details['phi_i_std']:.6f}")
    print(f"  最小值: {details['phi_i_min']:.6f}")
    print(f"  最大值: {details['phi_i_max']:.6f}")
    print()
    print("sigma_i (时间标准差) 统计:")
    print(f"  均值: {details['sigma_i_mean']:.6f}")
    print(f"  标准差: {details['sigma_i_std']:.6f}")
    print(f"  最小值: {details['sigma_i_min']:.6f}")
    print(f"  最大值: {details['sigma_i_max']:.6f}")
    print()
    
    # 分析相关性
    print("相关性分析:")
    print(f"  Pearson相关系数: {details['correlation']:.6f}")
    
    # 检查是否有明显的模式
    # 找出phi_i和sigma_i都较大的agent
    phi_threshold = np.percentile(details['phi_i'], 75)
    sigma_threshold = np.percentile(details['sigma_i'], 75)
    
    high_phi_high_sigma = np.sum((details['phi_i'] > phi_threshold) & (details['sigma_i'] > sigma_threshold))
    high_phi_low_sigma = np.sum((details['phi_i'] > phi_threshold) & (details['sigma_i'] < np.percentile(details['sigma_i'], 25)))
    low_phi_high_sigma = np.sum((details['phi_i'] < np.percentile(details['phi_i'], 25)) & (details['sigma_i'] > sigma_threshold))
    
    print(f"  高贡献高不稳定性agent数: {high_phi_high_sigma}")
    print(f"  高贡献低不稳定性agent数: {high_phi_low_sigma}")
    print(f"  低贡献高不稳定性agent数: {low_phi_high_sigma}")
    print()
    
    # 绘制散点图
    print("生成可视化图表...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 散点图：phi_i vs sigma_i
    ax = axes[0, 0]
    ax.scatter(details['phi_i'], details['sigma_i'], alpha=0.6, s=20)
    ax.set_xlabel('phi_i (总贡献绝对值)')
    ax.set_ylabel('sigma_i (时间标准差)')
    ax.set_title(f'C_ag = {details["correlation"]:.6f}')
    ax.grid(True, alpha=0.3)
    
    # 添加趋势线
    if len(details['phi_i']) > 1 and np.std(details['phi_i']) > 0:
        z = np.polyfit(details['phi_i'], details['sigma_i'], 1)
        p = np.poly1d(z)
        ax.plot(details['phi_i'], p(details['phi_i']), "r--", alpha=0.5, label=f'趋势线: y={z[0]:.4f}x+{z[1]:.4f}')
        ax.legend()
    
    # phi_i分布直方图
    ax = axes[0, 1]
    ax.hist(details['phi_i'], bins=30, alpha=0.7, edgecolor='black')
    ax.set_xlabel('phi_i (总贡献绝对值)')
    ax.set_ylabel('频数')
    ax.set_title('phi_i分布')
    ax.grid(True, alpha=0.3, axis='y')
    
    # sigma_i分布直方图
    ax = axes[1, 0]
    ax.hist(details['sigma_i'], bins=30, alpha=0.7, edgecolor='black')
    ax.set_xlabel('sigma_i (时间标准差)')
    ax.set_ylabel('频数')
    ax.set_title('sigma_i分布')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 时间序列：前10个agent的Shapley值
    ax = axes[1, 1]
    num_agents_to_plot = min(10, len(agent_ids))
    for i in range(num_agents_to_plot):
        ax.plot(timesteps, shapley_matrix[i, :], alpha=0.6, label=f'Agent {agent_ids[i]}', linewidth=1)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Shapley Value')
    ax.set_title(f'前{num_agents_to_plot}个Agent的Shapley时间序列')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / "c_ag_diagnosis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  图表已保存: {plot_path}")
    
    # 保存详细数据
    data_path = output_dir / "c_ag_diagnosis_data.json"
    diagnosis_data = {
        'c_ag': float(details['correlation']),
        'phi_i_stats': {
            'mean': float(details['phi_i_mean']),
            'std': float(details['phi_i_std']),
            'min': float(details['phi_i_min']),
            'max': float(details['phi_i_max']),
        },
        'sigma_i_stats': {
            'mean': float(details['sigma_i_mean']),
            'std': float(details['sigma_i_std']),
            'min': float(details['sigma_i_min']),
            'max': float(details['sigma_i_max']),
        },
        'agent_patterns': {
            'high_phi_high_sigma': int(high_phi_high_sigma),
            'high_phi_low_sigma': int(high_phi_low_sigma),
            'low_phi_high_sigma': int(low_phi_high_sigma),
        },
        'num_agents': len(agent_ids),
        'num_timesteps': len(timesteps),
    }
    
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(diagnosis_data, f, indent=2, ensure_ascii=False)
    print(f"  数据已保存: {data_path}")
    
    print()
    print("=" * 60)
    print("诊断完成")
    print("=" * 60)
    
    # 解释结果
    print("\n可能的原因分析:")
    if abs(details['correlation']) < 0.1:
        print("  - C_ag接近0，说明总贡献和时间不稳定性之间几乎没有线性关系")
        print("  - 这可能意味着：贡献大的agent不一定时间波动大，反之亦然")
        print("  - 在SocialLLM系统中，这可能反映了agent行为的复杂性")
    elif details['correlation'] < 0:
        print("  - C_ag为负，说明总贡献大的agent反而时间波动小")
        print("  - 这可能意味着：稳定贡献的agent（持续影响）比波动大的agent更重要")
    else:
        print("  - C_ag为正，说明总贡献大的agent时间波动也大")
        print("  - 这可能意味着：高风险贡献往往集中在某些关键时刻")


if __name__ == '__main__':
    main()
