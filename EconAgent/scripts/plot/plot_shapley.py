#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shapley Value 可视化脚本
生成热力图和累计风险折线图
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap


def str_to_bool(v):
    """将字符串转换为布尔值."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_white_blue_red_colormap():
    """创建白色-蓝色-红色的颜色映射（白色表示最低值）."""
    # 创建从白色到浅蓝到深蓝到红色的渐变（20个颜色点，使渐变更丰富）
    # 颜色点在整个范围内均匀分布，确保过渡更明显
    # 顺序：白色(最低) → 浅蓝 → 深蓝 → 浅红 → 深红(最高)
    colors = [
        '#FFFFFF',         # 0: 纯白色 (最低值)
        '#F0F8FF',         # 1: 极浅蓝白 (Alice Blue)
        '#E0F2FE',         # 2: 很浅的蓝色
        '#BAE6FD',         # 3: 浅蓝色
        '#93C5FD',         # 4: 浅蓝色
        '#60A5FA',         # 5: 中浅蓝色
        '#3B82F6',         # 6: 中浅蓝色
        '#2563EB',         # 7: 标准蓝色
        '#1D4ED8',         # 8: 中蓝色
        '#1E40AF',         # 9: 中深蓝色
        '#1E3A8A',         # 10: 深蓝色
        '#172554',         # 11: 很深蓝色
        '#1E1B4B',         # 12: 极深蓝色（过渡）
        '#4C1D95',         # 13: 蓝紫色（过渡）
        '#7C2D12',         # 14: 深棕红色（过渡）
        '#DC2626',         # 15: 深红色
        '#EF4444',         # 16: 中深红色
        '#F87171',         # 17: 中红色
        '#FB923C',         # 18: 橙红色
        '#FF5252',         # 19: 中浅红色
        '#FF1744',         # 20: 中深红色
        '#C62828'          # 21: 深红色 (最高值)
    ]
    # 增加插值精度，使颜色过渡更平滑
    n_bins = 512
    cmap = LinearSegmentedColormap.from_list('white_blue_red', colors, N=n_bins)
    return cmap


def load_shapley_results(shapley_values_path, stats_json_path):
    """加载 Shapley 值结果和统计信息."""
    shapley_values = np.load(shapley_values_path)
    
    with open(stats_json_path, 'r') as f:
        stats = json.load(f)
    
    return shapley_values, stats


def visualize_shapley_heatmap(shapley_values, output_path, title="Shapley Value Heatmap"):
    """可视化 Shapley 值热力图."""
    num_agents, episode_length = shapley_values.shape
    
    # 使用白-蓝-红颜色映射（白色表示最低值）
    vmin = np.min(shapley_values)
    vmax = np.max(shapley_values)
    
    if vmin < 0:
        # 有负值：使用 RdBu_r colormap（红-白-蓝）
        # 红色表示正值（最高），白色表示0，蓝色表示负值（最低）
        cmap = 'RdBu_r'  # 红色（正值，最高）-> 白色（0）-> 蓝色（负值，最低）
        abs_max = max(abs(vmin), abs(vmax))
        vmin_plot = -abs_max
        vmax_plot = abs_max
    else:
        # 只有正值：使用自定义的白-蓝-红渐变
        # 白色表示最低，蓝色表示中间，红色表示最高
        cmap = create_white_blue_red_colormap()
        vmin_plot = vmin
        vmax_plot = vmax
    
    # 创建图形，使用更紧凑的布局
    fig, ax = plt.subplots(figsize=(12, 3))
    
    # 绘制热力图
    im = ax.imshow(shapley_values, cmap=cmap, aspect='auto', 
                   interpolation='nearest', origin='lower',
                   vmin=vmin_plot, vmax=vmax_plot)
    
    # 添加 colorbar，调整位置和大小以避免重叠
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Shapley Value', fontsize=10, rotation=270, labelpad=15)
    
    # 设置标签和标题，使用较小的字体
    ax.set_xlabel('Timestep', fontsize=10)
    ax.set_ylabel('Agent ID', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=5)
    
    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 设置刻度
    if num_agents <= 20:
        ax.set_yticks(range(num_agents))
        ax.set_yticklabels(range(num_agents), fontsize=8)
    else:
        # 如果 agent 太多，只显示部分刻度
        step = max(1, num_agents // 10)
        ax.set_yticks(range(0, num_agents, step))
        ax.set_yticklabels(range(0, num_agents, step), fontsize=8)
    
    if episode_length <= 20:
        ax.set_xticks(range(episode_length))
        ax.set_xticklabels(range(1, episode_length + 1), fontsize=8)
    else:
        # 如果时间步太多，只显示部分刻度
        step = max(1, episode_length // 10)
        tick_positions = range(0, episode_length, step)
        tick_labels = [str(i + 1) for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=8)
    
    # 使用 tight_layout 并调整边距
    plt.tight_layout(pad=1.0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Heatmap saved to: {output_path}")
    plt.close()


def plot_cumulative_risk(shapley_values, baseline_risk, real_risk, output_path, 
                        title="Cumulative Risk Over Time"):
    """绘制累计风险折线图."""
    num_agents, episode_length = shapley_values.shape
    
    # 计算每个时间步的总贡献（所有 agent 在该时间步的 Shapley 值之和）
    shapley_per_timestep = np.sum(shapley_values, axis=0)
    
    # 累计风险 = baseline_risk + 累计 Shapley 值
    cumulative_risk = baseline_risk + np.cumsum(shapley_per_timestep)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    
    # 子图1: 累计风险（按时间步累计）
    timesteps = np.arange(1, episode_length + 1)
    ax1.plot(timesteps, cumulative_risk, 'b-', linewidth=2, marker='o', markersize=4, 
             label='Cumulative Risk')
    ax1.axhline(y=baseline_risk, color='g', linestyle='--', linewidth=1.5, 
                label=f'Baseline Risk ({baseline_risk:.6f})')
    ax1.axhline(y=real_risk, color='r', linestyle='--', linewidth=1.5, 
                label=f'Real Risk ({real_risk:.6f})')
    ax1.set_xlabel('Timestep', fontsize=12)
    ax1.set_ylabel('Cumulative Risk', fontsize=12)
    ax1.set_title('Cumulative Risk Over Time', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(timesteps)
    # 使用科学计数法格式化y轴
    ax1.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 子图2: 每个时间步的风险增量
    # 正值为红色系，负值为蓝色系
    colors = ['#EF4444' if x >= 0 else '#3B82F6' for x in shapley_per_timestep]
    ax2.bar(timesteps, shapley_per_timestep, alpha=0.6, color=colors, 
            label='Risk Contribution per Timestep')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('Shapley Value Sum', fontsize=12)
    ax2.set_title('Risk Contribution per Timestep (Sum of All Agents)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax2.set_xticks(timesteps)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Cumulative risk plot saved to: {output_path}")
    plt.close()


def plot_baseline_metrics(shapley_values_path, output_path, variable='price_inflation_rate', threshold=None):
    """绘制 baseline 的 metrics 折线图，参照 plot_world_metrics 的方式.
    
    Args:
        shapley_values_path: shapley_values.npy 文件路径，用于推断 baseline 目录
        output_path: 输出图片路径
        variable: 要绘制的变量名（默认：price_inflation_rate）
        threshold: 阈值（可选），如果提供会画红色虚线
    """
    # 从 shapley_values 路径推断 baseline 目录
    # 新路径: .../data/{simulation_name}/shapley/baseline/world_metrics.csv
    # 旧路径（向后兼容）: .../data/{simulation_name}/baseline/world_metrics.csv
    shapley_dir = os.path.dirname(shapley_values_path)
    
    # 首先尝试新路径（在 shapley 目录下的 baseline 子目录）
    baseline_csv_path = os.path.join(shapley_dir, 'baseline', 'world_metrics.csv')
    
    # 如果新路径不存在，尝试旧路径（在 simulation 目录下的 baseline 目录）
    if not os.path.exists(baseline_csv_path):
        simulation_dir = os.path.dirname(shapley_dir)  # 获取 simulation 目录
        baseline_csv_path = os.path.join(simulation_dir, 'baseline', 'world_metrics.csv')
    
    # 检查文件是否存在
    if not os.path.exists(baseline_csv_path):
        print(f"Warning: Baseline metrics file not found. Tried:")
        print(f"  - {os.path.join(shapley_dir, 'baseline', 'world_metrics.csv')}")
        print(f"  - {os.path.join(os.path.dirname(shapley_dir), 'baseline', 'world_metrics.csv')}")
        return False
    
    # 读取CSV文件
    try:
        df = pd.read_csv(baseline_csv_path)
        
        if df.empty:
            print(f"Error: Baseline CSV file is empty")
            return False
        
        # 检查变量是否存在
        if variable not in df.columns:
            print(f"Error: Variable '{variable}' not found in baseline CSV file")
            print(f"  Available variables: {', '.join(df.columns.tolist())}")
            return False
        
        # 提取数据
        timestep = df['timestep'].values
        values = df[variable].values
        
    except Exception as e:
        print(f"Error: Failed to read baseline CSV file {baseline_csv_path}: {e}")
        return False
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    plt.plot(timestep, values, marker='o', markersize=4, linewidth=2, 
             label=f'Baseline {variable.replace("_", " ").title()}', color='#1E88E5')
    
    # 添加阈值线（如果提供）
    if threshold is not None:
        try:
            threshold_value = float(threshold)
            plt.axhline(y=threshold_value, color='red', linestyle='--', linewidth=2, 
                       label=f'Threshold: {threshold_value}')
        except ValueError:
            print(f"Warning: Threshold '{threshold}' is not a valid number, skipping threshold line")
    
    # 添加零线
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # 设置标签和标题
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel(variable.replace('_', ' ').title(), fontsize=12)
    plt.title(f'Baseline {variable.replace("_", " ").title()} Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Baseline metrics plot saved to: {output_path}")
    return True


def plot_agent_total_attribution(shapley_values, output_path, title="Agent Total Attribution"):
    """绘制每个 agent 的总 Shapley 值柱状图.
    
    Args:
        shapley_values: Shapley 值数组，形状为 (num_agents, episode_length)
        output_path: 输出路径
        title: 图表标题
    """
    num_agents, episode_length = shapley_values.shape
    
    # 计算每个 agent 的总贡献（所有时间步的总和）
    agent_total_contributions = np.sum(shapley_values, axis=1)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(max(10, num_agents * 0.8), 6))
    
    # 准备数据
    agent_ids = np.arange(num_agents)
    
    # 根据值的正负选择颜色：正值用红色系，负值用蓝色系
    colors = ['#EF4444' if x >= 0 else '#3B82F6' for x in agent_total_contributions]
    
    # 绘制柱状图
    bars = ax.bar(agent_ids, agent_total_contributions, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # 添加零线
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # 设置标签和标题
    ax.set_xlabel('Agent ID', fontsize=12)
    ax.set_ylabel('Total Shapley Value', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    # 设置 x 轴刻度
    ax.set_xticks(agent_ids)
    ax.set_xticklabels([f'Agent {i}' for i in agent_ids], fontsize=10)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # 添加图例说明
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#EF4444', alpha=0.7, label='Positive Contribution'),
        Patch(facecolor='#3B82F6', alpha=0.7, label='Negative Contribution')
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Agent total attribution plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize Shapley Value results')
    parser.add_argument('--shapley_values', type=str, required=True,
                       help='Path to shapley_values.npy file')
    parser.add_argument('--stats_json', type=str, required=True,
                       help='Path to shapley_stats.json file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: same as shapley_values directory)')
    parser.add_argument('--plot_heatmap', nargs='?', const=True, default=True, type=str_to_bool,
                       help='Generate heatmap (default: True, use --plot_heatmap False to disable)')
    parser.add_argument('--plot_cumulative', nargs='?', const=True, default=True, type=str_to_bool,
                       help='Generate cumulative risk plot (default: True, use --plot_cumulative False to disable)')
    parser.add_argument('--plot_agent_total', nargs='?', const=True, default=True, type=str_to_bool,
                       help='Generate agent total attribution bar chart (default: True, use --plot_agent_total False to disable)')
    parser.add_argument('--plot_baseline', nargs='?', const=True, default=True, type=str_to_bool,
                       help='Generate baseline metrics plot (default: True, use --plot_baseline False to disable)')
    parser.add_argument('--baseline_variable', type=str, default='price_inflation_rate',
                       help='Variable to plot for baseline (default: price_inflation_rate)')
    parser.add_argument('--baseline_threshold', type=float, default=None,
                       help='Threshold value for baseline plot (optional)')
    
    args = parser.parse_args()
    
    # 确定输出目录
    if args.output_dir is None:
        # 自动推断输出目录：从 shapley_values 路径推断到对应的 simulation 文件夹下的 shapley 目录
        shapley_values_path = os.path.abspath(args.shapley_values)
        
        # 检查是否在 data 目录下
        if 'data' in shapley_values_path:
            # 找到 data 目录
            path_parts = shapley_values_path.split(os.sep)
            data_idx = None
            for i, part in enumerate(path_parts):
                if part == 'data' and i + 1 < len(path_parts):
                    data_idx = i
                    break
            
            if data_idx is not None and data_idx + 1 < len(path_parts):
                # 获取 simulation 名称（data 后面的文件夹）
                simulation_name = path_parts[data_idx + 1]
                # 构建输出路径：.../data/{simulation_name}/shapley/
                data_dir = os.sep.join(path_parts[:data_idx + 1])
                args.output_dir = os.path.join(data_dir, simulation_name, 'shapley')
            else:
                # 回退到 shapley_values 所在目录
                args.output_dir = os.path.dirname(shapley_values_path)
        else:
            # 如果不在 data 目录下，使用 shapley_values 所在目录
            args.output_dir = os.path.dirname(shapley_values_path)
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # 加载数据
    print(f"Loading Shapley values from: {args.shapley_values}")
    print(f"Loading stats from: {args.stats_json}")
    shapley_values, stats = load_shapley_results(args.shapley_values, args.stats_json)
    
    num_agents, episode_length = shapley_values.shape
    baseline_risk = stats.get('baseline_risk', 0.0)
    real_risk = stats.get('real_risk', 0.0)
    inflation_threshold = stats.get('inflation_threshold', 0.1)
    n_samples = stats.get('n_samples', 1000)
    method = stats.get('method', 'mc')
    
    print(f"\nShapley Value Statistics:")
    print(f"  Shape: {num_agents} agents × {episode_length} timesteps")
    print(f"  Baseline Risk: {baseline_risk:.6f}")
    print(f"  Real Risk: {real_risk:.6f}")
    print(f"  Total Shapley Sum: {np.sum(shapley_values):.6f}")
    print(f"  Method: {method}, Samples: {n_samples}")
    print()
    
    # 生成热力图
    if args.plot_heatmap:
        heatmap_path = os.path.join(args.output_dir, 'shapley_heatmap.png')
        title = f"Shapley Value Heatmap\n(Method: {method}, Samples: {n_samples}, Threshold: {inflation_threshold})"
        visualize_shapley_heatmap(shapley_values, heatmap_path, title=title)
    
    # 生成累计风险图
    if args.plot_cumulative:
        cumulative_path = os.path.join(args.output_dir, 'cumulative_risk.png')
        title = f"Cumulative Risk Over Time\n(Baseline: {baseline_risk:.6f}, Real: {real_risk:.6f})"
        plot_cumulative_risk(shapley_values, baseline_risk, real_risk, cumulative_path, title=title)
    
    # 生成 agent 总归因柱状图
    if args.plot_agent_total:
        agent_total_path = os.path.join(args.output_dir, 'agent_total_attribution.png')
        title = f"Agent Total Attribution\n(Method: {method}, Samples: {n_samples})"
        plot_agent_total_attribution(shapley_values, agent_total_path, title=title)
    
    # 生成 baseline metrics 图
    if args.plot_baseline:
        baseline_path = os.path.join(args.output_dir, 'baseline_metrics.png')
        success = plot_baseline_metrics(args.shapley_values, baseline_path, 
                                       variable=args.baseline_variable, 
                                       threshold=args.baseline_threshold)
        if not success:
            print(f"Warning: Failed to generate baseline metrics plot. Check error messages above.")
    
    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

