#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制SocialLLM的风险曲线和累积Shapley值图

参考TwinMarket的batch_plot.py中的plot_risk_and_shapley函数

用法:
    python3 scripts/plot/plot_risk_and_shapley.py --result_dir results/deepseek-v3.2/deepseek-v3.2_47
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
from pathlib import Path


def load_risk_data(results_dir):
    """加载风险数据
    
    Args:
        results_dir: 结果目录路径
        
    Returns:
        risk_evolution: 风险演化曲线，形状为 (num_timesteps,)
        baseline_risk: baseline风险（从shapley_stats.json读取，如果不存在则使用initial_risk）
        real_risk: 最高风险（从shapley_stats.json读取，如果不存在则使用max_risk）
        max_risk_timestep: 最高风险时间步
    """
    results_file = Path(results_dir) / "results.json"
    
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    timestep_results = data.get('timestep_results', [])
    if not timestep_results:
        raise ValueError("timestep_results is empty")
    
    # 提取风险值
    risk_evolution = np.array([t['risk'] for t in timestep_results])
    
    # 尝试从shapley_stats.json读取baseline_risk和real_risk
    shapley_stats_file = Path(results_dir) / "shapley" / "shapley_stats.json"
    if shapley_stats_file.exists():
        with open(shapley_stats_file, 'r', encoding='utf-8') as f:
            shapley_stats = json.load(f)
        baseline_risk = shapley_stats.get('baseline_risk', data.get('initial_risk', 0.0))
        real_risk = shapley_stats.get('real_risk', data.get('max_risk', 0.0))
    else:
        # 如果shapley_stats.json不存在，使用results.json中的值
        baseline_risk = data.get('initial_risk', 0.0)
        real_risk = data.get('max_risk', 0.0)
    
    max_risk_timestep = data.get('max_risk_timestep', 0)
    
    return risk_evolution, baseline_risk, real_risk, max_risk_timestep


def load_shapley_data(results_dir):
    """加载Shapley数据
    
    Args:
        results_dir: 结果目录路径
        
    Returns:
        shapley_values: Shapley值数组，形状为 (num_agents, num_timesteps)
        time_aggregated: 按时间聚合的shapley值，形状为 (num_timesteps,)
    """
    shapley_file = Path(results_dir) / "shapley" / "shapley_attribution_timeseries.csv"
    
    if not shapley_file.exists():
        raise FileNotFoundError(f"Shapley file not found: {shapley_file}")
    
    df = pd.read_csv(shapley_file)
    
    # 确保有必要的列
    required_cols = ['agent_id', 'timestep', 'shapley_value']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in Shapley CSV")
    
    # 获取agent和timestep的数量
    num_agents = df['agent_id'].nunique()
    num_timesteps = df['timestep'].max() + 1  # timestep从0开始
    
    # 构建shapley_values矩阵 (num_agents, num_timesteps)
    shapley_values = np.zeros((num_agents, num_timesteps))
    for _, row in df.iterrows():
        agent_id = int(row['agent_id'])
        timestep = int(row['timestep'])
        shapley_value = float(row['shapley_value'])
        shapley_values[agent_id, timestep] = shapley_value
    
    # 计算按时间聚合的shapley值（每个timestep所有agent的shapley值之和）
    time_aggregated = np.sum(shapley_values, axis=0)
    
    return shapley_values, time_aggregated


def plot_risk_and_shapley(shapley_values, risk_evolution, time_aggregated,
                          baseline_risk, real_risk, output_path,
                          max_risk_timestep=None,
                          title="Risk Evolution and Cumulative Shapley Value"):
    """绘制图: 风险曲线和累积shapley value的双子图
    
    Args:
        shapley_values: Shapley值数组，形状为 (num_agents, num_timesteps)
        risk_evolution: 风险演化曲线，形状为 (num_timesteps,)
        time_aggregated: 按时间聚合的shapley值，形状为 (num_timesteps,)
        baseline_risk: 基线风险
        real_risk: 实际风险（最高风险）
        output_path: 输出路径
        max_risk_timestep: 风险最高点的时间步（0-based），如果提供则只画到此时间步
        title: 图表标题
    """
    num_agents, num_timesteps = shapley_values.shape
    
    # 计算累积shapley value
    # 注意：Shapley值总和等于 real_risk - baseline_risk
    # 为了在图上正确显示，我们需要从baseline_risk开始累积
    # 这样累积Shapley值最终应该等于 real_risk
    shapley_per_timestep = time_aggregated  # 每个时间步所有agent的shapley值之和
    cumulative_shapley_raw = np.cumsum(shapley_per_timestep)  # 从0开始累积
    cumulative_shapley = baseline_risk + cumulative_shapley_raw  # 从baseline_risk开始累积
    
    # 确定整个图的时间步范围（画到风险最高点）
    # 注意：Shapley数据只包含到max_risk_timestep-1的actions（因为target_timestep是max_risk_timestep）
    # 但是，风险数据包含到max_risk_timestep的风险值
    # 所以，我们需要：
    # 1. 风险数据画到max_risk_timestep+1（因为risk_evolution包含timestep 0到max_risk_timestep）
    # 2. 累积Shapley值应该画到max_risk_timestep（与风险数据的最后一个点对齐）
    if max_risk_timestep is not None:
        # 风险数据画到max_risk_timestep+1（因为max_risk_timestep是0-based，risk_evolution[0]是初始状态，risk_evolution[max_risk_timestep+1]是max_risk_timestep之后的状态）
        risk_plot_length = min(int(max_risk_timestep) + 1, len(risk_evolution))
        # 累积Shapley值应该画到max_risk_timestep（与风险最高点对齐）
        # Shapley数据只到max_risk_timestep-1，但我们需要在max_risk_timestep处添加一个点，其值等于real_risk
        shapley_plot_length = int(max_risk_timestep) + 1
    else:
        risk_plot_length = len(risk_evolution)
        shapley_plot_length = num_timesteps
    
    # 扩展累积Shapley值到max_risk_timestep
    # 如果Shapley数据只到max_risk_timestep-1，我们需要在max_risk_timestep处添加real_risk
    if shapley_plot_length > len(cumulative_shapley):
        # 在末尾添加real_risk，确保累积Shapley值最终等于real_risk
        cumulative_shapley_extended = np.append(cumulative_shapley, real_risk)
    else:
        cumulative_shapley_extended = cumulative_shapley[:shapley_plot_length]
    
    # 使用扩展后的cumulative shapley曲线
    cumulative_shapley_plot = cumulative_shapley_extended
    cumulative_timesteps = np.arange(shapley_plot_length)  # 0-based
    
    # 风险数据画到risk_plot_length
    risk_evolution_plot = risk_evolution[:risk_plot_length]
    risk_timesteps = np.arange(risk_plot_length)  # 0-based
    
    # Shapley数据画到shapley_plot_length（但time_aggregated只到max_risk_timestep-1）
    # 对于bar chart，我们只画到实际的Shapley数据范围
    actual_shapley_length = min(len(time_aggregated), shapley_plot_length)
    time_aggregated_plot = time_aggregated[:actual_shapley_length]
    
    # 创建图形，使用上下两个子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 上子图: 风险演化曲线和累积shapley value
    ax1.plot(risk_timesteps, risk_evolution_plot, 'r-', linewidth=2, marker='o', markersize=4,
             label='Risk Evolution', alpha=0.8)
    ax1.plot(cumulative_timesteps, cumulative_shapley_plot, 'b-', linewidth=2, marker='s', markersize=4,
             label='Cumulative Shapley Value (from baseline)', alpha=0.8)
    ax1.axhline(y=real_risk * 0.9, color='orange', linestyle='--', linewidth=1.5,
                label=f'90% Real Risk ({real_risk * 0.9:.6f})')
    ax1.axhline(y=real_risk, color='orange', linestyle='-', linewidth=1.5,
                label=f'Real Risk ({real_risk:.6f})')
    ax1.set_xlabel('Timestep', fontsize=12)
    ax1.set_ylabel('Risk / Cumulative Shapley Value', fontsize=12)
    ax1.set_title('Risk Evolution and Cumulative Shapley Value', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    # 设置x轴范围，确保两个曲线都能完整显示
    max_timestep = max(len(risk_timesteps), len(cumulative_timesteps))
    if max_timestep > 0:
        ax1.set_xlim(-0.5, max_timestep - 0.5)
        ax1.set_xticks(np.arange(0, max_timestep, max(1, max_timestep//10)))  # 只显示部分刻度
    ax1.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 下子图: 按时间聚合的shapley value
    shapley_timesteps = np.arange(len(time_aggregated_plot))  # 0-based，只画到实际数据范围
    colors = ['#EF4444' if x >= 0 else '#3B82F6' for x in time_aggregated_plot]
    ax2.bar(shapley_timesteps, time_aggregated_plot, alpha=0.6, color=colors,
            edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('Time Aggregated Shapley Value', fontsize=12)
    ax2.set_title('Time Aggregated Shapley Value', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    if len(shapley_timesteps) > 0:
        # 设置x轴范围，使其与上子图对齐（都画到max_risk_timestep）
        ax2.set_xlim(-0.5, shapley_plot_length - 0.5)
        ax2.set_xticks(np.arange(0, shapley_plot_length, max(1, shapley_plot_length//10)))  # 只显示部分刻度
    ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 添加图例说明（仅在下子图）
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#EF4444', alpha=0.6, label='Positive'),
        Patch(facecolor='#3B82F6', alpha=0.6, label='Negative')
    ]
    ax2.legend(handles=legend_elements, fontsize=10, loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Plot risk evolution and cumulative Shapley value for SocialLLM'
    )
    parser.add_argument(
        '--result_dir',
        type=str,
        required=True,
        help='Result directory path (e.g., results/gpt-4o-mini/gpt-4o-mini_42)'
    )
    
    args = parser.parse_args()
    
    # 解析结果目录路径
    result_dir = Path(args.result_dir)
    if not result_dir.is_absolute():
        # 假设相对于项目根目录
        script_dir = Path(__file__).parent.parent.parent
        result_dir = script_dir / result_dir
    
    result_dir = result_dir.resolve()
    
    if not result_dir.exists():
        print(f"Error: Result directory not found: {result_dir}", file=os.sys.stderr)
        os.sys.exit(1)
    
    if not result_dir.is_dir():
        print(f"Error: Path is not a directory: {result_dir}", file=os.sys.stderr)
        os.sys.exit(1)
    
    print(f"Processing result directory: {result_dir}")
    print("=" * 60)
    
    try:
        # 加载数据
        print("Loading risk data...")
        risk_evolution, baseline_risk, real_risk, max_risk_timestep = load_risk_data(result_dir)
        
        print("Loading Shapley data...")
        shapley_values, time_aggregated = load_shapley_data(result_dir)
        
        print(f"  Risk evolution length: {len(risk_evolution)}")
        print(f"  Shapley values shape: {shapley_values.shape}")
        print(f"  Baseline risk: {baseline_risk:.6f}")
        print(f"  Real risk: {real_risk:.6f}")
        print(f"  Max risk timestep: {max_risk_timestep}")
        
        # 创建输出目录
        output_dir = result_dir / "plot"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 绘制图形
        output_path = output_dir / "risk_and_cumulative_shapley.png"
        print(f"\nPlotting...")
        plot_risk_and_shapley(
            shapley_values, risk_evolution, time_aggregated,
            baseline_risk, real_risk, output_path,
            max_risk_timestep=max_risk_timestep
        )
        
        print("\nDone!")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=os.sys.stderr)
        import traceback
        traceback.print_exc()
        os.sys.exit(1)


if __name__ == '__main__':
    main()
