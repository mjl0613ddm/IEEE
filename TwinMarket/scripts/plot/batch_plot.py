#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量绘图脚本
为TwinMarket项目的每个模型和种子生成四张图表
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


def find_latest_shapley_files(shapley_dir):
    """查找shapley目录中最新的shapley文件.
    
    Args:
        shapley_dir: shapley目录路径
        
    Returns:
        shapley_matrix_path: shapley_matrix文件路径
        stats_path: shapley_stats文件路径
        date_range: 日期范围字符串（如 "2023-06-20_2023-08-02"）
    """
    shapley_dir = Path(shapley_dir)
    
    # 查找所有shapley_matrix文件（TwinMarket使用shapley_matrix_*.npy）
    shapley_files = list(shapley_dir.glob("shapley_matrix_*.npy"))
    if not shapley_files:
        raise FileNotFoundError(f"No shapley_matrix file found in {shapley_dir}")
    
    # 按文件名排序，选择最新的（假设文件名包含日期范围）
    shapley_files.sort(reverse=True)
    latest_shapley_file = shapley_files[0]
    
    # 从文件名提取日期范围
    # 格式: shapley_matrix_2023-06-20_2023-08-02.npy
    filename = latest_shapley_file.stem
    if '_' in filename:
        parts = filename.split('_')
        if len(parts) >= 4:
            date_range = f"{parts[2]}_{parts[3]}"
        else:
            date_range = None
    else:
        date_range = None
    
    # 构建对应的stats文件路径
    if date_range:
        stats_path = shapley_dir / f"shapley_stats_{date_range}.json"
    else:
        stats_path = None
    
    if stats_path and not stats_path.exists():
        raise FileNotFoundError(f"Shapley stats file not found: {stats_path}")
    
    return str(latest_shapley_file), str(stats_path) if stats_path else None, date_range


def load_shapley_data(shapley_dir):
    """加载Shapley数据.
    
    Args:
        shapley_dir: shapley目录路径
        
    Returns:
        shapley_values: Shapley值数组，形状为 (num_agents, episode_length)
        stats: 统计信息字典
    """
    shapley_matrix_path, stats_path, date_range = find_latest_shapley_files(shapley_dir)
    
    if not os.path.exists(shapley_matrix_path):
        raise FileNotFoundError(f"Shapley matrix file not found: {shapley_matrix_path}")
    if not stats_path or not os.path.exists(stats_path):
        raise FileNotFoundError(f"Shapley stats file not found: {stats_path}")
    
    # TwinMarket使用shapley_matrix_*.npy文件，可以直接加载（不需要allow_pickle）
    shapley_values = np.load(shapley_matrix_path)
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    return shapley_values, stats


def load_risk_features_data(risk_features_dir, market_metrics_path, date_range=None):
    """加载Risk features数据.
    
    Args:
        risk_features_dir: risk_features目录路径
        market_metrics_path: market_metrics.csv文件路径
        date_range: 日期范围字符串（如 "2023-06-19_2023-07-03"），如果提供则只使用该范围内的数据
        
    Returns:
        agent_aggregated: 按agent聚合的shapley值，形状为 (num_agents,)
        time_aggregated: 按时间聚合的shapley值，形状为 (episode_length,)
        behaviour_aggregated: 按behavior聚合的shapley值
        risk_evolution: 风险演化曲线（从market_metrics.csv读取risk_indicator_simple），形状为 (episode_length,)
        risk_features: 风险特征指标字典
    """
    agent_aggregated_path = os.path.join(risk_features_dir, 'agent_aggregated.npy')
    time_aggregated_path = os.path.join(risk_features_dir, 'time_aggregated.npy')
    behaviour_aggregated_path = os.path.join(risk_features_dir, 'behaviour_aggregated.npy')
    risk_features_json_path = os.path.join(risk_features_dir, 'risk_features.json')
    
    # 检查文件是否存在
    required_files = [
        agent_aggregated_path, time_aggregated_path, behaviour_aggregated_path,
        risk_features_json_path, market_metrics_path
    ]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    agent_aggregated = np.load(agent_aggregated_path)
    time_aggregated = np.load(time_aggregated_path)
    behaviour_aggregated = np.load(behaviour_aggregated_path)
    
    # 从market_metrics.csv读取risk_indicator_simple作为risk_evolution
    df = pd.read_csv(market_metrics_path)
    if 'risk_indicator_simple' not in df.columns:
        raise ValueError(f"Column 'risk_indicator_simple' not found in {market_metrics_path}")
    
    # 按照date排序（和plot_risk_metrics.py保持一致）
    df = df.sort_values('date')
    
    # 如果提供了date_range，只使用该范围内的数据（对齐shapley的日期范围）
    if date_range and '_' in date_range:
        start_date, end_date = date_range.split('_')
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    
    # 只保留有risk_indicator_simple值的行（交易日），因为shapley_values的时间步对应的是交易日
    # market_metrics.csv包含所有日期（包括非交易日），但只有交易日有risk_indicator_simple值
    df_valid = df[df['risk_indicator_simple'].notna()].copy()
    risk_evolution = df_valid['risk_indicator_simple'].values
    # 保持NaN值，不替换为0，这样前几步没有值的地方会显示为空（和risk_indicator_simple曲线一样）
    # 注意：这里已经过滤了NaN，所以risk_evolution只包含有值的交易日
    
    with open(risk_features_json_path, 'r') as f:
        risk_features = json.load(f)
    
    return agent_aggregated, time_aggregated, behaviour_aggregated, risk_evolution, risk_features


def find_max_risk_timestep(market_metrics_path):
    """从market_metrics.csv中找到risk_indicator_simple最大值对应的时间步.
    
    Args:
        market_metrics_path: market_metrics.csv文件路径
        
    Returns:
        max_risk_timestep: 最大风险时间步（1-based，基于原始DataFrame的索引），如果找不到则返回None
    """
    try:
        df = pd.read_csv(market_metrics_path)
        if 'risk_indicator_simple' not in df.columns:
            return None
        
        # 找到最大值对应的原始索引（0-based）
        # 使用idxmax()获取最大值对应的原始索引
        max_idx = df['risk_indicator_simple'].idxmax()
        
        # 如果所有值都是NaN，idxmax()会返回NaN
        if pd.isna(max_idx):
            return None
        
        # 转换为1-based的时间步（基于原始DataFrame的索引）
        max_risk_timestep = int(max_idx) + 1
        
        return max_risk_timestep
    except Exception as e:
        print(f"Warning: Failed to find max_risk_timestep from {market_metrics_path}: {e}")
        return None


def plot_risk_and_shapley(shapley_values, risk_evolution, time_aggregated, 
                          baseline_risk, real_risk, output_path, 
                          max_risk_timestep=None,
                          title="Risk Evolution and Cumulative Shapley Value"):
    """绘制图1: 风险曲线和累积shapley value的双子图.
    
    Args:
        shapley_values: Shapley值数组，形状为 (num_agents, episode_length)
        risk_evolution: 风险演化曲线，形状为 (episode_length,)
        time_aggregated: 按时间聚合的shapley值，形状为 (episode_length,)
        baseline_risk: 基线风险
        real_risk: 实际风险
        output_path: 输出路径
        max_risk_timestep: 风险最高点的时间步（1-based），如果提供则只画到此时间步
        title: 图表标题
    """
    num_agents, episode_length = shapley_values.shape
    
    # 计算累积shapley value（不包含baseline_risk）
    shapley_per_timestep = np.sum(shapley_values, axis=0)  # 每个时间步所有agent的shapley值之和
    cumulative_shapley = np.cumsum(shapley_per_timestep)
    
    # 找到cumulative shapley value第一次超过90% real_risk的时间步（用于截断cumulative shapley曲线）
    # 注意：cumulative_shapley不包含baseline，但阈值应该基于real_risk的90%
    # 因为我们要看cumulative shapley value何时超过real_risk的90%
    threshold_90 = real_risk * 0.9
    cumulative_truncate_idx = None
    for i in range(len(cumulative_shapley)):
        if cumulative_shapley[i] > threshold_90:
            cumulative_truncate_idx = i + 1  # 保留到超过90%的那个时间步（包含该时间步，i是0-based，i+1是1-based）
            break
    
    # 确定整个图的时间步范围（画到风险最高点）
    if max_risk_timestep is not None:
        plot_length = min(int(max_risk_timestep), episode_length)
    else:
        plot_length = episode_length
    
    # 截断cumulative shapley value曲线（如果超过90%）
    if cumulative_truncate_idx is not None:
        cumulative_shapley_plot = cumulative_shapley[:cumulative_truncate_idx]
        cumulative_timesteps = np.arange(1, cumulative_truncate_idx + 1)
    else:
        cumulative_shapley_plot = cumulative_shapley[:plot_length]
        cumulative_timesteps = np.arange(1, plot_length + 1)
    
    # 其他曲线画到plot_length
    risk_evolution_plot = risk_evolution[:plot_length] if len(risk_evolution) >= plot_length else risk_evolution
    time_aggregated_plot = time_aggregated[:plot_length] if len(time_aggregated) >= plot_length else time_aggregated
    
    # 创建图形，使用上下两个子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    timesteps = np.arange(1, plot_length + 1)
    
    # 上子图: 风险演化曲线和累积shapley value
    ax1.plot(timesteps, risk_evolution_plot, 'r-', linewidth=2, marker='o', markersize=4, 
             label='Risk Evolution', alpha=0.8)
    ax1.plot(cumulative_timesteps, cumulative_shapley_plot, 'b-', linewidth=2, marker='s', markersize=4, 
             label='Cumulative Shapley Value', alpha=0.8)
    ax1.axhline(y=real_risk * 0.9, color='orange', linestyle='--', linewidth=1.5, 
                label=f'90% Real Risk ({real_risk * 0.9:.6f})')
    ax1.axhline(y=real_risk, color='orange', linestyle='-', linewidth=1.5, 
                label=f'Real Risk ({real_risk:.6f})')
    ax1.set_xlabel('Timestep', fontsize=12)
    ax1.set_ylabel('Risk / Cumulative Shapley Value', fontsize=12)
    ax1.set_title('Risk Evolution and Cumulative Shapley Value', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(timesteps[::max(1, len(timesteps)//10)])  # 只显示部分刻度
    ax1.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 下子图: 按时间聚合的shapley value
    colors = ['#EF4444' if x >= 0 else '#3B82F6' for x in time_aggregated_plot]
    ax2.bar(timesteps, time_aggregated_plot, alpha=0.6, color=colors, 
            edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('Time Aggregated Shapley Value', fontsize=12)
    ax2.set_title('Time Aggregated Shapley Value', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax2.set_xticks(timesteps[::max(1, len(timesteps)//10)])  # 只显示部分刻度
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
    print(f"Plot 1 saved to: {output_path}")
    plt.close()


def plot_shapley_instability_scatter(shapley_values, agent_aggregated, 
                                     output_path, title="Shapley Value vs Instability (C_ag)"):
    """绘制图2: Agents的shapley绝对值与instability散点图.
    
    Args:
        shapley_values: Shapley值数组，形状为 (num_agents, episode_length)
        agent_aggregated: 按agent聚合的shapley值，形状为 (num_agents,)
        output_path: 输出路径
        title: 图表标题
    """
    num_agents, episode_length = shapley_values.shape
    
    # 计算每个agent的shapley value绝对值
    abs_shapley = np.abs(agent_aggregated)
    
    # 计算每个agent的instability（时间维度上的标准差）
    instability = np.std(shapley_values, axis=1)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制散点图
    scatter = ax.scatter(abs_shapley, instability, s=100, alpha=0.6, 
                        c=range(num_agents), cmap='viridis', edgecolors='black', linewidths=1)
    
    # 添加每个点的标签（agent ID）
    for i in range(num_agents):
        ax.annotate(f'Agent {i}', (abs_shapley[i], instability[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # 设置标签和标题
    ax.set_xlabel('Absolute Shapley Value', fontsize=12)
    ax.set_ylabel('Instability (Std Dev)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 添加colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Agent ID', fontsize=10, rotation=270, labelpad=15)
    
    # 使用科学计数法格式化坐标轴
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot 2 saved to: {output_path}")
    plt.close()


def plot_agent_aggregated(agent_aggregated, output_path, title="Agent Aggregated Shapley Value"):
    """绘制图3: 按agent聚合的Shapley value柱状图.
    
    Args:
        agent_aggregated: 按agent聚合的shapley值，形状为 (num_agents,)
        output_path: 输出路径
        title: 图表标题
    """
    num_agents = len(agent_aggregated)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(max(10, num_agents * 0.8), 6))
    
    # 准备数据
    agent_ids = np.arange(num_agents)
    
    # 根据值的正负选择颜色：正值用红色系，负值用蓝色系
    colors = ['#EF4444' if x >= 0 else '#3B82F6' for x in agent_aggregated]
    
    # 绘制柱状图
    bars = ax.bar(agent_ids, agent_aggregated, color=colors, alpha=0.7, 
                  edgecolor='black', linewidth=0.5)
    
    # 添加零线
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # 设置标签和标题
    ax.set_xlabel('Agent ID', fontsize=12)
    ax.set_ylabel('Aggregated Shapley Value', fontsize=12)
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
    
    # 使用科学计数法格式化y轴
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot 3 saved to: {output_path}")
    plt.close()


def plot_behavior_aggregated(behaviour_aggregated, output_path, title="Behavior Aggregated Shapley Value"):
    """绘制图4: 按behavior聚合的柱状图.
    
    Args:
        behaviour_aggregated: 按behavior聚合的shapley值（10种股票）
        output_path: 输出路径
        title: 图表标题
    """
    num_behaviors = len(behaviour_aggregated)
    
    # 创建图形（增加宽度以容纳股票代码）
    fig, ax = plt.subplots(figsize=(max(12, num_behaviors * 1.2), 6))
    
    # 准备数据
    behavior_ids = np.arange(num_behaviors)
    
    # 根据值的正负选择颜色：正值用红色系，负值用蓝色系
    colors = ['#EF4444' if x >= 0 else '#3B82F6' for x in behaviour_aggregated]
    
    # 绘制柱状图
    bars = ax.bar(behavior_ids, behaviour_aggregated, color=colors, alpha=0.7, 
                  edgecolor='black', linewidth=0.5)
    
    # 添加零线
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # TwinMarket的10种股票代码
    STOCK_LIST = ['TLEI', 'MEI', 'CPEI', 'IEEI', 'REEI', 'TSEI', 'CGEI', 'TTEI', 'EREI', 'FSEI']
    
    # 设置标签和标题
    ax.set_xlabel('Stock', fontsize=12)
    ax.set_ylabel('Aggregated Shapley Value', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    # 设置 x 轴刻度，使用股票代码
    ax.set_xticks(behavior_ids)
    behavior_labels = [STOCK_LIST[i] if i < len(STOCK_LIST) else f'Stock {i}' 
                       for i in behavior_ids]
    ax.set_xticklabels(behavior_labels, fontsize=9, rotation=45, ha='right')
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # 添加图例说明
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#EF4444', alpha=0.7, label='Positive Contribution'),
        Patch(facecolor='#3B82F6', alpha=0.7, label='Negative Contribution')
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='best')
    
    # 使用科学计数法格式化y轴
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot 4 saved to: {output_path}")
    plt.close()


def process_model_seed(base_path, model_name, seed_name):
    """处理单个模型-种子组合.
    
    Args:
        base_path: 项目根路径
        model_name: 模型名称
        seed_name: 种子名称
    """
    # 构建路径
    shapley_dir = os.path.join(base_path, 'results', model_name, seed_name, 'shapley')
    risk_features_dir = os.path.join(base_path, 'results', 'risk_feature', model_name, seed_name)
    market_metrics_path = os.path.join(base_path, 'results', model_name, seed_name, 'analysis', 'market_metrics.csv')
    output_dir = os.path.join(base_path, 'results', model_name, seed_name, 'plot')
    
    # 检查必要目录是否存在
    if not os.path.exists(shapley_dir):
        print(f"Warning: Shapley directory not found: {shapley_dir}, skipping...")
        return False
    
    if not os.path.exists(risk_features_dir):
        print(f"Warning: Risk features directory not found: {risk_features_dir}, skipping...")
        return False
    
    if not os.path.exists(market_metrics_path):
        print(f"Warning: Market metrics file not found: {market_metrics_path}, skipping...")
        return False
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 加载数据
        shapley_values, stats = load_shapley_data(shapley_dir)
        
        # 获取shapley的日期范围，用于对齐risk_evolution
        date_range = stats.get('date_range', None)
        
        agent_aggregated, time_aggregated, behaviour_aggregated, risk_evolution, risk_features = \
            load_risk_features_data(risk_features_dir, market_metrics_path, date_range=date_range)
        
        # TwinMarket使用baseline_metric和real_metric（而非baseline_risk和real_risk）
        baseline_risk = stats.get('baseline_metric', 0.0)
        
        # 从market_metrics.csv计算max_risk_timestep
        max_risk_timestep = find_max_risk_timestep(market_metrics_path)
        
        # real_risk应该等于risk_evolution的最后一个值（目标日期end_date的risk_indicator_simple值）
        # 这样在图表上，红色的risk_evolution曲线能够到达黄色的real_risk线
        # 使用risk_evolution的最后一个值作为real_risk，确保它们在图表上对齐
        if len(risk_evolution) > 0:
            real_risk = float(risk_evolution[-1])
        else:
            # 如果risk_evolution为空，使用stats中的real_metric作为后备
            real_risk = stats.get('real_metric', 0.0)
        
        # 绘制四张图
        # 图1: 风险曲线和累积shapley value的双子图
        plot1_path = os.path.join(output_dir, 'risk_and_cumulative_shapley.png')
        plot_risk_and_shapley(shapley_values, risk_evolution, time_aggregated, 
                             baseline_risk, real_risk, plot1_path,
                             max_risk_timestep=max_risk_timestep)
        
        # 图2: Agents的shapley绝对值与instability散点图
        plot2_path = os.path.join(output_dir, 'shapley_instability_scatter.png')
        plot_shapley_instability_scatter(shapley_values, agent_aggregated, plot2_path)
        
        # 图3: 按agent聚合的Shapley value柱状图
        plot3_path = os.path.join(output_dir, 'agent_aggregated.png')
        plot_agent_aggregated(agent_aggregated, plot3_path)
        
        # 图4: 按behavior聚合的柱状图
        plot4_path = os.path.join(output_dir, 'behavior_aggregated.png')
        plot_behavior_aggregated(behaviour_aggregated, plot4_path)
        
        print(f"Successfully processed {model_name}/{seed_name}")
        return True
        
    except Exception as e:
        print(f"Error processing {model_name}/{seed_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数."""
    parser = argparse.ArgumentParser(description='Batch plot script for TwinMarket')
    parser.add_argument('--base_path', type=str, 
                       default='/mnt/shared-storage-user/meijilin/Economic_System_Attribution/TwinMarket',
                       help='Base path of the project (default: /mnt/shared-storage-user/meijilin/Economic_System_Attribution/TwinMarket)')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='List of models to process (default: all models in results/)')
    parser.add_argument('--seeds', type=str, nargs='+', default=None,
                       help='List of seeds to process (default: all seeds excluding *_rm)')
    
    args = parser.parse_args()
    
    base_path = args.base_path
    results_dir = os.path.join(base_path, 'results')
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    # 获取所有模型
    if args.models:
        models = args.models
    else:
        models = [d for d in os.listdir(results_dir) 
                 if os.path.isdir(os.path.join(results_dir, d)) 
                 and d != 'risk_feature' 
                 and d != 'accuracy' 
                 and d != 'faithfulness_exp'
                 and not d.startswith('logs_')
                 and not d.startswith('shapley_error')
                 and not d.startswith('risk_timestep')]
    
    print(f"Processing models: {models}")
    
    total_processed = 0
    total_success = 0
    
    # 遍历每个模型
    for model_name in models:
        model_dir = os.path.join(results_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
        
        # 获取所有种子（排除*_rm后缀的）
        if args.seeds:
            seeds = args.seeds
        else:
            seeds = [d for d in os.listdir(model_dir) 
                    if os.path.isdir(os.path.join(model_dir, d)) and not d.endswith('_rm')]
        
        print(f"  Processing {len(seeds)} seeds for model {model_name}")
        
        # 遍历每个种子
        for seed_name in seeds:
            seed_dir = os.path.join(model_dir, seed_name)
            if not os.path.isdir(seed_dir):
                continue
            
            total_processed += 1
            print(f"\nProcessing {model_name}/{seed_name}...")
            
            success = process_model_seed(base_path, model_name, seed_name)
            if success:
                total_success += 1
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total processed: {total_processed}")
    print(f"  Successful: {total_success}")
    print(f"  Failed: {total_processed - total_success}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()