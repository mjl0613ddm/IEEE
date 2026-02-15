#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量绘图脚本
为ACL24-EconAgent项目的每个模型和种子生成四张图表
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


def load_shapley_data(shapley_dir):
    """加载Shapley数据.
    
    Args:
        shapley_dir: shapley目录路径
        
    Returns:
        shapley_values: Shapley值数组，形状为 (num_agents, episode_length)
        stats: 统计信息字典
    """
    shapley_values_path = os.path.join(shapley_dir, 'shapley_values.npy')
    stats_json_path = os.path.join(shapley_dir, 'shapley_stats.json')
    
    if not os.path.exists(shapley_values_path):
        raise FileNotFoundError(f"Shapley values file not found: {shapley_values_path}")
    if not os.path.exists(stats_json_path):
        raise FileNotFoundError(f"Shapley stats file not found: {stats_json_path}")
    
    shapley_values = np.load(shapley_values_path)
    
    with open(stats_json_path, 'r') as f:
        stats = json.load(f)
    
    return shapley_values, stats


def calculate_risk_indicator_naive(df, lambda_param=0.94):
    """
    计算风险指标（基于Engle (1982)和Bollerslev (1986)）
    使用naive forecast方法：E_{t-1}[π_t] = π_{t-1}
    
    Args:
        df: 包含price列的DataFrame
        lambda_param: RiskMetrics参数λ，默认0.94
    
    Returns:
        risk_values数组
    """
    if 'price' not in df.columns:
        raise ValueError("CSV文件中缺少 'price' 列，无法计算风险指标")
    
    # 计算通胀率 π_t = log P_t - log P_{t-1}
    prices = df['price'].values
    log_prices = np.log(prices)
    pi_t = np.diff(log_prices)  # π_t = log P_t - log P_{t-1}
    
    # 在开头插入NaN以保持长度一致（第一个时间步没有前一个价格）
    pi_t = np.insert(pi_t, 0, np.nan)
    
    n = len(pi_t)
    
    # 初始化数组
    E_pi = np.full(n, np.nan)  # 预期通胀率
    e_t = np.full(n, np.nan)   # 预测误差
    h_t = np.full(n, np.nan)   # 风险指标
    
    # 计算预期和误差（naive forecast方法）
    for t in range(1, n):
        # Naive forecast: E_{t-1}[π_t] = π_{t-1}
        E_pi[t] = pi_t[t-1]
        
        # 计算预测误差 e_t = π_t - E_{t-1}[π_t]
        if not np.isnan(E_pi[t]) and not np.isnan(pi_t[t]):
            e_t[t] = pi_t[t] - E_pi[t]
    
    # 计算风险指标 h_t = λ * h_{t-1} + (1-λ) * e_{t-1}^2
    # 找到第一个有效的 e_t
    first_valid_idx = None
    for i in range(1, n):
        if not np.isnan(e_t[i]):
            first_valid_idx = i
            break
    
    if first_valid_idx is not None:
        # 初始化：h_t[first_valid_idx] = e_t[first_valid_idx]^2
        h_t[first_valid_idx] = e_t[first_valid_idx] ** 2
        
        # 递归计算：h_t = λ * h_{t-1} + (1-λ) * e_{t-1}^2
        for t in range(first_valid_idx + 1, n):
            if not np.isnan(e_t[t-1]) and not np.isnan(h_t[t-1]):
                h_t[t] = lambda_param * h_t[t-1] + (1 - lambda_param) * (e_t[t-1] ** 2)
    
    return h_t


def load_risk_features_data(risk_features_dir, world_metrics_path):
    """加载Risk features数据.
    
    Args:
        risk_features_dir: risk_features目录路径
        world_metrics_path: world_metrics.csv文件路径
        
    Returns:
        agent_aggregated: 按agent聚合的shapley值，形状为 (num_agents,)
        time_aggregated: 按时间聚合的shapley值，形状为 (episode_length,)
        behaviour_aggregated: 按behavior聚合的shapley值
        risk_evolution: 风险演化曲线（从world_metrics.csv读取），形状为 (episode_length,)
        risk_features: 风险特征指标字典
    """
    agent_aggregated_path = os.path.join(risk_features_dir, 'agent_aggregated.npy')
    time_aggregated_path = os.path.join(risk_features_dir, 'time_aggregated.npy')
    behaviour_aggregated_path = os.path.join(risk_features_dir, 'behaviour_aggregated.npy')
    risk_features_json_path = os.path.join(risk_features_dir, 'risk_features.json')
    
    # 检查文件是否存在
    required_files = [
        agent_aggregated_path, time_aggregated_path, behaviour_aggregated_path,
        risk_features_json_path, world_metrics_path
    ]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    agent_aggregated = np.load(agent_aggregated_path)
    time_aggregated = np.load(time_aggregated_path)
    behaviour_aggregated = np.load(behaviour_aggregated_path)
    
    # 从world_metrics.csv读取或计算risk_indicator_naive作为risk_evolution
    df = pd.read_csv(world_metrics_path)
    if 'risk_indicator_naive' in df.columns:
        # 如果CSV中已有risk_indicator_naive列，直接使用
        risk_evolution = df['risk_indicator_naive'].values
    else:
        # 如果CSV中没有该列，根据price列计算
        risk_evolution = calculate_risk_indicator_naive(df, lambda_param=0.94)
        # 处理NaN值，将NaN替换为0
        risk_evolution = np.nan_to_num(risk_evolution, nan=0.0)
    
    with open(risk_features_json_path, 'r') as f:
        risk_features = json.load(f)
    
    return agent_aggregated, time_aggregated, behaviour_aggregated, risk_evolution, risk_features


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
        behaviour_aggregated: 按behavior聚合的shapley值
        output_path: 输出路径
        title: 图表标题
    """
    num_behaviors = len(behaviour_aggregated)
    
    # 创建图形（增加宽度以容纳behavior名称）
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
    
    # Behavior名称定义
    ACTION_CATEGORY_NAMES = [
        'work=0, cons=0-25%',   # 类型0
        'work=0, cons=25-50%',  # 类型1
        'work=0, cons=50-75%',  # 类型2
        'work=0, cons=75-100%', # 类型3
        'work=1, cons=0-25%',   # 类型4
        'work=1, cons=25-50%',  # 类型5
        'work=1, cons=50-75%',  # 类型6
        'work=1, cons=75-100%', # 类型7
    ]
    
    # 设置标签和标题
    ax.set_xlabel('Behavior', fontsize=12)
    ax.set_ylabel('Aggregated Shapley Value', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    # 设置 x 轴刻度，使用behavior名称
    ax.set_xticks(behavior_ids)
    behavior_labels = [ACTION_CATEGORY_NAMES[i] if i < len(ACTION_CATEGORY_NAMES) else f'Behavior {i}' 
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
    shapley_dir = os.path.join(base_path, 'datas', model_name, seed_name, 'shapley')
    risk_features_dir = os.path.join(base_path, 'results', 'risk_features_exp', model_name, seed_name)
    world_metrics_path = os.path.join(base_path, 'datas', model_name, seed_name, 'metrics_csv', 'world_metrics.csv')
    output_dir = os.path.join(base_path, 'datas', model_name, seed_name, 'plot')
    
    # 检查必要目录是否存在
    if not os.path.exists(shapley_dir):
        print(f"Warning: Shapley directory not found: {shapley_dir}, skipping...")
        return False
    
    if not os.path.exists(risk_features_dir):
        print(f"Warning: Risk features directory not found: {risk_features_dir}, skipping...")
        return False
    
    if not os.path.exists(world_metrics_path):
        print(f"Warning: World metrics file not found: {world_metrics_path}, skipping...")
        return False
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 加载数据
        shapley_values, stats = load_shapley_data(shapley_dir)
        agent_aggregated, time_aggregated, behaviour_aggregated, risk_evolution, risk_features = \
            load_risk_features_data(risk_features_dir, world_metrics_path)
        
        baseline_risk = stats.get('baseline_risk', 0.0)
        real_risk = stats.get('real_risk', 0.0)
        max_risk_timestep = stats.get('max_risk_timestep', None)  # 获取风险最高点时间步
        
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
    parser = argparse.ArgumentParser(description='Batch plot script for ACL24-EconAgent')
    parser.add_argument('--base_path', type=str, 
                       default='/mnt/shared-storage-user/meijilin/Economic_System_Attribution/ACL24-EconAgent',
                       help='Base path of the project (default: /mnt/shared-storage-user/meijilin/Economic_System_Attribution/ACL24-EconAgent)')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='List of models to process (default: all models in datas/)')
    parser.add_argument('--seeds', type=str, nargs='+', default=None,
                       help='List of seeds to process (default: all seeds excluding *_rm)')
    
    args = parser.parse_args()
    
    base_path = args.base_path
    datas_dir = os.path.join(base_path, 'datas')
    
    if not os.path.exists(datas_dir):
        print(f"Error: Datas directory not found: {datas_dir}")
        return
    
    # 获取所有模型
    if args.models:
        models = args.models
    else:
        models = [d for d in os.listdir(datas_dir) 
                 if os.path.isdir(os.path.join(datas_dir, d))]
    
    print(f"Processing models: {models}")
    
    total_processed = 0
    total_success = 0
    
    # 遍历每个模型
    for model_name in models:
        model_dir = os.path.join(datas_dir, model_name)
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