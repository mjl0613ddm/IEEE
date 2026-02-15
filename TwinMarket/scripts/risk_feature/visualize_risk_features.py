#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Risk Feature可视化脚本
包括：
1. L_tm（相对风险延迟）指标的可视化（单模型和汇总对比）
2. 动作聚类风险（G_be指标，按股票类别）的可视化（单模型汇总和模型对比）
3. 单个seed的风险累积过程可视化
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
import seaborn as sns

# 设置样式
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# 10种股票的固定顺序
STOCK_LIST = ['TLEI', 'MEI', 'CPEI', 'IEEI', 'REEI', 'TSEI', 'CGEI', 'TTEI', 'EREI', 'FSEI']

# 设置项目根目录
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # TwinMarket
RESULTS_ROOT = PROJECT_ROOT / "results"
RISK_FEATURE_ROOT = RESULTS_ROOT / "risk_feature"


def visualize_risk_accumulation_by_seed(seed_dir: Path, output_path: Path, model_name: str, seed_name: str):
    """
    单个seed的风险累积过程可视化
    
    显示从第0步开始到风险步，曲线显示风险累积过程，并加上阈值线
    
    Args:
        seed_dir: seed结果目录（包含attribution_matrix.npy或risk_evolution.npy）
        output_path: 输出图片路径
        model_name: 模型名称
        seed_name: seed名称
    """
    # 尝试加载attribution_matrix.npy或risk_evolution.npy
    attribution_matrix_path = seed_dir / "attribution_matrix.npy"
    risk_evolution_path = seed_dir / "risk_evolution.npy"
    
    if attribution_matrix_path.exists():
        # 使用attribution_matrix，需要按日期聚合
        attribution_matrix = np.load(attribution_matrix_path)
        # 计算时间Shapley值（沿agent维度求和）
        time_shapley = np.sum(attribution_matrix, axis=0)
    elif risk_evolution_path.exists():
        # 直接使用risk_evolution（已经按日期聚合）
        time_shapley = np.load(risk_evolution_path)
    else:
        print(f"Warning: No attribution_matrix.npy or risk_evolution.npy found in {seed_dir}")
        return
    
    T = len(time_shapley)
    if T == 0:
        print(f"Warning: Empty time_shapley array in {seed_dir}")
        return
    
    # 计算总风险（直接和，不使用绝对值）
    rho = np.sum(time_shapley)
    
    # 如果总风险为0，返回
    if abs(rho) < 1e-10:
        print(f"Warning: Total risk rho is near zero in {seed_dir}")
        return
    
    # 计算累积和
    cumulative = np.cumsum(time_shapley)
    
    # 计算阈值（90%）
    q = 0.9
    threshold = q * rho
    
    # 找到第一次累积到90%的时刻：T* = min{t' | cumulative[t'] > 0.9 * rho}
    T_star = None
    for t in range(T):
        if cumulative[t] > threshold:
            T_star = t + 1  # 转换为1-based索引（公式中的t'）
            break
    
    # 如果从未达到90%，使用T
    if T_star is None:
        T_star = T
    
    # 计算L_tm
    L_tm = (T - T_star) / T
    
    # 读取risk_features.json获取L_tm值进行验证
    risk_features_path = seed_dir / "risk_features.json"
    reported_L_tm = None
    if risk_features_path.exists():
        with open(risk_features_path, 'r') as f:
            risk_features = json.load(f)
            reported_L_tm = risk_features.get('L_tm')
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 时间步（从第0步开始到T步）
    timesteps = np.arange(0, T + 1)  # 从0到T，共T+1个点
    
    # 累积风险曲线（第0步为0，第1步开始是cumulative值）
    cumulative_plot = np.concatenate([[0], cumulative])  # 第0步累积为0
    
    # 绘制累积风险曲线
    ax.plot(timesteps, cumulative_plot, 'b-', linewidth=2, marker='o', markersize=4,
            label=f'Cumulative Risk (L_tm={L_tm:.3f})', zorder=3)
    
    # 绘制阈值线
    ax.axhline(y=threshold, color='r', linestyle='--', linewidth=2,
               label=f'Threshold (90% × ρ = {threshold:.3f})', zorder=2)
    
    # 标记T*时刻（第一次超过阈值的点，注意T*是1-based，对应timesteps中的索引）
    if T_star <= T:
        # T*是1-based（第T*个时间步），在timesteps中的索引也是T*
        ax.axvline(x=T_star, color='g', linestyle='--', linewidth=1.5,
                   label=f"T* = {T_star} (90% accumulated at step {T_star})", zorder=2, alpha=0.7)
        # 标记交点（T*时刻的累积值）
        if T_star > 0 and T_star <= len(cumulative):
            ax.plot(T_star, cumulative[T_star - 1], 'go', markersize=10, zorder=4,
                   label=f'90% Risk Point (t={T_star})', markeredgecolor='darkgreen', markeredgewidth=2)
    
    # 添加总风险线（100%）
    ax.axhline(y=rho, color='gray', linestyle=':', linewidth=1,
               label=f'Total Risk (ρ = {rho:.3f})', alpha=0.7)
    
    # 标记L_tm值
    ax.text(0.02, 0.98, f'L_tm = {L_tm:.3f}\nT* = {T_star}/{T}\nρ = {rho:.3f}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 如果计算出的L_tm与报告的不同，显示警告
    if reported_L_tm is not None and abs(L_tm - reported_L_tm) > 1e-5:
        ax.text(0.98, 0.98, f'Reported L_tm: {reported_L_tm:.3f}',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    ax.set_xlabel('Time Step (t, starting from 0)', fontsize=11)
    ax.set_ylabel('Cumulative Risk', fontsize=11)
    ax.set_title(f'Risk Accumulation Process\n{model_name}/{seed_name}', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 设置x轴刻度（从0开始）
    if T <= 30:
        ax.set_xticks(timesteps)
    else:
        step = max(1, T // 10)
        ax.set_xticks(range(0, T + 1, step))
    
    # 确保x轴从0开始
    ax.set_xlim(-0.5, T + 0.5)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Risk accumulation plot saved to: {output_path}")
    plt.close()


def collect_seed_L_tm_values(model_dir: Path) -> List[float]:
    """
    收集单个模型下所有seed的L_tm值
    
    Args:
        model_dir: 模型目录路径
        
    Returns:
        L_tm值列表
    """
    L_tm_values = []
    
    # 遍历该模型下的所有seed目录
    for seed_dir in model_dir.iterdir():
        if not seed_dir.is_dir():
            continue
        
        # 跳过汇总文件
        if seed_dir.name.endswith('_summary.json') or seed_dir.name.endswith('_summary.csv'):
            continue
        
        risk_features_path = seed_dir / "risk_features.json"
        if risk_features_path.exists():
            try:
                with open(risk_features_path, 'r') as f:
                    risk_features = json.load(f)
                    L_tm = risk_features.get('L_tm')
                    if L_tm is not None:
                        L_tm_values.append(L_tm)
            except Exception as e:
                print(f"Warning: Failed to load {risk_features_path}: {e}")
                continue
    
    return L_tm_values


def visualize_L_tm_by_model(model_dir: Path, model_name: str, output_path: Path):
    """
    单个模型的L_tm可视化（显示所有seed的数据点）
    
    Args:
        model_dir: 模型目录路径
        model_name: 模型名称
        output_path: 输出图片路径
    """
    # 收集所有seed的L_tm值
    L_tm_values = collect_seed_L_tm_values(model_dir)
    
    if not L_tm_values:
        print(f"Warning: No L_tm values found for model {model_name}")
        return
    
    # 读取汇总统计信息（如果存在）
    summary_path = model_dir / f"{model_name}_summary.json"
    summary_stats = None
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary_data = json.load(f)
            if 'metrics' in summary_data and 'L_tm' in summary_data['metrics']:
                summary_stats = summary_data['metrics']['L_tm']
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制散点图（显示所有seed的数据点）
    seed_indices = np.arange(1, len(L_tm_values) + 1)
    ax.scatter(seed_indices, L_tm_values, s=100, alpha=0.7, color='steelblue',
               edgecolors='darkblue', linewidths=1.5, zorder=3, label='Seed Values')
    
    # 计算并显示统计信息
    mean_val = np.mean(L_tm_values)
    std_val = np.std(L_tm_values)
    median_val = np.median(L_tm_values)
    min_val = np.min(L_tm_values)
    max_val = np.max(L_tm_values)
    
    # 绘制均值线
    ax.axhline(y=mean_val, color='r', linestyle='-', linewidth=2,
               label=f'Mean = {mean_val:.3f}', zorder=2)
    
    # 绘制均值±标准差区间
    ax.fill_between([0, len(L_tm_values) + 1], 
                     mean_val - std_val, mean_val + std_val,
                     alpha=0.2, color='red', label=f'Mean ± Std ({mean_val:.3f} ± {std_val:.3f})')
    
    # 绘制中位数线
    ax.axhline(y=median_val, color='g', linestyle='--', linewidth=1.5,
               label=f'Median = {median_val:.3f}', zorder=2)
    
    # 绘制最小值和最大值
    ax.scatter([np.argmin(L_tm_values) + 1], [min_val], s=150, marker='v', 
               color='orange', edgecolors='darkorange', linewidths=2, zorder=4,
               label=f'Min = {min_val:.3f}')
    ax.scatter([np.argmax(L_tm_values) + 1], [max_val], s=150, marker='^',
               color='purple', edgecolors='darkviolet', linewidths=2, zorder=4,
               label=f'Max = {max_val:.3f}')
    
    # 如果汇总统计存在，显示在文本框中
    stats_text = f'Count: {len(L_tm_values)}\nMean: {mean_val:.3f}\nStd: {std_val:.3f}\nMedian: {median_val:.3f}\nMin: {min_val:.3f}\nMax: {max_val:.3f}'
    if summary_stats:
        if abs(mean_val - summary_stats.get('mean', 0)) < 1e-5:
            stats_text += '\n✓ Matches summary'
    
    ax.text(0.98, 0.98, stats_text,
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('Seed Index', fontsize=11)
    ax.set_ylabel('L_tm (Relative Risk Latency)', fontsize=11)
    ax.set_title(f'L_tm Distribution - {model_name}', fontsize=12, fontweight='bold')
    ax.set_xlim(0, len(L_tm_values) + 1)
    ax.set_xticks(seed_indices)
    ax.set_xticklabels([f'S{i}' for i in seed_indices])
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"L_tm distribution plot saved to: {output_path}")
    plt.close()


def visualize_L_tm_all_models(all_models_summary_path: Path, output_path: Path):
    """
    所有模型的L_tm对比可视化（汇总对比）
    
    Args:
        all_models_summary_path: 所有模型汇总JSON文件路径
        output_path: 输出图片路径
    """
    if not all_models_summary_path.exists():
        print(f"Warning: All models summary file not found: {all_models_summary_path}")
        return
    
    with open(all_models_summary_path, 'r') as f:
        all_models_data = json.load(f)
    
    models = []
    means = []
    stds = []
    mins = []
    maxs = []
    medians = []
    counts = []
    
    for model_data in all_models_data.get('models', []):
        model_name = model_data.get('model')
        if not model_name:
            continue
        
        if 'metrics' in model_data and 'L_tm' in model_data['metrics']:
            L_tm_stats = model_data['metrics']['L_tm']
            models.append(model_name)
            means.append(L_tm_stats.get('mean', 0))
            stds.append(L_tm_stats.get('std', 0))
            mins.append(L_tm_stats.get('min', 0))
            maxs.append(L_tm_stats.get('max', 0))
            medians.append(L_tm_stats.get('median', 0))
            counts.append(L_tm_stats.get('count', 0))
    
    if not models:
        print("Warning: No model data found for L_tm comparison")
        return
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 图1：条形图（均值+误差棒）
    x_pos = np.arange(len(models))
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    bars = ax1.bar(x_pos, means, yerr=stds, alpha=0.7, color=colors,
                   edgecolor='black', linewidth=1.5, 
                   error_kw={'elinewidth': 2, 'ecolor': 'black'})
    
    ax1.set_xlabel('Model', fontsize=11)
    ax1.set_ylabel('L_tm (Mean ± Std)', fontsize=11)
    ax1.set_title('L_tm Comparison Across Models (Mean ± Std)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # 在柱状图上添加数值标签
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 图2：箱线图风格的对比（使用最小、第一四分位、中位数、第三四分位、最大值）
    # 由于我们没有原始数据，使用汇总统计构建箱线图
    box_data = []
    box_labels = []
    for i, model_name in enumerate(models):
        # 使用中位数、min、max、mean构建近似箱线图数据
        # 注意：这不是真正的箱线图，因为缺少25%和75%分位数
        box_data.append([mins[i], medians[i], means[i], maxs[i]])
        box_labels.append(model_name)
    
    # 使用小提琴图或散点图显示分布
    # 这里使用简化的方式：显示均值、中位数、最小值和最大值
    for i, model_name in enumerate(models):
        y_pos = i
        # 绘制范围
        ax2.plot([mins[i], maxs[i]], [y_pos, y_pos], 'k-', linewidth=2, alpha=0.7)
        # 绘制均值
        ax2.scatter(means[i], y_pos, s=150, marker='o', color='red', 
                   edgecolors='darkred', linewidths=2, zorder=3, label='Mean' if i == 0 else '')
        # 绘制中位数
        ax2.scatter(medians[i], y_pos, s=100, marker='s', color='blue',
                   edgecolors='darkblue', linewidths=2, zorder=3, label='Median' if i == 0 else '')
        # 绘制最小值和最大值
        ax2.scatter(mins[i], y_pos, s=80, marker='|', color='green', zorder=3)
        ax2.scatter(maxs[i], y_pos, s=80, marker='|', color='green', zorder=3)
    
    ax2.set_xlabel('L_tm Value', fontsize=11)
    ax2.set_ylabel('Model', fontsize=11)
    ax2.set_title('L_tm Range Comparison Across Models', fontsize=12, fontweight='bold')
    ax2.set_yticks(range(len(models)))
    ax2.set_yticklabels(models, fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax2.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"All models L_tm comparison plot saved to: {output_path}")
    plt.close()


def collect_seed_behavior_risk(model_dir: Path) -> Tuple[List[np.ndarray], List[str]]:
    """
    收集单个模型下所有seed的动作聚类风险数据，并对每个seed归一化
    
    Args:
        model_dir: 模型目录路径
        
    Returns:
        (behaviour_data_list, seed_names) - behaviour_data_list是包含每个seed的10种股票风险贡献的列表（每个seed已按自身有符号总和归一化）
    """
    behaviour_data_list = []
    seed_names = []
    model_name = model_dir.name
    
    # 遍历该模型下的所有seed目录
    for seed_dir in model_dir.iterdir():
        if not seed_dir.is_dir():
            continue
        
        # 跳过汇总文件
        if seed_dir.name.endswith('_summary.json') or seed_dir.name.endswith('_summary.csv'):
            continue
        
        behaviour_path = seed_dir / "behaviour_aggregated.npy"
        if behaviour_path.exists():
            try:
                behaviour_data = np.load(behaviour_path)
                if behaviour_data.shape == (10,):  # 确保是10种股票
                    # 归一化：按自身的有符号总和归一化
                    total_sum = float(np.sum(behaviour_data))  # 有符号总和
                    if abs(total_sum) < 1e-10:
                        print(f"Warning: behaviour_aggregated sum is near zero in {seed_dir.name}, skip seed")
                        continue
                    # 使用有符号总和做归一化
                    normalized = behaviour_data / total_sum
                    
                    # 特殊过滤：llama模型归一化后超过±2000%（即>20.0或<-20.0）的过滤掉
                    if 'llama' in model_name.lower() and (np.any(normalized > 20.0) or np.any(normalized < -20.0)):
                        print(f"Warning: Llama seed {seed_dir.name} has normalized values out of range [-20.0, 20.0] "
                              f"(min: {np.min(normalized):.4f}, max: {np.max(normalized):.4f}), skip seed")
                        continue
                    
                    # 特殊过滤：qwen模型在EREI类别（索引8）超过500%（即>5.0）的过滤掉
                    if 'qwen' in model_name.lower() and normalized[8] > 5.0:
                        print(f"Warning: Qwen seed {seed_dir.name} has EREI category value {normalized[8]:.4f} > 5.0 (500%), skip seed")
                        continue
                    
                    behaviour_data_list.append(normalized)
                    seed_names.append(seed_dir.name)
            except Exception as e:
                print(f"Warning: Failed to load {behaviour_path}: {e}")
                continue
    
    return behaviour_data_list, seed_names


def visualize_behavior_risk_by_model(model_dir: Path, model_name: str, output_path: Path):
    """
    单个模型的动作聚类风险可视化（显示所有seed的数据）
    
    Args:
        model_dir: 模型目录路径
        model_name: 模型名称
        output_path: 输出图片路径
    """
    # 收集所有seed的动作聚类风险数据
    behaviour_data_list, seed_names = collect_seed_behavior_risk(model_dir)
    
    if not behaviour_data_list:
        print(f"Warning: No behaviour_aggregated data found for model {model_name}")
        return
    
    # 转换为数组：shape (num_seeds, 10)
    behaviour_array = np.array(behaviour_data_list)  # shape: (num_seeds, 10)
    num_seeds = behaviour_array.shape[0]
    
    # 计算每个股票类别的统计信息
    stock_means = np.mean(behaviour_array, axis=0)
    stock_stds = np.std(behaviour_array, axis=0)
    stock_mins = np.min(behaviour_array, axis=0)
    stock_maxs = np.max(behaviour_array, axis=0)
    stock_medians = np.median(behaviour_array, axis=0)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x_pos = np.arange(len(STOCK_LIST))
    width = 0.6
    
    # 确定颜色（正负值用不同颜色）
    colors = []
    for mean_val in stock_means:
        if mean_val >= 0:
            colors.append('#EF4444')  # 红色表示正贡献
        else:
            colors.append('#3B82F6')  # 蓝色表示负贡献
    
    # 绘制均值柱状图（带误差棒）
    # ax.bar()的yerr会自动绘制误差棒，通过error_kw传递样式参数
    bars = ax.bar(x_pos, stock_means, width, alpha=0.7, color=colors,
                  edgecolor='black', linewidth=1, yerr=stock_stds,
                  label='Mean ± Std', zorder=2, 
                  error_kw={'elinewidth': 1.5, 'ecolor': 'darkred'})
    
    # 绘制所有seed的数据点（散点图叠加）
    # 使用固定的小偏移，按顺序排列，避免点重叠
    jitter_step = 0.1
    for i, stock in enumerate(STOCK_LIST):
        seed_values = behaviour_array[:, i]
        # 在柱状图位置周围添加固定偏移，避免点重叠
        # 为每个seed创建一个小的x偏移，从-i*jitter_step/2到i*jitter_step/2
        x_offsets = np.linspace(-jitter_step * (num_seeds - 1) / 2, 
                                jitter_step * (num_seeds - 1) / 2, 
                                num_seeds)
        x_positions = i + x_offsets
        ax.scatter(x_positions, seed_values, s=50, alpha=0.5, color='gray',
                  edgecolors='darkgray', linewidths=0.5, zorder=3, marker='o')
    
    # 添加零线
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, zorder=1)
    
    # 在柱状图上添加均值标签
    for i, (bar, mean, std) in enumerate(zip(bars, stock_means, stock_stds)):
        if abs(mean) > 0.01:  # 只显示较大的值
            height = bar.get_height()
            label_y = height + std + abs(max(stock_means) - min(stock_means)) * 0.05 if height >= 0 else height - std - abs(max(stock_means) - min(stock_means)) * 0.05
            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                   f'{mean:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Stock Type', fontsize=11)
    ax.set_ylabel('Risk Contribution', fontsize=11)
    ax.set_title(f'Behavior Risk Distribution (G_be) - {model_name}\n(Mean ± Std, all {num_seeds} seeds)', 
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(STOCK_LIST, rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # 添加图例
    legend_elements = [
        Patch(facecolor='#EF4444', alpha=0.7, label='Positive Contribution (Increase Risk)'),
        Patch(facecolor='#3B82F6', alpha=0.7, label='Negative Contribution (Decrease Risk)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                  markersize=8, alpha=0.5, label='Individual Seed Values')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=9)
    
    # 添加统计信息文本框
    # 尝试从summary文件读取G_be统计信息
    summary_path = model_dir / f"{model_name}_summary.json"
    G_be_mean = None
    if summary_path.exists():
        try:
            with open(summary_path, 'r') as f:
                summary_data = json.load(f)
                if 'metrics' in summary_data and 'G_be' in summary_data['metrics']:
                    G_be_mean = summary_data['metrics']['G_be'].get('mean')
        except:
            pass
    
    stats_text = f'Number of Seeds: {num_seeds}'
    if G_be_mean is not None:
        stats_text += f'\nG_be Mean: {G_be_mean:.3f}'
    ax.text(0.98, 0.02, stats_text,
            transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Behavior risk distribution plot saved to: {output_path}")
    plt.close()


def visualize_behavior_risk_all_models(risk_feature_root: Path, output_path: Path):
    """
    所有模型的动作聚类风险对比可视化
    
    Args:
        risk_feature_root: risk_feature根目录
        output_path: 输出图片路径
    """
    # 收集所有模型的数据
    model_data_dict = {}
    
    for model_dir in risk_feature_root.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        # 跳过汇总目录
        if model_name == 'visualizations' or model_name.endswith('_summary'):
            continue
        
        behaviour_data_list, seed_names = collect_seed_behavior_risk(model_dir)
        if behaviour_data_list:
            # 计算每个股票类别的均值
            behaviour_array = np.array(behaviour_data_list)
            stock_means = np.mean(behaviour_array, axis=0)
            model_data_dict[model_name] = stock_means
    
    if not model_data_dict:
        print("Warning: No model data found for behavior risk comparison")
        return
    
    # 创建分组柱状图
    fig, ax = plt.subplots(figsize=(16, 8))
    
    models = list(model_data_dict.keys())
    x_pos = np.arange(len(STOCK_LIST))
    width = 0.8 / len(models)  # 根据模型数量调整柱宽
    
    # 为每个模型分配颜色
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    # 绘制每个模型的柱状图
    for i, (model_name, stock_means) in enumerate(model_data_dict.items()):
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x_pos + offset, stock_means, width, alpha=0.7,
                     color=colors[i], edgecolor='black', linewidth=0.5,
                     label=model_name)
        
        # 在柱状图上添加数值标签（如果值较大）
        for j, (bar, val) in enumerate(zip(bars, stock_means)):
            if abs(val) > 0.05:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}', ha='center', va='bottom' if height >= 0 else 'top',
                       fontsize=7, rotation=90)
    
    ax.set_xlabel('Stock Type', fontsize=11)
    ax.set_ylabel('Risk Contribution (Mean)', fontsize=11)
    ax.set_title('Behavior Risk Distribution Comparison Across Models', 
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(STOCK_LIST, rotation=45, ha='right', fontsize=9)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, zorder=1)
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"All models behavior risk comparison plot saved to: {output_path}")
    plt.close()
    
    # 保存数据文件
    data_output_path = output_path.parent / "behavior_risk_data.json"
    data_array_path = output_path.parent / "behavior_risk_data.npy"
    
    # 保存为JSON格式（便于查看）
    data_dict = {
        "models": list(model_data_dict.keys()),
        "stocks": STOCK_LIST,
        "data": {}
    }
    for model_name, stock_means in model_data_dict.items():
        data_dict["data"][model_name] = {
            "mean_values": stock_means.tolist()
        }
    
    with open(data_output_path, 'w') as f:
        json.dump(data_dict, f, indent=2)
    print(f"Behavior risk data (JSON) saved to: {data_output_path}")
    
    # 保存为numpy数组格式（便于后续分析）
    # 形状: (5 models, 10 stocks)
    data_array = np.array([model_data_dict[model] for model in data_dict["models"]])
    np.save(data_array_path, data_array)
    print(f"Behavior risk data (numpy array) saved to: {data_array_path}")
    print(f"  Array shape: {data_array.shape} (models × stocks)")


def main():
    parser = argparse.ArgumentParser(
        description='Risk Feature可视化脚本'
    )
    
    parser.add_argument(
        '--risk_feature_root',
        type=str,
        default=None,
        help='Risk feature根目录（默认: PROJECT_ROOT/results/risk_feature）'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='只可视化指定模型（默认: 所有模型）'
    )
    
    parser.add_argument(
        '--skip_accumulation',
        action='store_true',
        help='跳过单个seed的风险累积过程可视化（可能较慢）'
    )
    
    args = parser.parse_args()
    
    # 设置risk feature根目录
    if args.risk_feature_root:
        risk_feature_root = Path(args.risk_feature_root)
    else:
        risk_feature_root = RISK_FEATURE_ROOT
    
    print("=" * 60)
    print("Risk Feature可视化")
    print("=" * 60)
    print(f"Risk feature root: {risk_feature_root}")
    print(f"Model filter: {args.model if args.model else 'All models'}")
    print("=" * 60 + "\n")
    
    if not risk_feature_root.exists():
        print(f"❌ 错误: Risk feature根目录不存在: {risk_feature_root}")
        return 1
    
    # 1. 遍历所有模型，生成每个模型的可视化
    print("Step 1: Generating visualizations for each model...")
    model_count = 0
    
    for model_dir in risk_feature_root.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        # 跳过汇总目录和visualizations目录
        if model_name == 'visualizations' or model_name.endswith('_summary'):
            continue
        
        # 如果指定了模型过滤，只处理匹配的模型
        if args.model and args.model not in model_name:
            continue
        
        print(f"\nProcessing model: {model_name}")
        model_count += 1
        
        # 1.1 生成单模型L_tm分布图
        L_tm_output_path = model_dir / "L_tm_distribution.png"
        visualize_L_tm_by_model(model_dir, model_name, L_tm_output_path)
        
        # 1.2 生成单模型动作聚类风险分布图
        behavior_output_path = model_dir / "behavior_risk_distribution.png"
        visualize_behavior_risk_by_model(model_dir, model_name, behavior_output_path)
        
        # 1.3 生成每个seed的风险累积过程图
        if not args.skip_accumulation:
            seed_count = 0
            for seed_dir in model_dir.iterdir():
                if not seed_dir.is_dir():
                    continue
                
                # 跳过汇总文件
                if seed_dir.name.endswith('_summary.json') or seed_dir.name.endswith('_summary.csv'):
                    continue
                
                # 检查是否有risk_feature数据
                risk_features_path = seed_dir / "risk_features.json"
                attribution_matrix_path = seed_dir / "attribution_matrix.npy"
                
                if risk_features_path.exists() or attribution_matrix_path.exists():
                    seed_count += 1
                    seed_name = seed_dir.name
                    accumulation_output_path = seed_dir / "risk_accumulation.png"
                    visualize_risk_accumulation_by_seed(
                        seed_dir, accumulation_output_path, model_name, seed_name
                    )
            
            print(f"  Generated {seed_count} risk accumulation plots for {model_name}")
    
    print(f"\nProcessed {model_count} models")
    
    # 2. 生成汇总对比可视化
    print("\nStep 2: Generating comparison visualizations...")
    
    # 2.1 所有模型的L_tm对比
    all_models_summary_path = risk_feature_root / "all_models_summary.json"
    if all_models_summary_path.exists():
        L_tm_comparison_path = risk_feature_root / "visualizations" / "L_tm" / "all_models_L_tm_comparison.png"
        visualize_L_tm_all_models(all_models_summary_path, L_tm_comparison_path)
    else:
        print(f"Warning: All models summary file not found: {all_models_summary_path}")
    
    # 2.2 所有模型的动作聚类风险对比（可选）
    behavior_comparison_path = risk_feature_root / "visualizations" / "behavior_risk" / "all_models_behavior_comparison.png"
    visualize_behavior_risk_all_models(risk_feature_root, behavior_comparison_path)
    
    print("\n" + "=" * 60)
    print("所有可视化完成！")
    print("=" * 60)
    print(f"可视化结果保存在: {risk_feature_root}")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
