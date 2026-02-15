#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SocialLLM Behavior Risk可视化脚本
包括：
1. 动作聚类风险（G_be指标，按行为类别）的可视化（单模型汇总和模型对比）
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

# 6种行为模式定义
# 0: posted=0, like
# 1: posted=0, no_interaction
# 2: posted=0, dislike
# 3: posted=1, like
# 4: posted=1, no_interaction
# 5: posted=1, dislike
ACTION_CATEGORY_NAMES = [
    'posted=0, like',
    'posted=0, no_interaction',
    'posted=0, dislike',
    'posted=1, like',
    'posted=1, no_interaction',
    'posted=1, dislike',
]

# 设置项目根目录
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # SocialLLM
RESULTS_ROOT = PROJECT_ROOT / "results"
RISK_FEATURE_ROOT = RESULTS_ROOT / "risk_feature"


def collect_seed_behavior_risk(model_dir: Path, filter_outliers: bool = False, 
                               strict_threshold: float = 1.8, normal_threshold: float = 2.0) -> Tuple[List[np.ndarray], List[str], List[Tuple[str, str]]]:
    """
    收集单个模型下所有seed的动作聚类风险数据，并对每个seed归一化
    
    Args:
        model_dir: 模型目录路径（results/{model_name}）
        filter_outliers: 是否启用异常值过滤（默认: False）
        strict_threshold: 严格类别的Z-score阈值（默认: 1.8）
        normal_threshold: 普通类别的Z-score阈值（默认: 2.0）
        
    Returns:
        (behaviour_data_list, seed_names, filtered_seeds) - 
        behaviour_data_list是包含每个seed的6种行为类别风险贡献的列表（每个seed已按自身有符号总和归一化）
        filtered_seeds是被过滤的seed列表，格式为[(seed_name, reason), ...]
    """
    model_name = model_dir.name
    behaviour_data_list = []
    seed_names = []
    filtered_seeds = []  # 记录被过滤的seed及其原因
    raw_data_list = []  # 临时存储原始数据，用于异常值检测
    
    # 第一遍：收集所有原始数据
    # SocialLLM的数据结构：results/{model_name}/{model_name}_{seed}/risk_feature/risk_features.json
    for seed_dir in model_dir.iterdir():
        if not seed_dir.is_dir():
            continue
        
        # 跳过汇总文件
        if seed_dir.name.endswith('_summary.json') or seed_dir.name.endswith('_summary.csv'):
            continue
        
        # 检查是否是seed目录（格式：{model_name}_{seed_number}）
        if not seed_dir.name.startswith(model_name + '_'):
            continue
        
        risk_features_path = seed_dir / "risk_feature" / "risk_features.json"
        if risk_features_path.exists():
            try:
                with open(risk_features_path, 'r', encoding='utf-8') as f:
                    risk_features = json.load(f)
                    behavior_shapley = risk_features.get('behavior_shapley', {})
                    
                    # 将字典转换为数组（按顺序）
                    behaviour_data = np.array([
                        behavior_shapley.get(ACTION_CATEGORY_NAMES[i], 0.0)
                        for i in range(6)
                    ], dtype=np.float64)
                    
                    if behaviour_data.shape == (6,):  # 确保是6种行为类别
                        raw_data_list.append((behaviour_data, seed_dir.name))
            except Exception as e:
                print(f"Warning: Failed to load {risk_features_path}: {e}")
                continue
    
    # 归一化并检测异常值
    if not raw_data_list:
        return behaviour_data_list, seed_names, filtered_seeds
    
    # 先归一化所有数据，并过滤掉归一化后超出合理范围的seed
    normalized_list = []
    
    for behaviour_data, seed_name in raw_data_list:
        total_sum = float(np.sum(behaviour_data))  # 有符号总和
        if abs(total_sum) < 1e-10:
            reason = f"Sum near zero (sum={total_sum:.6e})"
            print(f"Warning: behaviour_shapley sum is near zero in {seed_name}, skip seed")
            filtered_seeds.append((seed_name, reason))
            continue
        # 使用有符号总和做归一化
        normalized = behaviour_data / total_sum
        
        # 检查归一化后的值是否超出合理范围（-1.0 到 1.0）
        # 如果任何一个类别超出范围，过滤掉这个seed
        if np.any(normalized > 1.0) or np.any(normalized < -1.0):
            reason = f"Normalized values out of range [-1.0, 1.0] (min={np.min(normalized):.4f}, max={np.max(normalized):.4f})"
            print(f"Warning: Normalized values out of range [-1.0, 1.0] in {seed_name} "
                  f"(min: {np.min(normalized):.4f}, max: {np.max(normalized):.4f}), skip seed")
            filtered_seeds.append((seed_name, reason))
            continue
        
        normalized_list.append((normalized, seed_name))
    
    if not normalized_list:
        return behaviour_data_list, seed_names, filtered_seeds
    
    # 如果启用异常值过滤，进行检测
    if filter_outliers:
        # 检测异常值：使用多层检测方法
        outlier_indices = []
        
        # 先检测明显的极端值（归一化后绝对值超过5）
        for i, (normalized, seed_name) in enumerate(normalized_list):
            max_abs_val = np.max(np.abs(normalized))
            if max_abs_val > 5.0:  # 归一化后绝对值超过5认为是明显异常
                reason = f"Extreme outlier (max abs normalized value: {max_abs_val:.2f})"
                print(f"Warning: Detected extreme outlier seed {seed_name} (max abs normalized value: {max_abs_val:.2f}), skipping")
                filtered_seeds.append((seed_name, reason))
                outlier_indices.append(i)
        
        # 对于剩余的seed，使用迭代Z-score方法检测异常值（对每个类别单独检测）
        max_iterations = 5
        
        for iteration in range(max_iterations):
            remaining_list = [(nv, name) for i, (nv, name) in enumerate(normalized_list) if i not in outlier_indices]
            if len(remaining_list) <= 2:  # 至少需要3个seed才能计算有意义的Z-score
                break
            
            remaining_array = np.array([nv for nv, _ in remaining_list])
            new_outliers = []
            
            # 对每个类别单独计算均值和标准差
            for category_idx in range(remaining_array.shape[1]):
                category_values = remaining_array[:, category_idx]
                mean_val = np.mean(category_values)
                std_val = np.std(category_values)
                
                if std_val < 1e-10:  # 标准差太小，跳过
                    continue
                
                # 使用普通阈值
                threshold = normal_threshold
                
                # 检查每个seed在这个类别上的Z-score
                for idx, (normalized, seed_name) in enumerate(normalized_list):
                    if idx in outlier_indices or idx in new_outliers:
                        continue
                    z_score = abs((normalized[category_idx] - mean_val) / std_val)
                    if z_score > threshold:
                        reason = f"Z-score outlier in category {category_idx} ({ACTION_CATEGORY_NAMES[category_idx]}) (value: {normalized[category_idx]:.4f}, Z-score: {z_score:.2f})"
                        print(f"Warning: Detected outlier seed {seed_name} in category {category_idx} ({ACTION_CATEGORY_NAMES[category_idx]}) "
                              f"(value: {normalized[category_idx]:.4f}, Z-score: {z_score:.2f}), skipping")
                        filtered_seeds.append((seed_name, reason))
                        new_outliers.append(idx)
            
            if not new_outliers:
                break  # 没有新的异常值，停止迭代
            
            outlier_indices.extend(new_outliers)
        
        # 过滤掉异常值
        for i, (normalized, seed_name) in enumerate(normalized_list):
            if i not in outlier_indices:
                behaviour_data_list.append(normalized)
                seed_names.append(seed_name)
    else:
        # 不过滤，直接添加所有归一化后的数据
        for normalized, seed_name in normalized_list:
            behaviour_data_list.append(normalized)
            seed_names.append(seed_name)
    
    return behaviour_data_list, seed_names, filtered_seeds


def visualize_behavior_risk_by_model(model_dir: Path, model_name: str, output_path: Path,
                                     filter_outliers: bool = False, strict_threshold: float = 1.8, normal_threshold: float = 2.0):
    """
    单个模型的动作聚类风险可视化（显示所有seed的数据）
    
    Args:
        model_dir: 模型目录路径
        model_name: 模型名称
        output_path: 输出图片路径
        filter_outliers: 是否启用异常值过滤（默认: False）
        strict_threshold: 严格类别的Z-score阈值（默认: 1.8）
        normal_threshold: 普通类别的Z-score阈值（默认: 2.0）
    """
    # 收集所有seed的动作聚类风险数据
    behaviour_data_list, seed_names, filtered_seeds = collect_seed_behavior_risk(model_dir, filter_outliers, strict_threshold, normal_threshold)
    
    # 输出过滤信息
    if filtered_seeds:
        print(f"\n{model_name} - Filtered seeds ({len(filtered_seeds)}):")
        for seed_name, reason in filtered_seeds:
            print(f"  - {seed_name}: {reason}")
    else:
        print(f"\n{model_name} - No seeds filtered")
    
    if not behaviour_data_list:
        print(f"Warning: No behaviour_shapley data found for model {model_name}")
        return filtered_seeds if filtered_seeds else []
    
    # 转换为数组：shape (num_seeds, 6)
    behaviour_array = np.array(behaviour_data_list)  # shape: (num_seeds, 6)
    num_seeds = behaviour_array.shape[0]
    
    # 计算每个行为类别的统计信息
    category_means = np.mean(behaviour_array, axis=0)
    category_stds = np.std(behaviour_array, axis=0)
    category_mins = np.min(behaviour_array, axis=0)
    category_maxs = np.max(behaviour_array, axis=0)
    category_medians = np.median(behaviour_array, axis=0)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x_pos = np.arange(len(ACTION_CATEGORY_NAMES))
    width = 0.6
    
    # 确定颜色（正负值用不同颜色）
    colors = []
    for mean_val in category_means:
        if mean_val >= 0:
            colors.append('#EF4444')  # 红色表示正贡献
        else:
            colors.append('#3B82F6')  # 蓝色表示负贡献
    
    # 绘制均值柱状图（带误差棒）
    bars = ax.bar(x_pos, category_means, width, alpha=0.7, color=colors,
                  edgecolor='black', linewidth=1, yerr=category_stds,
                  label='Mean ± Std', zorder=2, 
                  error_kw={'elinewidth': 1.5, 'ecolor': 'darkred'})
    
    # 绘制所有seed的数据点（散点图叠加）
    jitter_step = 0.1
    for i, category_name in enumerate(ACTION_CATEGORY_NAMES):
        seed_values = behaviour_array[:, i]
        x_offsets = np.linspace(-jitter_step * (num_seeds - 1) / 2, 
                                jitter_step * (num_seeds - 1) / 2, 
                                num_seeds)
        x_positions = i + x_offsets
        ax.scatter(x_positions, seed_values, s=50, alpha=0.5, color='gray',
                  edgecolors='darkgray', linewidths=0.5, zorder=3, marker='o')
    
    # 添加零线
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, zorder=1)
    
    # 在柱状图上添加均值标签
    for i, (bar, mean, std) in enumerate(zip(bars, category_means, category_stds)):
        if abs(mean) > 0.01:  # 只显示较大的值
            height = bar.get_height()
            label_y = height + std + abs(max(category_means) - min(category_means)) * 0.05 if height >= 0 else height - std - abs(max(category_means) - min(category_means)) * 0.05
            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                   f'{mean:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Behavior Category', fontsize=11)
    ax.set_ylabel('Risk Contribution', fontsize=11)
    ax.set_title(f'Behavior Risk Distribution (G_be) - {model_name}\n(Mean ± Std, all {num_seeds} seeds)', 
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ACTION_CATEGORY_NAMES, rotation=45, ha='right', fontsize=9)
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
    summary_path = RISK_FEATURE_ROOT / model_name / f"{model_name}_summary.json"
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
    
    # 保存单模型的JSON数据文件
    json_output_path = output_path.parent / f"{model_name}_behavior_risk_data.json"
    model_data_dict = {
        "model": model_name,
        "stocks": ACTION_CATEGORY_NAMES,
        "data": {
            model_name: {
                "mean_values": category_means.tolist(),
                "std_values": category_stds.tolist(),
                "min_values": category_mins.tolist(),
                "max_values": category_maxs.tolist(),
                "median_values": category_medians.tolist(),
                "num_seeds": num_seeds
            }
        }
    }
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(model_data_dict, f, indent=2, ensure_ascii=False)
    print(f"Behavior risk data (JSON) saved to: {json_output_path}")
    
    return filtered_seeds


def visualize_behavior_risk_all_models(results_root: Path, output_path: Path,
                                       filter_outliers: bool = False, strict_threshold: float = 1.8, normal_threshold: float = 2.0):
    """
    所有模型的动作聚类风险对比可视化
    
    注意：每个模型先计算其所有seed的均值，然后进行对比。
    这样确保每个模型有相等的权重，不会因为seed数量不同而影响对比结果。
    
    Args:
        results_root: results根目录
        output_path: 输出图片路径
        filter_outliers: 是否启用异常值过滤（默认: False）
        strict_threshold: 严格类别的Z-score阈值（默认: 1.8）
        normal_threshold: 普通类别的Z-score阈值（默认: 2.0）
    """
    # 收集所有模型的数据
    model_data_dict = {}
    model_seed_counts = {}  # 记录每个模型的seed数量，用于图例显示
    
    # SocialLLM的数据结构：results/{model_name}/{model_name}_{seed}/risk_feature/risk_features.json
    # 所以需要从results目录遍历模型
    
    for model_dir in results_root.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        # 跳过汇总目录和visualizations目录
        if model_name == 'risk_feature' or model_name.endswith('_summary'):
            continue
        
        behaviour_data_list, seed_names, filtered_seeds = collect_seed_behavior_risk(model_dir, filter_outliers, strict_threshold, normal_threshold)
        if behaviour_data_list:
            # 计算每个行为类别的均值（对每个模型先归一化，确保模型间权重相等）
            behaviour_array = np.array(behaviour_data_list)
            category_means = np.mean(behaviour_array, axis=0)
            model_data_dict[model_name] = category_means
            model_seed_counts[model_name] = len(behaviour_data_list)
            
            # 输出过滤信息
            if filtered_seeds:
                print(f"  {model_name}: {len(filtered_seeds)} seed(s) filtered")
                for seed_name, reason in filtered_seeds:
                    print(f"    - {seed_name}: {reason}")
    
    if not model_data_dict:
        print("Warning: No model data found for behavior risk comparison")
        return
    
    # 创建分组柱状图
    fig, ax = plt.subplots(figsize=(16, 8))
    
    models = list(model_data_dict.keys())
    x_pos = np.arange(len(ACTION_CATEGORY_NAMES))
    width = 0.8 / len(models)  # 根据模型数量调整柱宽
    
    # 为每个模型分配颜色
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    # 绘制每个模型的柱状图
    for i, (model_name, category_means) in enumerate(model_data_dict.items()):
        offset = (i - len(models) / 2 + 0.5) * width
        seed_count = model_seed_counts.get(model_name, 0)
        label = f"{model_name} (n={seed_count})"
        bars = ax.bar(x_pos + offset, category_means, width, alpha=0.7,
                     color=colors[i], edgecolor='black', linewidth=0.5,
                     label=label)
        
        # 在柱状图上添加数值标签（如果值较大）
        for j, (bar, val) in enumerate(zip(bars, category_means)):
            if abs(val) > 0.05:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}', ha='center', va='bottom' if height >= 0 else 'top',
                       fontsize=7, rotation=90)
    
    ax.set_xlabel('Behavior Category', fontsize=11)
    ax.set_ylabel('Risk Contribution (Mean)', fontsize=11)
    ax.set_title('Behavior Risk Distribution Comparison Across Models', 
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ACTION_CATEGORY_NAMES, rotation=45, ha='right', fontsize=9)
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
    # 按照用户要求的格式：使用 "stocks" 字段名（虽然实际上是行为类别），只包含 mean_values
    data_dict = {
        "models": list(model_data_dict.keys()),
        "stocks": ACTION_CATEGORY_NAMES,  # 使用 "stocks" 字段名以匹配格式要求
        "data": {}
    }
    for model_name, category_means in model_data_dict.items():
        data_dict["data"][model_name] = {
            "mean_values": category_means.tolist()
        }
    
    with open(data_output_path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, indent=2, ensure_ascii=False)
    print(f"Behavior risk data (JSON) saved to: {data_output_path}")
    
    # 保存为numpy数组格式（便于后续分析）
    data_array = np.array([model_data_dict[model] for model in data_dict["models"]])
    np.save(data_array_path, data_array)
    print(f"Behavior risk data (numpy array) saved to: {data_array_path}")
    print(f"  Array shape: {data_array.shape} (models × categories)")


def main():
    parser = argparse.ArgumentParser(
        description='SocialLLM Behavior Risk可视化脚本'
    )
    
    parser.add_argument(
        '--risk_feature_root',
        type=str,
        default=None,
        help='Risk feature根目录（默认: PROJECT_ROOT/results/risk_feature）'
    )
    
    parser.add_argument(
        '--results_root',
        type=str,
        default=None,
        help='Results根目录（默认: PROJECT_ROOT/results）'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='只可视化指定模型（默认: 所有模型）'
    )
    
    parser.add_argument(
        '--filter_outliers',
        action='store_true',
        help='启用异常值过滤（默认: 不过滤）'
    )
    
    parser.add_argument(
        '--strict_threshold',
        type=float,
        default=1.8,
        help='严格类别的Z-score阈值（默认: 1.8）'
    )
    
    parser.add_argument(
        '--normal_threshold',
        type=float,
        default=2.0,
        help='普通类别的Z-score阈值（默认: 2.0）'
    )
    
    args = parser.parse_args()
    
    # 设置risk feature根目录
    if args.risk_feature_root:
        risk_feature_root = Path(args.risk_feature_root)
    else:
        risk_feature_root = RISK_FEATURE_ROOT
    
    # 设置results根目录
    if args.results_root:
        results_root = Path(args.results_root)
    else:
        results_root = RESULTS_ROOT
    
    print("=" * 60)
    print("SocialLLM Behavior Risk可视化")
    print("=" * 60)
    print(f"Risk feature root: {risk_feature_root}")
    print(f"Results root: {results_root}")
    print(f"Model filter: {args.model if args.model else 'All models'}")
    print("=" * 60 + "\n")
    
    if not results_root.exists():
        print(f"❌ 错误: Results根目录不存在: {results_root}")
        return 1
    
    # 收集所有模型的过滤信息
    all_filtered_info = {}  # {model_name: [(seed_name, reason), ...]}
    
    # 1. 遍历所有模型，生成每个模型的可视化
    print("Step 1: Generating visualizations for each model...")
    model_count = 0
    
    for model_dir in results_root.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        # 跳过汇总目录和visualizations目录
        if model_name == 'risk_feature' or model_name.endswith('_summary'):
            continue
        
        # 如果指定了模型过滤，只处理匹配的模型
        if args.model and args.model not in model_name:
            continue
        
        print(f"\nProcessing model: {model_name}")
        model_count += 1
        
        # 1.1 生成单模型动作聚类风险分布图
        behavior_output_path = risk_feature_root / model_name / "behavior_risk_distribution.png"
        filtered_seeds = visualize_behavior_risk_by_model(model_dir, model_name, behavior_output_path,
                                       args.filter_outliers, args.strict_threshold, args.normal_threshold)
        if filtered_seeds:
            all_filtered_info[model_name] = filtered_seeds
    
    print(f"\nProcessed {model_count} models")
    
    # 2. 生成汇总对比可视化
    print("\nStep 2: Generating comparison visualizations...")
    
    # 2.1 所有模型的动作聚类风险对比
    behavior_comparison_path = risk_feature_root / "visualizations" / "behavior_risk" / "all_models_behavior_comparison.png"
    visualize_behavior_risk_all_models(results_root, behavior_comparison_path,
                                     args.filter_outliers, args.strict_threshold, args.normal_threshold)
    
    # 3. 保存过滤信息汇总
    if all_filtered_info:
        filter_summary_path = risk_feature_root / "visualizations" / "behavior_risk" / "filtered_seeds_summary.txt"
        filter_summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(filter_summary_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("Filtered Seeds Summary\n")
            f.write("=" * 60 + "\n\n")
            total_filtered = 0
            for model_name, filtered_seeds in sorted(all_filtered_info.items()):
                f.write(f"Model: {model_name}\n")
                f.write(f"  Filtered seeds: {len(filtered_seeds)}\n")
                for seed_name, reason in filtered_seeds:
                    f.write(f"    - {seed_name}: {reason}\n")
                f.write("\n")
                total_filtered += len(filtered_seeds)
            f.write(f"Total filtered seeds across all models: {total_filtered}\n")
        print(f"\nFiltered seeds summary saved to: {filter_summary_path}")
    
    print("\n" + "=" * 60)
    print("所有可视化完成！")
    print("=" * 60)
    print(f"可视化结果保存在: {risk_feature_root}")
    if all_filtered_info:
        total_filtered = sum(len(seeds) for seeds in all_filtered_info.values())
        print(f"总共过滤了 {total_filtered} 个seed")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
