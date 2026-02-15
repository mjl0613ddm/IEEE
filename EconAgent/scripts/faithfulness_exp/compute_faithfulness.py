#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EconAgent Faithfulness实验脚本

计算Deletion和Insertion指标，验证不同归因方法的准确性。

使用方法:
    python scripts/faithfulness_exp/compute_faithfulness.py gpt/gpt_42 --method shapley --metric_type deletion_top_5
"""

import os
import sys
import json
import argparse
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 添加项目路径
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_ROOT = PROJECT_ROOT / "datas"
RESULTS_ROOT = PROJECT_ROOT / "results"
sys.path.insert(0, str(PROJECT_ROOT))

# 导入shapley.py中的函数
from scripts.core.shapley import (
    run_counterfactual_simulation,
    _compute_risk_from_metrics,
    load_real_actions
)


def load_shapley_stats(model_path: str) -> Dict:
    """加载shapley stats获取实验参数"""
    sim_dir = DATA_ROOT / model_path
    shapley_dir = sim_dir / "shapley"
    
    stats_file = shapley_dir / "shapley_stats.json"
    if not stats_file.exists():
        raise FileNotFoundError(f"Shapley stats file not found: {stats_file}")
    
    with open(stats_file, 'r') as f:
        shapley_stats = json.load(f)
    
    return shapley_stats


def load_score_data(model_path: str, method: str) -> Tuple[np.ndarray, Dict]:
    """
    加载指定方法的分数数据
    
    Args:
        model_path: 模型路径，如 "gpt/gpt_42"
        method: 方法名称，可选 'shapley', 'random', 'loo', 'llm', 'mast'
    
    Returns:
        (score_matrix, stats): 分数矩阵和统计信息
    """
    sim_dir = DATA_ROOT / model_path
    
    if method == 'shapley':
        score_file = sim_dir / "shapley" / "shapley_values.npy"
        stats_file = sim_dir / "shapley" / "shapley_stats.json"
    else:
        score_file = sim_dir / "faithfulness_exp" / method / f"{method}_scores.npy"
        stats_file = sim_dir / "faithfulness_exp" / method / f"{method}_stats.json"
    
    if not score_file.exists():
        raise FileNotFoundError(f"{method} scores file not found: {score_file}")
    
    if not stats_file.exists():
        raise FileNotFoundError(f"{method} stats file not found: {stats_file}")
    
    score_matrix = np.load(score_file)
    
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    # 对于baseline方法，需要从shapley_stats获取一些配置
    if method != 'shapley':
        shapley_stats = load_shapley_stats(model_path)
        # 合并配置信息
        for key in ['baseline_work', 'baseline_consumption', 'use_probabilistic_baseline',
                    'metric_name', 'inflation_threshold', 'use_metric_directly',
                    'risk_aggregation', 'include_both_risks', 'risk_lambda', 'seed',
                    'num_agents', 'episode_length', 'target_timesteps']:
            if key in shapley_stats and key not in stats:
                stats[key] = shapley_stats[key]
    
    return score_matrix, stats


def sort_actions_by_scores(score_matrix: np.ndarray, num_agents: int, max_timestep: int) -> List[Tuple[Tuple[int, int], float]]:
    """
    按分数从高到低排序所有动作对
    
    Args:
        score_matrix: 分数矩阵，形状为 (num_agents, max_timestep)
        num_agents: Agent数量
        max_timestep: 最大时间步
    
    Returns:
        排序后的(action, score)列表，action是(agent_id, timestep)元组
    """
    action_score_list = []
    
    for agent_id in range(num_agents):
        for timestep in range(max_timestep):
            score_value = float(score_matrix[agent_id, timestep])
            action_score_list.append(((agent_id, timestep), score_value))
    
    # 按分数从高到低排序
    action_score_list.sort(key=lambda x: x[1], reverse=True)
    
    return action_score_list


def compute_deletion_top_n(
    sorted_actions: List[Tuple[Tuple[int, int], float]],
    model_path: str,
    stats: Dict,
    n: int,
    real_risk: float
) -> float:
    """
    计算Deletion指标：移除分数最高的n个动作后，风险下降比例
    
    Args:
        sorted_actions: 排序后的(action, score)列表
        model_path: 模型路径
        stats: 统计信息字典
        n: 要移除的动作数量（5或10）
        real_risk: 真实风险值
    
    Returns:
        风险下降比例: (real_risk - risk_after_removal) / real_risk
    """
    print(f"计算Deletion Top {n}...")
    
    sim_dir = DATA_ROOT / model_path
    
    # 加载真实动作
    actions_json_file = sim_dir / "actions_json" / "all_actions.json"
    if not actions_json_file.exists():
        raise FileNotFoundError(f"Actions JSON file not found: {actions_json_file}")
    real_actions_dict = load_real_actions(str(actions_json_file))
    
    num_agents = stats['num_agents']
    episode_length = stats['episode_length']
    target_timesteps = stats.get('target_timesteps', None)
    
    if target_timesteps is None or len(target_timesteps) == 0:
        max_timestep = episode_length
    else:
        max_timestep = max(target_timesteps)
    
    # 创建所有动作的集合
    all_actions_set = set()
    for agent_id in range(num_agents):
        for timestep in range(max_timestep):
            all_actions_set.add((agent_id, timestep))
    
    # 获取要移除的n个最高分动作
    actions_to_remove = sorted_actions[:n]
    removed_actions_set = {action for action, score in actions_to_remove}
    
    # 从全部动作中移除这n个动作
    current_subset = all_actions_set - removed_actions_set
    
    print(f"  移除前动作数: {len(all_actions_set)}")
    print(f"  移除了 {len(removed_actions_set)} 个动作")
    print(f"  移除后动作数: {len(current_subset)}")
    
    # 运行反事实模拟
    print(f"  运行反事实模拟...")
    metrics = run_counterfactual_simulation(
        num_agents=num_agents,
        episode_length=episode_length,
        real_actions_dict=real_actions_dict,
        baseline_work=stats.get('baseline_work', 1.0),
        baseline_consumption=stats.get('baseline_consumption', 0.8),
        subset_U=current_subset,
        seed=stats.get('seed', None),
        use_probabilistic_baseline=stats.get('use_probabilistic_baseline', False)
    )
    
    # 计算风险值
    target_t = target_timesteps[0] if target_timesteps and len(target_timesteps) > 0 else None
    
    risk_after_removal = _compute_risk_from_metrics(
        metrics,
        target_timestep=target_t,
        metric_name=stats.get('metric_name', 'risk_indicator_naive'),
        inflation_threshold=stats.get('inflation_threshold', 0.0),
        use_metric_directly=stats.get('use_metric_directly', False),
        risk_aggregation=stats.get('risk_aggregation', 'max'),
        include_both_risks=stats.get('include_both_risks', True),
        verbose=False,
        risk_lambda=stats.get('risk_lambda', 0.94)
    )
    
    # 计算相对风险下降
    if real_risk == 0:
        risk_decrease_relative = 0.0
        print(f"  警告: real_risk为0，无法计算相对风险下降")
    else:
        risk_decrease_relative = (real_risk - risk_after_removal) / real_risk
    
    print(f"  原始风险 (real_risk): {real_risk:.6f}")
    print(f"  移除后风险: {risk_after_removal:.6f}")
    print(f"  相对风险下降: {risk_decrease_relative:.6f} ({risk_decrease_relative * 100:.2f}%)")
    
    return risk_decrease_relative


def compute_insertion_top_n(
    sorted_actions: List[Tuple[Tuple[int, int], float]],
    model_path: str,
    stats: Dict,
    n: int,
    real_risk: float,
    baseline_risk: float
) -> float:
    """
    计算Insertion指标：从baseline开始添加分数最高的n个动作后，风险上升到real_risk的比例
    
    Args:
        sorted_actions: 排序后的(action, score)列表
        model_path: 模型路径
        stats: 统计信息字典
        n: 要添加的动作数量（5或10）
        real_risk: 真实风险值
        baseline_risk: Baseline风险值
    
    Returns:
        风险上升比例: (risk_after_insertion - real_risk) / real_risk
    """
    print(f"计算Insertion Top {n}...")
    
    sim_dir = DATA_ROOT / model_path
    
    # 加载真实动作
    actions_json_file = sim_dir / "actions_json" / "all_actions.json"
    if not actions_json_file.exists():
        raise FileNotFoundError(f"Actions JSON file not found: {actions_json_file}")
    real_actions_dict = load_real_actions(str(actions_json_file))
    
    num_agents = stats['num_agents']
    episode_length = stats['episode_length']
    target_timesteps = stats.get('target_timesteps', None)
    
    if target_timesteps is None or len(target_timesteps) == 0:
        max_timestep = episode_length
    else:
        max_timestep = max(target_timesteps)
    
    # 获取要添加的n个最高分动作
    actions_to_add = sorted_actions[:n]
    added_actions_set = {action for action, score in actions_to_add}
    
    # 从baseline开始（subset_U为空），只添加这n个动作
    current_subset = added_actions_set.copy()
    
    print(f"  添加了 {len(current_subset)} 个动作（从baseline开始）")
    
    # 运行反事实模拟
    print(f"  运行反事实模拟...")
    metrics = run_counterfactual_simulation(
        num_agents=num_agents,
        episode_length=episode_length,
        real_actions_dict=real_actions_dict,
        baseline_work=stats.get('baseline_work', 1.0),
        baseline_consumption=stats.get('baseline_consumption', 0.8),
        subset_U=current_subset,
        seed=stats.get('seed', None),
        use_probabilistic_baseline=stats.get('use_probabilistic_baseline', False)
    )
    
    # 计算风险值
    target_t = target_timesteps[0] if target_timesteps and len(target_timesteps) > 0 else None
    
    risk_after_insertion = _compute_risk_from_metrics(
        metrics,
        target_timestep=target_t,
        metric_name=stats.get('metric_name', 'risk_indicator_naive'),
        inflation_threshold=stats.get('inflation_threshold', 0.0),
        use_metric_directly=stats.get('use_metric_directly', False),
        risk_aggregation=stats.get('risk_aggregation', 'max'),
        include_both_risks=stats.get('include_both_risks', True),
        verbose=False,
        risk_lambda=stats.get('risk_lambda', 0.94)
    )
    
    # 计算相对于real_risk的比例
    if real_risk == 0:
        risk_increase_relative = 0.0
        print(f"  警告: real_risk为0，无法计算相对风险上升")
    else:
        risk_increase_relative = (risk_after_insertion - real_risk) / real_risk
    
    print(f"  Baseline风险: {baseline_risk:.6f}")
    print(f"  真实风险 (real_risk): {real_risk:.6f}")
    print(f"  添加后风险: {risk_after_insertion:.6f}")
    print(f"  相对风险变化（相对于real_risk）: {risk_increase_relative:.6f} ({risk_increase_relative * 100:.2f}%)")
    
    return risk_increase_relative


def compute_faithfulness(
    model_path: str,
    method: str = 'shapley',
    metric_type: str = 'deletion_top_5',
    verbose: bool = False
) -> Dict:
    """
    计算faithfulness指标
    
    Args:
        model_path: 模型路径，如 "gpt/gpt_42"
        method: 方法名称，可选 'shapley', 'random', 'llm', 'mast', 'loo'
        metric_type: 指标类型，支持动态格式：
            - deletion_top_n (n为任意正整数，如 deletion_top_3, deletion_top_15, deletion_top_20)
            - insertion_top_n (n为任意正整数，如 insertion_top_3, insertion_top_15, insertion_top_20)
            也支持旧的固定格式（向后兼容）: deletion_top_5, deletion_top_10, insertion_top_5, insertion_top_10
        verbose: 是否输出详细信息
    
    Returns:
        计算结果字典
    """
    print(f"处理: {model_path}")
    print(f"方法: {method}")
    print(f"指标类型: {metric_type}")
    print("=" * 60)
    
    # 验证metric_type并提取n值
    # 支持动态格式（如 deletion_top_n, insertion_top_n）和旧的固定格式（向后兼容）
    valid_metric_patterns = {
        'deletion_top': r'^deletion_top_(\d+)$',
        'insertion_top': r'^insertion_top_(\d+)$'
    }
    
    n_value = None
    metric_category = None
    
    # 检查是否是动态格式
    for pattern_name, pattern in valid_metric_patterns.items():
        match = re.match(pattern, metric_type)
        if match:
            n_value = int(match.group(1))
            metric_category = pattern_name
            break
    
    # 如果没匹配到动态格式，检查是否是旧的固定格式（保持向后兼容）
    if n_value is None:
        if metric_type in ['deletion_top_5', 'insertion_top_5']:
            n_value = 5
            metric_category = 'deletion_top' if 'deletion' in metric_type else 'insertion_top'
        elif metric_type in ['deletion_top_10', 'insertion_top_10']:
            n_value = 10
            metric_category = 'deletion_top' if 'deletion' in metric_type else 'insertion_top'
        else:
            raise ValueError(f"不支持的指标类型: '{metric_type}'。支持的格式: deletion_top_n, insertion_top_n (n为任意正整数)，或旧的固定格式: deletion_top_5, deletion_top_10, insertion_top_5, insertion_top_10")
    
    # 加载数据
    print("加载数据...")
    score_matrix, stats = load_score_data(model_path, method)
    
    num_agents = stats['num_agents']
    episode_length = stats['episode_length']
    target_timesteps = stats.get('target_timesteps', None)
    
    if target_timesteps is None or len(target_timesteps) == 0:
        max_timestep = episode_length
    else:
        max_timestep = max(target_timesteps)
    
    print(f"  分数矩阵形状: {score_matrix.shape}")
    print(f"  Num agents: {num_agents}")
    print(f"  Max timestep: {max_timestep}")
    
    # 从shapley_stats获取real_risk和baseline_risk
    shapley_stats = load_shapley_stats(model_path)
    real_risk = float(shapley_stats.get('real_risk', shapley_stats.get('real_metric', 0.0)))
    baseline_risk = float(shapley_stats.get('baseline_risk', shapley_stats.get('baseline_metric', 0.0)))
    
    print(f"  Real risk: {real_risk:.6f}")
    print(f"  Baseline risk: {baseline_risk:.6f}")
    
    # 排序动作
    print("排序动作...")
    sorted_actions = sort_actions_by_scores(score_matrix, num_agents, max_timestep)
    print(f"  排序完成，共 {len(sorted_actions)} 个动作")
    if sorted_actions:
        print(f"  最高分: {sorted_actions[0][1]:.6f}")
        print(f"  最低分: {sorted_actions[-1][1]:.6f}")
    
    # 根据metric_type执行不同的计算
    results = {
        "model_path": model_path,
        "method": method,
        "metric_type": metric_type,
        "baseline_risk": baseline_risk,
        "real_risk": real_risk,
        "total_actions": len(sorted_actions),
    }
    
    if metric_category == 'deletion_top':
        risk_decrease = compute_deletion_top_n(sorted_actions, model_path, stats, n_value, real_risk)
        results["risk_decrease_relative"] = float(risk_decrease)
        results["n"] = n_value
        
    elif metric_category == 'insertion_top':
        risk_increase = compute_insertion_top_n(sorted_actions, model_path, stats, n_value, real_risk, baseline_risk)
        results["risk_increase_relative"] = float(risk_increase)
        results["n"] = n_value
    
    # 保存结果
    model_name, model_id = model_path.split('/')
    output_dir = RESULTS_ROOT / "faithfulness_exp" / model_name / model_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f"faithfulness_results_{method}_{metric_type}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存到: {results_file}")
    
    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='计算EconAgent的faithfulness指标',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 计算shapley方法的deletion_top_5指标
  python scripts/faithfulness_exp/compute_faithfulness.py gpt/gpt_42 --method shapley --metric_type deletion_top_5
  
  # 计算llm方法的insertion_top_10指标
  python scripts/faithfulness_exp/compute_faithfulness.py gpt/gpt_42 --method llm --metric_type insertion_top_10
        """
    )
    
    parser.add_argument(
        'model_path',
        type=str,
        help='模型路径，格式为 "model/model_id"，如 "gpt/gpt_42"'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        default='shapley',
        choices=['shapley', 'random', 'llm', 'mast', 'loo'],
        help='归因方法（默认: shapley）'
    )
    
    parser.add_argument(
        '--metric_type',
        type=str,
        default='deletion_top_5',
        # 支持动态格式：deletion_top_n, insertion_top_n (n为任意正整数)
        # 也支持旧的固定格式：deletion_top_5, deletion_top_10, insertion_top_5, insertion_top_10
        help='指标类型（默认: deletion_top_5）。支持格式: deletion_top_n, insertion_top_n (n为任意正整数)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='输出详细信息'
    )
    
    args = parser.parse_args()
    
    try:
        results = compute_faithfulness(
            model_path=args.model_path,
            method=args.method,
            metric_type=args.metric_type,
            verbose=args.verbose
        )
        return 0
    except Exception as e:
        print(f"错误: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
