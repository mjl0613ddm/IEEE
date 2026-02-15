#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leave-One-Out (LOO) Baseline方法：通过移除每个action并计算风险下降值来评估action的重要性
用于faithfulness实验的baseline对比

使用方法:
    python scripts/faithfulness_exp/compute_loo_baseline.py gpt/gpt_42
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

# 添加项目路径
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_ROOT = PROJECT_ROOT / "datas"
sys.path.insert(0, str(PROJECT_ROOT))

# 导入shapley.py中的函数
from scripts.core.shapley import (
    run_counterfactual_simulation,
    _compute_risk_from_metrics,
    load_real_actions
)


def load_shapley_stats(model_path):
    """加载shapley stats获取实验参数"""
    sim_dir = DATA_ROOT / model_path
    shapley_dir = sim_dir / "shapley"
    
    stats_file = shapley_dir / "shapley_stats.json"
    if not stats_file.exists():
        raise FileNotFoundError(f"Shapley stats file not found: {stats_file}")
    
    with open(stats_file, 'r') as f:
        shapley_stats = json.load(f)
    
    return shapley_stats


def compute_loo_baseline(model_path):
    """使用Leave-One-Out方法计算baseline分数"""
    print(f"处理: {model_path}")
    print("="*60)
    
    # 加载数据
    print("\n加载数据...")
    shapley_stats = load_shapley_stats(model_path)
    
    num_agents = shapley_stats['num_agents']
    episode_length = shapley_stats['episode_length']
    target_timesteps = shapley_stats.get('target_timesteps', None)
    baseline_work = shapley_stats.get('baseline_work', 1.0)
    baseline_consumption = shapley_stats.get('baseline_consumption', 0.8)
    
    if target_timesteps is None or len(target_timesteps) == 0:
        max_timestep = episode_length
    else:
        max_timestep = max(target_timesteps)
    
    print(f"  Num agents: {num_agents}")
    print(f"  Max timestep: {max_timestep}")
    print(f"  Episode length: {episode_length}")
    
    # 加载真实动作
    sim_dir = DATA_ROOT / model_path
    actions_json_file = sim_dir / "actions_json" / "all_actions.json"
    if not actions_json_file.exists():
        raise FileNotFoundError(f"Actions JSON file not found: {actions_json_file}")
    real_actions_dict = load_real_actions(str(actions_json_file))
    
    # 创建所有action的集合（在target_timesteps范围内）
    all_actions = set()
    for agent_id in range(num_agents):
        for timestep in range(1, max_timestep + 1):  # timestep从1开始
            all_actions.add((agent_id, timestep))
    
    print(f"  总action数: {len(all_actions)}")
    
    # 计算real_risk（全部真实动作）
    print("\n计算real_risk（全部真实动作）...")
    real_metrics = run_counterfactual_simulation(
        num_agents=num_agents,
        episode_length=episode_length,
        real_actions_dict=real_actions_dict,
        baseline_work=baseline_work,
        baseline_consumption=baseline_consumption,
        subset_U=all_actions,  # 全部动作
        seed=shapley_stats.get('seed', None),
        use_probabilistic_baseline=shapley_stats.get('use_probabilistic_baseline', False)
    )
    
    target_t = target_timesteps[0] if target_timesteps and len(target_timesteps) > 0 else None
    real_risk = _compute_risk_from_metrics(
        real_metrics,
        target_timestep=target_t,
        metric_name=shapley_stats.get('metric_name', 'risk_indicator_naive'),
        inflation_threshold=shapley_stats.get('inflation_threshold', 0.0),
        use_metric_directly=shapley_stats.get('use_metric_directly', False),
        risk_aggregation=shapley_stats.get('risk_aggregation', 'max'),
        include_both_risks=shapley_stats.get('include_both_risks', True),
        verbose=False,
        risk_lambda=shapley_stats.get('risk_lambda', 0.94)
    )
    
    print(f"  Real risk: {real_risk:.6f}")
    
    # 计算baseline_risk（全部baseline动作）
    print("\n计算baseline_risk（全部baseline动作）...")
    baseline_metrics = run_counterfactual_simulation(
        num_agents=num_agents,
        episode_length=episode_length,
        real_actions_dict=real_actions_dict,
        baseline_work=baseline_work,
        baseline_consumption=baseline_consumption,
        subset_U=set(),  # 空集表示全部使用baseline
        seed=shapley_stats.get('seed', None),
        use_probabilistic_baseline=shapley_stats.get('use_probabilistic_baseline', False)
    )
    
    baseline_risk = _compute_risk_from_metrics(
        baseline_metrics,
        target_timestep=target_t,
        metric_name=shapley_stats.get('metric_name', 'risk_indicator_naive'),
        inflation_threshold=shapley_stats.get('inflation_threshold', 0.0),
        use_metric_directly=shapley_stats.get('use_metric_directly', False),
        risk_aggregation=shapley_stats.get('risk_aggregation', 'max'),
        include_both_risks=shapley_stats.get('include_both_risks', True),
        verbose=False,
        risk_lambda=shapley_stats.get('risk_lambda', 0.94)
    )
    
    print(f"  Baseline risk: {baseline_risk:.6f}")
    
    # 计算每个action的LOO分数
    print("\n计算LOO分数（Leave-One-Out）...")
    print(f"  需要运行 {len(all_actions)} 次模拟...")
    
    loo_scores_detailed = []
    risk_drops = []
    
    for idx, (agent_id, timestep) in enumerate(sorted(all_actions)):
        # 创建subset_U，包含除当前action外的所有动作
        subset_U = all_actions - {(agent_id, timestep)}
        
        # 运行counterfactual simulation
        metrics = run_counterfactual_simulation(
            num_agents=num_agents,
            episode_length=episode_length,
            real_actions_dict=real_actions_dict,
            baseline_work=baseline_work,
            baseline_consumption=baseline_consumption,
            subset_U=subset_U,
            seed=shapley_stats.get('seed', None),
            use_probabilistic_baseline=shapley_stats.get('use_probabilistic_baseline', False)
        )
        
        # 计算风险
        risk_without = _compute_risk_from_metrics(
            metrics,
            target_timestep=target_t,
            metric_name=shapley_stats.get('metric_name', 'risk_indicator_naive'),
            inflation_threshold=shapley_stats.get('inflation_threshold', 0.0),
            use_metric_directly=shapley_stats.get('use_metric_directly', False),
            risk_aggregation=shapley_stats.get('risk_aggregation', 'max'),
            include_both_risks=shapley_stats.get('include_both_risks', True),
            verbose=False,
            risk_lambda=shapley_stats.get('risk_lambda', 0.94)
        )
        
        # 计算风险下降值
        risk_drop = real_risk - risk_without
        risk_drops.append(risk_drop)
        
        loo_scores_detailed.append({
            "agent_id": agent_id,
            "timestep": timestep,
            "risk_drop": float(risk_drop),
            "risk_without": float(risk_without)
        })
        
        if (idx + 1) % 10 == 0:
            print(f"  处理进度: {idx + 1}/{len(all_actions)}, 当前action: agent_{agent_id}, timestep_{timestep}, risk_drop: {risk_drop:.6f}")
    
    print(f"  完成！共处理 {len(all_actions)} 个action")
    
    # 归一化分数到0-1范围
    print("\n归一化分数...")
    risk_drops_array = np.array(risk_drops)
    min_drop = np.min(risk_drops_array)
    max_drop = np.max(risk_drops_array)
    
    print(f"  Min risk drop: {min_drop:.6f}")
    print(f"  Max risk drop: {max_drop:.6f}")
    
    if max_drop > min_drop:
        normalized_scores = (risk_drops_array - min_drop) / (max_drop - min_drop)
    else:
        # 如果所有risk_drop相同，归一化到0.5
        normalized_scores = np.ones_like(risk_drops_array) * 0.5
    
    # 更新详细分数列表，添加归一化分数
    for i, score_info in enumerate(loo_scores_detailed):
        score_info["normalized_score"] = float(normalized_scores[i])
    
    # 创建分数矩阵
    print("\n创建分数矩阵...")
    score_matrix = np.zeros((num_agents, max_timestep))
    
    for i, score_info in enumerate(loo_scores_detailed):
        agent_id = score_info["agent_id"]
        timestep = score_info["timestep"]
        normalized_score = score_info["normalized_score"]
        # timestep从1开始，矩阵索引从0开始
        score_matrix[agent_id, timestep - 1] = normalized_score
    
    print(f"  分数矩阵形状: {score_matrix.shape}")
    
    # 计算统计信息
    score_stats = {
        "mean": float(np.mean(normalized_scores)),
        "std": float(np.std(normalized_scores)),
        "min": float(np.min(normalized_scores)),
        "max": float(np.max(normalized_scores)),
        "sum": float(np.sum(normalized_scores)),
        "median": float(np.median(normalized_scores))
    }
    
    print(f"  分数统计:")
    print(f"    Mean: {score_stats['mean']:.6f}")
    print(f"    Std: {score_stats['std']:.6f}")
    print(f"    Min: {score_stats['min']:.6f}")
    print(f"    Max: {score_stats['max']:.6f}")
    print(f"    Median: {score_stats['median']:.6f}")
    
    # 保存结果
    output_dir = sim_dir / "faithfulness_exp" / "loo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存分数矩阵
    scores_file = output_dir / "loo_scores.npy"
    np.save(scores_file, score_matrix)
    print(f"\n分数矩阵已保存到: {scores_file}")
    
    # 保存详细分数
    detailed_file = output_dir / "loo_scores_detailed.json"
    with open(detailed_file, 'w') as f:
        json.dump({"scores": loo_scores_detailed}, f, indent=2)
    print(f"详细分数已保存到: {detailed_file}")
    
    # 保存统计信息
    stats = {
        "method": "loo",
        "baseline_risk": float(baseline_risk),
        "real_risk": float(real_risk),
        "num_agents": num_agents,
        "max_timestep": max_timestep,
        "episode_length": episode_length,
        "target_timesteps": target_timesteps if target_timesteps else [],
        "score_stats": score_stats,
        "risk_drop_stats": {
            "mean": float(np.mean(risk_drops_array)),
            "std": float(np.std(risk_drops_array)),
            "min": float(min_drop),
            "max": float(max_drop)
        }
    }
    
    # 从shapley_stats复制配置参数
    for key in ['metric_name', 'baseline_type', 'baseline_work', 'baseline_consumption',
                'risk_lambda', 'seed', 'inflation_threshold', 'use_metric_directly',
                'risk_aggregation', 'include_both_risks', 'use_probabilistic_baseline']:
        if key in shapley_stats:
            stats[key] = shapley_stats[key]
    
    stats_file = output_dir / "loo_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"统计信息已保存到: {stats_file}")
    
    print("\n" + "="*60)
    print("完成!")
    print("="*60)
    
    return score_matrix, stats


def main():
    parser = argparse.ArgumentParser(
        description='使用Leave-One-Out方法计算baseline分数矩阵',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 计算gpt/gpt_42的LOO baseline
  python scripts/faithfulness_exp/compute_loo_baseline.py gpt/gpt_42
        """
    )
    
    parser.add_argument(
        'model_path',
        type=str,
        help='模型路径，格式为 "model/model_id"，如 "gpt/gpt_42"'
    )
    
    args = parser.parse_args()
    
    try:
        score_matrix, stats = compute_loo_baseline(args.model_path)
        return 0
    except Exception as e:
        print(f"\n错误: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
