#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Baseline方法：为所有action随机生成分数矩阵（0-1之间）
用于faithfulness实验的baseline对比

使用方法:
    python scripts/faithfulness_exp/compute_random_baseline.py gpt/gpt_42
    python scripts/faithfulness_exp/compute_random_baseline.py gpt/gpt_42 --seed 42
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


def compute_random_baseline(model_path, seed=None):
    """使用Random方法计算baseline分数"""
    print(f"处理: {model_path}")
    print("="*60)
    
    # 加载数据
    print("\n加载数据...")
    shapley_stats = load_shapley_stats(model_path)
    
    num_agents = shapley_stats['num_agents']
    episode_length = shapley_stats['episode_length']
    target_timesteps = shapley_stats.get('target_timesteps', None)
    baseline_risk = shapley_stats.get('baseline_risk', 0.0)
    real_risk = shapley_stats.get('real_risk', 0.0)
    
    if target_timesteps is None or len(target_timesteps) == 0:
        max_timestep = episode_length
    else:
        max_timestep = max(target_timesteps)
    
    print(f"  Num agents: {num_agents}")
    print(f"  Max timestep: {max_timestep}")
    print(f"  Episode length: {episode_length}")
    
    # 使用seed（如果提供）或从shapley_stats获取
    if seed is None:
        seed = shapley_stats.get('seed', 42)
    
    print(f"  Random seed: {seed}")
    
    # 生成随机分数矩阵（0-1之间均匀分布）
    print("\n生成随机分数矩阵...")
    np.random.seed(seed)
    score_matrix = np.random.uniform(0.0, 1.0, size=(num_agents, max_timestep))
    
    print(f"  分数矩阵形状: {score_matrix.shape}")
    
    # 计算统计信息
    score_stats = {
        "mean": float(np.mean(score_matrix)),
        "std": float(np.std(score_matrix)),
        "min": float(np.min(score_matrix)),
        "max": float(np.max(score_matrix)),
        "sum": float(np.sum(score_matrix)),
        "median": float(np.median(score_matrix))
    }
    
    print(f"  分数统计:")
    print(f"    Mean: {score_stats['mean']:.6f}")
    print(f"    Std: {score_stats['std']:.6f}")
    print(f"    Min: {score_stats['min']:.6f}")
    print(f"    Max: {score_stats['max']:.6f}")
    print(f"    Median: {score_stats['median']:.6f}")
    
    # 保存结果
    sim_dir = DATA_ROOT / model_path
    output_dir = sim_dir / "faithfulness_exp" / "random"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存分数矩阵
    scores_file = output_dir / "random_scores.npy"
    np.save(scores_file, score_matrix)
    print(f"\n分数矩阵已保存到: {scores_file}")
    
    # 保存统计信息
    stats = {
        "method": "random",
        "baseline_risk": float(baseline_risk),
        "real_risk": float(real_risk),
        "num_agents": num_agents,
        "max_timestep": max_timestep,
        "episode_length": episode_length,
        "target_timesteps": target_timesteps if target_timesteps else [],
        "score_stats": score_stats,
        "random_seed": seed
    }
    
    # 从shapley_stats复制配置参数
    for key in ['metric_name', 'baseline_type', 'baseline_work', 'baseline_consumption',
                'risk_lambda', 'seed', 'inflation_threshold', 'use_metric_directly',
                'risk_aggregation', 'include_both_risks', 'use_probabilistic_baseline']:
        if key in shapley_stats:
            stats[key] = shapley_stats[key]
    
    stats_file = output_dir / "random_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"统计信息已保存到: {stats_file}")
    
    print("\n" + "="*60)
    print("完成!")
    print("="*60)
    
    return score_matrix, stats


def main():
    parser = argparse.ArgumentParser(
        description='使用Random方法计算baseline分数矩阵',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 计算gpt/gpt_42的Random baseline（使用shapley_stats中的seed）
  python scripts/faithfulness_exp/compute_random_baseline.py gpt/gpt_42
  
  # 使用指定的seed
  python scripts/faithfulness_exp/compute_random_baseline.py gpt/gpt_42 --seed 123
        """
    )
    
    parser.add_argument(
        'model_path',
        type=str,
        help='模型路径，格式为 "model/model_id"，如 "gpt/gpt_42"'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='随机数种子（如果不指定，使用shapley_stats中的seed）'
    )
    
    args = parser.parse_args()
    
    try:
        score_matrix, stats = compute_random_baseline(args.model_path, seed=args.seed)
        return 0
    except Exception as e:
        print(f"\n错误: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
