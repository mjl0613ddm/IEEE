#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Baseline方法：为所有action随机生成分数（0-1之间）
用于faithfulness实验的baseline对比

使用方法:
    python scripts/faithfulness_exp/compute_random_baseline.py --result_dir results/gpt-4o-mini/gpt-4o-mini_42
    python scripts/faithfulness_exp/compute_random_baseline.py --result_dir results/gpt-4o-mini/gpt-4o-mini_42 --seed 123
"""

import os
import sys
import json
import argparse
import csv
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Union

# 尝试导入pandas，如果不可用则使用csv模块
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_results(results_file: Path) -> Dict:
    """加载results.json文件"""
    if not results_file.exists():
        raise FileNotFoundError(f"Results文件不存在: {results_file}")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_shapley_stats(shapley_dir: Path) -> Optional[Dict]:
    """加载shapley stats获取实验参数"""
    if not shapley_dir.exists():
        return None
    
    stats_file = shapley_dir / "shapley_stats.json"
    if not stats_file.exists():
        return None
    
    with open(stats_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_random_baseline(result_dir: Path, seed: Optional[int] = None) -> pd.DataFrame:
    """
    使用Random方法计算baseline分数
    
    Args:
        result_dir: 结果目录路径
        seed: 随机种子（如果为None，则从shapley_stats中读取）
    
    Returns:
        DataFrame with columns: agent_id, timestep, random_value
    """
    results_file = result_dir / "results.json"
    shapley_dir = result_dir / "shapley"
    
    # 加载数据
    results_data = load_results(results_file)
    shapley_stats = load_shapley_stats(shapley_dir)
    
    # 获取参数
    num_agents = results_data.get('num_agents', 20)
    max_risk_timestep = results_data.get('max_risk_timestep')
    if max_risk_timestep is None:
        max_risk_timestep = results_data.get('num_steps', 30)
    
    # 获取随机种子
    if seed is None:
        if shapley_stats:
            seed = shapley_stats.get('seed', 42)
        else:
            seed = 42
    
    print(f"处理: {result_dir.name}")
    print("="*60)
    print(f"  Num agents: {num_agents}")
    print(f"  Max timestep: {max_risk_timestep}")
    print(f"  Random seed: {seed}")
    
    # 生成随机分数矩阵（0-1之间均匀分布）
    print("\n生成随机分数矩阵...")
    np.random.seed(seed)
    score_matrix = np.random.uniform(0.0, 1.0, size=(num_agents, max_risk_timestep))
    
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
    
    # 转换为DataFrame
    rows = []
    for agent_id in range(num_agents):
        for timestep in range(max_risk_timestep):
            rows.append({
                'agent_id': agent_id,
                'timestep': timestep,
                'random_value': float(score_matrix[agent_id, timestep])
            })
    
    df = pd.DataFrame(rows)
    
    return df, score_stats


def main():
    parser = argparse.ArgumentParser(
        description='使用Random方法计算baseline分数',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--result_dir',
        type=str,
        required=True,
        help='结果目录路径（包含results.json）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='随机种子（可选，默认从shapley_stats.json读取或使用42）'
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='如果结果文件已存在，则跳过'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='输出详细信息'
    )
    
    args = parser.parse_args()
    
    # 转换为Path对象
    result_dir = Path(args.result_dir).resolve()
    
    if not result_dir.exists():
        print(f"错误: 结果目录不存在: {result_dir}", file=sys.stderr)
        sys.exit(1)
    
    # 检查输出文件
    output_dir = result_dir / "faithfulness_exp" / "random"
    output_file = output_dir / "random_attribution_timeseries.csv"
    stats_file = output_dir / "random_stats.json"
    
    if args.skip_existing and output_file.exists():
        print(f"跳过已存在的文件: {output_file}")
        return
    
    # 计算random baseline
    try:
        df, score_stats = compute_random_baseline(result_dir, args.seed)
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存CSV文件
        df.to_csv(output_file, index=False)
        print(f"\n✓ CSV文件已保存到: {output_file}")
        
        # 保存统计信息
        results_data = load_results(result_dir / "results.json")
        stats = {
            "method": "random",
            "num_agents": results_data.get('num_agents', 20),
            "max_risk_timestep": results_data.get('max_risk_timestep'),
            "max_risk": float(results_data.get('max_risk', 0.0)),
            "initial_risk": float(results_data.get('initial_risk', 0.0)),
            "score_stats": score_stats,
            "random_seed": args.seed if args.seed is not None else (load_shapley_stats(result_dir / "shapley") or {}).get('seed', 42)
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 统计信息已保存到: {stats_file}")
    
    except Exception as e:
        print(f"错误: 计算random baseline失败: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
