#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
保存 SocialLLM 的风险数据为 npy 文件
参考 TwinMarket 的数据格式，保存以下文件：
1. shapley_values.npy: Shapley值矩阵 (num_agents × num_timesteps)
2. risk_evolution.npy: 风险演化曲线 (num_timesteps,) - 按时间聚合的Shapley值
3. cumulative_risk.npy: 累积风险曲线 (num_timesteps,) - risk_evolution的累积和
4. risk_timeseries.npy: 风险时间序列 (num_timesteps,) - 从results.json读取的每一步风险值
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_shapley_data(result_dir: Path) -> Tuple[np.ndarray, Dict, List[int], List[int]]:
    """
    加载Shapley数据
    
    Args:
        result_dir: 结果目录路径（包含shapley/目录）
        
    Returns:
        (shapley_matrix, stats, agent_ids, timesteps)
    """
    shapley_file = result_dir / "shapley" / "shapley_attribution_timeseries.csv"
    shapley_stats_file = result_dir / "shapley" / "shapley_stats.json"
    
    if not shapley_file.exists():
        raise FileNotFoundError(f"Shapley CSV file not found: {shapley_file}")
    
    if not shapley_stats_file.exists():
        raise FileNotFoundError(f"Shapley stats file not found: {shapley_stats_file}")
    
    # 加载CSV文件
    print(f"Loading shapley data from: {shapley_file}")
    
    # 读取CSV文件并收集数据
    agent_ids_set = set()
    timesteps_set = set()
    shapley_data = []
    
    with open(shapley_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            agent_id = int(row['agent_id'])
            timestep = int(row['timestep'])
            shapley_value = float(row['shapley_value'])
            
            agent_ids_set.add(agent_id)
            timesteps_set.add(timestep)
            shapley_data.append((agent_id, timestep, shapley_value))
    
    # 获取agent_ids和timesteps
    agent_ids = sorted(list(agent_ids_set))
    timesteps = sorted(list(timesteps_set))
    
    num_agents = len(agent_ids)
    num_timesteps = len(timesteps)
    
    print(f"  Loaded {len(shapley_data)} records")
    print(f"  Agents: {num_agents}, Timesteps: {num_timesteps}")
    
    # 加载统计信息
    with open(shapley_stats_file, 'r', encoding='utf-8') as f:
        stats = json.load(f)
    print(f"Loaded stats from: {shapley_stats_file}")
    
    # 创建agent_id和timestep到索引的映射
    agent_id_to_idx = {aid: idx for idx, aid in enumerate(agent_ids)}
    timestep_to_idx = {t: idx for idx, t in enumerate(timesteps)}
    
    # 构建Shapley矩阵 (num_agents × num_timesteps)
    shapley_matrix = np.zeros((num_agents, num_timesteps), dtype=np.float64)
    
    for agent_id, timestep, shapley_value in shapley_data:
        if agent_id in agent_id_to_idx and timestep in timestep_to_idx:
            agent_idx = agent_id_to_idx[agent_id]
            timestep_idx = timestep_to_idx[timestep]
            shapley_matrix[agent_idx, timestep_idx] = shapley_value
    
    return shapley_matrix, stats, agent_ids, timesteps


def load_risk_timeseries(result_dir: Path) -> Tuple[np.ndarray, List[int]]:
    """
    从results.json加载风险时间序列数据
    
    Args:
        result_dir: 结果目录路径（包含results.json）
        
    Returns:
        (risk_timeseries, timesteps) - 风险时间序列数组和时间步列表
    """
    results_file = result_dir / "results.json"
    
    if not results_file.exists():
        print(f"Warning: results.json not found: {results_file}")
        return None, None
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    timestep_results = data.get('timestep_results', [])
    
    if not timestep_results:
        print(f"Warning: timestep_results is empty in {results_file}")
        return None, None
    
    # 提取时间步和风险值
    timesteps = [t['timestep'] for t in timestep_results]
    risks = [t['risk'] for t in timestep_results]
    
    risk_timeseries = np.array(risks, dtype=np.float64)
    
    return risk_timeseries, timesteps


def save_risk_data(result_dir: Path, output_dir: Path = None):
    """
    保存风险数据为 npy 文件
    
    Args:
        result_dir: 结果目录路径（包含shapley/目录）
        output_dir: 输出目录（默认：result_dir/data）
    """
    if output_dir is None:
        output_dir = result_dir / "data"
    else:
        output_dir = Path(output_dir)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"Processing: {result_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # 1. 加载Shapley数据
    print("\nStep 1: Loading Shapley data...")
    shapley_matrix, stats, agent_ids, timesteps = load_shapley_data(result_dir)
    print(f"  Shapley matrix shape: {shapley_matrix.shape}")
    
    # 2. 计算 risk_evolution（按时间聚合，沿着agent维度求和）
    print("\nStep 2: Calculating risk_evolution...")
    risk_evolution = np.sum(shapley_matrix, axis=0)  # shape: (num_timesteps,)
    print(f"  Risk evolution shape: {risk_evolution.shape}")
    print(f"  Risk evolution sum: {np.sum(risk_evolution):.6f}")
    
    # 3. 计算 cumulative_risk（累积和）
    print("\nStep 3: Calculating cumulative risk...")
    cumulative_risk = np.cumsum(risk_evolution)  # shape: (num_timesteps,)
    print(f"  Cumulative risk shape: {cumulative_risk.shape}")
    print(f"  Final cumulative risk: {cumulative_risk[-1]:.6f}")
    
    # 4. 加载风险时间序列（从results.json）
    print("\nStep 4: Loading risk timeseries from results.json...")
    risk_timeseries, risk_timesteps = load_risk_timeseries(result_dir)
    if risk_timeseries is not None:
        print(f"  Risk timeseries shape: {risk_timeseries.shape}")
        print(f"  Risk timeseries range: [{np.min(risk_timeseries):.6f}, {np.max(risk_timeseries):.6f}]")
    else:
        print("  Warning: Could not load risk timeseries from results.json")
    
    # 5. 保存数据
    print("\nStep 5: Saving data...")
    
    # 5.1 保存 shapley_values.npy
    shapley_output_path = output_dir / "shapley_values.npy"
    np.save(shapley_output_path, shapley_matrix)
    print(f"  ✓ Saved shapley_values: {shapley_output_path}")
    print(f"    Shape: {shapley_matrix.shape} (agents × timesteps)")
    
    # 5.2 保存 risk_evolution.npy
    risk_evolution_output_path = output_dir / "risk_evolution.npy"
    np.save(risk_evolution_output_path, risk_evolution)
    print(f"  ✓ Saved risk_evolution: {risk_evolution_output_path}")
    print(f"    Shape: {risk_evolution.shape} (timesteps,)")
    
    # 5.3 保存 cumulative_risk.npy
    cumulative_output_path = output_dir / "cumulative_risk.npy"
    np.save(cumulative_output_path, cumulative_risk)
    print(f"  ✓ Saved cumulative_risk: {cumulative_output_path}")
    print(f"    Shape: {cumulative_risk.shape} (timesteps,)")
    
    # 5.4 保存 risk_timeseries.npy（从results.json读取的每一步风险值）
    if risk_timeseries is not None:
        risk_timeseries_output_path = output_dir / "risk_timeseries.npy"
        np.save(risk_timeseries_output_path, risk_timeseries)
        print(f"  ✓ Saved risk_timeseries: {risk_timeseries_output_path}")
        print(f"    Shape: {risk_timeseries.shape} (timesteps,)")
    
    # 5.5 保存 stats.json（用于参考）
    stats_output_path = output_dir / "stats.json"
    # 添加额外的元数据
    stats_with_metadata = {
        **stats,
        "num_agents": len(agent_ids),
        "num_timesteps": len(timesteps),
        "agent_ids": agent_ids,
        "timesteps": timesteps,
        "risk_evolution_sum": float(np.sum(risk_evolution)),
        "cumulative_risk_final": float(cumulative_risk[-1]) if len(cumulative_risk) > 0 else 0.0,
    }
    # 如果成功加载了风险时间序列，添加到stats中
    if risk_timeseries is not None:
        stats_with_metadata["risk_timeseries"] = {
            "shape": list(risk_timeseries.shape),
            "min": float(np.min(risk_timeseries)),
            "max": float(np.max(risk_timeseries)),
            "mean": float(np.mean(risk_timeseries)),
            "timesteps": risk_timesteps,
        }
    with open(stats_output_path, 'w', encoding='utf-8') as f:
        json.dump(stats_with_metadata, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved stats: {stats_output_path}")
    
    print("\n" + "=" * 60)
    print("✓ All data saved successfully!")
    print("=" * 60)
    
    result = {
        "shapley_values": shapley_matrix,
        "risk_evolution": risk_evolution,
        "cumulative_risk": cumulative_risk,
        "stats": stats_with_metadata
    }
    
    if risk_timeseries is not None:
        result["risk_timeseries"] = risk_timeseries
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='保存 SocialLLM 的风险数据为 npy 文件',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--result_dir',
        type=str,
        required=True,
        help='结果目录路径（包含shapley/目录）'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='输出目录（默认：{result_dir}/data）'
    )
    
    args = parser.parse_args()
    
    # 解析结果目录路径
    result_dir = Path(args.result_dir)
    if not result_dir.is_absolute():
        result_dir = project_root / result_dir
    result_dir = result_dir.resolve()
    
    if not result_dir.exists():
        print(f"❌ 错误: 结果目录不存在: {result_dir}", file=sys.stderr)
        sys.exit(1)
    
    # 解析输出目录路径
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir
        output_dir = output_dir.resolve()
    
    # 保存数据
    try:
        save_risk_data(result_dir, output_dir)
    except Exception as e:
        print(f"❌ 错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
