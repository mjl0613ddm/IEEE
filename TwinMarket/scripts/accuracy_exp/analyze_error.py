#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析MC Shapley值的误差
计算 ||Φ - Φ*||₂ / ||Φ*||₂
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def compute_relative_error(phi_mc: np.ndarray, phi_exact: np.ndarray) -> float:
    """
    计算相对误差：||Φ - Φ*||₂ / ||Φ*||₂
    使用spectral norm（TwinMarket原来的方法）
    
    Args:
        phi_mc: MC Shapley值 (num_agents, num_dates)
        phi_exact: Exact Shapley值 (num_agents, num_dates)
        
    Returns:
        相对误差值
    """
    # 计算L2范数（对2D数组使用ord=2计算spectral norm）
    diff = phi_mc - phi_exact
    error_norm = np.linalg.norm(diff, ord=2)
    exact_norm = np.linalg.norm(phi_exact, ord=2)
    
    if exact_norm == 0:
        # 如果exact值为0，使用绝对误差
        return error_norm
    
    return error_norm / exact_norm


def analyze_error(
    output_dir: Path,
    n_samples_list: List[int],
    seeds: List[int],
    metric_name: str = 'risk_indicator_simple',
    baseline_type: str = 'hold',
    verbose: bool = False
) -> Dict:
    """
    分析MC Shapley值的误差
    
    Args:
        output_dir: 输出目录（包含exact和MC Shapley值文件）
        n_samples_list: MC采样次数列表
        seeds: 随机种子列表
        metric_name: 风险指标名称
        baseline_type: Baseline策略类型
        verbose: 是否输出详细信息
        
    Returns:
        分析结果字典
    """
    # 加载exact Shapley值
    exact_shapley_files = list(output_dir.glob("exact_shapley_seed*.npy"))
    if not exact_shapley_files:
        raise FileNotFoundError(f"未找到exact Shapley值文件: {output_dir}")
    
    # 加载所有种子的exact值（如果有多个种子）
    exact_shapley_all = []
    exact_seeds = []
    for exact_file in sorted(exact_shapley_files):
        seed_str = exact_file.stem.split('seed')[1]
        seed = int(seed_str)
        phi_exact = np.load(exact_file)
        exact_shapley_all.append(phi_exact)
        exact_seeds.append(seed)
    
    # 如果只有一个种子，使用它；如果有多个，使用均值
    if len(exact_shapley_all) == 1:
        phi_exact_mean = exact_shapley_all[0]
        exact_seed_used = exact_seeds[0]
    else:
        phi_exact_mean = np.mean(exact_shapley_all, axis=0)
        exact_seed_used = None
    
    # 保存mean文件（如果不存在）
    mean_file = output_dir / "exact_shapley_values_mean.npy"
    if not mean_file.exists():
        np.save(mean_file, phi_exact_mean)
    
    if verbose:
        print(f"加载Exact Shapley值:")
        print(f"  文件数: {len(exact_shapley_all)}")
        print(f"  种子: {exact_seeds}")
        print(f"  形状: {phi_exact_mean.shape}")
        print(f"  总和: {np.sum(phi_exact_mean):.6f}")
        print(f"  均值: {np.mean(phi_exact_mean):.6f}")
        print(f"  标准差: {np.std(phi_exact_mean):.6f}")
    
    # 计算每个采样次数、每个种子的误差
    results = []
    all_errors = {n_samples: [] for n_samples in n_samples_list}
    
    for n_samples in n_samples_list:
        errors_for_n_samples = []
        
        for seed in seeds:
            mc_file = output_dir / f"mc_shapley_n{n_samples}_seed{seed}.npy"
            if not mc_file.exists():
                if verbose:
                    print(f"⚠️  警告: MC文件不存在: {mc_file}")
                continue
            
            phi_mc = np.load(mc_file)
            
            # 使用相同的exact值（如果exact也有对应的种子，优先使用）
            if exact_seed_used and exact_seed_used == seed and len(exact_shapley_all) > 1:
                # 找到对应种子的exact值
                seed_idx = exact_seeds.index(seed)
                phi_exact = exact_shapley_all[seed_idx]
            else:
                phi_exact = phi_exact_mean
            
            # 计算误差
            error = compute_relative_error(phi_mc, phi_exact)
            errors_for_n_samples.append(error)
            all_errors[n_samples].append(error)
            
            if verbose:
                print(f"  n_samples={n_samples}, seed={seed}: error={error:.6f}")
        
        if errors_for_n_samples:
            mean_error = np.mean(errors_for_n_samples)
            std_error = np.std(errors_for_n_samples)
            min_error = np.min(errors_for_n_samples)
            max_error = np.max(errors_for_n_samples)
            
            results.append({
                'n_samples': n_samples,
                'mean_error': float(mean_error),
                'std_error': float(std_error),
                'min_error': float(min_error),
                'max_error': float(max_error),
                'errors': [float(e) for e in errors_for_n_samples],
                'mean_mc_shapley_path': None
            })
    
    # 准备结果字典
    num_agents, num_dates = phi_exact_mean.shape
    
    analysis_results = {
        'exact_shapley_path': str(output_dir / "exact_shapley_values_mean.npy"),
        'exact_shapley_sum': float(np.sum(phi_exact_mean)),
        'exact_shapley_mean': float(np.mean(phi_exact_mean)),
        'exact_shapley_std': float(np.std(phi_exact_mean)),
        'num_agents': num_agents,
        'episode_length': num_dates,
        'seeds': seeds,
        'metric_name': metric_name,
        'baseline_type': baseline_type,
        'computation_method': 'per_seed',
        'results': results
    }
    
    return analysis_results


def main():
    parser = argparse.ArgumentParser(description='分析MC Shapley值的误差')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录（包含exact和MC Shapley值文件）')
    parser.add_argument('--base_path', type=str, default=None, help='项目根目录路径（默认：自动检测）')
    parser.add_argument('--n_samples_list', type=int, nargs='+', default=[10, 100, 1000, 10000], help='MC采样次数列表')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 401, 45, 46, 613, 48, 49, 50, 51], help='随机种子列表')
    parser.add_argument('--metric_name', type=str, default='risk_indicator_simple', help='风险指标名称')
    parser.add_argument('--baseline_type', type=str, default='hold', help='Baseline策略类型')
    parser.add_argument('--verbose', action='store_true', help='输出详细信息')
    
    args = parser.parse_args()
    
    # 确定项目根目录
    if args.base_path is None:
        base_path = Path(__file__).parent.parent.parent
    else:
        base_path = Path(args.base_path)
    
    # 确定输出目录
    if args.output_dir.startswith('results/'):
        output_dir = base_path / args.output_dir
    elif Path(args.output_dir).is_absolute():
        output_dir = Path(args.output_dir)
    else:
        output_dir = base_path / "results" / args.output_dir
    
    if not output_dir.exists():
        raise FileNotFoundError(f"输出目录不存在: {output_dir}")
    
    # 分析误差
    analysis_results = analyze_error(
        output_dir=output_dir,
        n_samples_list=args.n_samples_list,
        seeds=args.seeds,
        metric_name=args.metric_name,
        baseline_type=args.baseline_type,
        verbose=args.verbose
    )
    
    # 保存JSON结果
    json_file = output_dir / "error_analysis_results.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    print(f"✅ 误差分析结果已保存到: {json_file}")
    
    # 保存CSV结果
    csv_data = []
    for result in analysis_results['results']:
        csv_data.append({
            'n_samples': result['n_samples'],
            'mean_error': result['mean_error'],
            'std_error': result['std_error'],
            'min_error': result['min_error'],
            'max_error': result['max_error']
        })
    
    csv_file = output_dir / "error_analysis.csv"
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, index=False)
    print(f"✅ 误差分析CSV已保存到: {csv_file}")
    
    # 打印摘要
    print("\n" + "="*50)
    print("误差分析摘要")
    print("="*50)
    print(f"Agent数量: {analysis_results['num_agents']}")
    print(f"时间步数: {analysis_results['episode_length']}")
    print(f"种子数: {len(analysis_results['seeds'])}")
    print(f"Exact Shapley值总和: {analysis_results['exact_shapley_sum']:.6f}")
    print(f"Exact Shapley值均值: {analysis_results['exact_shapley_mean']:.6f}")
    print(f"Exact Shapley值标准差: {analysis_results['exact_shapley_std']:.6f}")
    print("\nMC采样误差统计:")
    for result in analysis_results['results']:
        print(f"  n_samples={result['n_samples']:5d}: "
              f"mean={result['mean_error']:.6f}, "
              f"std={result['std_error']:.6f}, "
              f"min={result['min_error']:.6f}, "
              f"max={result['max_error']:.6f}")


if __name__ == "__main__":
    main()

