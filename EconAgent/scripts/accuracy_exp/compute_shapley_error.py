#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算完全Shapley值和蒙特卡罗方法的误差
验证蒙特卡罗方法的近似准确性
"""

import os
import sys
import copy
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations
import yaml
from multiprocessing import Pool, Manager
from functools import partial

# 添加项目根目录（EconAgent）到路径
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

import ai_economist.foundation as foundation

# 导入shapley.py中的函数
from scripts.core.shapley import (
    _extract_world_metrics,
    calculate_risk_indicator_from_metrics,
    load_metrics_from_csv,
    run_counterfactual_simulation,
    _compute_risk_from_metrics,
    _compute_risk_cached,
    load_real_actions,
    compute_baseline_values
)

# 加载配置
_config_path = os.path.join(_PROJECT_ROOT, 'config.yaml')
with open(_config_path, "r") as f:
    run_configuration = yaml.safe_load(f)
base_env_config = run_configuration.get('env')


def _compute_risk_for_subset(args_tuple):
    """辅助函数：在并行计算中计算单个子集的风险值"""
    (subset_key_tuple, num_agents, episode_length, real_actions_dict, baseline_work, 
     baseline_consumption, inflation_threshold, seed, target_timestep, metric_name,
     use_metric_directly, use_probabilistic_baseline, risk_aggregation, 
     include_both_risks, risk_lambda) = args_tuple
    
    # 将tuple转换回frozenset
    subset_key = frozenset(subset_key_tuple)
    
    # 计算风险值
    risk = _compute_risk_cached(
        subset_key, num_agents, episode_length,
        real_actions_dict, baseline_work, baseline_consumption,
        inflation_threshold, seed,
        target_timestep=target_timestep,
        metric_name=metric_name,
        use_metric_directly=use_metric_directly,
        use_probabilistic_baseline=use_probabilistic_baseline,
        risk_aggregation=risk_aggregation,
        include_both_risks=include_both_risks,
        risk_lambda=risk_lambda
    )
    
    return (subset_key_tuple, risk)


def compute_exact_shapley(
    num_agents, episode_length,
    real_actions_dict,
    baseline_work, baseline_consumption,
    inflation_threshold=0.1,
    seed=None,
    baseline_metrics_csv=None,
    real_metrics_csv=None,
    target_timestep=None,
    metric_name='risk_indicator_naive',
    use_metric_directly=False,
    use_probabilistic_baseline=False,
    risk_aggregation='max',
    include_both_risks=True,
    risk_lambda=0.94,
    n_jobs=1
):
    """
    计算完全Shapley值（枚举所有子集）
    
    说明：
    - 将每个(agent_id, timestep)组合看作一个player，总共M=num_agents*episode_length个player
    - 例如：5个agent，5个时间步 = 25个player
    - 模拟按照时间顺序进行：t=0,1,2,...,episode_length-1
    - 对于每个子集S，运行完整的episode_length步模拟，其中：
      * 如果(agent_id, t)在S中，使用real action
      * 否则，使用baseline action
    - 归因值：如果target_timestep指定，使用该时间步的risk_indicator_naive值
      （注意：timestep是1-indexed，所以最后一个时间步是episode_length）
    
    Returns:
        shapley_values: np.ndarray of shape (num_agents, episode_length)
        baseline_risk: float
        real_risk: float
    """
    # 创建所有player：每个(agent_id, timestep)组合是一个player
    # 例如：5个agent，5个时间步 = 25个player
    all_players = [(i, t) for i in range(num_agents) for t in range(episode_length)]
    M = len(all_players)
    
    print(f"Computing exact Shapley values (enumerating all {2**M} subsets)...")
    print(f"Total players: {M} (agents={num_agents}, timesteps={episode_length})")
    
    # 计算baseline risk
    if baseline_metrics_csv and os.path.exists(baseline_metrics_csv):
        print(f"Loading baseline metrics from: {baseline_metrics_csv}")
        baseline_metrics = load_metrics_from_csv(baseline_metrics_csv, metric_name, risk_lambda=risk_lambda)
        if baseline_metrics:
            baseline_risk = _compute_risk_from_metrics(
                baseline_metrics, target_timestep, metric_name, inflation_threshold, use_metric_directly,
                risk_aggregation=risk_aggregation, include_both_risks=include_both_risks, verbose=False, risk_lambda=risk_lambda
            )
            print(f"Baseline risk (from file): {baseline_risk:.6f}")
        else:
            baseline_metrics = run_counterfactual_simulation(
                num_agents, episode_length, real_actions_dict,
                baseline_work, baseline_consumption, subset_U=set(), seed=seed,
                use_probabilistic_baseline=use_probabilistic_baseline
            )
            baseline_risk = _compute_risk_from_metrics(
                baseline_metrics, target_timestep, metric_name, inflation_threshold, use_metric_directly,
                risk_aggregation=risk_aggregation, include_both_risks=include_both_risks, verbose=False, risk_lambda=risk_lambda
            )
            print(f"Baseline risk (computed): {baseline_risk:.6f}")
    else:
        print("Computing baseline risk...")
        baseline_metrics = run_counterfactual_simulation(
            num_agents, episode_length, real_actions_dict,
            baseline_work, baseline_consumption, subset_U=set(), seed=seed,
            use_probabilistic_baseline=use_probabilistic_baseline
        )
        baseline_risk = _compute_risk_from_metrics(
            baseline_metrics, target_timestep, metric_name, inflation_threshold, use_metric_directly,
            risk_aggregation=risk_aggregation, include_both_risks=include_both_risks, verbose=False, risk_lambda=risk_lambda
        )
        print(f"Baseline risk: {baseline_risk:.6f}")
    
    # 计算real risk
    if real_metrics_csv and os.path.exists(real_metrics_csv):
        print(f"Loading real metrics from: {real_metrics_csv}")
        real_metrics = load_metrics_from_csv(real_metrics_csv, metric_name, risk_lambda=risk_lambda)
        if real_metrics:
            real_risk = _compute_risk_from_metrics(
                real_metrics, target_timestep, metric_name, inflation_threshold, use_metric_directly,
                risk_aggregation=risk_aggregation, include_both_risks=include_both_risks, verbose=False, risk_lambda=risk_lambda
            )
            print(f"Real risk (from file): {real_risk:.6f}")
        else:
            real_metrics = run_counterfactual_simulation(
                num_agents, episode_length, real_actions_dict,
                baseline_work, baseline_consumption, subset_U=set(all_players), seed=seed,
                use_probabilistic_baseline=use_probabilistic_baseline
            )
            real_risk = _compute_risk_from_metrics(
                real_metrics, target_timestep, metric_name, inflation_threshold, use_metric_directly,
                risk_aggregation=risk_aggregation, include_both_risks=include_both_risks, verbose=False, risk_lambda=risk_lambda
            )
            print(f"Real risk (computed): {real_risk:.6f}")
    else:
        print("Computing real risk...")
        real_metrics = run_counterfactual_simulation(
            num_agents, episode_length, real_actions_dict,
            baseline_work, baseline_consumption, subset_U=set(all_players), seed=seed,
            use_probabilistic_baseline=use_probabilistic_baseline
        )
        real_risk = _compute_risk_from_metrics(
            real_metrics, target_timestep, metric_name, inflation_threshold, use_metric_directly,
            risk_aggregation=risk_aggregation, include_both_risks=include_both_risks, verbose=False, risk_lambda=risk_lambda
        )
        print(f"Real risk: {real_risk:.6f}")
    
    # 初始化Shapley值累加器
    shapley_values = np.zeros((num_agents, episode_length))
    
    # 缓存风险值
    risk_cache = {frozenset(): baseline_risk}
    
    # 计算阶乘缓存
    def factorial(n):
        if n < 0:
            return 0
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result
    
    # 预计算所有阶乘
    fact_cache = {i: factorial(i) for i in range(M + 1)}
    
    # Shapley值公式：phi_i = sum_{S subseteq N\{i}} [|S|!(M-|S|-1)!/M!] * [v(S union {i}) - v(S)]
    # 对于每个玩家i，枚举所有不包含i的子集S
    # 总迭代次数：M * 2^(M-1)，因为每个player需要枚举2^(M-1)个子集
    total_iterations = M * (2 ** (M - 1))
    print(f"Enumerating all subsets for each player...")
    print(f"Total iterations: {total_iterations:,} (M={M} players, 2^(M-1)={2**(M-1):,} subsets per player)")
    if n_jobs > 1:
        print(f"Using {n_jobs} parallel processes for acceleration")
    
    # 收集所有需要计算的子集（去重）
    all_subset_keys = set()
    all_tasks = []  # (player, subset_key_tuple, subset_with_i_key_tuple, subset_size)
    
    for player_idx, player in enumerate(all_players):
        agent_id, timestep = player
        other_players = [p for p in all_players if p != player]
        num_other = len(other_players)
        
        for subset_mask in range(2 ** num_other):
            subset = set()
            for i in range(num_other):
                if subset_mask & (1 << i):
                    subset.add(other_players[i])
            
            subset_key = frozenset(subset)
            subset_with_i = subset | {player}
            subset_with_i_key = frozenset(subset_with_i)
            
            # 转换为tuple以便序列化（用于多进程）
            subset_key_tuple = tuple(sorted(subset_key))
            subset_with_i_key_tuple = tuple(sorted(subset_with_i_key))
            
            all_subset_keys.add(subset_key_tuple)
            all_subset_keys.add(subset_with_i_key_tuple)
            
            all_tasks.append((player, subset_key_tuple, subset_with_i_key_tuple, len(subset)))
    
    # 并行计算所有唯一子集的风险值
    print(f"Computing risks for {len(all_subset_keys)} unique subsets...")
    
    if n_jobs > 1:
        # 准备并行任务参数
        tasks_args = [
            (subset_key_tuple, num_agents, episode_length, real_actions_dict, baseline_work,
             baseline_consumption, inflation_threshold, seed, target_timestep, metric_name,
             use_metric_directly, use_probabilistic_baseline, risk_aggregation,
             include_both_risks, risk_lambda)
            for subset_key_tuple in all_subset_keys
        ]
        
        # 并行计算
        with Pool(processes=n_jobs) as pool:
            results = list(tqdm(
                pool.imap(_compute_risk_for_subset, tasks_args),
                total=len(tasks_args),
                desc="Computing risks (parallel)"
            ))
        
        # 更新缓存
        for subset_key_tuple, risk in results:
            subset_key = frozenset(subset_key_tuple)
            risk_cache[subset_key] = risk
    else:
        # 串行计算（原有逻辑）
        for subset_key_tuple in tqdm(all_subset_keys, desc="Computing risks (serial)"):
            subset_key = frozenset(subset_key_tuple)
            if subset_key not in risk_cache:
                risk = _compute_risk_cached(
                    subset_key, num_agents, episode_length,
                    real_actions_dict, baseline_work, baseline_consumption,
                    inflation_threshold, seed,
                    target_timestep=target_timestep,
                    metric_name=metric_name,
                    use_metric_directly=use_metric_directly,
                    use_probabilistic_baseline=use_probabilistic_baseline,
                    risk_aggregation=risk_aggregation,
                    include_both_risks=include_both_risks,
                    risk_lambda=risk_lambda
                )
                risk_cache[subset_key] = risk
    
    # 计算Shapley值
    print("Computing Shapley values from cached risks...")
    for player, subset_key_tuple, subset_with_i_key_tuple, subset_size in tqdm(all_tasks, desc="Computing Shapley values"):
        agent_id, timestep = player
        
        subset_key = frozenset(subset_key_tuple)
        subset_with_i_key = frozenset(subset_with_i_key_tuple)
        
        risk_S = risk_cache[subset_key]
        risk_S_with_i = risk_cache[subset_with_i_key]
        
        # 边际贡献
        marginal_contrib = risk_S_with_i - risk_S
        
        # 权重：|S|!(M-|S|-1)!/M!
        weight = (fact_cache[subset_size] * fact_cache[M - subset_size - 1]) / fact_cache[M]
        
        shapley_values[agent_id, timestep] += weight * marginal_contrib
    
    print(f"Computed exact Shapley values using {len(risk_cache)} unique subsets")
    
    return shapley_values, baseline_risk, real_risk


def compute_mc_shapley(
    num_agents, episode_length,
    real_actions_dict,
    baseline_work, baseline_consumption,
    inflation_threshold=0.1,
    n_samples=1000,
    seed=None,
    baseline_metrics_csv=None,
    real_metrics_csv=None,
    target_timestep=None,
    metric_name='risk_indicator_naive',
    use_metric_directly=False,
    use_probabilistic_baseline=False,
    risk_aggregation='max',
    include_both_risks=True,
    risk_lambda=0.94
):
    """
    计算蒙特卡罗Shapley值（从shapley.py导入的函数）
    """
    from shapley import monte_carlo_shapley
    
    result = monte_carlo_shapley(
        num_agents, episode_length, real_actions_dict,
        baseline_work, baseline_consumption,
        inflation_threshold, n_samples, seed,
        baseline_metrics_csv=baseline_metrics_csv,
        real_metrics_csv=real_metrics_csv,
        target_timestep=target_timestep,
        target_timesteps=None,
        metric_name=metric_name,
        use_metric_directly=use_metric_directly,
        use_probabilistic_baseline=use_probabilistic_baseline,
        risk_aggregation=risk_aggregation,
        include_both_risks=include_both_risks,
        risk_lambda=risk_lambda
    )
    
    shapley_values, baseline_risk, real_risk, _ = result
    return shapley_values, baseline_risk, real_risk


def compute_relative_error(phi_mc, phi_exact):
    """
    计算相对误差：||Phi - Phi*||_2 / ||Phi*||_2
    
    Args:
        phi_mc: 蒙特卡罗Shapley值 (num_agents, episode_length)
        phi_exact: 完全Shapley值 (num_agents, episode_length)
    
    Returns:
        relative_error: float
    """
    diff = phi_mc - phi_exact
    error_norm = np.linalg.norm(diff.flatten(), ord=2)
    exact_norm = np.linalg.norm(phi_exact.flatten(), ord=2)
    
    if exact_norm == 0:
        if error_norm == 0:
            return 0.0
        else:
            return np.inf
    
    relative_error = error_norm / exact_norm
    return relative_error


def main():
    parser = argparse.ArgumentParser(description='Compute exact Shapley values and MC error')
    parser.add_argument('--num_agents', type=int, default=5)
    parser.add_argument('--episode_length', type=int, default=5)
    parser.add_argument('--baseline_type', type=str, default='average',
                       choices=['fixed', 'average', 'stable'])
    parser.add_argument('--baseline_work', type=float, default=1.0)
    parser.add_argument('--baseline_consumption', type=float, default=0.5)
    parser.add_argument('--stable_period', type=int, default=None)
    parser.add_argument('--use_probabilistic_baseline', action='store_true')
    parser.add_argument('--inflation_threshold', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                       help='Multiple seeds for averaging (default: use --seed)')
    parser.add_argument('--real_actions_json', type=str, required=True,
                       help='Path to real actions JSON (e.g., gpt-4o-verify/actions_json/all_actions.json)')
    parser.add_argument('--baseline_actions_json', type=str, required=True,
                       help='Path to baseline actions JSON (e.g., baseline-verify/actions_json/all_actions.json)')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--baseline_metrics_csv', type=str, default=None)
    parser.add_argument('--real_metrics_csv', type=str, default=None)
    parser.add_argument('--target_timestep', type=int, default=None,
                       help='Target timestep for Shapley calculation (1-indexed). If None, will use the last timestep (episode_length) for risk_indicator_naive.')
    parser.add_argument('--metric_name', type=str, default='risk_indicator_naive',
                       choices=['price_inflation_rate', 'price', 'interest_rate', 'unemployment_rate', 'risk_indicator_naive'])
    parser.add_argument('--risk_lambda', type=float, default=0.94)
    parser.add_argument('--use_metric_directly', action='store_true')
    parser.add_argument('--risk_aggregation', type=str, default='max', choices=['max', 'sum'])
    parser.add_argument('--exclude_both_risks', action='store_true')
    parser.add_argument('--mc_samples_list', type=int, nargs='+', default=[10, 50, 100, 500, 1000, 2000, 5000],
                       help='List of MC sample sizes to test')
    parser.add_argument('--skip_exact', action='store_true',
                       help='Skip exact Shapley computation (use cached if available)')
    parser.add_argument('--exact_shapley_file', type=str, default=None,
                       help='Path to existing exact Shapley values file (.npy). If provided, will use this file instead of computing or loading from output_dir.')
    parser.add_argument('--n_jobs', type=int, default=1,
                       help='Number of parallel processes for exact Shapley computation (default: 1, use CPU count if > 1)')
    
    args = parser.parse_args()
    
    args.include_both_risks = not args.exclude_both_risks
    
    # 确定使用的种子
    if args.seeds is not None:
        seeds = args.seeds
    else:
        seeds = [args.seed]
    
    print(f"Will run {len(seeds)} experiments with seeds: {seeds}")
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = 'results/shapley_error_analysis'
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Output directory: {args.output_dir}")
    
    # 加载real actions
    print(f"Loading real actions from: {args.real_actions_json}")
    real_actions_dict = load_real_actions(args.real_actions_json)
    
    # 加载baseline actions（用于计算baseline值）
    print(f"Loading baseline actions from: {args.baseline_actions_json}")
    baseline_actions_dict = load_real_actions(args.baseline_actions_json)
    
    # 计算baseline值（从baseline actions计算）
    if args.baseline_type == 'fixed':
        baseline_work = args.baseline_work
        baseline_consumption = args.baseline_consumption
        print(f"Using fixed baseline: work={baseline_work:.4f}, consumption={baseline_consumption:.4f}")
    else:
        baseline_work, baseline_consumption = compute_baseline_values(
            baseline_actions_dict, args.num_agents, args.episode_length,
            baseline_type=args.baseline_type, stable_period=args.stable_period,
            actions_json_path=args.baseline_actions_json,
            use_willingness=args.use_probabilistic_baseline
        )
    
    # 设置metrics CSV路径
    if args.baseline_metrics_csv is None:
        baseline_dir = os.path.dirname(os.path.dirname(args.baseline_actions_json))
        args.baseline_metrics_csv = os.path.join(baseline_dir, 'metrics_csv', 'world_metrics.csv')
    
    if args.real_metrics_csv is None:
        real_dir = os.path.dirname(os.path.dirname(args.real_actions_json))
        args.real_metrics_csv = os.path.join(real_dir, 'metrics_csv', 'world_metrics.csv')
    
    print(f"Baseline metrics CSV: {args.baseline_metrics_csv}")
    print(f"Real metrics CSV: {args.real_metrics_csv}")
    
    # 如果target_timestep为None且metric_name是risk_indicator_naive，默认使用最后一个时间步
    if args.target_timestep is None and args.metric_name == 'risk_indicator_naive':
        args.target_timestep = args.episode_length
        print(f"Using last timestep ({args.episode_length}) for risk_indicator_naive")
    
    # 对每个种子分别计算exact和MC，然后汇总
    print("\n" + "="*60)
    print("Computing Exact and MC Shapley Values for Each Seed")
    print("="*60)
    print(f"Will process {len(seeds)} seeds, computing exact Shapley for each seed separately")
    
    # 设置并行进程数
    n_jobs = args.n_jobs
    if n_jobs < 1:
        import multiprocessing
        n_jobs = multiprocessing.cpu_count()
        print(f"Using all available CPU cores: {n_jobs}")
    
    # Exact Shapley value是确定性的，只计算一次（使用第一个种子）
    # 种子只用于MC采样的随机性
    exact_shapley_path = os.path.join(args.output_dir, 'exact_shapley_values.npy')
    
    # 检查exact Shapley value是否存在
    if os.path.exists(exact_shapley_path) and not args.skip_exact:
        print(f"\nLoading exact Shapley values from: {exact_shapley_path}")
        phi_exact = np.load(exact_shapley_path)
        print(f"✅ Exact Shapley values loaded (sum: {np.sum(phi_exact):.6f})")
    else:
        # 计算exact Shapley value（只计算一次，使用第一个种子）
        exact_seed = seeds[0] if seeds else args.seed
        print(f"\n{'='*60}")
        print(f"Computing Exact Shapley Values (Deterministic)")
        print(f"{'='*60}")
        print(f"Note: Exact Shapley value is deterministic, computed once using seed {exact_seed}")
        print(f"      Seeds {seeds} will only be used for MC sampling randomness")
        
        if args.skip_exact:
            print("⚠️  --skip_exact is set, but exact Shapley values not found. Cannot proceed without exact values.")
            sys.exit(1)
        
        phi_exact, baseline_risk, real_risk = compute_exact_shapley(
            args.num_agents, args.episode_length, real_actions_dict,
            baseline_work, baseline_consumption,
            args.inflation_threshold, exact_seed,
            args.baseline_metrics_csv, args.real_metrics_csv,
            args.target_timestep, args.metric_name,
            args.use_metric_directly, args.use_probabilistic_baseline,
            args.risk_aggregation, args.include_both_risks, args.risk_lambda,
            n_jobs=n_jobs
        )
        # 保存exact Shapley值（只保存一次）
        np.save(exact_shapley_path, phi_exact)
        print(f"✅ Saved exact Shapley values to: {exact_shapley_path}")
        print(f"   Exact Shapley sum: {np.sum(phi_exact):.6f}")
    
    # 检查哪些种子的MC Shapley值已经计算过
    print("\nChecking existing MC Shapley results...")
    completed_seeds = []
    incomplete_seeds = []
    
    for seed in seeds:
        # 检查所有MC文件是否都存在
        all_mc_exist = True
        for n_samples in args.mc_samples_list:
            mc_shapley_seed_path = os.path.join(args.output_dir, f'mc_shapley_n{n_samples}_seed{seed}.npy')
            if not os.path.exists(mc_shapley_seed_path):
                all_mc_exist = False
                break
        
        if all_mc_exist:
            completed_seeds.append(seed)
            print(f"  ✓ Seed {seed}: All MC files exist, will skip computation")
        else:
            incomplete_seeds.append(seed)
            print(f"  ⚠ Seed {seed}: Some MC files missing, will recompute")
    
    print(f"\nSummary: {len(completed_seeds)} seeds completed, {len(incomplete_seeds)} seeds need MC computation")
    
    # 存储所有种子的结果（包括已有的和新计算的）
    all_seed_results = {}  # {n_samples: [error1, error2, ...]}
    
    # 加载已完成的种子的结果
    if completed_seeds:
        print(f"\nLoading results for {len(completed_seeds)} completed seeds...")
        for seed in completed_seeds:
            for n_samples in args.mc_samples_list:
                mc_shapley_seed_path = os.path.join(args.output_dir, f'mc_shapley_n{n_samples}_seed{seed}.npy')
                phi_mc = np.load(mc_shapley_seed_path)
                
                # 计算误差（使用同一个exact Shapley value）
                error = compute_relative_error(phi_mc, phi_exact)
                
                if n_samples not in all_seed_results:
                    all_seed_results[n_samples] = []
                all_seed_results[n_samples].append(error)
    
    # 计算不完整的种子的MC Shapley值
    if incomplete_seeds:
        print(f"\nComputing MC Shapley values for {len(incomplete_seeds)} incomplete seeds...")
        print(f"Note: Using the same exact Shapley values for all seeds (deterministic)")
        for seed_idx, seed in enumerate(incomplete_seeds):
            print(f"\n{'='*60}")
            print(f"Processing Seed {seed_idx+1}/{len(incomplete_seeds)}: {seed} (Total: {seed_idx+1+len(completed_seeds)}/{len(seeds)})")
            print(f"{'='*60}")
            
            # 用这个种子计算不同采样次数的MC Shapley值
            print(f"Computing MC Shapley values for seed {seed}...")
            for n_samples in args.mc_samples_list:
                mc_shapley_seed_path = os.path.join(args.output_dir, f'mc_shapley_n{n_samples}_seed{seed}.npy')
                
                # 检查MC文件是否已存在
                if os.path.exists(mc_shapley_seed_path):
                    print(f"  Loading MC Shapley values for seed={seed}, n_samples={n_samples} from cache...")
                    phi_mc = np.load(mc_shapley_seed_path)
                else:
                    print(f"  Running MC with seed={seed}, n_samples={n_samples}...")
                    phi_mc, _, _ = compute_mc_shapley(
                        args.num_agents, args.episode_length, real_actions_dict,
                        baseline_work, baseline_consumption,
                        args.inflation_threshold, n_samples, seed,
                        args.baseline_metrics_csv, args.real_metrics_csv,
                        args.target_timestep, args.metric_name,
                        args.use_metric_directly, args.use_probabilistic_baseline,
                        args.risk_aggregation, args.include_both_risks, args.risk_lambda
                    )
                    # 保存该种子的MC Shapley值
                    np.save(mc_shapley_seed_path, phi_mc)
                
                # 计算误差（使用同一个exact Shapley value）
                error = compute_relative_error(phi_mc, phi_exact)
                
                # 存储结果
                if n_samples not in all_seed_results:
                    all_seed_results[n_samples] = []
                all_seed_results[n_samples].append(error)
                
                print(f"    Relative error: {error:.6f}")
    else:
        print("\nAll seeds already completed! No computation needed.")
    
    # 汇总所有种子的结果（只包含脚本中指定的种子）
    print("\n" + "="*60)
    print("Summary: Aggregating Results Across All Seeds")
    print("="*60)
    print(f"Seeds specified in script: {seeds}")
    print(f"Total seeds included: {len(seeds)} ({len(completed_seeds)} from cache, {len(incomplete_seeds)} newly computed)")
    print(f"Note: Only seeds specified in --seeds argument are included in the final results")
    
    results = []
    
    for n_samples in args.mc_samples_list:
        errors_list = all_seed_results.get(n_samples, [])
        
        if len(errors_list) != len(seeds):
            print(f"Warning: n_samples={n_samples} has {len(errors_list)} errors but {len(seeds)} seeds expected")
        
        if len(errors_list) == 0:
            print(f"Warning: No errors found for n_samples={n_samples}, skipping...")
            continue
        
        mean_error = np.mean(errors_list)
        std_error = np.std(errors_list)
        
        results.append({
            'n_samples': n_samples,
            'mean_error': mean_error,
            'std_error': std_error,
            'errors': errors_list,
            'mean_mc_shapley_path': None  # 不再保存平均MC值，因为每个种子都有独立的
        })
        
        print(f"n_samples={n_samples}: mean_error={mean_error:.6f} ± {std_error:.6f} (from {len(errors_list)} seeds)")
        print(f"  Error range: [{np.min(errors_list):.6f}, {np.max(errors_list):.6f}]")
    
    # Exact Shapley值已经计算并保存，不需要再聚合
    # 因为exact Shapley值是确定性的，所有种子应该得到相同的结果
    print(f"\nExact Shapley values (deterministic, computed once):")
    print(f"  File: {exact_shapley_path}")
    print(f"  Sum: {np.sum(phi_exact):.6f}")
    print(f"  Mean: {np.mean(phi_exact):.6f}")
    print(f"  Std: {np.std(phi_exact):.6f}")
    
    # 保存结果
    results_summary = {
        'exact_shapley_path': exact_shapley_path,
        'exact_shapley_sum': float(np.sum(phi_exact)),
        'exact_shapley_mean': float(np.mean(phi_exact)),
        'exact_shapley_std': float(np.std(phi_exact)),
        'num_agents': args.num_agents,
        'episode_length': args.episode_length,
        'seeds': seeds,
        'metric_name': args.metric_name,
        'risk_lambda': args.risk_lambda,
        'computation_method': 'single_exact_multi_mc',  # 标记：exact计算一次，MC用多个种子
        'results': [
            {
                'n_samples': r['n_samples'],
                'mean_error': float(r['mean_error']),
                'std_error': float(r['std_error']),
                'errors': [float(e) for e in r['errors']],
                'mean_mc_shapley_path': r['mean_mc_shapley_path']
            }
            for r in results
        ]
    }
    
    results_json_path = os.path.join(args.output_dir, 'error_analysis_results.json')
    with open(results_json_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # 保存CSV格式的结果（便于绘图）
    error_df = pd.DataFrame([
        {
            'n_samples': r['n_samples'],
            'mean_error': r['mean_error'],
            'std_error': r['std_error'],
            'min_error': np.min(r['errors']),
            'max_error': np.max(r['errors'])
        }
        for r in results
    ])
    error_csv_path = os.path.join(args.output_dir, 'error_analysis.csv')
    error_df.to_csv(error_csv_path, index=False)
    
    # 打印总结
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    print(f"Exact Shapley (deterministic, computed once):")
    print(f"  Sum: {np.sum(phi_exact):.6f}")
    print(f"  Mean: {np.mean(phi_exact):.6f}")
    print(f"  Std: {np.std(phi_exact):.6f}")
    print(f"\nRelative Error Analysis (||Phi_MC - Phi*||_2 / ||Phi*||_2):")
    print(f"Each seed computed separately, then aggregated across {len(seeds)} seeds")
    print(f"{'n_samples':<12} {'mean_error':<15} {'std_error':<15} {'min_error':<15} {'max_error':<15}")
    print("-" * 75)
    for r in results:
        print(f"{r['n_samples']:<12} {r['mean_error']:<15.6f} {r['std_error']:<15.6f} "
              f"{np.min(r['errors']):<15.6f} {np.max(r['errors']):<15.6f}")
    print("="*60)
    
    print(f"\nAll results saved to: {args.output_dir}")
    print(f"  - Exact Shapley values (deterministic): {exact_shapley_path}")
    print(f"  - Error analysis JSON: {results_json_path}")
    print(f"  - Error analysis CSV: {error_csv_path}")
    print(f"  - Per-seed MC Shapley values: mc_shapley_n*_seed*.npy")


if __name__ == '__main__':
    main()

