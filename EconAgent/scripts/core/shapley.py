#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蒙特卡洛方法计算 Shapley Value - 经济风险归因
"""

import os
import sys
import copy
import json
import argparse
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

# Add project root (EconAgent) to path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

import ai_economist.foundation as foundation
import yaml

_config_path = os.path.join(_PROJECT_ROOT, 'config.yaml')
with open(_config_path, "r") as f:
    run_configuration = yaml.safe_load(f)
base_env_config = run_configuration.get('env')


def _extract_world_metrics(env, timestep):
    """Extract world-level metrics."""
    world_metrics = {'timestep': timestep}
    
    if hasattr(env, 'world') and hasattr(env.world, 'price'):
        prices = env.world.price
        if len(prices) > 0:
            world_metrics['price'] = float(prices[-1])
            if len(prices) > 1:
                price_inflation = (prices[-1] - prices[-2]) / (prices[-2] + 1e-8)
                world_metrics['price_inflation_rate'] = float(price_inflation)
            else:
                world_metrics['price_inflation_rate'] = 0.0
        else:
            world_metrics['price'] = None
            world_metrics['price_inflation_rate'] = None
    else:
        world_metrics['price'] = None
        world_metrics['price_inflation_rate'] = None
    
    if hasattr(env, 'world') and hasattr(env.world, 'interest_rate'):
        interest_rates = env.world.interest_rate
        if len(interest_rates) > 0:
            world_metrics['interest_rate'] = float(interest_rates[-1])
        else:
            world_metrics['interest_rate'] = None
    
    return world_metrics


def calculate_risk_indicator_from_metrics(world_metrics_list, method='naive', lambda_param=0.94):
    """
    从world_metrics_list计算风险指标（基于Engle (1982)和Bollerslev (1986)）
    
    Args:
        world_metrics_list: List of world metrics dictionaries (must contain 'price')
        method: 'rolling' 或 'naive'，表示预期规则（目前只支持'naive'）
        lambda_param: RiskMetrics参数λ，默认0.94
    
    Returns:
        risk_values: List[float] - 每个时间步的风险指标值（与world_metrics_list对应）
    """
    if not world_metrics_list:
        return []
    
    # 提取价格数据
    prices = []
    for metrics in world_metrics_list:
        price = metrics.get('price')
        if price is not None:
            prices.append(float(price))
        else:
            return []  # 如果缺少价格数据，返回空列表
    
    if len(prices) < 2:
        return [0.0] * len(prices)  # 至少需要2个价格点才能计算
    
    # 计算通胀率 π_t = log P_t - log P_{t-1}
    log_prices = [np.log(p) for p in prices]
    pi_t = []
    pi_t.append(np.nan)  # 第一个时间步没有前一个价格
    for i in range(1, len(log_prices)):
        pi_t.append(log_prices[i] - log_prices[i-1])
    
    n = len(pi_t)
    
    # 初始化数组
    E_pi = [np.nan] * n  # 预期通胀率
    e_t = [np.nan] * n   # 预测误差
    h_t = [np.nan] * n   # 风险指标
    
    # 计算预期和误差（使用naive方法）
    if method == 'naive':
        for t in range(1, n):
            # 方式2：naive forecast E_{t-1}[π_t] = π_{t-1}
            if not np.isnan(pi_t[t-1]):
                E_pi[t] = pi_t[t-1]
            
            # 计算预测误差 e_t = π_t - E_{t-1}[π_t]
            if not np.isnan(E_pi[t]) and not np.isnan(pi_t[t]):
                e_t[t] = pi_t[t] - E_pi[t]
    else:
        # 目前只支持naive方法
        raise ValueError(f"Method '{method}' not supported. Only 'naive' is currently supported.")
    
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
    
    # 将NaN替换为0.0
    risk_values = [0.0 if np.isnan(h) else float(h) for h in h_t]
    
    return risk_values


def calculate_inflation_risk(world_metrics_list, threshold=0.1, verbose=False, 
                             risk_aggregation='max', include_both_risks=True):
    """
    Calculate inflation/deflation risk.
    
    Args:
        world_metrics_list: List of world metrics dictionaries
        threshold: Inflation threshold (default: 0.1)
        verbose: Whether to print detailed information
        risk_aggregation: 'max' or 'sum' - how to aggregate risks across timesteps
        include_both_risks: If True, count both inflation > threshold AND deflation < 0.
                          If False, when both inflation and deflation risks exist, only count inflation risks (prioritize inflation over deflation)
    
    Risk calculation:
    - High inflation (> threshold): risk = inflation - threshold
    - Deflation (< 0): risk = |deflation| = -deflation
    - Normal range (0 to threshold): risk = 0
    
    Returns:
        Aggregated risk value (max or sum depending on risk_aggregation)
    """
    inflation_rates = [m.get('price_inflation_rate', 0.0) for m in world_metrics_list]
    if not inflation_rates:
        return 0.0
    
    max_risk = 0.0
    max_risk_timestep = None
    risk_details = []
    total_risk = 0.0
    
    # Determine which risk types to include
    if include_both_risks:
        # Include both inflation and deflation risks
        include_inflation = True
        include_deflation = True
    else:
        # Only include inflation risks (> threshold), ignore deflation
        include_inflation = True
        include_deflation = False
    
    # Second pass: calculate risks
    for t, inflation in enumerate(inflation_rates):
        if inflation > threshold:
            risk = inflation - threshold
            risk_type = "high_inflation"
        elif inflation < 0:
            risk = -inflation
            risk_type = "deflation"
        else:
            risk = 0.0
            risk_type = "normal"
        
        risk_details.append((t+1, inflation, risk, risk_type))
        
        # Check if we should include this risk
        should_include = False
        if risk > 0:
            if risk_type == "high_inflation" and include_inflation:
                should_include = True
            elif risk_type == "deflation" and include_deflation:
                should_include = True
        
        if should_include:
            total_risk += risk
        if risk > max_risk:
            max_risk = risk
            max_risk_timestep = t + 1
    
    # Determine final risk value based on aggregation method
    if risk_aggregation == 'sum':
        final_risk = total_risk
    else:  # 'max'
        final_risk = max_risk
    
    if verbose:
        included_risks = []
        for t, inflation in enumerate(inflation_rates):
            if inflation > threshold:
                risk = inflation - threshold
                risk_type = "high_inflation"
                if include_inflation:
                    included_risks.append((t+1, inflation, risk, risk_type))
            elif inflation < 0:
                risk = -inflation
                risk_type = "deflation"
                if include_deflation:
                    included_risks.append((t+1, inflation, risk, risk_type))
        
        print(f"  Risk calculation (threshold={threshold}, aggregation={risk_aggregation}, include_both={include_both_risks}):")
        if risk_aggregation == 'sum':
            print(f"    Total risk (sum): {final_risk:.6f}")
            print(f"    Included risk timesteps: {len(included_risks)}")
        else:
            print(f"    Max risk: {final_risk:.6f} at timestep {max_risk_timestep}")
        if final_risk > 0 and max_risk_timestep:
            metric_at_max = next((inf for t, inf, _, _ in included_risks if t == max_risk_timestep), None)
            if metric_at_max is not None:
                print(f"    Inflation rate at max risk timestep: {metric_at_max:.6f}")
        sorted_risks = sorted(included_risks, key=lambda x: x[2], reverse=True)[:5]
        if sorted_risks:
            print(f"    Top included risk timesteps:")
        for t, inf, risk, rtype in sorted_risks:
                print(f"      Timestep {t}: inflation={inf:.6f}, risk={risk:.6f} ({rtype})")
    
    return final_risk


def find_max_risk_timestep(metrics, metric_name, risk_lambda=0.94, 
                           inflation_threshold=0.1, use_metric_directly=False,
                           include_both_risks=True):
    """
    Find timestep with maximum risk value.
    
    Args:
        metrics: List of world metrics dictionaries
        metric_name: Name of the metric to use
        risk_lambda: RiskMetrics lambda parameter for risk indicator calculation
        inflation_threshold: Inflation threshold for risk calculation
        use_metric_directly: Whether to use metric value directly
        include_both_risks: Whether to include both inflation and deflation risks
    
    Returns:
        (max_risk_timestep, max_risk_value): Tuple of (timestep (1-indexed), risk value)
        Returns (None, None) if no valid risk values found
    """
    if not metrics:
        return None, None
    
    # 计算风险指标
    if metric_name == 'risk_indicator_naive':
        risk_values = calculate_risk_indicator_from_metrics(metrics, method='naive', lambda_param=risk_lambda)
    else:
        # 对于其他指标，计算每个时间点的风险值
        risk_values = []
        for t in range(1, len(metrics) + 1):
            risk = _compute_risk_from_metrics(
                metrics, t, metric_name, inflation_threshold, 
                use_metric_directly, 'max', include_both_risks, False, risk_lambda
            )
            risk_values.append(risk)
    
    if not risk_values or all(r == 0.0 for r in risk_values):
        return None, None
    
    max_risk_value = max(risk_values)
    # 找到最大值的索引（通常只有一个最大值，但如果有多个相同的，选择最晚的时间步）
    max_risk_timestep = risk_values.index(max_risk_value) + 1  # 1-indexed
    
    # 调试：检查是否有多个相同的最大值
    max_count = risk_values.count(max_risk_value)
    if max_count > 1:
        # 如果有多个相同的最大值，选择最后一个（最晚的时间步）
        for i in range(len(risk_values) - 1, -1, -1):
            if risk_values[i] == max_risk_value:
                max_risk_timestep = i + 1  # 转换为1-indexed
                break
        print(f"Warning: Found {max_count} identical maximum risk values ({max_risk_value:.6f}), using latest timestep {max_risk_timestep}")
    
    return max_risk_timestep, max_risk_value


def load_metrics_from_csv(csv_path, metric_name='price_inflation_rate', risk_lambda=0.94):
    """Load metrics from world_metrics.csv file."""
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        
        # 特殊处理：如果metric_name是risk_indicator_naive，需要计算
        if metric_name == 'risk_indicator_naive':
            if 'price' not in df.columns:
                print(f"Warning: CSV file {csv_path} does not contain 'price' column, cannot calculate risk indicator")
                return None
            
            # 检查是否已有计算好的风险指标列
            if 'risk_indicator_naive' in df.columns:
                # 直接读取
                metrics_list = []
                for _, row in df.iterrows():
                    metrics = {'timestep': int(row['timestep'])}
                    metrics['risk_indicator_naive'] = float(row['risk_indicator_naive']) if pd.notna(row['risk_indicator_naive']) else 0.0
                    for col in ['price', 'interest_rate', 'unemployment_rate']:
                        if col in df.columns and pd.notna(row[col]):
                            metrics[col] = float(row[col])
                    metrics_list.append(metrics)
                return metrics_list
            else:
                # 从价格数据计算风险指标
                metrics_list = []
                for _, row in df.iterrows():
                    metrics = {'timestep': int(row['timestep'])}
                    for col in ['price', 'interest_rate', 'unemployment_rate']:
                        if col in df.columns and pd.notna(row[col]):
                            metrics[col] = float(row[col])
                    metrics_list.append(metrics)
                
                # 计算风险指标
                risk_values = calculate_risk_indicator_from_metrics(metrics_list, method='naive', lambda_param=risk_lambda)
                for i, metrics in enumerate(metrics_list):
                    if i < len(risk_values):
                        metrics['risk_indicator_naive'] = risk_values[i]
                    else:
                        metrics['risk_indicator_naive'] = 0.0
                
                return metrics_list
        
        # 原有逻辑：直接读取metric_name列
        if metric_name not in df.columns:
            return None
        metrics_list = []
        for _, row in df.iterrows():
            metrics = {'timestep': int(row['timestep'])}
            metrics[metric_name] = float(row[metric_name]) if pd.notna(row[metric_name]) else 0.0
            for col in ['price', 'interest_rate', 'unemployment_rate']:
                if col in df.columns and pd.notna(row[col]):
                    metrics[col] = float(row[col])
            metrics_list.append(metrics)
        return metrics_list
    except Exception as e:
        print(f"Warning: Failed to load metrics from {csv_path}: {e}")
        return None


def save_metrics_to_csv(world_metrics_list, csv_path):
    """Save world_metrics_list to CSV file."""
    try:
        data = []
        for metrics in world_metrics_list:
            row = {'timestep': metrics.get('timestep', 0)}
            for key in ['price', 'price_inflation_rate', 'interest_rate', 'unemployment_rate',
                       'total_wealth', 'total_income', 'total_consumption',
                       'avg_wealth', 'avg_income', 'avg_consumption',
                       'gini_wealth', 'gini_income']:
                if key in metrics:
                    row[key] = metrics[key]
            data.append(row)
        
        df = pd.DataFrame(data)
        cols = ['timestep'] + [c for c in df.columns if c != 'timestep']
        df = df[cols]
        
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Saved metrics to: {csv_path}")
        return True
    except Exception as e:
        print(f"Warning: Failed to save metrics to {csv_path}: {e}")
        return False


def get_metric_at_timestep(world_metrics_list, target_timestep, metric_name='price_inflation_rate'):
    """Get metric value at specific timestep."""
    if target_timestep < 1 or target_timestep > len(world_metrics_list):
        return 0.0
    metric = world_metrics_list[target_timestep - 1].get(metric_name, 0.0)
    return float(metric) if metric is not None else 0.0


def run_counterfactual_simulation(
    num_agents, episode_length, 
    real_actions_dict,
    baseline_work, baseline_consumption,
    subset_U,
    seed=None,
    use_probabilistic_baseline=False
):
    """Run counterfactual simulation.
    
    If use_probabilistic_baseline=True and baseline_work is in [0, 1] (not just 0 or 1),
    will use random sampling to determine work action based on baseline_work probability.
    Otherwise, uses deterministic conversion (baseline_work >= 0.5 -> 1, else 0).
    """
    cfg = copy.deepcopy(base_env_config)
    cfg['n_agents'] = num_agents
    cfg['episode_length'] = episode_length
    if seed is not None:
        cfg['seed'] = seed
    
    env = foundation.make_env_instance(**cfg)
    obs = env.reset()
    
    # Determine if baseline_work represents willingness (0-1) or executed action (0/1)
    # If using probabilistic baseline and baseline_work is in (0, 1), treat it as willingness
    is_willingness = use_probabilistic_baseline and 0 < baseline_work < 1
    
    world_metrics_list = []
    
    # Create a deterministic RNG for probabilistic baseline if needed
    base_seed = seed if seed is not None else 0
    
    for t in range(episode_length):
        actions = {}
        step_key = f'step_{t+1}'
        
        if step_key in real_actions_dict:
            real_actions = real_actions_dict[step_key]
            for agent_id in range(num_agents):
                agent_key = str(agent_id)
                if (agent_id, t) in subset_U:
                    if agent_key in real_actions:
                        actions[agent_key] = real_actions[agent_key]
                    else:
                        # Use baseline action
                        if is_willingness:
                            # Use deterministic seed for reproducibility
                            rng = np.random.RandomState(base_seed + agent_id * 10000 + t)
                            work_action_agent = int(rng.uniform() <= baseline_work)
                        else:
                            work_action_agent = int(np.clip(baseline_work, 0, 1))
                        actions[agent_key] = [
                            work_action_agent,
                            int(np.clip(baseline_consumption / 0.02, 0, 50)),
                        ]
                else:
                    # Use baseline action
                    if is_willingness:
                        # Use deterministic seed for reproducibility
                        rng = np.random.RandomState(base_seed + agent_id * 10000 + t)
                        work_action_agent = int(rng.uniform() <= baseline_work)
                    else:
                        work_action_agent = int(np.clip(baseline_work, 0, 1))
                    actions[agent_key] = [
                        work_action_agent,
                        int(np.clip(baseline_consumption / 0.02, 0, 50)),
                    ]
        else:
            for agent_id in range(num_agents):
                if is_willingness:
                    # Use deterministic seed for reproducibility
                    rng = np.random.RandomState(base_seed + agent_id * 10000 + t)
                    work_action_agent = int(rng.uniform() <= baseline_work)
                else:
                    work_action_agent = int(np.clip(baseline_work, 0, 1))
                actions[str(agent_id)] = [
                    work_action_agent,
                    int(np.clip(baseline_consumption / 0.02, 0, 50)),
                ]
        
        actions['p'] = [0]
        obs, _, _, _ = env.step(actions)
        world_metrics = _extract_world_metrics(env, t + 1)
        world_metrics_list.append(world_metrics)
    
    return world_metrics_list


def _compute_risk_from_metrics(metrics, target_timestep, metric_name, inflation_threshold, use_metric_directly,
                                risk_aggregation='max', include_both_risks=True, verbose=False, risk_lambda=0.94):
    """Compute risk from metrics based on parameters."""
    # 处理risk_indicator_naive
    if metric_name == 'risk_indicator_naive':
        # 计算风险指标
        risk_values = calculate_risk_indicator_from_metrics(metrics, method='naive', lambda_param=risk_lambda)
        if not risk_values:
            return 0.0
        
        if target_timestep is not None:
            # 获取指定时间点的风险值
            if target_timestep < 1 or target_timestep > len(risk_values):
                return 0.0
            return risk_values[target_timestep - 1]
        else:
            # 如果没有指定时间点，返回最大值
            return max(risk_values) if risk_values else 0.0
    
    # 原有逻辑
    if target_timestep is not None:
        metric_value = get_metric_at_timestep(metrics, target_timestep, metric_name)
        if use_metric_directly:
            return metric_value
        else:
            if metric_name == 'price_inflation_rate':
                if metric_value > inflation_threshold:
                    return metric_value - inflation_threshold
                elif metric_value < 0:
                    return -metric_value
                else:
                    return 0.0
            else:
                return metric_value
    elif use_metric_directly:
        metric_values = [m.get(metric_name, 0.0) for m in metrics]
        return max(metric_values) if metric_values else 0.0
    else:
        return calculate_inflation_risk(metrics, inflation_threshold, verbose=verbose,
                                       risk_aggregation=risk_aggregation, include_both_risks=include_both_risks)


def _compute_risk_cached(
    subset_U_frozen,
    num_agents, episode_length,
    real_actions_dict,
    baseline_work, baseline_consumption,
    inflation_threshold,
    seed,
    target_timestep=None,
    metric_name='price_inflation_rate',
    use_metric_directly=False,
    use_probabilistic_baseline=False,
    risk_aggregation='max',
    include_both_risks=True,
    risk_lambda=0.94
):
    """Compute risk for a subset with caching."""
    subset_U = set(subset_U_frozen)
    metrics = run_counterfactual_simulation(
        num_agents, episode_length, real_actions_dict,
        baseline_work, baseline_consumption, subset_U=subset_U, seed=seed,
        use_probabilistic_baseline=use_probabilistic_baseline
    )
    return _compute_risk_from_metrics(metrics, target_timestep, metric_name, inflation_threshold, use_metric_directly,
                                     risk_aggregation=risk_aggregation, include_both_risks=include_both_risks, verbose=False, risk_lambda=risk_lambda)


def monte_carlo_shapley(
    num_agents, episode_length,
    real_actions_dict,
    baseline_work, baseline_consumption,
    inflation_threshold=0.1,
    n_samples=1000,
    seed=None,
    baseline_metrics_csv=None,
    real_metrics_csv=None,
    target_timestep=None,
    target_timesteps=None,
    metric_name='price_inflation_rate',
    use_metric_directly=False,
    use_probabilistic_baseline=False,
    risk_aggregation='max',
    include_both_risks=True,
    risk_lambda=0.94
):
    """Monte Carlo Shapley value calculation with caching.
    
    Supports multiple target timesteps. If target_timesteps is provided, returns
    separate Shapley values for each timestep and an aggregated result.
    """
    all_players = [(i, t) for i in range(num_agents) for t in range(episode_length)]
    
    # 处理时间点参数：支持单个或多个时间点
    if target_timesteps is not None:
        if isinstance(target_timesteps, (int, float)):
            target_timesteps = [int(target_timesteps)]
        else:
            target_timesteps = [int(t) for t in target_timesteps]
    elif target_timestep is not None:
        target_timesteps = [int(target_timestep)]
    else:
        target_timesteps = None
    
    # 如果指定了单个target_timestep，过滤掉该时间点及之后的所有玩家
    # 只计算该时间点之前的动作的Shapley值
    if target_timesteps is not None and len(target_timesteps) == 1:
        cutoff_timestep = target_timesteps[0]
        original_count = len(all_players)
        all_players = [(i, t) for i, t in all_players if t < cutoff_timestep]
        filtered_count = len(all_players)
        print(f"Filtered players: only computing Shapley for timesteps < {cutoff_timestep}")
        print(f"  Original players: {original_count}, Filtered players: {filtered_count}")
        print(f"  Reduction: {original_count - filtered_count} players excluded ({100.0 * (original_count - filtered_count) / original_count:.1f}%)")
    
    M = len(all_players)
    
    is_multi_timestep = target_timesteps is not None and len(target_timesteps) > 1
    
    # Compute baseline risk
    if baseline_metrics_csv and os.path.exists(baseline_metrics_csv):
        print(f"Loading baseline metrics from: {baseline_metrics_csv}")
        baseline_metrics = load_metrics_from_csv(baseline_metrics_csv, metric_name, risk_lambda=risk_lambda)
        if baseline_metrics:
            if is_multi_timestep:
                baseline_risks = {}
                for t in target_timesteps:
                    baseline_risks[t] = _compute_risk_from_metrics(
                        baseline_metrics, t, metric_name, inflation_threshold, use_metric_directly,
                        risk_aggregation=risk_aggregation, include_both_risks=include_both_risks, verbose=True, risk_lambda=risk_lambda
                    )
                baseline_risk = np.mean(list(baseline_risks.values()))
                print(f"Baseline risk (from file, multi-timestep): {baseline_risk:.6f}")
                for t, risk in baseline_risks.items():
                    print(f"  Timestep {t}: {risk:.6f}")
            else:
                target_t = target_timesteps[0] if target_timesteps else None
                baseline_risk = _compute_risk_from_metrics(
                    baseline_metrics, target_t, metric_name, inflation_threshold, use_metric_directly,
                    risk_aggregation=risk_aggregation, include_both_risks=include_both_risks, verbose=True, risk_lambda=risk_lambda
                )
                print(f"Baseline risk (from file): {baseline_risk:.6f}")
        else:
            print("Failed to load baseline metrics, computing...")
            baseline_metrics = run_counterfactual_simulation(
                num_agents, episode_length, real_actions_dict,
                baseline_work, baseline_consumption, subset_U=set(), seed=seed,
                use_probabilistic_baseline=use_probabilistic_baseline
            )
            if is_multi_timestep:
                baseline_risks = {}
                for t in target_timesteps:
                    baseline_risks[t] = _compute_risk_from_metrics(
                        baseline_metrics, t, metric_name, inflation_threshold, use_metric_directly,
                        risk_aggregation=risk_aggregation, include_both_risks=include_both_risks, verbose=True, risk_lambda=risk_lambda
                    )
                baseline_risk = np.mean(list(baseline_risks.values()))
                print(f"Baseline risk (computed, multi-timestep): {baseline_risk:.6f}")
            else:
                target_t = target_timesteps[0] if target_timesteps else None
                baseline_risk = _compute_risk_from_metrics(
                    baseline_metrics, target_t, metric_name, inflation_threshold, use_metric_directly,
                    risk_aggregation=risk_aggregation, include_both_risks=include_both_risks, verbose=True, risk_lambda=risk_lambda
                )
                print(f"Baseline risk (computed): {baseline_risk:.6f}")
    else:
        print("Computing baseline risk...")
        print(f"  Baseline strategy: work={baseline_work:.4f}, consumption={baseline_consumption:.4f}")
        baseline_metrics = run_counterfactual_simulation(
            num_agents, episode_length, real_actions_dict,
            baseline_work, baseline_consumption, subset_U=set(), seed=seed,
            use_probabilistic_baseline=use_probabilistic_baseline
        )
        if baseline_metrics:
            prices = [m.get('price', None) for m in baseline_metrics if m.get('price') is not None]
            if prices:
                print(f"  Price values (first 5, last 5): {prices[:5]} ... {prices[-5:]}")
                print(f"  Price range: min={min(prices):.6f}, max={max(prices):.6f}")
        if is_multi_timestep:
            baseline_risks = {}
            for t in target_timesteps:
                baseline_risks[t] = _compute_risk_from_metrics(
                    baseline_metrics, t, metric_name, inflation_threshold, use_metric_directly,
                    risk_aggregation=risk_aggregation, include_both_risks=include_both_risks, verbose=True, risk_lambda=risk_lambda
                )
            baseline_risk = np.mean(list(baseline_risks.values()))
            print(f"Baseline risk (multi-timestep): {baseline_risk:.6f}")
            for t, risk in baseline_risks.items():
                print(f"  Timestep {t}: {risk:.6f}")
        else:
            target_t = target_timesteps[0] if target_timesteps else None
            baseline_risk = _compute_risk_from_metrics(
                baseline_metrics, target_t, metric_name, inflation_threshold, use_metric_directly,
                risk_aggregation=risk_aggregation, include_both_risks=include_both_risks, verbose=True, risk_lambda=risk_lambda
            )
            if target_t is not None:
                if metric_name == 'risk_indicator_naive':
                    risk_values = calculate_risk_indicator_from_metrics(baseline_metrics, method='naive', lambda_param=risk_lambda)
                    print(f"  All risk_indicator_naive values: {[f'{x:.6f}' for x in risk_values]}")
                else:
                    all_metric_values = [m.get(metric_name, 0.0) for m in baseline_metrics]
                    print(f"  All {metric_name} values: {[f'{x:.6f}' for x in all_metric_values]}")
                metric_value = get_metric_at_timestep(baseline_metrics, target_t, metric_name)
                print(f"  Value at timestep {target_t}: {metric_value:.6f}")
                if use_metric_directly:
                    print(f"  Using metric directly: baseline_risk = {baseline_risk:.6f}")
            print(f"Baseline risk: {baseline_risk:.6f}")
    
    # Compute real risk
    if real_metrics_csv and os.path.exists(real_metrics_csv):
        print(f"Loading real metrics from: {real_metrics_csv}")
        real_metrics = load_metrics_from_csv(real_metrics_csv, metric_name, risk_lambda=risk_lambda)
        if real_metrics:
            if is_multi_timestep:
                real_risks = {}
                for t in target_timesteps:
                    real_risks[t] = _compute_risk_from_metrics(
                        real_metrics, t, metric_name, inflation_threshold, use_metric_directly,
                        risk_aggregation=risk_aggregation, include_both_risks=include_both_risks, verbose=True, risk_lambda=risk_lambda
                    )
                real_risk = np.mean(list(real_risks.values()))
                print(f"Real risk (from file, multi-timestep): {real_risk:.6f}")
                for t, risk in real_risks.items():
                    print(f"  Timestep {t}: {risk:.6f}")
            else:
                target_t = target_timesteps[0] if target_timesteps else None
                real_risk = _compute_risk_from_metrics(
                    real_metrics, target_t, metric_name, inflation_threshold, use_metric_directly,
                    risk_aggregation=risk_aggregation, include_both_risks=include_both_risks, verbose=True, risk_lambda=risk_lambda
                )
                print(f"Real risk (from file): {real_risk:.6f}")
        else:
            print("Failed to load real metrics, computing...")
            real_metrics = run_counterfactual_simulation(
                num_agents, episode_length, real_actions_dict,
                baseline_work, baseline_consumption, subset_U=set(all_players), seed=seed,
                use_probabilistic_baseline=use_probabilistic_baseline
            )
            if is_multi_timestep:
                real_risks = {}
                for t in target_timesteps:
                    real_risks[t] = _compute_risk_from_metrics(
                        real_metrics, t, metric_name, inflation_threshold, use_metric_directly,
                        risk_aggregation=risk_aggregation, include_both_risks=include_both_risks, verbose=True, risk_lambda=risk_lambda
                    )
                real_risk = np.mean(list(real_risks.values()))
                print(f"Real risk (computed, multi-timestep): {real_risk:.6f}")
            else:
                target_t = target_timesteps[0] if target_timesteps else None
                real_risk = _compute_risk_from_metrics(
                    real_metrics, target_t, metric_name, inflation_threshold, use_metric_directly,
                    risk_aggregation=risk_aggregation, include_both_risks=include_both_risks, verbose=True, risk_lambda=risk_lambda
                )
                print(f"Real risk (computed): {real_risk:.6f}")
    else:
        print("Computing real risk...")
        real_metrics = run_counterfactual_simulation(
            num_agents, episode_length, real_actions_dict,
            baseline_work, baseline_consumption, subset_U=set(all_players), seed=seed,
            use_probabilistic_baseline=use_probabilistic_baseline
        )
        if is_multi_timestep:
            real_risks = {}
            for t in target_timesteps:
                real_risks[t] = _compute_risk_from_metrics(
                    real_metrics, t, metric_name, inflation_threshold, use_metric_directly,
                    risk_aggregation=risk_aggregation, include_both_risks=include_both_risks, verbose=True, risk_lambda=risk_lambda
                )
            real_risk = np.mean(list(real_risks.values()))
            print(f"Real risk (multi-timestep): {real_risk:.6f}")
            for t, risk in real_risks.items():
                print(f"  Timestep {t}: {risk:.6f}")
        else:
            target_t = target_timesteps[0] if target_timesteps else None
            real_risk = _compute_risk_from_metrics(
                real_metrics, target_t, metric_name, inflation_threshold, use_metric_directly,
                risk_aggregation=risk_aggregation, include_both_risks=include_both_risks, verbose=True, risk_lambda=risk_lambda
            )
            if target_t is not None:
                if metric_name == 'risk_indicator_naive':
                    risk_values = calculate_risk_indicator_from_metrics(real_metrics, method='naive', lambda_param=risk_lambda)
                    print(f"  All risk_indicator_naive values: {[f'{x:.6f}' for x in risk_values]}")
                else:
                    all_metric_values = [m.get(metric_name, 0.0) for m in real_metrics]
                    print(f"  All {metric_name} values: {[f'{x:.6f}' for x in all_metric_values]}")
                metric_value = get_metric_at_timestep(real_metrics, target_t, metric_name)
                print(f"  Value at timestep {target_t}: {metric_value:.6f}")
                if use_metric_directly:
                    print(f"  Using metric directly: real_risk = {real_risk:.6f}")
            print(f"Real risk: {real_risk:.6f}")
    
    # Monte Carlo sampling
    print(f"Running Monte Carlo sampling ({n_samples} permutations)...")
    if is_multi_timestep:
        print(f"Computing Shapley values for {len(target_timesteps)} timesteps: {target_timesteps}")
    np.random.seed(seed)
    
    # 初始化Shapley值累加器
    if is_multi_timestep:
        # 为每个时间点创建独立的累加器
        marginal_contributions_dict = {t: np.zeros((num_agents, episode_length)) for t in target_timesteps}
        risk_cache_dict = {t: {frozenset(): baseline_risk} for t in target_timesteps}
    else:
        marginal_contributions = np.zeros((num_agents, episode_length))
        risk_cache = {frozenset(): baseline_risk}
    
    print(f"Using caching optimization...")
    for sample_idx in tqdm(range(n_samples), desc="MC Sampling"):
        permutation = np.random.permutation(M)
        permuted_players = [all_players[i] for i in permutation]
        
        current_subset = set()
        if is_multi_timestep:
            prev_risks = {t: baseline_risk for t in target_timesteps}
        else:
            prev_risk = baseline_risk
        
        for pos, player in enumerate(permuted_players):
            agent_id, timestep = player
            current_subset.add(player)
            subset_key = frozenset(current_subset)
            
            if is_multi_timestep:
                # 为每个时间点计算风险
                for t in target_timesteps:
                    if subset_key not in risk_cache_dict[t]:
                        current_risk = _compute_risk_cached(
                            subset_key, num_agents, episode_length,
                            real_actions_dict, baseline_work, baseline_consumption,
                            inflation_threshold, seed,
                            target_timestep=t,
                            metric_name=metric_name,
                            use_metric_directly=use_metric_directly,
                            use_probabilistic_baseline=use_probabilistic_baseline,
                            risk_aggregation=risk_aggregation,
                            include_both_risks=include_both_risks,
                            risk_lambda=risk_lambda
                        )
                        risk_cache_dict[t][subset_key] = current_risk
                    else:
                        current_risk = risk_cache_dict[t][subset_key]
                    
                    marginal_contrib = current_risk - prev_risks[t]
                    marginal_contributions_dict[t][agent_id, timestep] += marginal_contrib
                    prev_risks[t] = current_risk
            else:
                # 单个时间点逻辑
                target_t = target_timesteps[0] if target_timesteps else None
                if subset_key not in risk_cache:
                    current_risk = _compute_risk_cached(
                        subset_key, num_agents, episode_length,
                        real_actions_dict, baseline_work, baseline_consumption,
                        inflation_threshold, seed,
                        target_timestep=target_t,
                        metric_name=metric_name,
                        use_metric_directly=use_metric_directly,
                        use_probabilistic_baseline=use_probabilistic_baseline,
                        risk_aggregation=risk_aggregation,
                        include_both_risks=include_both_risks,
                        risk_lambda=risk_lambda
                    )
                    risk_cache[subset_key] = current_risk
                else:
                    current_risk = risk_cache[subset_key]
                
                marginal_contrib = current_risk - prev_risk
                marginal_contributions[agent_id, timestep] += marginal_contrib
                prev_risk = current_risk
    
    # 计算最终的Shapley值
    if is_multi_timestep:
        shapley_values_dict = {t: marginal_contributions_dict[t] / n_samples for t in target_timesteps}
        # 计算聚合的Shapley值（所有时间点的平均值）
        aggregated_shapley_values = np.mean([shapley_values_dict[t] for t in target_timesteps], axis=0)
        
        total_cache_size = sum(len(cache) for cache in risk_cache_dict.values())
        print(f"Cache hit rate: {total_cache_size} unique subsets computed across {len(target_timesteps)} timesteps")
        
        return shapley_values_dict, aggregated_shapley_values, baseline_risk, real_risk, baseline_metrics
    else:
        total_cache_size = len(risk_cache)
        print(f"Cache hit rate: {total_cache_size} unique subsets computed out of {n_samples * M} total evaluations")
        
        shapley_values = marginal_contributions / n_samples
        return shapley_values, baseline_risk, real_risk, baseline_metrics


def load_real_actions(actions_json_path):
    """Load real actions from JSON."""
    with open(actions_json_path, 'r') as f:
        all_actions = json.load(f)
    return all_actions


def load_work_willingness_from_dialogs(actions_json_path, num_agents, episode_length, 
                                       baseline_type='average', stable_period=None):
    """Load original work willingness (0-1) from dialog pickles.
    
    Returns a dict mapping (agent_id, timestep) to work willingness, or None if not available.
    """
    # Infer dialog path from actions_json path
    actions_dir = os.path.dirname(actions_json_path)
    simulation_dir = os.path.dirname(actions_dir)
    dialog_dir = os.path.join(simulation_dir, 'dialog_pickles')
    
    if not os.path.exists(dialog_dir):
        return None
    
    willingness_dict = {}
    
    if baseline_type == 'stable':
        if stable_period is None:
            stable_period = episode_length // 3
        timesteps_to_use = range(1, min(stable_period + 1, episode_length + 1))
    else:  # baseline_type == 'average'
        timesteps_to_use = range(1, episode_length + 1)
    
    for t in timesteps_to_use:
        dialog_path = os.path.join(dialog_dir, f'dialog_{t}.pkl')
        if not os.path.exists(dialog_path):
            continue
        
        try:
            with open(dialog_path, 'rb') as f:
                dialog_queue = pickle.load(f)
            
            if len(dialog_queue) != num_agents:
                continue
            
            for agent_id in range(num_agents):
                if len(dialog_queue[agent_id]) > 0:
                    last_msg = dialog_queue[agent_id][-1]
                    if last_msg.get('role') == 'assistant':
                        try:
                            content = last_msg['content']
                            # Parse JSON from content
                            if '{' in content and 'work' in content:
                                # Extract JSON part
                                start = content.find('{')
                                end = content.rfind('}') + 1
                                if start >= 0 and end > start:
                                    json_str = content[start:end]
                                    parsed = json.loads(json_str)
                                    if 'work' in parsed:
                                        willingness = float(parsed['work'])
                                        willingness_dict[(agent_id, t - 1)] = willingness
                        except:
                            pass
        except Exception as e:
            continue
    
    return willingness_dict if willingness_dict else None


def compute_baseline_values(real_actions_dict, num_agents, episode_length, 
                            baseline_type='average', stable_period=None,
                            actions_json_path=None, use_willingness=False):
    """Compute baseline work and consumption values based on baseline_type.
    
    If use_willingness=True and actions_json_path is provided, will try to load
    original work willingness from dialog pickles to preserve probabilistic behavior.
    Otherwise, uses the executed actions (0/1) which are deterministic.
    """
    if baseline_type == 'fixed':
        return 1.0, 0.5
    
    # Try to load original work willingness from dialogs if requested
    willingness_dict = None
    if use_willingness and actions_json_path:
        willingness_dict = load_work_willingness_from_dialogs(
            actions_json_path, num_agents, episode_length, baseline_type, stable_period
        )
        if willingness_dict:
            print("Found work willingness values in dialogs, using them for baseline calculation")
        else:
            print("Could not load work willingness from dialogs, using executed actions (0/1)")
    
    all_work = []
    all_consumption = []
    
    if baseline_type == 'stable':
        if stable_period is None:
            stable_period = episode_length // 3
        timesteps_to_use = range(min(stable_period, episode_length))
        print(f"Using stable baseline: averaging first {len(timesteps_to_use)} timesteps")
    else:  # baseline_type == 'average'
        timesteps_to_use = range(episode_length)
        print(f"Using average baseline: averaging all {episode_length} timesteps")
    
    for t in timesteps_to_use:
        step_key = f'step_{t+1}'
        if step_key in real_actions_dict:
            real_actions = real_actions_dict[step_key]
            for agent_id in range(num_agents):
                agent_key = str(agent_id)
                if agent_key in real_actions:
                    action = real_actions[agent_key]
                    if isinstance(action, list) and len(action) >= 2:
                        # Use original willingness if available, otherwise use executed action (0/1)
                        if willingness_dict and (agent_id, t) in willingness_dict:
                            work = willingness_dict[(agent_id, t)]
                        else:
                            work = action[0]  # This is 0 or 1
                        
                        consumption_scaled = action[1]
                        consumption = consumption_scaled * 0.02
                        all_work.append(work)
                        all_consumption.append(consumption)
    
    if all_work and all_consumption:
        baseline_work = np.mean(all_work)
        baseline_consumption = np.mean(all_consumption)
        if use_willingness and willingness_dict:
            print(f"Computed baseline (from work willingness): work={baseline_work:.4f}, consumption={baseline_consumption:.4f}")
            print(f"  (from {len(all_work)} action samples, work values are 0-1 willingness)")
        else:
            print(f"Computed baseline (from executed actions): work={baseline_work:.4f}, consumption={baseline_consumption:.4f}")
            print(f"  (from {len(all_work)} action samples, work values are 0/1)")
            if baseline_work >= 0.5:
                print(f"  Note: Average work={baseline_work:.4f} >= 0.5, so all agents will work in baseline (deterministic)")
            else:
                print(f"  Note: Average work={baseline_work:.4f} < 0.5, so all agents will not work in baseline (deterministic)")
        return baseline_work, baseline_consumption
    else:
        print("Warning: No actions found, using default values")
        return 1.0, 0.5


def main():
    parser = argparse.ArgumentParser(description='Monte Carlo Shapley Value Calculation')
    parser.add_argument('--num_agents', type=int, default=20)
    parser.add_argument('--episode_length', type=int, default=20)
    parser.add_argument('--baseline_type', type=str, default='average',
                       choices=['fixed', 'average', 'stable'],
                       help='Baseline strategy type')
    parser.add_argument('--baseline_work', type=float, default=1.0,
                       help='Fixed baseline work value (only used when baseline_type=fixed)')
    parser.add_argument('--baseline_consumption', type=float, default=0.5,
                       help='Fixed baseline consumption value (only used when baseline_type=fixed)')
    parser.add_argument('--stable_period', type=int, default=None,
                       help='Number of timesteps to use for stable baseline (only used when baseline_type=stable)')
    parser.add_argument('--use_probabilistic_baseline', action='store_true',
                       help='Use probabilistic baseline based on work willingness from dialogs (if available). '
                            'If enabled and dialogs are available, will use original work willingness values (0-1) '
                            'instead of executed actions (0/1), and sample work decisions probabilistically. '
                            'Otherwise, uses deterministic baseline (work >= 0.5 -> 1, else 0).')
    parser.add_argument('--inflation_threshold', type=float, default=0.0)
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--actions_json', type=str, required=True,
                       help='Path to all_actions.json from simulation')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (auto-detected from actions_json if not specified)')
    parser.add_argument('--baseline_metrics_csv', type=str, default=None,
                       help='Path to baseline world_metrics.csv (optional)')
    parser.add_argument('--real_metrics_csv', type=str, default=None,
                       help='Path to real world_metrics.csv (optional)')
    parser.add_argument('--target_timestep', type=int, default=None,
                       help='Target timestep for Shapley calculation (1-indexed). Deprecated: use --target_timesteps instead.')
    parser.add_argument('--target_timesteps', type=int, nargs='+', default=None,
                       help='Target timesteps for Shapley calculation (1-indexed). Can specify multiple timesteps.')
    parser.add_argument('--metric_name', type=str, default='price_inflation_rate',
                       choices=['price_inflation_rate', 'price', 'interest_rate', 'unemployment_rate', 'risk_indicator_naive'],
                       help='Metric name to use for Shapley calculation')
    parser.add_argument('--risk_lambda', type=float, default=0.94,
                       help='RiskMetrics lambda parameter for risk indicator calculation (default: 0.94)')
    parser.add_argument('--use_metric_directly', action='store_true',
                       help='Use metric value directly instead of threshold-based risk calculation')
    parser.add_argument('--risk_aggregation', type=str, default='max',
                       choices=['max', 'sum'],
                       help='Risk aggregation method: "max" for maximum risk across timesteps, "sum" for cumulative risk (sum of all risks above threshold)')
    parser.add_argument('--exclude_both_risks', action='store_true',
                       help='If set, when both inflation and deflation risks exist, only count inflation risks. '
                            'By default (if not set), both inflation > threshold AND deflation < 0 are counted.')
    
    args = parser.parse_args()
    
    # Set include_both_risks based on exclude_both_risks flag (default: True)
    args.include_both_risks = not args.exclude_both_risks
    
    # Auto-detect output directory
    if args.output_dir is None:
        actions_json_path = os.path.abspath(args.actions_json)
        if 'data' in actions_json_path:
            path_parts = actions_json_path.split(os.sep)
            data_idx = None
            for i, part in enumerate(path_parts):
                if part == 'data' and i + 1 < len(path_parts):
                    data_idx = i
                    break
            
            if data_idx is not None and data_idx + 1 < len(path_parts):
                simulation_name = path_parts[data_idx + 1]
                data_dir = os.sep.join(path_parts[:data_idx + 1])
                args.output_dir = os.path.join(data_dir, simulation_name, 'shapley')
            else:
                args.output_dir = 'results/shapley_mc'
        else:
            args.output_dir = 'results/shapley_mc'
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Output directory: {args.output_dir}")
    print(f"Loading real actions from: {args.actions_json}")
    real_actions_dict = load_real_actions(args.actions_json)
    
    # Compute baseline values
    actions_json_path = os.path.abspath(args.actions_json)
    if args.baseline_type == 'fixed':
        baseline_work = args.baseline_work
        baseline_consumption = args.baseline_consumption
        print(f"Using fixed baseline: work={baseline_work:.4f}, consumption={baseline_consumption:.4f}")
    else:
        baseline_work, baseline_consumption = compute_baseline_values(
            real_actions_dict, args.num_agents, args.episode_length,
            baseline_type=args.baseline_type, stable_period=args.stable_period,
            actions_json_path=actions_json_path,
            use_willingness=args.use_probabilistic_baseline
        )
    
    # 处理时间点参数：支持--target_timestep（向后兼容）和--target_timesteps
    target_timesteps = args.target_timesteps
    if target_timesteps is None and args.target_timestep is not None:
        target_timesteps = [args.target_timestep]
        print(f"Note: --target_timestep is deprecated, use --target_timesteps instead")
    
    # 如果没有指定target_timesteps，自动查找最高风险点
    is_auto_max_risk = False
    max_risk_timestep = None
    max_risk_value = None
    if target_timesteps is None:
        print("\nAuto-detecting maximum risk timestep...")
        # 优先从real_metrics_csv加载真实模拟的metrics
        # 如果不存在，则运行真实模拟来计算real_metrics（使用所有真实动作）
        real_metrics = None
        if args.real_metrics_csv and os.path.exists(args.real_metrics_csv):
            print(f"Loading real metrics from: {args.real_metrics_csv}")
            real_metrics = load_metrics_from_csv(args.real_metrics_csv, args.metric_name, risk_lambda=args.risk_lambda)
        
        if real_metrics is None:
            # 如果real_metrics_csv不存在，运行真实模拟来计算real_metrics
            # 使用所有真实动作（subset_U=set(all_players)）
            print("Real metrics CSV not found. Computing real metrics from simulation...")
            all_players = [(i, t) for i in range(args.num_agents) for t in range(args.episode_length)]
            real_metrics = run_counterfactual_simulation(
                args.num_agents, args.episode_length, real_actions_dict,
                baseline_work, baseline_consumption, subset_U=set(all_players), seed=args.seed,
                use_probabilistic_baseline=args.use_probabilistic_baseline
            )
        
        if real_metrics:
            max_risk_timestep, max_risk_value = find_max_risk_timestep(
                real_metrics, args.metric_name, args.risk_lambda,
                args.inflation_threshold, args.use_metric_directly, args.include_both_risks
            )
            if max_risk_timestep is not None:
                target_timesteps = [max_risk_timestep]
                is_auto_max_risk = True
                print(f"Found maximum risk at timestep {max_risk_timestep} (risk value: {max_risk_value:.6f})")
            else:
                print("Warning: Could not find maximum risk timestep. Using last timestep.")
                target_timesteps = [args.episode_length]
        else:
            print("Warning: Could not compute metrics for auto-detection. Using last timestep.")
            target_timesteps = [args.episode_length]
    
    print("\n" + "="*60)
    print("Shapley Value Calculation - Monte Carlo Method")
    print("="*60)
    print(f"Agents: {args.num_agents}, Timesteps: {args.episode_length}")
    print(f"Baseline Type: {args.baseline_type}")
    if args.baseline_type == 'stable':
        stable_period = args.stable_period if args.stable_period is not None else args.episode_length // 3
        print(f"Stable Period: {stable_period}")
    print(f"MC Samples: {args.n_samples}")
    print(f"Metric Name: {args.metric_name}")
    if args.metric_name == 'risk_indicator_naive':
        print(f"Risk Lambda: {args.risk_lambda}")
    print(f"Inflation Threshold: {args.inflation_threshold}")
    print(f"Risk Aggregation: {args.risk_aggregation}")
    print(f"Include Both Risks: {args.include_both_risks}")
    if target_timesteps:
        if len(target_timesteps) > 1:
            print(f"Target Timesteps: {target_timesteps} (multi-timestep mode)")
        else:
            if is_auto_max_risk:
                print(f"Target Timestep: {target_timesteps[0]} (auto-detected max risk, value: {max_risk_value:.6f})")
            else:
                print(f"Target Timestep: {target_timesteps[0]}")
    print("="*60 + "\n")
    
    # Compute baseline_metrics for saving
    baseline_metrics = None
    if args.baseline_metrics_csv and os.path.exists(args.baseline_metrics_csv):
        print(f"Loading baseline metrics from: {args.baseline_metrics_csv}")
        baseline_metrics = load_metrics_from_csv(args.baseline_metrics_csv, args.metric_name, risk_lambda=args.risk_lambda)
    else:
        print("Computing baseline metrics...")
        baseline_metrics = run_counterfactual_simulation(
            args.num_agents, args.episode_length, real_actions_dict,
            baseline_work, baseline_consumption, subset_U=set(), seed=args.seed,
            use_probabilistic_baseline=args.use_probabilistic_baseline
        )
    
    # 判断是否为多时间点模式
    is_multi_timestep = target_timesteps is not None and len(target_timesteps) > 1
    
    # 检查是否已经计算过最高风险点的结果（如果是自动查找的最高风险点）
    if is_auto_max_risk:
        max_risk_output_file = os.path.join(args.output_dir, 'shapley_values_max_risk.npy')
        stats_file = os.path.join(args.output_dir, 'shapley_stats.json')
        
        # 检查文件是否存在
        if os.path.exists(max_risk_output_file):
            # 检查stats.json中是否标记为最高风险点计算
            skip_computation = False
            if os.path.exists(stats_file):
                try:
                    with open(stats_file, 'r') as f:
                        stats = json.load(f)
                        if stats.get('is_max_risk_timestep', False) and stats.get('max_risk_timestep') == max_risk_timestep:
                            skip_computation = True
                            print(f"\nFound existing results for max risk timestep {max_risk_timestep}")
                            print(f"  Output file: {max_risk_output_file}")
                            print(f"  Stats file: {stats_file}")
                            print("Skipping computation...")
                except Exception as e:
                    print(f"Warning: Could not read stats file: {e}")
            
            if skip_computation:
                print("\nComputation skipped. Use existing results.")
                return
        
        print(f"\nComputing Shapley values for auto-detected max risk timestep {max_risk_timestep}...")
    
    # Compute Shapley values
    result = monte_carlo_shapley(
        args.num_agents, args.episode_length, real_actions_dict,
        baseline_work, baseline_consumption,
        args.inflation_threshold, args.n_samples, args.seed,
        baseline_metrics_csv=args.baseline_metrics_csv,
        real_metrics_csv=args.real_metrics_csv,
        target_timestep=args.target_timestep,
        target_timesteps=target_timesteps,
        use_probabilistic_baseline=args.use_probabilistic_baseline,
        metric_name=args.metric_name,
        use_metric_directly=args.use_metric_directly,
        risk_aggregation=args.risk_aggregation,
        include_both_risks=args.include_both_risks,
        risk_lambda=args.risk_lambda
    )
    
    # 处理返回结果：可能是单个结果或多时间点结果
    if is_multi_timestep:
        shapley_values_dict, aggregated_shapley_values, baseline_risk, real_risk, _ = result
        shapley_values = aggregated_shapley_values  # 用于后续统计
    else:
        shapley_values, baseline_risk, real_risk, _ = result
        shapley_values_dict = None
        aggregated_shapley_values = None
    
    # Save results
    if is_multi_timestep:
        # 保存每个时间点的结果
        for t in target_timesteps:
            np.save(os.path.join(args.output_dir, f'shapley_values_t{t}.npy'), shapley_values_dict[t])
        # 保存聚合结果
        np.save(os.path.join(args.output_dir, 'shapley_values_aggregated.npy'), aggregated_shapley_values)
        # 也保存为默认名称（使用聚合结果）
        np.save(os.path.join(args.output_dir, 'shapley_values.npy'), aggregated_shapley_values)
    else:
        # 如果使用自动查找的最高风险点，保存为特殊文件名
        if is_auto_max_risk:
            np.save(os.path.join(args.output_dir, 'shapley_values_max_risk.npy'), shapley_values)
            # 也保存为默认名称以便兼容
            np.save(os.path.join(args.output_dir, 'shapley_values.npy'), shapley_values)
        else:
            np.save(os.path.join(args.output_dir, 'shapley_values.npy'), shapley_values)
    
    # Save baseline metrics
    # Save baseline in a subdirectory of output_dir for easier visualization
    baseline_dir = os.path.join(args.output_dir, 'baseline')
    os.makedirs(baseline_dir, exist_ok=True)
    baseline_metrics_path = os.path.join(baseline_dir, 'world_metrics.csv')
    save_metrics_to_csv(baseline_metrics, baseline_metrics_path)
    
    stats = {
        'method': 'mc',
        'baseline_risk': float(baseline_risk),
        'real_risk': float(real_risk),
        'total_shapley_sum': float(np.sum(shapley_values)),
        'mean_shapley': float(np.mean(shapley_values)),
        'std_shapley': float(np.std(shapley_values)),
        'min_shapley': float(np.min(shapley_values)),
        'max_shapley': float(np.max(shapley_values)),
        'n_samples': args.n_samples,
        'inflation_threshold': args.inflation_threshold,
        # Baseline configuration
        'baseline_type': args.baseline_type,
        'baseline_work': float(baseline_work),
        'baseline_consumption': float(baseline_consumption),
        'use_probabilistic_baseline': args.use_probabilistic_baseline,
        'stable_period': args.stable_period if args.baseline_type == 'stable' else None,
        # Other parameters
        'target_timestep': args.target_timestep,
        'target_timesteps': target_timesteps,
        'is_multi_timestep': is_multi_timestep,
        'metric_name': args.metric_name,
        'use_metric_directly': args.use_metric_directly,
        'risk_aggregation': args.risk_aggregation,
        'include_both_risks': args.include_both_risks,
        'risk_lambda': args.risk_lambda,
        'num_agents': args.num_agents,
        'episode_length': args.episode_length,
        'seed': args.seed,
        # Max risk detection
        'is_max_risk_timestep': is_auto_max_risk,
        'max_risk_timestep': max_risk_timestep if is_auto_max_risk else None,
        'max_risk_value': float(max_risk_value) if is_auto_max_risk and max_risk_value is not None else None,
    }
    
    # 如果是多时间点，添加每个时间点的统计信息
    if is_multi_timestep:
        stats['timestep_stats'] = {}
        for t in target_timesteps:
            sv = shapley_values_dict[t]
            stats['timestep_stats'][t] = {
                'total_shapley_sum': float(np.sum(sv)),
                'mean_shapley': float(np.mean(sv)),
                'std_shapley': float(np.std(sv)),
                'min_shapley': float(np.min(sv)),
                'max_shapley': float(np.max(sv)),
            }
    
    with open(os.path.join(args.output_dir, 'shapley_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    print(f"Baseline Risk: {baseline_risk:.6f}")
    print(f"Real Risk: {real_risk:.6f}")
    print(f"Risk Difference: {real_risk - baseline_risk:.6f}")
    if is_multi_timestep:
        print(f"\nMulti-timestep Results (aggregated across {len(target_timesteps)} timesteps):")
        print(f"Total Shapley Sum: {np.sum(shapley_values):.6f}")
        print(f"Mean Shapley Value: {np.mean(shapley_values):.6f}")
        print(f"\nPer-timestep Results:")
        for t in target_timesteps:
            sv = shapley_values_dict[t]
            print(f"  Timestep {t}: Total={np.sum(sv):.6f}, Mean={np.mean(sv):.6f}")
    else:
        print(f"Total Shapley Sum: {np.sum(shapley_values):.6f}")
        print(f"Mean Shapley Value: {np.mean(shapley_values):.6f}")
    print("="*60 + "\n")
    
    print(f"\nAll results saved to: {args.output_dir}")
    print(f"\nTo generate visualizations, run:")
    print(f"  python scripts/plot/plot_shapley.py \\")
    print(f"    --shapley_values {os.path.join(args.output_dir, 'shapley_values.npy')} \\")
    print(f"    --stats_json {os.path.join(args.output_dir, 'shapley_stats.json')} \\")
    print(f"    --output_dir {args.output_dir}")


if __name__ == '__main__':
    main()
