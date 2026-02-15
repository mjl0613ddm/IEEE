#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TwinMarket 风险特征计算脚本
计算5个风险特征指标，适配TwinMarket的数据结构
修改了两个指标：
1. 时间指标（L_tm）：改为计算90%风险累积时刻
2. 风险-不稳定性相关系数（C_ag）：先取绝对值再计算
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

# 10种股票的固定顺序
STOCK_LIST = ['TLEI', 'MEI', 'CPEI', 'IEEI', 'REEI', 'TSEI', 'CGEI', 'TTEI', 'EREI', 'FSEI']
STOCK_TO_IDX = {stock: idx for idx, stock in enumerate(STOCK_LIST)}


def generate_dates_from_range(date_range_str: str) -> List[str]:
    """
    从日期范围字符串生成日期列表
    
    Args:
        date_range_str: 格式为 "2023-06-15_2023-07-07" 的字符串
    
    Returns:
        日期字符串列表，格式为 "YYYY-MM-DD"
    """
    if '_' not in date_range_str:
        return []
    
    try:
        start_str, end_str = date_range_str.split('_', 1)
        start_date = datetime.strptime(start_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_str, '%Y-%m-%d')
        
        dates = []
        current_date = start_date
        while current_date <= end_date:
            dates.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)
        
        return dates
    except Exception as e:
        print(f"Warning: Failed to parse date range '{date_range_str}': {e}")
        return []


def create_white_blue_red_colormap():
    """创建白色-蓝色-红色的颜色映射（白色表示最低值）."""
    colors = [
        '#FFFFFF', '#F0F8FF', '#E0F2FE', '#BAE6FD', '#93C5FD',
        '#60B5FA', '#3B82F6', '#2563EB', '#1D4ED8', '#1E40AF',
        '#1E3A8A', '#172554', '#1E1B4B', '#4C1D95', '#7C2D12',
        '#DC2626', '#EF4444', '#F87171', '#FB923C', '#FF5252',
        '#FF1744', '#C62828'
    ]
    n_bins = 512
    cmap = LinearSegmentedColormap.from_list('white_blue_red', colors, N=n_bins)
    return cmap


def find_latest_shapley_files(shapley_dir: Path) -> Tuple[Optional[Path], Optional[Path], Optional[Path], Optional[str]]:
    """
    在shapley目录中查找最新的shapley文件
    
    Returns:
        (shapley_matrix_path, shapley_labels_path, shapley_stats_path, date_range)
    """
    shapley_files = list(shapley_dir.glob("shapley_matrix_*.npy"))
    if not shapley_files:
        return None, None, None, None
    
    # 按文件名排序，选择最新的（假设文件名包含日期范围）
    shapley_files.sort(reverse=True)
    latest_matrix = shapley_files[0]
    
    # 从文件名提取日期范围
    # 格式: shapley_matrix_2023-06-15_2023-07-07.npy
    filename = latest_matrix.stem
    if '_' in filename:
        parts = filename.split('_')
        if len(parts) >= 4:
            date_range = f"{parts[2]}_{parts[3]}"
        else:
            date_range = None
    else:
        date_range = None
    
    # 构建对应的labels和stats文件路径
    labels_path = shapley_dir / f"shapley_labels_{date_range}.npy" if date_range else None
    stats_path = shapley_dir / f"shapley_stats_{date_range}.json" if date_range else None
    
    if labels_path and not labels_path.exists():
        labels_path = None
    if stats_path and not stats_path.exists():
        stats_path = None
    
    return latest_matrix, labels_path, stats_path, date_range


def load_shapley_data(shapley_dir: Path) -> Tuple[np.ndarray, Dict, List[str], List[str]]:
    """
    加载Shapley数据
    
    Returns:
        (shapley_matrix, stats, user_ids, dates)
    """
    # 查找最新的shapley文件
    matrix_path, labels_path, stats_path, date_range = find_latest_shapley_files(shapley_dir)
    
    if matrix_path is None:
        raise FileNotFoundError(f"No shapley matrix file found in {shapley_dir}")
    
    # 加载矩阵
    shapley_matrix = np.load(matrix_path)
    print(f"Loaded shapley matrix from: {matrix_path}")
    print(f"  Shape: {shapley_matrix.shape}")
    
    # 加载标签
    user_ids = []
    dates = []
    if labels_path and labels_path.exists():
        labels_data = np.load(labels_path, allow_pickle=True).item()
        user_ids = labels_data.get('user_ids', [])
        dates = labels_data.get('dates', [])
        print(f"Loaded labels from: {labels_path}")
        print(f"  User IDs: {len(user_ids)}, Dates: {len(dates)}")
    else:
        # 如果没有labels文件，尝试从stats推断
        print(f"Warning: Labels file not found, will try to infer from stats")
    
    # 加载统计信息
    stats = {}
    if stats_path and stats_path.exists():
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        print(f"Loaded stats from: {stats_path}")
        
        # 如果dates为空，尝试从stats的date_range生成
        if not dates and 'date_range' in stats:
            date_range_str = stats['date_range']
            dates = generate_dates_from_range(date_range_str)
            if dates:
                print(f"Generated {len(dates)} dates from date_range: {date_range_str}")
    else:
        print(f"Warning: Stats file not found")
    
    return shapley_matrix, stats, user_ids, dates


def load_transactions_by_date(simulation_results_dir: Path, dates: List[str]) -> pd.DataFrame:
    """
    按日期加载交易记录
    
    Returns:
        包含所有日期的交易记录的DataFrame
    """
    all_transactions = []
    
    for date in dates:
        trans_file = simulation_results_dir / date / f"transactions_{date}.csv"
        if trans_file.exists():
            df = pd.read_csv(trans_file)
            df['date'] = date  # 添加日期列
            all_transactions.append(df)
        else:
            print(f"Warning: Transaction file not found: {trans_file}")
    
    if not all_transactions:
        return pd.DataFrame()
    
    combined_df = pd.concat(all_transactions, ignore_index=True)
    print(f"Loaded {len(combined_df)} transactions from {len(all_transactions)} dates")
    return combined_df


def classify_actions_by_stock(transactions_df: pd.DataFrame, dates: List[str], 
                              user_ids: List[str]) -> Dict[Tuple[str, str], List[str]]:
    """
    按股票类型分类动作（同时考虑买入和卖出）
    
    如果一个agent一天对多支股票进行了交易（买入或卖出），
    返回该agent当天交易过的所有股票列表（包括买入和卖出）
    
    Returns:
        {(user_id, date): [stock1, stock2, ...]} 的字典
    """
    if transactions_df.empty:
        return {}
    
    # 同时考虑买入和卖出交易
    all_transactions = transactions_df.copy()
    
    # 创建user_id到索引的映射（如果user_ids是字符串）
    user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)} if user_ids else {}
    
    # 按(user_id, date)分组，收集交易过的股票（买入和卖出都计入）
    actions_dict = {}
    
    for (user_id, date), group in all_transactions.groupby(['user_id', 'date']):
        # 获取该agent当天交易过的所有唯一股票（包括买入和卖出）
        stocks = group['stock_code'].unique().tolist()
        # 只保留在STOCK_LIST中的股票
        stocks = [s for s in stocks if s in STOCK_LIST]
        
        if stocks:
            actions_dict[(user_id, date)] = stocks
    
    print(f"Classified actions for {len(actions_dict)} (agent, date) pairs (buy and sell)")
    return actions_dict


def calculate_gini_coefficient(values: np.ndarray) -> float:
    """
    计算基尼系数（Gini Coefficient）
    
    使用公式：G = Σ_i Σ_j |x_i - x_j| / (2N * Σ_i |x_i|)
    """
    if len(values) == 0:
        return 0.0
    
    abs_values = np.abs(values)
    n = len(abs_values)
    sum_abs = np.sum(abs_values)
    
    if sum_abs == 0:
        return 0.0
    
    diff_sum = 0.0
    for i in range(n):
        for j in range(n):
            diff_sum += abs(abs_values[i] - abs_values[j])
    
    gini = diff_sum / (2 * n * sum_abs)
    return float(gini)


def calculate_agent_sparsity(shapley_matrix: np.ndarray) -> float:
    """
    指标2：风险个体稀疏性 (G_ag)
    
    基于个体风险归因绝对值的基尼系数
    """
    # 计算每个agent的总Shapley值
    agent_total_shapley = np.sum(shapley_matrix, axis=1)
    return calculate_gini_coefficient(agent_total_shapley)


def calculate_system_vulnerability(shapley_matrix: np.ndarray) -> float:
    """
    指标3：风险-不稳定性相关系数 (C_ag)
    
    修改：先对贡献值取绝对值，然后计算贡献绝对值与方差的相关系数
    
    个体风险贡献绝对值 |φ^ag_i| 与行为不确定性（标准差）σ^ag_i 之间的皮尔森相关系数
    """
    # 计算每个agent的总Shapley值，取绝对值
    phi_i = np.abs(np.sum(shapley_matrix, axis=1))
    
    # 计算每个agent的标准差（不稳定性）
    sigma_i = np.std(shapley_matrix, axis=1, ddof=0)
    
    # 计算Pearson相关系数
    if np.std(phi_i) == 0 or np.std(sigma_i) == 0:
        return 0.0
    
    correlation = np.corrcoef(phi_i, sigma_i)[0, 1]
    
    if np.isnan(correlation):
        return 0.0
    
    return float(correlation)


def calculate_agent_synergy(shapley_matrix: np.ndarray) -> float:
    """
    指标4：风险个体协同性 (Z_ag)
    
    系统在单位时间内，所有个体沙普利值矢量和的模与标量和之比的时间平均值
    """
    T = shapley_matrix.shape[1]
    
    if T == 0:
        return 0.0
    
    ratios = []
    for t in range(T):
        # 矢量和的模
        vector_sum = np.abs(np.sum(shapley_matrix[:, t]))
        
        # 标量和
        scalar_sum = np.sum(np.abs(shapley_matrix[:, t]))
        
        if scalar_sum == 0:
            ratio = 0.0
        else:
            ratio = vector_sum / scalar_sum
        
        ratios.append(ratio)
    
    return float(np.mean(ratios))


def calculate_time_delay(shapley_matrix: np.ndarray) -> float:
    """
    指标1：相对风险延迟 (L_tm)
    
    修改：计算第一次风险累积到90%的时刻
    
    公式：L_tm = (T - T*) / T
    其中 T* = min{t' | sum_{t=1}^{t'} φ^tm_t > q * ρ}
    q = 0.9, ρ = sum_{t=1}^T φ^tm_t（总风险，直接和，不使用绝对值）
    """
    # 计算时间Shapley值（沿agent维度求和）
    time_shapley = np.sum(shapley_matrix, axis=0)
    
    T = len(time_shapley)
    
    if T == 0:
        return 0.0
    
    # 计算总风险（直接和，不使用绝对值）
    rho = np.sum(time_shapley)
    
    # 如果总风险为0，返回0
    if abs(rho) < 1e-10:
        return 0.0
    
    # 计算累积和
    cumulative = np.cumsum(time_shapley)
    
    # 找到第一次累积到90%的时刻：T* = min{t' | cumulative[t'] > 0.9 * rho}
    q = 0.9
    threshold = q * rho
    T_star = None
    
    for t in range(T):
        if cumulative[t] > threshold:
            T_star = t + 1  # 转换为1-based索引（公式中的t'）
            break
    
    # 如果从未达到90%，使用T
    if T_star is None:
        T_star = T
    
    # 计算相对风险延迟
    L_tm = (T - T_star) / T
    
    return float(L_tm)


def calculate_action_sparsity(shapley_matrix: np.ndarray, dates: List[str], 
                              user_ids: List[str],
                              actions_dict: Dict[Tuple[str, str], List[str]]) -> Tuple[float, Dict[str, float]]:
    """
    指标5：行为风险集中度 (G_be)
    
    基于股票类别Shapley值的基尼系数（K=10）
    
    同时考虑买入和卖出交易，按股票种类分类
    如果一个agent一天对多支股票进行了交易（买入或卖出），
    将该agent当天的Shapley值均分到各个交易过的股票类别
    
    Returns:
        (sparsity, stock_shapley_dict)
    """
    # 创建user_id和date到索引的映射（转换为字符串以确保匹配）
    user_id_to_idx = {}
    if user_ids:
        for idx, uid in enumerate(user_ids):
            # 将user_id转换为字符串，支持多种格式
            uid_str = str(uid)
            user_id_to_idx[uid_str] = idx
    
    date_to_idx = {str(date): idx for idx, date in enumerate(dates)} if dates else {}
    
    # 计算每个股票类别的Shapley值
    stock_shapley = {stock: 0.0 for stock in STOCK_LIST}
    
    matched_count = 0
    for (user_id, date), stocks in actions_dict.items():
        # 转换为字符串以确保匹配
        user_id_str = str(user_id)
        date_str = str(date)
        
        # 获取agent和date的索引
        if user_id_str not in user_id_to_idx or date_str not in date_to_idx:
            continue
        
        agent_idx = user_id_to_idx[user_id_str]
        date_idx = date_to_idx[date_str]
        
        # 获取该agent当天的Shapley值
        shapley_value = shapley_matrix[agent_idx, date_idx]
        
        # 如果当天交易过多支股票（买入或卖出），均分Shapley值到各个股票类别
        n_stocks = len(stocks)
        if n_stocks > 0:
            shapley_per_stock = shapley_value / n_stocks
            for stock in stocks:
                if stock in stock_shapley:
                    stock_shapley[stock] += shapley_per_stock
            matched_count += 1
    
    if matched_count == 0:
        print(f"  Warning: No matched (agent, date) pairs for action sparsity calculation")
    
    # 转换为数组（按STOCK_LIST顺序）
    stock_shapley_array = np.array([stock_shapley[stock] for stock in STOCK_LIST])
    
    # 计算基尼系数
    sparsity = calculate_gini_coefficient(stock_shapley_array)
    
    return sparsity, stock_shapley


def calculate_all_risk_metrics(shapley_matrix: np.ndarray, dates: List[str], 
                               user_ids: List[str],
                               actions_dict: Dict[Tuple[str, str], List[str]]) -> Dict:
    """
    计算所有5个风险特征指标
    
    指标顺序：
    1. L_tm: 相对风险延迟 (Relative risk latency)
    2. G_ag: 风险个体稀疏性 (Agent risk concentration)
    3. C_ag: 风险-不稳定性相关系数 (Risk-instability correlation)
    4. Z_ag: 风险个体协同性 (Agent risk synchronization)
    5. G_be: 行为风险集中度 (Behavioral risk concentration)
    
    Returns:
        包含所有指标的字典
    """
    metrics = {}
    
    # 指标1：相对风险延迟 L_tm（已修改：90%累积时刻）
    metrics['L_tm'] = calculate_time_delay(shapley_matrix)
    
    # 指标2：风险个体稀疏性 G_ag
    metrics['G_ag'] = calculate_agent_sparsity(shapley_matrix)
    
    # 指标3：风险-不稳定性相关系数 C_ag（已修改：取绝对值）
    metrics['C_ag'] = calculate_system_vulnerability(shapley_matrix)
    
    # 指标4：风险个体协同性 Z_ag
    metrics['Z_ag'] = calculate_agent_synergy(shapley_matrix)
    
    # 指标5：行为风险集中度 G_be
    metrics['G_be'], stock_shapley = calculate_action_sparsity(
        shapley_matrix, dates, user_ids, actions_dict
    )
    metrics['stock_shapley'] = stock_shapley
    
    return metrics


def save_attribution_data(shapley_matrix: np.ndarray, dates: List[str],
                          user_ids: List[str],
                          stock_shapley: Dict[str, float],
                          output_dir: Path):
    """
    保存5个npy文件到risk_feature目录
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_agents, num_dates = shapley_matrix.shape
    
    # 1. risk_evolution.npy: T维向量（每个时间步的风险贡献，沿着agent聚合）
    risk_evolution = np.sum(shapley_matrix, axis=0)  # shape: (num_dates,)
    np.save(output_dir / "risk_evolution.npy", risk_evolution)
    
    # 2. behaviour_aggregated.npy: K维向量（K=10，沿着股票类别聚合）
    behaviour_aggregated = np.array([stock_shapley.get(stock, 0.0) for stock in STOCK_LIST], 
                                    dtype=np.float64)
    np.save(output_dir / "behaviour_aggregated.npy", behaviour_aggregated)
    
    # 3. attribution_matrix.npy: N×T矩阵（归因矩阵）
    np.save(output_dir / "attribution_matrix.npy", shapley_matrix)
    
    # 4. time_aggregated.npy: T维向量（沿着时间聚合，与risk_evolution相同）
    time_aggregated = risk_evolution.copy()
    np.save(output_dir / "time_aggregated.npy", time_aggregated)
    
    # 5. agent_aggregated.npy: N维向量（沿着agent聚合）
    agent_aggregated = np.sum(shapley_matrix, axis=1)  # shape: (num_agents,)
    np.save(output_dir / "agent_aggregated.npy", agent_aggregated)
    
    print(f"Saved attribution data to: {output_dir}")
    print(f"  - risk_evolution.npy: shape {risk_evolution.shape}")
    print(f"  - behaviour_aggregated.npy: shape {behaviour_aggregated.shape}")
    print(f"  - attribution_matrix.npy: shape {shapley_matrix.shape}")
    print(f"  - time_aggregated.npy: shape {time_aggregated.shape}")
    print(f"  - agent_aggregated.npy: shape {agent_aggregated.shape}")


def save_risk_features_json(metrics: Dict, stats: Dict, dates: List[str],
                            user_ids: List[str], output_path: Path):
    """
    保存风险特征指标JSON文件
    """
    # 构建日期范围字符串
    if dates:
        date_range = f"{dates[0]}_{dates[-1]}"
    else:
        date_range = "unknown"
    
    # 构建输出字典（按顺序：L_tm, G_ag, C_ag, Z_ag, G_be）
    output_data = {
        "L_tm": metrics.get('L_tm'),
        "G_ag": metrics.get('G_ag'),
        "C_ag": metrics.get('C_ag'),
        "Z_ag": metrics.get('Z_ag'),
        "G_be": metrics.get('G_be'),
        "metadata": {
            "num_agents": len(user_ids) if user_ids else 0,
            "num_dates": len(dates) if dates else 0,
            "date_range": date_range,
            "metric_name": stats.get('metric_name', 'unknown'),
            "baseline_metric": stats.get('baseline_metric', 0.0),
            "real_metric": stats.get('real_metric', 0.0)
        }
    }
    
    # 保存JSON文件
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Risk features JSON saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='TwinMarket 风险特征计算'
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='结果根目录（包含shapley文件夹和simulation_results文件夹的目录）'
    )
    
    parser.add_argument(
        '--shapley_dir',
        type=str,
        default=None,
        help='Shapley数据目录（默认：{results_dir}/shapley）'
    )
    
    parser.add_argument(
        '--simulation_results_dir',
        type=str,
        default=None,
        help='模拟结果目录（默认：{results_dir}/simulation_results）'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='输出目录（默认：{results_dir}/risk_feature）'
    )
    
    args = parser.parse_args()
    
    # 解析路径
    results_dir = Path(args.results_dir)
    
    if args.shapley_dir:
        shapley_dir = Path(args.shapley_dir)
    else:
        shapley_dir = results_dir / "shapley"
    
    if args.simulation_results_dir:
        simulation_results_dir = Path(args.simulation_results_dir)
    else:
        simulation_results_dir = results_dir / "simulation_results"
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_dir / "risk_feature"
    
    print("=" * 60)
    print("TwinMarket 风险特征计算")
    print("=" * 60)
    print(f"Results directory: {results_dir}")
    print(f"Shapley directory: {shapley_dir}")
    print(f"Simulation results directory: {simulation_results_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60 + "\n")
    
    # 1. 加载Shapley数据
    print("Step 1: Loading Shapley data...")
    shapley_matrix, stats, user_ids, dates = load_shapley_data(shapley_dir)
    
    print(f"  Matrix shape: {shapley_matrix.shape}")
    print(f"  User IDs: {len(user_ids)}")
    print(f"  Dates: {len(dates)}")
    
    # 验证数据一致性
    if dates and len(dates) != shapley_matrix.shape[1]:
        print(f"  Warning: Date count ({len(dates)}) doesn't match matrix columns ({shapley_matrix.shape[1]})")
        # 如果dates数量不匹配，只使用矩阵的列数对应的dates
        if len(dates) > shapley_matrix.shape[1]:
            dates = dates[:shapley_matrix.shape[1]]
        elif len(dates) < shapley_matrix.shape[1]:
            # 如果dates不够，生成占位符
            print(f"  Warning: Not enough dates, using date indices")
            dates = [f"date_{i}" for i in range(shapley_matrix.shape[1])]
    
    # 如果没有user_ids，生成占位符
    if not user_ids:
        user_ids = [f"user_{i}" for i in range(shapley_matrix.shape[0])]
        print(f"  Generated {len(user_ids)} placeholder user IDs")
    
    print()
    
    # 2. 加载交易记录
    print("Step 2: Loading transactions...")
    if dates:
        transactions_df = load_transactions_by_date(simulation_results_dir, dates)
    else:
        print("  Warning: No dates available, skipping transaction loading")
        transactions_df = pd.DataFrame()
    print()
    
    # 3. 分类动作
    print("Step 3: Classifying actions by stock...")
    if not transactions_df.empty and dates and user_ids:
        actions_dict = classify_actions_by_stock(transactions_df, dates, user_ids)
    else:
        print("  Warning: Cannot classify actions (missing data)")
        actions_dict = {}
    print()
    
    # 4. 计算风险特征指标
    print("Step 4: Calculating risk metrics...")
    metrics = calculate_all_risk_metrics(shapley_matrix, dates, user_ids, actions_dict)
    print(f"  L_tm (Relative risk latency): {metrics['L_tm']:.6f}")
    print(f"  G_ag (Agent risk concentration): {metrics['G_ag']:.6f}")
    print(f"  C_ag (Risk-instability correlation): {metrics['C_ag']:.6f}")
    print(f"  Z_ag (Agent risk synchronization): {metrics['Z_ag']:.6f}")
    print(f"  G_be (Behavioral risk concentration): {metrics['G_be']:.6f}")
    print()
    
    # 5. 保存数据
    print("Step 5: Saving data files...")
    
    # 5.1 保存5个npy文件
    save_attribution_data(
        shapley_matrix, dates, user_ids,
        metrics.get('stock_shapley', {}),
        output_dir
    )
    
    # 5.2 保存风险特征指标JSON文件
    risk_features_json_path = output_dir / "risk_features.json"
    save_risk_features_json(metrics, stats, dates, user_ids, risk_features_json_path)
    
    print()
    print("=" * 60)
    print("All tasks completed!")
    print("=" * 60)
    print(f"Risk features saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
