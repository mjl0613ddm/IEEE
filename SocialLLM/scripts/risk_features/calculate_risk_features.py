#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SocialLLM 风险特征计算脚本
计算5个风险特征指标，适配SocialLLM的数据结构

5个指标：
1. L_tm: 相对风险延迟（90%风险累积时刻）
2. G_ag: 风险个体稀疏性（基于agent的基尼系数）
3. C_ag: 风险-不稳定性相关系数（先取绝对值）
4. Z_ag: 风险个体协同性
5. G_be: 行为风险集中度（基于6种行为模式的基尼系数）

使用方法:
    python scripts/risk_feature/calculate_risk_features.py --result_dir results/gpt-4o-mini/gpt-4o-mini_42
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 6种行为模式定义
# 0: posted=0, like
# 1: posted=0, no_interaction
# 2: posted=0, dislike
# 3: posted=1, like
# 4: posted=1, no_interaction
# 5: posted=1, dislike
BEHAVIOR_CATEGORY_NAMES = [
    'posted=0, like',
    'posted=0, no_interaction',
    'posted=0, dislike',
    'posted=1, like',
    'posted=1, no_interaction',
    'posted=1, dislike',
]


def load_shapley_data(result_dir: Path) -> Tuple[np.ndarray, Dict, List[int], List[int]]:
    """
    加载Shapley数据
    
    Args:
        result_dir: 结果目录路径
        
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
    df_shapley = pd.read_csv(shapley_file)
    print(f"Loaded shapley data from: {shapley_file}")
    print(f"  Shape: {df_shapley.shape}")
    
    # 加载统计信息
    with open(shapley_stats_file, 'r', encoding='utf-8') as f:
        stats = json.load(f)
    print(f"Loaded stats from: {shapley_stats_file}")
    
    # 获取agent_ids和timesteps
    agent_ids = sorted(df_shapley['agent_id'].unique().tolist())
    timesteps = sorted(df_shapley['timestep'].unique().tolist())
    
    num_agents = len(agent_ids)
    num_timesteps = len(timesteps)
    
    print(f"  Agents: {num_agents}, Timesteps: {num_timesteps}")
    
    # 创建agent_id和timestep到索引的映射
    agent_id_to_idx = {aid: idx for idx, aid in enumerate(agent_ids)}
    timestep_to_idx = {t: idx for idx, t in enumerate(timesteps)}
    
    # 构建Shapley矩阵 (num_agents × num_timesteps)
    shapley_matrix = np.zeros((num_agents, num_timesteps), dtype=np.float64)
    
    for _, row in df_shapley.iterrows():
        agent_id = int(row['agent_id'])
        timestep = int(row['timestep'])
        shapley_value = float(row['shapley_value'])
        
        if agent_id in agent_id_to_idx and timestep in timestep_to_idx:
            agent_idx = agent_id_to_idx[agent_id]
            timestep_idx = timestep_to_idx[timestep]
            shapley_matrix[agent_idx, timestep_idx] = shapley_value
    
    return shapley_matrix, stats, agent_ids, timesteps


def load_action_table(result_dir: Path) -> pd.DataFrame:
    """
    加载action_table数据
    
    Args:
        result_dir: 结果目录路径
        
    Returns:
        DataFrame with columns: agent_id, timestep, posted, view_count, like_count, dislike_count, belief
    """
    action_table_file = result_dir / "action_table" / "action_table.csv"
    
    if not action_table_file.exists():
        raise FileNotFoundError(f"Action table file not found: {action_table_file}")
    
    df = pd.read_csv(action_table_file)
    print(f"Loaded action table from: {action_table_file}")
    print(f"  Shape: {df.shape}")
    
    return df


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


def calculate_time_delay(shapley_matrix: np.ndarray) -> float:
    """
    指标1：相对风险延迟 (L_tm)
    
    计算第一次风险累积到90%的时刻
    
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


def classify_actions_by_post_interaction(action_table_df: pd.DataFrame,
                                        shapley_df: pd.DataFrame,
                                        agent_ids: List[int],
                                        timesteps: List[int]) -> Dict[int, float]:
    """
    按发帖和互动类型分类，返回6种模式的Shapley值分配
    
    Args:
        action_table_df: action_table DataFrame
        shapley_df: shapley DataFrame with columns: agent_id, timestep, shapley_value
        agent_ids: agent ID列表
        timesteps: timestep列表
        
    Returns:
        {behavior_category: shapley_value} 字典，category为0-5
    """
    # 创建agent_id和timestep到Shapley值的映射
    shapley_dict = {}
    for _, row in shapley_df.iterrows():
        agent_id = int(row['agent_id'])
        timestep = int(row['timestep'])
        shapley_value = float(row['shapley_value'])
        shapley_dict[(agent_id, timestep)] = shapley_value
    
    # 创建agent_id和timestep到action的映射
    action_dict = {}
    for _, row in action_table_df.iterrows():
        agent_id = int(row['agent_id'])
        timestep = int(row['timestep'])
        action_dict[(agent_id, timestep)] = {
            'posted': int(row.get('posted', 0)),
            'view_count': int(row.get('view_count', 0)),
            'like_count': int(row.get('like_count', 0)),
            'dislike_count': int(row.get('dislike_count', 0)),
        }
    
    # 初始化6种行为模式的Shapley值
    behavior_shapley = {i: 0.0 for i in range(6)}
    
    matched_count = 0
    
    # 遍历所有(agent_id, timestep)对
    for agent_id in agent_ids:
        for timestep in timesteps:
            key = (agent_id, timestep)
            
            # 获取Shapley值
            shapley_value = shapley_dict.get(key, 0.0)
            if abs(shapley_value) < 1e-10:
                continue
            
            # 获取action信息
            action_info = action_dict.get(key)
            if action_info is None:
                continue
            
            posted = action_info['posted']
            view_count = action_info['view_count']
            like_count = action_info['like_count']
            dislike_count = action_info['dislike_count']
            
            # 计算不互动数量
            no_interaction_count = view_count - like_count - dislike_count
            if no_interaction_count < 0:
                no_interaction_count = 0
            
            # 情况1：没发帖但看了帖子（posted=0, view_count>0）
            if posted == 0 and view_count > 0:
                # 按三个行为的比例分配
                if view_count > 0:
                    like_ratio = like_count / view_count
                    no_interaction_ratio = no_interaction_count / view_count
                    dislike_ratio = dislike_count / view_count
                    
                    # category 0: posted=0, like
                    behavior_shapley[0] += shapley_value * like_ratio
                    # category 1: posted=0, no_interaction
                    behavior_shapley[1] += shapley_value * no_interaction_ratio
                    # category 2: posted=0, dislike
                    behavior_shapley[2] += shapley_value * dislike_ratio
                    matched_count += 1
            
            # 情况2：发帖但没有互动（posted=1, view_count=0）
            elif posted == 1 and view_count == 0:
                # 100%分配给"posted=1, no_interaction"（category 4）
                behavior_shapley[4] += shapley_value
                matched_count += 1
            
            # 情况3：发帖并且有互动（posted=1, view_count>0）
            elif posted == 1 and view_count > 0:
                # 25%分配给"posted=1"相关的行为（按互动类型比例分配）
                # 75%按比例分配给三个互动行为
                if view_count > 0:
                    like_ratio = like_count / view_count
                    no_interaction_ratio = no_interaction_count / view_count
                    dislike_ratio = dislike_count / view_count
                    
                    # 25%部分：按互动类型比例分配到posted=1的三种类型
                    # category 3: posted=1, like
                    behavior_shapley[3] += shapley_value * 0.25 * like_ratio
                    # category 4: posted=1, no_interaction
                    behavior_shapley[4] += shapley_value * 0.25 * no_interaction_ratio
                    # category 5: posted=1, dislike
                    behavior_shapley[5] += shapley_value * 0.25 * dislike_ratio
                    
                    # 75%部分：按比例分配给三个互动行为
                    behavior_shapley[3] += shapley_value * 0.75 * like_ratio
                    behavior_shapley[4] += shapley_value * 0.75 * no_interaction_ratio
                    behavior_shapley[5] += shapley_value * 0.75 * dislike_ratio
                    matched_count += 1
            
            # 情况4：发帖但没有看任何帖子（posted=1, view_count=0）- 这种情况已经在情况2中处理
            # 情况5：既没发帖也没看帖子（posted=0, view_count=0）- 不分配Shapley值
    
    if matched_count == 0:
        print(f"  Warning: No matched (agent, timestep) pairs for action sparsity calculation")
    
    return behavior_shapley


def calculate_action_sparsity(shapley_matrix: np.ndarray,
                              action_table_df: pd.DataFrame,
                              shapley_df: pd.DataFrame,
                              agent_ids: List[int],
                              timesteps: List[int]) -> Tuple[float, Dict[int, float]]:
    """
    指标5：行为风险集中度 (G_be)
    
    基于6种行为模式的Shapley值的基尼系数
    
    Args:
        shapley_matrix: Shapley矩阵 (N×T)
        action_table_df: action_table DataFrame
        shapley_df: shapley DataFrame
        agent_ids: agent ID列表
        timesteps: timestep列表
        
    Returns:
        (sparsity, behavior_shapley_dict) 元组
    """
    # 分类行为并分配Shapley值
    behavior_shapley = classify_actions_by_post_interaction(
        action_table_df, shapley_df, agent_ids, timesteps
    )
    
    # 转换为6维数组
    behavior_shapley_array = np.array([behavior_shapley[i] for i in range(6)], dtype=np.float64)
    
    # 计算基尼系数
    sparsity = calculate_gini_coefficient(behavior_shapley_array)
    
    return sparsity, behavior_shapley


def calculate_all_risk_metrics(shapley_matrix: np.ndarray,
                               action_table_df: pd.DataFrame,
                               shapley_df: pd.DataFrame,
                               agent_ids: List[int],
                               timesteps: List[int]) -> Dict:
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
    
    # 指标1：相对风险延迟 L_tm（90%累积时刻）
    metrics['L_tm'] = calculate_time_delay(shapley_matrix)
    
    # 指标2：风险个体稀疏性 G_ag
    metrics['G_ag'] = calculate_agent_sparsity(shapley_matrix)
    
    # 指标3：风险-不稳定性相关系数 C_ag（取绝对值）
    metrics['C_ag'] = calculate_system_vulnerability(shapley_matrix)
    
    # 指标4：风险个体协同性 Z_ag
    metrics['Z_ag'] = calculate_agent_synergy(shapley_matrix)
    
    # 指标5：行为风险集中度 G_be
    metrics['G_be'], behavior_shapley = calculate_action_sparsity(
        shapley_matrix, action_table_df, shapley_df, agent_ids, timesteps
    )
    metrics['behavior_shapley'] = behavior_shapley
    
    return metrics


def save_risk_features_json(metrics: Dict, stats: Dict, agent_ids: List[int],
                            timesteps: List[int], output_path: Path):
    """
    保存风险特征指标JSON文件
    
    Args:
        metrics: 包含5个指标的字典
        stats: Shapley统计信息
        agent_ids: agent ID列表
        timesteps: timestep列表
        output_path: 输出JSON文件路径
    """
    behavior_shapley = metrics.get('behavior_shapley', {})
    behavior_shapley_named = {
        BEHAVIOR_CATEGORY_NAMES[i]: float(behavior_shapley.get(i, 0.0))
        for i in range(6)
    }
    
    data = {
        "L_tm": metrics.get('L_tm'),
        "G_ag": metrics.get('G_ag'),
        "C_ag": metrics.get('C_ag'),
        "Z_ag": metrics.get('Z_ag'),
        "G_be": metrics.get('G_be'),
        "behavior_shapley": behavior_shapley_named,
        "metadata": {
            "num_agents": len(agent_ids),
            "num_timesteps": len(timesteps),
            "agent_ids": agent_ids,
            "timesteps": timesteps,
            "max_risk_timestep": stats.get('max_risk_timestep'),
            "max_risk": stats.get('max_risk'),
            "initial_risk": stats.get('initial_risk'),
        }
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Risk features JSON saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='计算SocialLLM风险特征指标',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--result_dir',
        type=str,
        required=True,
        help='结果目录路径（包含shapley/和action_table/目录）'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='输出目录（默认：{result_dir}/risk_feature）'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='输出详细信息'
    )
    
    args = parser.parse_args()
    
    # 解析结果目录路径
    result_dir = Path(args.result_dir)
    if not result_dir.is_absolute():
        result_dir = project_root / result_dir
    result_dir = result_dir.resolve()
    
    if not result_dir.exists():
        print(f"错误: 结果目录不存在: {result_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"处理结果目录: {result_dir}")
    print("=" * 60)
    
    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = result_dir / "risk_feature"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载Shapley数据
    print("Step 1: Loading Shapley data...")
    shapley_matrix, stats, agent_ids, timesteps = load_shapley_data(result_dir)
    print(f"  Shapley matrix shape: {shapley_matrix.shape}")
    
    # 2. 加载action_table数据
    print("\nStep 2: Loading action table data...")
    action_table_df = load_action_table(result_dir)
    
    # 创建shapley DataFrame用于行为分类
    shapley_file = result_dir / "shapley" / "shapley_attribution_timeseries.csv"
    shapley_df = pd.read_csv(shapley_file)
    
    # 3. 计算风险特征指标
    print("\nStep 3: Calculating risk metrics...")
    metrics = calculate_all_risk_metrics(
        shapley_matrix, action_table_df, shapley_df, agent_ids, timesteps
    )
    
    print(f"  L_tm (Relative risk latency): {metrics['L_tm']:.6f}")
    print(f"  G_ag (Agent risk concentration): {metrics['G_ag']:.6f}")
    print(f"  C_ag (Risk-instability correlation): {metrics['C_ag']:.6f}")
    print(f"  Z_ag (Agent risk synchronization): {metrics['Z_ag']:.6f}")
    print(f"  G_be (Behavioral risk concentration): {metrics['G_be']:.6f}")
    
    # 打印行为Shapley值分布
    behavior_shapley = metrics.get('behavior_shapley', {})
    print(f"\n  Behavior Shapley distribution:")
    for i in range(6):
        category_name = BEHAVIOR_CATEGORY_NAMES[i]
        value = behavior_shapley.get(i, 0.0)
        print(f"    {category_name}: {value:.6f}")
    
    # 4. 保存结果
    print("\nStep 4: Saving results...")
    risk_features_json_path = output_dir / "risk_features.json"
    save_risk_features_json(metrics, stats, agent_ids, timesteps, risk_features_json_path)
    
    print(f"\n✓ Risk features saved to: {output_dir}")


if __name__ == "__main__":
    main()