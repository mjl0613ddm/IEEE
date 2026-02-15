#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EconAgent 风险特征计算脚本
计算5个风险特征指标，适配EconAgent的数据结构
修改了两个指标：
1. 时间指标（L_tm）：改为计算90%风险累积时刻
2. 风险-不稳定性相关系数（C_ag）：先取绝对值再计算
3. 行为风险集中度（G_be）：按工作状态和消费比例分8个类型
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# 消费比例区间定义（4个区间）
CONSUMPTION_CATEGORIES = [
    (0.0, 0.25, 0),   # 0-25%
    (0.25, 0.50, 1),  # 25-50%
    (0.50, 0.75, 2),  # 50-75%
    (0.75, 1.0, 3),   # 75-100%
]

# 8种行为类型定义（工作状态 × 消费比例区间）
# 类型0-3：不工作(0) + 4个consumption区间
# 类型4-7：工作(1) + 4个consumption区间
ACTION_CATEGORY_NAMES = [
    'work=0, cons=0-25%',   # 类型0
    'work=0, cons=25-50%',  # 类型1
    'work=0, cons=50-75%',  # 类型2
    'work=0, cons=75-100%', # 类型3
    'work=1, cons=0-25%',   # 类型4
    'work=1, cons=25-50%',  # 类型5
    'work=1, cons=50-75%',  # 类型6
    'work=1, cons=75-100%', # 类型7
]


def load_shapley_data(shapley_dir: Path) -> Tuple[np.ndarray, Dict, List[int], List[int]]:
    """
    加载Shapley数据
    
    Args:
        shapley_dir: Shapley数据目录
        
    Returns:
        (shapley_matrix, stats, agent_ids, timesteps)
    """
    shapley_matrix_path = shapley_dir / "shapley_values.npy"
    shapley_stats_path = shapley_dir / "shapley_stats.json"
    
    if not shapley_matrix_path.exists():
        raise FileNotFoundError(f"Shapley matrix file not found: {shapley_matrix_path}")
    
    if not shapley_stats_path.exists():
        raise FileNotFoundError(f"Shapley stats file not found: {shapley_stats_path}")
    
    # 加载矩阵
    shapley_matrix = np.load(shapley_matrix_path)
    print(f"Loaded shapley matrix from: {shapley_matrix_path}")
    print(f"  Shape: {shapley_matrix.shape}")
    
    # 加载统计信息
    with open(shapley_stats_path, 'r') as f:
        stats = json.load(f)
    print(f"Loaded stats from: {shapley_stats_path}")
    
    # 从stats获取agent_ids和timesteps
    num_agents = stats.get('num_agents', shapley_matrix.shape[0])
    episode_length = stats.get('episode_length', shapley_matrix.shape[1])
    seed = stats.get('seed', None)
    
    agent_ids = list(range(num_agents))  # [0, 1, ..., 9]
    timesteps = list(range(1, episode_length + 1))  # [1, 2, ..., 50]
    
    print(f"  Agents: {num_agents}, Timesteps: {episode_length}, Seed: {seed}")
    
    return shapley_matrix, stats, agent_ids, timesteps


def load_actions_json(actions_json_path: Path) -> Dict[Tuple[int, int], int]:
    """
    从actions_json文件加载work状态
    
    Args:
        actions_json_path: all_actions.json文件路径
        
    Returns:
        {(agent_id, timestep): work_status} 字典，work_status为0或1
    """
    if not actions_json_path.exists():
        print(f"Warning: Actions JSON file not found: {actions_json_path}")
        return {}
    
    with open(actions_json_path, 'r') as f:
        all_actions = json.load(f)
    
    work_dict = {}
    
    # 遍历所有时间步
    for step_key, step_actions in all_actions.items():
        if not step_key.startswith('step_'):
            continue
        
        # 提取timestep（从"step_1"提取1）
        try:
            timestep = int(step_key.split('_')[1])
        except (IndexError, ValueError):
            continue
        
        # 遍历所有agent
        for agent_id_str, action in step_actions.items():
            if not isinstance(action, list) or len(action) < 1:
                continue
            
            try:
                agent_id = int(agent_id_str)
                work_status = int(action[0])  # work是第一个元素（0或1）
                work_dict[(agent_id, timestep)] = work_status
            except (ValueError, IndexError):
                continue
    
    print(f"Loaded work status for {len(work_dict)} (agent, timestep) pairs")
    return work_dict


def get_consumption_category(consumption_rate: float) -> int:
    """
    根据consumption rate确定consumption category (0-3)
    
    Args:
        consumption_rate: 消费比例（0-1之间）
        
    Returns:
        category索引 (0-3)
    """
    for min_val, max_val, category_idx in CONSUMPTION_CATEGORIES:
        if min_val <= consumption_rate < max_val or (category_idx == 3 and consumption_rate == 1.0):
            return category_idx
    # 默认返回最后一个category
    return CONSUMPTION_CATEGORIES[-1][2]


def classify_actions_by_work_consumption(action_table_path: Path,
                                         actions_json_path: Path,
                                         timesteps: List[int],
                                         agent_ids: List[int]) -> Dict[Tuple[int, int], int]:
    """
    按work状态和consumption rate分类动作到8个类型
    类型 = work_status * 4 + consumption_category
    - work=0: 类型0-3（不工作）
    - work=1: 类型4-7（工作）
    
    Args:
        action_table_path: action_table.csv文件路径
        actions_json_path: all_actions.json文件路径
        timesteps: timestep列表 [1, 2, ..., 50]
        agent_ids: agent ID列表 [0, 1, ..., 9]
        
    Returns:
        {(agent_id, timestep): action_category} 字典，category为0-7
    """
    # 1. 加载work状态
    work_dict = load_actions_json(actions_json_path)
    
    # 2. 加载consumption rate
    if not action_table_path.exists():
        print(f"Warning: Action table file not found: {action_table_path}")
        return {}
    
    try:
        action_table_df = pd.read_csv(action_table_path)
    except Exception as e:
        print(f"Warning: Failed to load action table: {e}")
        return {}
    
    # 检查必需的列
    required_columns = ['agent_id', 'timestep', 'endogenous_Consumption Rate']
    missing_columns = [col for col in required_columns if col not in action_table_df.columns]
    if missing_columns:
        print(f"Warning: Missing columns in action table: {missing_columns}")
        return {}
    
    # 3. 分类动作
    actions_dict = {}
    
    for _, row in action_table_df.iterrows():
        agent_id = int(row['agent_id'])
        timestep = int(row['timestep'])
        consumption_rate = float(row['endogenous_Consumption Rate'])
        
        # 检查agent_id和timestep是否在有效范围内
        if agent_id not in agent_ids or timestep not in timesteps:
            continue
        
        # 获取work状态
        work_status = work_dict.get((agent_id, timestep), None)
        if work_status is None:
            # 如果没有work状态，跳过
            continue
        
        # 计算consumption category (0-3)
        consumption_category = get_consumption_category(consumption_rate)
        
        # 计算最终类型: category = work_status * 4 + consumption_category (0-7)
        action_category = work_status * 4 + consumption_category
        
        actions_dict[(agent_id, timestep)] = action_category
    
    print(f"Classified actions for {len(actions_dict)} (agent, timestep) pairs")
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


def calculate_action_sparsity(shapley_matrix: np.ndarray, timesteps: List[int],
                              agent_ids: List[int],
                              actions_dict: Dict[Tuple[int, int], int]) -> Tuple[float, Dict[int, float]]:
    """
    指标5：行为风险集中度 (G_be)
    
    基于work+consumption类型的Shapley值的基尼系数（K=8）
    
    Args:
        shapley_matrix: Shapley矩阵 (N×T)
        timesteps: timestep列表
        agent_ids: agent ID列表
        actions_dict: {(agent_id, timestep): action_category} 字典
        
    Returns:
        (sparsity, category_shapley_dict) 元组
    """
    # 创建agent_id和timestep到索引的映射
    agent_id_to_idx = {aid: idx for idx, aid in enumerate(agent_ids)}
    timestep_to_idx = {t: idx for idx, t in enumerate(timesteps)}
    
    # 创建8个类型的Shapley值字典
    category_shapley = {i: 0.0 for i in range(8)}
    
    matched_count = 0
    for (agent_id, timestep), category in actions_dict.items():
        # 获取agent和timestep的索引
        if agent_id not in agent_id_to_idx or timestep not in timestep_to_idx:
            continue
        
        agent_idx = agent_id_to_idx[agent_id]
        timestep_idx = timestep_to_idx[timestep]
        
        # 获取该agent该timestep的Shapley值
        shapley_value = shapley_matrix[agent_idx, timestep_idx]
        
        # 累加到对应类型
        if 0 <= category < 8:
            category_shapley[category] += shapley_value
            matched_count += 1
    
    if matched_count == 0:
        print(f"  Warning: No matched (agent, timestep) pairs for action sparsity calculation")
    
    # 转换为8维数组
    category_shapley_array = np.array([category_shapley[i] for i in range(8)], dtype=np.float64)
    
    # 计算基尼系数
    sparsity = calculate_gini_coefficient(category_shapley_array)
    
    return sparsity, category_shapley


def calculate_all_risk_metrics(shapley_matrix: np.ndarray, timesteps: List[int], 
                               agent_ids: List[int],
                               actions_dict: Dict[Tuple[int, int], int]) -> Dict:
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
    metrics['G_be'], category_shapley = calculate_action_sparsity(
        shapley_matrix, timesteps, agent_ids, actions_dict
    )
    metrics['category_shapley'] = category_shapley
    
    return metrics


def save_attribution_data(shapley_matrix: np.ndarray, timesteps: List[int],
                          agent_ids: List[int],
                          category_shapley: Dict[int, float],
                          output_dir: Path):
    """
    保存5个npy文件到risk_features_exp目录
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_agents, num_timesteps = shapley_matrix.shape
    
    # 1. risk_evolution.npy: T维向量（每个时间步的风险贡献，沿着agent聚合）
    risk_evolution = np.sum(shapley_matrix, axis=0)  # shape: (num_timesteps,)
    np.save(output_dir / "risk_evolution.npy", risk_evolution)
    
    # 2. behaviour_aggregated.npy: 8维向量（沿着work+consumption类型聚合）
    behaviour_aggregated = np.array([category_shapley.get(i, 0.0) for i in range(8)], 
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


def save_risk_features_json(metrics: Dict, stats: Dict, timesteps: List[int],
                            agent_ids: List[int], output_path: Path):
    """
    保存风险特征指标JSON文件
    """
    # 构建输出字典（按顺序：L_tm, G_ag, C_ag, Z_ag, G_be）
    output_data = {
        "L_tm": metrics.get('L_tm'),
        "G_ag": metrics.get('G_ag'),
        "C_ag": metrics.get('C_ag'),
        "Z_ag": metrics.get('Z_ag'),
        "G_be": metrics.get('G_be'),
        "metadata": {
            "num_agents": len(agent_ids) if agent_ids else 0,
            "num_timesteps": len(timesteps) if timesteps else 0,
            "episode_length": stats.get('episode_length', 0),
            "seed": stats.get('seed', None),
            "metric_name": stats.get('metric_name', 'unknown'),
            "baseline_risk": stats.get('baseline_risk', 0.0),
            "real_risk": stats.get('real_risk', 0.0)
        }
    }
    
    # 保存JSON文件
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Risk features JSON saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='EconAgent 风险特征计算'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='数据目录（包含shapley、action_table、actions_json文件夹的目录）'
    )
    
    parser.add_argument(
        '--shapley_dir',
        type=str,
        default=None,
        help='Shapley数据目录（默认：{data_dir}/shapley）'
    )
    
    parser.add_argument(
        '--action_table_dir',
        type=str,
        default=None,
        help='Action table目录（默认：{data_dir}/action_table）'
    )
    
    parser.add_argument(
        '--actions_json_dir',
        type=str,
        default=None,
        help='Actions JSON目录（默认：{data_dir}/actions_json）'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='输出目录（默认：{data_dir}/risk_features_exp）'
    )
    
    args = parser.parse_args()
    
    # 解析路径
    data_dir = Path(args.data_dir)
    
    if args.shapley_dir:
        shapley_dir = Path(args.shapley_dir)
    else:
        shapley_dir = data_dir / "shapley"
    
    if args.action_table_dir:
        action_table_dir = Path(args.action_table_dir)
    else:
        action_table_dir = data_dir / "action_table"
    
    if args.actions_json_dir:
        actions_json_dir = Path(args.actions_json_dir)
    else:
        actions_json_dir = data_dir / "actions_json"
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_dir / "risk_features_exp"
    
    print("=" * 60)
    print("EconAgent 风险特征计算")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Shapley directory: {shapley_dir}")
    print(f"Action table directory: {action_table_dir}")
    print(f"Actions JSON directory: {actions_json_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60 + "\n")
    
    # 1. 加载Shapley数据
    print("Step 1: Loading Shapley data...")
    shapley_matrix, stats, agent_ids, timesteps = load_shapley_data(shapley_dir)
    
    print(f"  Matrix shape: {shapley_matrix.shape}")
    print(f"  Agent IDs: {len(agent_ids)}")
    print(f"  Timesteps: {len(timesteps)}")
    
    # 验证数据一致性
    if len(timesteps) != shapley_matrix.shape[1]:
        print(f"  Warning: Timestep count ({len(timesteps)}) doesn't match matrix columns ({shapley_matrix.shape[1]})")
        if len(timesteps) > shapley_matrix.shape[1]:
            timesteps = timesteps[:shapley_matrix.shape[1]]
        elif len(timesteps) < shapley_matrix.shape[1]:
            timesteps = list(range(1, shapley_matrix.shape[1] + 1))
    
    if len(agent_ids) != shapley_matrix.shape[0]:
        print(f"  Warning: Agent count ({len(agent_ids)}) doesn't match matrix rows ({shapley_matrix.shape[0]})")
        agent_ids = list(range(shapley_matrix.shape[0]))
    
    print()
    
    # 2. 分类动作
    print("Step 2: Classifying actions by work and consumption...")
    action_table_path = action_table_dir / "action_table.csv"
    actions_json_path = actions_json_dir / "all_actions.json"
    
    actions_dict = classify_actions_by_work_consumption(
        action_table_path, actions_json_path, timesteps, agent_ids
    )
    print()
    
    # 3. 计算风险特征指标
    print("Step 3: Calculating risk metrics...")
    metrics = calculate_all_risk_metrics(shapley_matrix, timesteps, agent_ids, actions_dict)
    print(f"  L_tm (Relative risk latency): {metrics['L_tm']:.6f}")
    print(f"  G_ag (Agent risk concentration): {metrics['G_ag']:.6f}")
    print(f"  C_ag (Risk-instability correlation): {metrics['C_ag']:.6f}")
    print(f"  Z_ag (Agent risk synchronization): {metrics['Z_ag']:.6f}")
    print(f"  G_be (Behavioral risk concentration): {metrics['G_be']:.6f}")
    print()
    
    # 4. 保存数据
    print("Step 4: Saving data files...")
    
    # 4.1 保存5个npy文件
    save_attribution_data(
        shapley_matrix, timesteps, agent_ids,
        metrics.get('category_shapley', {}),
        output_dir
    )
    
    # 4.2 保存风险特征指标JSON文件
    risk_features_json_path = output_dir / "risk_features.json"
    save_risk_features_json(metrics, stats, timesteps, agent_ids, risk_features_json_path)
    
    print()
    print("=" * 60)
    print("All tasks completed!")
    print("=" * 60)
    print(f"Risk features saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
