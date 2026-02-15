"""
工具函数模块
包含极化风险计算、数据保存/加载等功能
"""
import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional


def clamp(value: float, min_val: float, max_val: float) -> float:
    """限制值在指定范围内"""
    return max(min_val, min(max_val, value))


def calculate_indicators(
    belief_value: float,
    base_post_preference: float,
    base_interaction_preference: float,
    base_sensitivity: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> Tuple[float, float, float]:
    """
    根据信念值计算三个指标
    
    Args:
        belief_value: 信念值 [-1, 1]
        base_post_preference: 基础发帖偏好
        base_interaction_preference: 基础互动偏好
        base_sensitivity: 基础敏感度
        alpha: 敏感度系数
        beta: 发帖偏好系数
        gamma: 互动偏好系数
    
    Returns:
        (post_preference, interaction_preference, belief_update_sensitivity)
    """
    abs_belief = abs(belief_value)
    
    # 信念更新敏感度 = base_sensitivity * (1 - α * |信念值|)
    belief_update_sensitivity = base_sensitivity * (1 - alpha * abs_belief)
    belief_update_sensitivity = clamp(belief_update_sensitivity, 0.0, 1.0)
    
    # 发帖偏好 = base_post_preference + β * |信念值|
    post_preference = base_post_preference + beta * abs_belief
    post_preference = clamp(post_preference, 0.0, 1.0)
    
    # 互动偏好 = base_interaction_preference + γ * |信念值|
    interaction_preference = base_interaction_preference + gamma * abs_belief
    interaction_preference = clamp(interaction_preference, 0.0, 1.0)
    
    return post_preference, interaction_preference, belief_update_sensitivity


def calculate_polarization_risk(belief_values: List[float]) -> float:
    """
    计算极化风险指标
    
    使用公式: R_polar(t) = (1/N) * sum((b_i^t - b_bar^t)^2)
    其中 b_bar^t 是所有agent在时刻t的信念值的平均值
    
    Args:
        belief_values: 所有agent的信念值列表
    
    Returns:
        极化风险值（方差）
    """
    if not belief_values:
        return 0.0
    
    beliefs = np.array(belief_values)
    N = len(beliefs)
    
    # 计算平均值 b_bar^t
    mean_belief = float(np.mean(beliefs))
    
    # 计算 R_polar(t) = (1/N) * sum((b_i^t - b_bar^t)^2)
    risk = float(np.mean((beliefs - mean_belief) ** 2))
    
    return risk


def save_actions(actions: Dict[Tuple[int, int], Dict[str, Any]], filepath: str):
    """
    保存动作历史到JSON文件
    
    Args:
        actions: 动作字典，格式为 {(agent_id, timestep): {...}}
        filepath: 保存路径
    """
    # 将tuple key转换为字符串
    actions_str = {f"({k[0]}, {k[1]})": v for k, v in actions.items()}
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(actions_str, f, indent=2, ensure_ascii=False)


def load_actions(filepath: str) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """
    从JSON文件加载动作历史
    
    Args:
        filepath: 文件路径
    
    Returns:
        动作字典，格式为 {(agent_id, timestep): {...}}
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        actions_str = json.load(f)
    
    # 将字符串key转换回tuple
    actions = {}
    for k, v in actions_str.items():
        # 解析 "(agent_id, timestep)" 格式
        k = k.strip('()')
        agent_id, timestep = map(int, k.split(','))
        actions[(agent_id, timestep)] = v
    
    return actions


def save_random_states(random_states: Dict[Tuple[int, int], Dict[str, Any]], filepath: str):
    """
    保存随机状态到JSON文件
    
    Args:
        random_states: 随机状态字典，格式为 {(agent_id, timestep): {...}}
        filepath: 保存路径
    """
    # 将tuple key转换为字符串
    states_str = {f"({k[0]}, {k[1]})": v for k, v in random_states.items()}
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(states_str, f, indent=2, ensure_ascii=False)


def load_random_states(filepath: str) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """
    从JSON文件加载随机状态
    
    Args:
        filepath: 文件路径
    
    Returns:
        随机状态字典，格式为 {(agent_id, timestep): {...}}
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        states_str = json.load(f)
    
    # 将字符串key转换回tuple
    random_states = {}
    for k, v in states_str.items():
        # 解析 "(agent_id, timestep)" 格式
        k = k.strip('()')
        agent_id, timestep = map(int, k.split(','))
        random_states[(agent_id, timestep)] = v
    
    return random_states
