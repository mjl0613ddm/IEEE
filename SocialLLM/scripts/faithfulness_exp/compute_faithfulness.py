#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SocialLLM Faithfulness实验脚本

计算Deletion指标，验证不同归因方法的准确性。
通过移除分数最高的N个动作，计算风险下降比例。

使用方法:
    python scripts/faithfulness_exp/compute_faithfulness.py --result_dir results/gpt-4o-mini/gpt-4o-mini_42 --method shapley --metric_type deletion_top_3
"""

import os
import sys
import json
import argparse
import re
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils import calculate_polarization_risk, load_actions, clamp
from agents import SocialMediaAgent


def run_counterfactual_simulation_direct(
    initial_beliefs: List[float],
    active_actions: Set[Tuple[int, int]],  # 使用原始actions的集合（不是masked_actions）
    num_agents: int,
    num_steps: int,
    config_params: Dict,
    target_timestep: int,
    original_actions: Dict,  # 原始actions字典
    seed: int,
) -> float:
    """
    直接运行rule-based反事实模拟（类似EconAgent/TwinMarket的实现）
    
    Args:
        initial_beliefs: 初始belief值列表
        active_actions: 使用原始actions的集合 {(agent_id, timestep)}
        num_agents: Agent数量
        num_steps: 总时间步数
        config_params: 配置参数字典
        target_timestep: 目标时间步
        original_actions: 原始actions字典
        seed: 随机种子
        
    Returns:
        目标时间步的风险值
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    
    # 初始化agents
    agents = []
    for i in range(num_agents):
        initial_belief = initial_beliefs[i] if i < len(initial_beliefs) else 0.0
        agent = SocialMediaAgent(
            agent_id=i,
            initial_belief_value=initial_belief,
            base_post_preference=config_params.get("base_post_preference", 0.5),
            base_interaction_preference=config_params.get("base_interaction_preference", 0.4),
            base_sensitivity=config_params.get("base_sensitivity", 0.5),
            alpha=config_params.get("alpha", 0.5),
            beta=config_params.get("beta", 0.5),
            gamma=config_params.get("gamma", 0.3),
            update_magnitude=config_params.get("update_magnitude", 0.1),
            reinforcement_coefficient=config_params.get("reinforcement_coefficient", 0.5),
        )
        agents.append(agent)
    
    # 获取配置参数
    min_view_ratio = config_params.get("min_view_ratio", 0.3)
    max_view_ratio = config_params.get("max_view_ratio", 1.0)
    no_post_multiplier = config_params.get("no_post_multiplier", 1.2)
    
    # 运行模拟到target_timestep
    risk_history = []
    
    # 记录初始状态的风险
    initial_risk = calculate_polarization_risk(initial_beliefs)
    risk_history.append(initial_risk)
    
    # 运行到target_timestep（需要运行target_timestep+1步，因为risk_history[0]是初始状态）
    for timestep in range(target_timestep + 1):
        current_posts = []
        post_interactions = {}
        
        # 阶段1: 所有agent决定发帖/不发帖
        for agent in agents:
            action_key = (agent.agent_id, timestep)
            
            if action_key in active_actions and action_key in original_actions:
                # 使用原始action
                original_action = original_actions[action_key]
                post_value = original_action.get("post")
                should_post = (post_value is not None)
                post_belief = post_value if should_post else None
            else:
                # 被mask，不发帖
                should_post = False
                post_belief = None
            
            if should_post and post_belief is not None:
                current_posts.append({
                    "author_id": agent.agent_id,
                    "belief_value": post_belief,
                })
                agent.record_post(timestep, post_belief)
                post_interactions[agent.agent_id] = {
                    "views": 0,
                    "likes": 0,
                    "dislikes": 0,
                }
        
        # 如果所有agent都不发帖
        if len(current_posts) == 0:
            # 所有agent的发帖偏好乘以乘数
            for agent in agents:
                agent.post_preference = clamp(
                    agent.post_preference * no_post_multiplier,
                    0.0, 1.0
                )
            # 记录当前时间步的belief和风险
            current_beliefs = [agent.belief_value for agent in agents]
            current_risk = calculate_polarization_risk(current_beliefs)
            risk_history.append(current_risk)
            continue
        
        # 阶段2: 生成可看的帖子列表（移除被mask agent发的帖子）
        # 只有active_actions中的agent发的帖子才能被看到
        available_posts = [
            post for post in current_posts
            if (post["author_id"], timestep) in active_actions
        ]
        available_post_ids = [post["author_id"] for post in available_posts]
        
        # 阶段3: 每个agent看帖子并互动
        for agent in agents:
            action_key = (agent.agent_id, timestep)
            
            # 如果被mask，不看帖子、不互动
            if action_key not in active_actions:
                continue
            
            # 使用原始action的看帖子和互动决策
            if action_key in original_actions:
                original_action = original_actions[action_key]
                interactions = original_action.get("interactions", [])
                
                # 按照原始actions中的interactions顺序处理
                for interaction in interactions:
                    post_id = interaction.get("post_id")
                    action = interaction.get("action")  # 可能是"like", "dislike", 或None
                    
                    # 检查帖子是否存在且没有被mask（在available_post_ids中）
                    if post_id not in available_post_ids:
                        continue
                    
                    # 找到对应的帖子
                    post = None
                    for p in available_posts:
                        if p["author_id"] == post_id:
                            post = p
                            break
                    
                    if post is None:
                        continue
                    
                    # 获取帖子当前的互动情况
                    post_stats = post_interactions.get(post_id, {"views": 0, "likes": 0, "dislikes": 0})
                    
                    # 增加浏览量（即使action是None，也增加了浏览量）
                    post_interactions[post_id]["views"] = post_stats["views"] + 1
                    
                    # 使用原始的互动决策
                    if action == "like":
                        post_interactions[post_id]["likes"] += 1
                        # 根据互动更新信念值
                        agent.update_belief_from_interaction(post["belief_value"], "like")
                    elif action == "dislike":
                        post_interactions[post_id]["dislikes"] += 1
                        # 根据互动更新信念值
                        agent.update_belief_from_interaction(post["belief_value"], "dislike")
                    # 如果action是None，表示看了帖子但没有互动，不更新信念值
        
        # 阶段4: 根据自己发的帖子的反馈更新信念值
        for agent in agents:
            if agent.agent_id in post_interactions:
                stats = post_interactions[agent.agent_id]
                agent.update_post_feedback(
                    timestep,
                    stats["views"],
                    stats["likes"],
                    stats["dislikes"],
                )
                # 根据反馈更新信念值
                agent.update_belief_from_feedback(timestep)
        
        # 记录当前时间步的belief和风险
        current_beliefs = [agent.belief_value for agent in agents]
        current_risk = calculate_polarization_risk(current_beliefs)
        risk_history.append(current_risk)
    
    # 返回target_timestep的风险值（risk_history[target_timestep+1]）
    if target_timestep + 1 < len(risk_history):
        return risk_history[target_timestep + 1]
    else:
        return risk_history[-1] if risk_history else 0.0


def load_results(result_dir: Path) -> Dict:
    """加载results.json"""
    results_file = result_dir / "results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"results.json not found: {results_file}")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_score_data(result_dir: Path, method: str) -> pd.DataFrame:
    """
    加载指定方法的分数数据
    
    Args:
        result_dir: 结果目录路径
        method: 方法名称，可选 'shapley', 'random', 'loo', 'llm', 'mast'
    
    Returns:
        DataFrame with columns: agent_id, timestep, score
    """
    if method == 'shapley':
        score_file = result_dir / "shapley" / "shapley_attribution_timeseries.csv"
        if not score_file.exists():
            raise FileNotFoundError(f"Shapley CSV file not found: {score_file}")
        
        df = pd.read_csv(score_file)
        # 确保有score列
        if 'shapley_value' in df.columns:
            df['score'] = df['shapley_value']
        elif 'score' not in df.columns:
            raise ValueError(f"Shapley CSV missing score column: {score_file}")
    else:
        score_file = result_dir / "faithfulness_exp" / method / f"{method}_attribution_timeseries.csv"
        if not score_file.exists():
            raise FileNotFoundError(f"{method} CSV file not found: {score_file}. Please run compute_{method}_baseline.py first.")
        
        df = pd.read_csv(score_file)
        # 确保有score列
        score_col = f'{method}_value'
        if score_col in df.columns:
            df['score'] = df[score_col]
        elif 'score' not in df.columns:
            raise ValueError(f"{method} CSV missing score column: {score_file}")
    
    # 确保有agent_id和timestep列
    if 'agent_id' not in df.columns or 'timestep' not in df.columns:
        raise ValueError(f"CSV missing required columns (agent_id, timestep): {score_file}")
    
    return df[['agent_id', 'timestep', 'score']]


def sort_actions_by_scores(df: pd.DataFrame, method: str) -> List[Tuple[Tuple[int, int], float]]:
    """
    按分数排序所有动作对
    
    对于Shapley值：按数值从小到大排序（负数优先，只移除高风险动作）
    对于其他方法（LOO/LLM/MAST/Random）：按分数从高到低排序（正数优先，只移除高风险动作）
    
    Args:
        df: DataFrame with columns: agent_id, timestep, score
        method: 方法名称，'shapley'或其他
    
    Returns:
        排序后的列表，每个元素是((agent_id, timestep), score)
    """
    actions = []
    for _, row in df.iterrows():
        agent_id = int(row['agent_id'])
        timestep = int(row['timestep'])
        score = float(row['score'])
        actions.append(((agent_id, timestep), score))
    
    # 根据方法类型选择排序方式
    if method == 'shapley':
        # Shapley值：正数 = 高风险（使风险上升），负数 = 低风险（使风险下降）
        # 按数值从大到小排序（正数在前，负数在后），移除最大的正数（高风险action）
        actions.sort(key=lambda x: x[1], reverse=True)
    else:
        # 其他方法：正数/高分 = 高风险，按分数从高到低排序
        actions.sort(key=lambda x: x[1], reverse=True)
    
    return actions


def load_config_params(result_dir: Path) -> Dict:
    """加载配置参数"""
    # 尝试多个可能的配置文件路径
    possible_config_paths = [
        result_dir.parent.parent / "config" / "config.yaml",
        project_root / "config" / "config.yaml",
    ]
    
    config_params = {}
    
    for config_file in possible_config_paths:
        if config_file.exists():
            try:
                import yaml
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                sim_config = config.get("simulation", {})
                indicators_config = config.get("indicators", {})
                
                config_params = {
                    "seed": sim_config.get("seed", 42),
                    "base_post_preference": sim_config.get("base_post_preference", 0.5),
                    "base_interaction_preference": sim_config.get("base_interaction_preference", 0.4),
                    "base_sensitivity": sim_config.get("base_sensitivity", 0.5),
                    "alpha": indicators_config.get("alpha", 0.5),
                    "beta": indicators_config.get("beta", 0.5),
                    "gamma": indicators_config.get("gamma", 0.3),
                    "update_magnitude": indicators_config.get("update_magnitude", 0.1),
                    "reinforcement_coefficient": indicators_config.get("reinforcement_coefficient", 0.5),
                    "no_post_multiplier": indicators_config.get("no_post_multiplier", 1.2),
                    "min_view_ratio": sim_config.get("min_view_ratio", 0.3),
                    "max_view_ratio": sim_config.get("max_view_ratio", 1.0),
                    "belief_range": indicators_config.get("belief_range", [-0.5, 0.5]),
                }
                break
            except Exception as e:
                print(f"警告: 无法加载配置文件 {config_file}: {e}")
                continue
    
    # 如果没找到配置文件，使用默认值
    if not config_params:
        config_params = {
            "seed": 42,
            "base_post_preference": 0.5,
            "base_interaction_preference": 0.4,
            "base_sensitivity": 0.5,
            "alpha": 0.5,
            "beta": 0.5,
            "gamma": 0.3,
            "update_magnitude": 0.1,
            "reinforcement_coefficient": 0.5,
            "no_post_multiplier": 1.2,
            "min_view_ratio": 0.3,
            "max_view_ratio": 1.0,
            "belief_range": (-0.5, 0.5),
        }
    
    return config_params


def compute_deletion_top_n(
    sorted_actions: List[Tuple[Tuple[int, int], float]],
    result_dir: Path,
    results_data: Dict,
    n: int,
    max_risk: float,
    max_risk_timestep: int,
    config_params: Dict,
    verbose: bool = False
) -> Tuple[float, float, List[Tuple[int, int]]]:
    """
    计算Deletion指标：移除分数最高的n个动作后，风险下降比例
    
    Args:
        sorted_actions: 排序后的(action, score)列表
        result_dir: 结果目录路径
        results_data: results.json的内容
        n: 要移除的动作数量
        max_risk: 原始最高风险值
        max_risk_timestep: 最高风险时间步
        config_params: 配置参数字典
        verbose: 是否输出详细信息
    
    Returns:
        (risk_after_removal, risk_decrease_relative, removed_actions_list)
    """
    print(f"计算Deletion Top {n}...")
    
    num_agents = results_data.get('num_agents', 20)
    num_steps = results_data.get('num_steps', 30)
    
    # 获取初始beliefs
    timestep_results = results_data.get('timestep_results', [])
    if not timestep_results or len(timestep_results) == 0:
        raise ValueError("results.json中缺少timestep_results")
    
    first_timestep = timestep_results[0]
    if first_timestep.get('timestep') != 0:
        raise ValueError("timestep_results的第一个元素不是timestep 0")
    
    initial_beliefs = first_timestep.get('beliefs', [])
    if not initial_beliefs:
        raise ValueError("timestep_results[0]中缺少beliefs")
    
    if len(initial_beliefs) != num_agents:
        raise ValueError(f"初始belief值数量 ({len(initial_beliefs)}) 与agent数量 ({num_agents}) 不匹配")
    
    # 加载原始actions
    actions_file = result_dir / "actions.json"
    if not actions_file.exists():
        raise FileNotFoundError(f"actions.json not found: {actions_file}")
    original_actions = load_actions(str(actions_file))
    
    # 创建所有actions的集合
    all_actions = set()
    for agent_id in range(num_agents):
        for timestep in range(max_risk_timestep + 1):
            all_actions.add((agent_id, timestep))
    
    # 获取要移除的n个最高分动作
    actions_to_remove = sorted_actions[:n]
    removed_actions_set = {action for action, score in actions_to_remove}
    removed_actions_list = [list(action) for action, score in actions_to_remove]
    
    print(f"  移除前动作数: {len(sorted_actions)}")
    print(f"  移除了 {len(removed_actions_set)} 个动作")
    
    if verbose:
        print(f"  被移除的动作: {removed_actions_list}")
    
    # 计算 active_actions（所有actions减去被移除的actions）
    active_actions = all_actions - removed_actions_set
    
    # 运行反事实模拟
    print(f"  运行反事实模拟（移除 {len(removed_actions_set)} 个动作）...")
    
    try:
        risk_after_removal = run_counterfactual_simulation_direct(
            initial_beliefs=initial_beliefs,
            active_actions=active_actions,
            num_agents=num_agents,
            num_steps=num_steps,
            config_params=config_params,
            target_timestep=max_risk_timestep,
            original_actions=original_actions,
            seed=config_params.get("seed", 42)
        )
    except Exception as e:
        print(f"  错误: 反事实模拟失败: {e}")
        raise
    
    # 计算相对风险下降
    if max_risk == 0:
        risk_decrease_relative = 0.0
        print(f"  警告: max_risk为0，无法计算相对风险下降")
    else:
        risk_decrease_relative = (max_risk - risk_after_removal) / max_risk
    
    print(f"  原始风险 (max_risk): {max_risk:.6f}")
    print(f"  移除后风险: {risk_after_removal:.6f}")
    print(f"  相对风险下降: {risk_decrease_relative:.6f} ({risk_decrease_relative * 100:.2f}%)")
    
    return risk_after_removal, risk_decrease_relative, removed_actions_list


def main():
    parser = argparse.ArgumentParser(
        description='计算SocialLLM Faithfulness指标',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--result_dir',
        type=str,
        required=True,
        help='结果目录路径（包含results.json）'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        required=True,
        choices=['shapley', 'random', 'loo', 'llm', 'mast'],
        help='归因方法'
    )
    
    parser.add_argument(
        '--metric_type',
        type=str,
        required=True,
        help='指标类型，支持格式: deletion_top_n (n为任意正整数，如 deletion_top_3, deletion_top_10)'
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
    
    # 解析metric_type
    metric_type = args.metric_type
    metric_match = re.match(r'^deletion_top_(\d+)$', metric_type)
    if not metric_match:
        print(f"错误: 不支持的指标类型: '{metric_type}'。支持的格式: deletion_top_n (n为任意正整数)", file=sys.stderr)
        sys.exit(1)
    
    n_value = int(metric_match.group(1))
    
    # 加载结果文件
    print("加载结果文件...")
    results_data = load_results(result_dir)
    
    num_agents = results_data.get("num_agents")
    num_steps = results_data.get("num_steps")
    max_risk_timestep = results_data.get("max_risk_timestep")
    max_risk = results_data.get("max_risk")
    initial_risk = results_data.get("initial_risk", 0.0)
    
    if max_risk_timestep is None:
        print("错误: results.json中缺少max_risk_timestep", file=sys.stderr)
        sys.exit(1)
    
    if max_risk is None:
        print("错误: results.json中缺少max_risk", file=sys.stderr)
        sys.exit(1)
    
    print(f"  Agent数量: {num_agents}")
    print(f"  模拟步数: {num_steps}")
    print(f"  最高风险时间步: {max_risk_timestep}")
    print(f"  最高风险值: {max_risk:.6f}")
    print(f"  初始风险值: {initial_risk:.6f}")
    
    # 加载分数数据
    print(f"\n加载{args.method}方法的分数数据...")
    df_scores = load_score_data(result_dir, args.method)
    print(f"  加载了 {len(df_scores)} 个(agent_id, timestep)对的分数")
    
    # 按分数排序
    print(f"\n按分数排序动作...")
    sorted_actions = sort_actions_by_scores(df_scores, args.method)
    if args.method == 'shapley':
        print(f"  排序完成（按绝对值），最大绝对值: {abs(sorted_actions[0][1]):.6f}, 最小绝对值: {abs(sorted_actions[-1][1]):.6f}")
    else:
        print(f"  排序完成，最高分: {sorted_actions[0][1]:.6f}, 最低分: {sorted_actions[-1][1]:.6f}")
    
    # 加载配置参数
    config_params = load_config_params(result_dir)
    
    # 计算deletion指标
    print(f"\n计算 {args.metric_type} 指标...")
    risk_after_removal, risk_decrease_relative, removed_actions_list = compute_deletion_top_n(
        sorted_actions=sorted_actions,
        result_dir=result_dir,
        results_data=results_data,
        n=n_value,
        max_risk=max_risk,
        max_risk_timestep=max_risk_timestep,
        config_params=config_params,
        verbose=args.verbose
    )
    
    # 保存结果
    output_dir = result_dir / "faithfulness_exp"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"faithfulness_results_{args.method}_{args.metric_type}.json"
    
    results = {
        "method": args.method,
        "metric_type": args.metric_type,
        "max_risk": float(max_risk),
        "initial_risk": float(initial_risk),
        "risk_after_removal": float(risk_after_removal),
        "risk_decrease_relative": float(risk_decrease_relative),
        "removed_actions": removed_actions_list,
        "removed_actions_count": len(removed_actions_list),
        "max_risk_timestep": int(max_risk_timestep),
        "num_agents": int(num_agents),
        "num_steps": int(num_steps)
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 结果已保存到: {output_file}")
    print(f"  风险下降比例: {risk_decrease_relative:.6f} ({risk_decrease_relative * 100:.2f}%)")


if __name__ == "__main__":
    main()
