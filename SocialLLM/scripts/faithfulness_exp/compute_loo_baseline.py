#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leave-One-Out (LOO) Baseline方法：通过移除每个action并计算风险下降值来评估action的重要性
用于faithfulness实验的baseline对比

使用方法:
    python scripts/faithfulness_exp/compute_loo_baseline.py --result_dir results/gpt-4o-mini/gpt-4o-mini_42
"""

import os
import sys
import json
import argparse
import random
import subprocess
import multiprocessing
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

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


def load_config_params(result_dir: Path) -> Dict:
    """加载配置参数（从config.yaml或使用默认值）"""
    # 尝试从shapley_stats中获取配置参数
    shapley_dir = result_dir / "shapley"
    shapley_stats = load_shapley_stats(shapley_dir)
    
    if shapley_stats and 'config_params' in shapley_stats:
        return shapley_stats['config_params']
    
    # 尝试从config.yaml加载
    config_files = [
        result_dir.parent.parent / "config" / "config.yaml",
        project_root / "config" / "config.yaml",
    ]
    
    config_params = None
    for config_file in config_files:
        if config_file.exists():
            try:
                import yaml
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    sim_config = config.get('simulation', {})
                    indicators_config = config.get('indicators', {})
                    init_config = config.get('initialization', {})
                    
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
                        "belief_range": tuple(init_config.get("belief_range", [-0.5, 0.5])),
                    }
                    break
            except Exception as e:
                continue
    
    # 如果所有配置文件都加载失败，使用默认值
    if config_params is None:
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


def get_num_threads(n_threads: int) -> int:
    """
    获取实际使用的线程数
    
    Args:
        n_threads: 用户指定的线程数（0表示自动检测）
    
    Returns:
        实际使用的线程数
    """
    if n_threads > 0:
        return n_threads
    
    # 自动检测CPU核心数
    try:
        # 方法1: 使用nproc命令
        try:
            result = subprocess.run(['nproc'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                detected_count = int(result.stdout.strip())
                if detected_count > 0:
                    return detected_count
        except Exception:
            pass
        
        # 方法2: 使用multiprocessing.cpu_count()
        detected_count = multiprocessing.cpu_count()
        if detected_count > 0:
            return detected_count
    except Exception:
        pass
    
    # 默认值
    return 1


def process_single_loo_action(
    args_tuple: Tuple[Tuple[int, int], List[float], List[Tuple[int, int]], int, int, Dict, int, Dict, float, int]
) -> Tuple[int, int, float, Optional[str]]:
    """
    处理单个LOO action的计算（用于并行执行）
    
    Args:
        args_tuple: (action, initial_beliefs, all_actions_list, num_agents, num_steps, config_params, max_risk_timestep, original_actions, real_risk, seed)
    
    Returns:
        (agent_id, timestep, loo_value, error_message)
    """
    try:
        (agent_id, timestep), initial_beliefs, all_actions_list, num_agents, num_steps, config_params, max_risk_timestep, original_actions, real_risk, seed = args_tuple
        
        # 将列表转换回集合
        all_actions = set(all_actions_list)
        
        # 计算 active_actions（所有actions减去当前action）
        active_actions = all_actions - {(agent_id, timestep)}
        
        # 运行counterfactual simulation
        risk_without = run_counterfactual_simulation_direct(
            initial_beliefs=initial_beliefs,
            active_actions=active_actions,
            num_agents=num_agents,
            num_steps=num_steps,
            config_params=config_params,
            target_timestep=max_risk_timestep,
            original_actions=original_actions,
            seed=seed
        )
        
        # 计算风险下降值（风险下降越多，说明该action越重要）
        risk_drop = real_risk - risk_without
        
        return (agent_id, timestep, risk_drop, None)
    except Exception as e:
        return (agent_id, timestep, 0.0, str(e))


def compute_loo_baseline(result_dir: Path, n_threads: int = 1) -> pd.DataFrame:
    """
    使用Leave-One-Out方法计算baseline分数
    
    Args:
        result_dir: 结果目录路径
    
    Returns:
        DataFrame with columns: agent_id, timestep, loo_value
    """
    results_file = result_dir / "results.json"
    
    # 加载数据
    results_data = load_results(results_file)
    
    # 获取参数
    num_agents = results_data.get('num_agents', 20)
    num_steps = results_data.get('num_steps', 30)
    max_risk_timestep = results_data.get('max_risk_timestep')
    if max_risk_timestep is None:
        max_risk_timestep = num_steps
    
    # 从timestep_results中提取初始beliefs（timestep 0的beliefs）
    initial_beliefs = results_data.get('initial_beliefs', [])
    if not initial_beliefs:
        # 如果没有initial_beliefs字段，从timestep_results中提取
        timestep_results = results_data.get('timestep_results', [])
        if timestep_results and len(timestep_results) > 0:
            # timestep 0的beliefs就是初始beliefs
            first_timestep = timestep_results[0]
            if first_timestep.get('timestep') == 0:
                initial_beliefs = first_timestep.get('beliefs', [])
        
        if not initial_beliefs:
            raise ValueError("results.json中缺少initial_beliefs字段，且无法从timestep_results中提取")
    
    # 加载配置参数
    config_params = load_config_params(result_dir)
    
    # 加载原始actions
    actions_file = result_dir / "actions.json"
    if not actions_file.exists():
        raise FileNotFoundError(f"actions.json not found: {actions_file}")
    original_actions = load_actions(str(actions_file))
    
    print(f"处理: {result_dir.name}")
    print("="*60)
    print(f"  Num agents: {num_agents}")
    print(f"  Max timestep: {max_risk_timestep}")
    print(f"  Num steps: {num_steps}")
    
    # 创建所有action的集合（修复范围：应该是max_risk_timestep + 1）
    all_actions = set()
    for agent_id in range(num_agents):
        for timestep in range(max_risk_timestep + 1):  # 修复：+1
            all_actions.add((agent_id, timestep))
    
    # 计算real_risk（全部真实动作）
    print("\n计算real_risk（全部真实动作）...")
    real_risk = run_counterfactual_simulation_direct(
        initial_beliefs=initial_beliefs,
        active_actions=all_actions,  # 所有actions
        num_agents=num_agents,
        num_steps=num_steps,
        config_params=config_params,
        target_timestep=max_risk_timestep,
        original_actions=original_actions,
        seed=config_params.get("seed", 42)
    )
    print(f"  Real risk: {real_risk:.6f}")
    
    print(f"\n计算LOO分数（Leave-One-Out）...")
    print(f"  需要运行 {len(all_actions)} 次模拟...")
    print(f"  并行线程数: {n_threads}")
    
    loo_scores = []
    risk_drops = []
    
    # 准备参数列表
    seed = config_params.get("seed", 42)
    all_actions_list = sorted(all_actions)
    
    if n_threads > 1:
        # 使用并行处理
        # 注意：all_actions是set，不能直接序列化，需要转换为list
        all_actions_list_for_parallel = list(all_actions)
        args_list = [
            (action, initial_beliefs, all_actions_list_for_parallel, num_agents, num_steps, config_params, max_risk_timestep, original_actions, real_risk, seed)
            for action in all_actions_list
        ]
        
        with ProcessPoolExecutor(max_workers=n_threads) as executor:
            futures = {executor.submit(process_single_loo_action, args): args[0] for args in args_list}
            
            with tqdm(total=len(all_actions_list), desc="LOO计算进度", unit="action") as pbar:
                for future in as_completed(futures):
                    agent_id, timestep, risk_drop, error = future.result()
                    if error:
                        print(f"警告: ({agent_id}, {timestep}) 计算失败: {error}", file=sys.stderr)
                    risk_drops.append(risk_drop)
                    loo_scores.append({
                        'agent_id': agent_id,
                        'timestep': timestep,
                        'loo_value': float(risk_drop)
                    })
                    pbar.update(1)
    else:
        # 串行处理（用于调试或单线程）
        for agent_id, timestep in tqdm(all_actions_list, desc="LOO计算进度"):
            # 计算 active_actions（所有actions减去当前action）
            active_actions = all_actions - {(agent_id, timestep)}
            
            # 运行counterfactual simulation
            risk_without = run_counterfactual_simulation_direct(
                initial_beliefs=initial_beliefs,
                active_actions=active_actions,
                num_agents=num_agents,
                num_steps=num_steps,
                config_params=config_params,
                target_timestep=max_risk_timestep,
                original_actions=original_actions,
                seed=seed
            )
            
            # 计算风险下降值（风险下降越多，说明该action越重要）
            risk_drop = real_risk - risk_without
            risk_drops.append(risk_drop)
            
            loo_scores.append({
                'agent_id': agent_id,
                'timestep': timestep,
                'loo_value': float(risk_drop)
            })
    
    # 转换为DataFrame
    df = pd.DataFrame(loo_scores)
    
    # 计算统计信息
    risk_drops_array = np.array(risk_drops)
    score_stats = {
        "mean": float(np.mean(risk_drops_array)),
        "std": float(np.std(risk_drops_array)),
        "min": float(np.min(risk_drops_array)),
        "max": float(np.max(risk_drops_array)),
        "sum": float(np.sum(risk_drops_array)),
        "median": float(np.median(risk_drops_array))
    }
    
    print(f"\nLOO分数统计:")
    print(f"    Mean: {score_stats['mean']:.6f}")
    print(f"    Std: {score_stats['std']:.6f}")
    print(f"    Min: {score_stats['min']:.6f}")
    print(f"    Max: {score_stats['max']:.6f}")
    print(f"    Median: {score_stats['median']:.6f}")
    
    return df, score_stats, real_risk


def main():
    parser = argparse.ArgumentParser(
        description='使用Leave-One-Out方法计算baseline分数',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--result_dir',
        type=str,
        required=True,
        help='结果目录路径（包含results.json）'
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='如果结果文件已存在，则跳过'
    )
    
    parser.add_argument(
        '--n-threads',
        type=int,
        default=0,
        help='并行线程数（0表示自动检测CPU核心数，1表示串行执行）'
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
    output_dir = result_dir / "faithfulness_exp" / "loo"
    output_file = output_dir / "loo_attribution_timeseries.csv"
    stats_file = output_dir / "loo_stats.json"
    
    if args.skip_existing and output_file.exists():
        print(f"跳过已存在的文件: {output_file}")
        return
    
    # 获取线程数
    n_threads = get_num_threads(args.n_threads)
    if args.verbose:
        print(f"使用线程数: {n_threads}")
    
    # 计算LOO baseline
    try:
        df, score_stats, real_risk = compute_loo_baseline(result_dir, n_threads=n_threads)
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存CSV文件
        df.to_csv(output_file, index=False)
        print(f"\n✓ CSV文件已保存到: {output_file}")
        
        # 保存统计信息
        results_data = load_results(result_dir / "results.json")
        stats = {
            "method": "loo",
            "num_agents": results_data.get('num_agents', 20),
            "max_risk_timestep": results_data.get('max_risk_timestep'),
            "max_risk": float(results_data.get('max_risk', 0.0)),
            "initial_risk": float(results_data.get('initial_risk', 0.0)),
            "real_risk": float(real_risk),
            "score_stats": score_stats
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 统计信息已保存到: {stats_file}")
    
    except Exception as e:
        print(f"错误: 计算LOO baseline失败: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
