#!/usr/bin/env python3
"""
SocialLLM Shapley Value 计算脚本

使用蒙特卡洛方法计算每个agent在每个时间步对最高风险点的贡献度。
Baseline策略：不行动（masked agent不发帖、不互动，且该agent发的帖子从其他agent的视野中移除）

实现方式：参考EconAgent和TwinMarket，为所有(agent_id, timestep)生成一个排列，
然后逐个添加，计算每个的边际贡献。所有(agent_id, timestep)共享同一个排列。
"""

import json
import os
import sys
import argparse
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils import calculate_polarization_risk, load_actions, clamp, calculate_indicators
from agents import SocialMediaAgent
import math


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='计算 SocialLLM 风险指标的 Shapley Value 归因分析'
    )
    
    parser.add_argument(
        '--result_dir',
        type=str,
        required=True,
        help='结果目录路径（包含results.json, actions.json, random_states.json）'
    )
    
    parser.add_argument(
        '--base_path',
        type=str,
        default=None,
        help='项目根目录路径（可选，默认：自动检测）'
    )
    
    parser.add_argument(
        '--n_samples',
        type=int,
        default=1000,
        help='蒙特卡洛采样次数（可选，默认：1000）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机数种子（可选，默认：42）'
    )
    
    parser.add_argument(
        '--n_threads',
        type=int,
        default=1,
        help='并行线程数（可选，默认：1，设为0则使用CPU核心数，建议设置为实际CPU核心数，如64）'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='输出详细信息（可选）'
    )
    
    return parser.parse_args()


def get_num_threads(n_threads: int) -> int:
    """获取实际使用的线程数"""
    if n_threads > 0:
        return n_threads
    
    # 检查环境变量（允许用户手动指定）
    import os
    env_threads = os.environ.get('SOCIALLLM_N_THREADS')
    if env_threads:
        try:
            env_count = int(env_threads)
            if env_count > 0:
                return env_count
        except:
            pass
    
    # 自动检测CPU核心数（使用多种方法）
    detected_count = None
    
    try:
        # 方法1: 使用nproc命令
        import subprocess
        result = subprocess.run(['nproc'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            detected_count = int(result.stdout.strip())
    except:
        pass
    
    if detected_count is None:
        # 方法2: 使用multiprocessing
        try:
            import multiprocessing
            detected_count = multiprocessing.cpu_count()
        except:
            pass
    
    if detected_count is None:
        detected_count = 1
    
    # 如果检测到的CPU数量很少（<4），给出警告
    if detected_count < 4:
        print(f"警告: 检测到的CPU核心数较少 ({detected_count})，可能被容器/环境限制。", file=sys.stderr)
        print(f"提示: 可以使用 --n_threads 参数手动指定，或设置环境变量 SOCIALLLM_N_THREADS", file=sys.stderr)
    
    return detected_count


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


def calculate_risk_for_subset(
    initial_beliefs: List[float],
    active_actions: Set[Tuple[int, int]],  # 改为active_actions而不是masked_actions
    num_agents: int,
    num_steps: int,
    config_params: Dict,
    target_timestep: int,
    seed: int,
    original_actions: Dict,  # 原始模拟的actions（必需）
) -> float:
    """
    计算给定子集的风险值（带缓存）
    
    Args:
        initial_beliefs: 初始belief值列表
        active_actions: 使用原始actions的集合 {(agent_id, timestep)}
        num_agents: agent数量
        num_steps: 总时间步数
        config_params: 配置参数字典
        target_timestep: 目标时间步
        seed: 随机种子
        original_actions: 原始模拟的actions（必需）
        
    Returns:
        目标时间步的风险值
    """
    try:
        return run_counterfactual_simulation_direct(
            initial_beliefs=initial_beliefs,
            active_actions=active_actions,
            num_agents=num_agents,
            num_steps=num_steps,
            config_params=config_params,
            target_timestep=target_timestep,
            original_actions=original_actions,
            seed=seed,
        )
    except Exception as e:
        print(f"警告: 计算风险失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 0.0


def process_single_sample(
    sample_idx: int,
    all_actions: List[Tuple[int, int]],
    initial_beliefs: List[float],
    baseline_risk: float,
    num_agents: int,
    num_steps: int,
    config_params: Dict,
    target_timestep: int,
    sample_seed: int,
    risk_cache: Dict,
    original_actions: Dict,  # 原始模拟的actions（必需）
    verbose: bool = False,
) -> Dict[Tuple[int, int], float]:
    """
    处理单个蒙特卡洛采样
    
    Args:
        sample_idx: 采样索引
        all_actions: 所有(agent_id, timestep)组合列表
        initial_beliefs: 初始belief值列表
        baseline_risk: baseline风险值
        num_agents: agent数量
        num_steps: 总时间步数
        config_params: 配置参数字典
        target_timestep: 目标时间步
        sample_seed: 该采样的随机种子
        risk_cache: 风险缓存字典
        verbose: 是否输出详细信息
        
    Returns:
        该采样中每个action的边际贡献字典
    """
    # 设置随机种子
    random.seed(sample_seed)
    np.random.seed(sample_seed)
    
    # 随机排列所有actions
    permuted_actions = all_actions.copy()
    random.shuffle(permuted_actions)
    
    # 逐个添加action，计算边际贡献
    current_subset = set()
    prev_risk = baseline_risk
    marginal_contribs = {}
    
    try:
        for action_idx, action in enumerate(permuted_actions):
            current_subset.add(action)
            subset_key = frozenset(current_subset)
            
            # 检查缓存
            if subset_key in risk_cache:
                current_risk = risk_cache[subset_key]
            else:
                # 计算当前子集的风险
                # current_subset是使用原始actions的集合
                current_risk = calculate_risk_for_subset(
                    initial_beliefs=initial_beliefs,
                    active_actions=current_subset,  # 改为active_actions
                    num_agents=num_agents,
                    num_steps=num_steps,
                    config_params=config_params,
                    target_timestep=target_timestep,
                    seed=sample_seed,
                    original_actions=original_actions,  # 传入原始actions
                )
                # 存入缓存
                risk_cache[subset_key] = current_risk
            
            # 计算边际贡献
            marginal_contrib = current_risk - prev_risk
            marginal_contribs[action] = marginal_contrib
            
            prev_risk = current_risk
            
    except Exception as e:
        # 即使非verbose模式，也输出错误信息（重要错误）
        print(f"错误: 采样 {sample_idx} 处理失败: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        # 返回空的边际贡献字典
        return {}
    
    return marginal_contribs


def calculate_shapley_values(
    all_actions: List[Tuple[int, int]],
    initial_beliefs: List[float],
    num_agents: int,
    num_steps: int,
    config_params: Dict,
    target_timestep: int,
    n_samples: int,
    seed: int,
    n_threads: int,
    original_actions: Dict,  # 原始模拟的actions（必需）
    max_risk: Optional[float] = None,  # 原始模拟的最高风险，用于作为real_risk
    verbose: bool = False,
) -> Tuple[Dict[Tuple[int, int], float], float, float]:
    """
    计算Shapley值（参考EconAgent/TwinMarket的实现方式）
    
    Args:
        all_actions: 所有(agent_id, timestep)组合列表
        initial_beliefs: 初始belief值列表
        num_agents: agent数量
        num_steps: 总时间步数
        config_params: 配置参数字典
        target_timestep: 目标时间步
        n_samples: 蒙特卡洛采样次数
        seed: 随机数种子
        n_threads: 并行线程数
        max_risk: 原始模拟的最高风险，用于作为real_risk（如果提供）
        original_actions: 原始模拟的actions（如果提供，将使用它们而不是重新生成）
        verbose: 是否输出详细信息
        
    Returns:
        (shapley_values字典, baseline_risk, real_risk)
    """
    # 计算baseline风险（所有actions都被mask，系统保持在初始状态）
    # 如果所有actions都被mask，所有人的belief都不会变，所以baseline_risk应该等于initial_risk
    # 直接使用timestep 0的风险值（initial_risk）
    baseline_risk = calculate_polarization_risk(initial_beliefs)
    if verbose:
        print(f"Baseline风险（使用initial_risk）: {baseline_risk:.6f}")
    
    # 计算真实风险（所有actions都不被mask）
    # 注意：real_risk应该等于原始模拟在max_risk_timestep的风险，也就是max_risk
    # 但是，由于随机种子的不同，重新计算可能得到不同的结果
    # 所以，如果提供了max_risk，直接使用它；否则重新计算
    if max_risk is not None:
        # 直接使用原始模拟的max_risk作为real_risk
        real_risk = max_risk
        if verbose:
            print(f"使用原始模拟的max_risk作为real_risk: {real_risk:.6f}")
    else:
        # 如果没有提供max_risk，重新计算
        if verbose:
            print("计算真实风险...")
        # 所有actions都使用原始值
        real_risk = calculate_risk_for_subset(
            initial_beliefs=initial_beliefs,
            active_actions=set(all_actions),  # 所有actions都使用原始值
            num_agents=num_agents,
            num_steps=num_steps,
            config_params=config_params,
            target_timestep=target_timestep,
            seed=seed,
            original_actions=original_actions,  # 传入原始actions
        )
        if verbose:
            print(f"计算的真实风险: {real_risk:.6f}")
    
    # 初始化Shapley值累加器
    shapley_values = defaultdict(float)
    
    # 初始化风险缓存（每个进程独立）
    risk_cache = {frozenset(): baseline_risk}
    
    # 为每个采样生成独立的随机种子
    sample_seeds = [seed + i for i in range(n_samples)]
    
    # 跟踪成功完成的采样数
    completed_samples = 0
    
    if n_threads > 1:
        # 多进程模式
        if verbose:
            print(f"使用 {n_threads} 个进程并行计算...")
        
        # 注意：多进程模式下，每个进程有独立的缓存
        # 将不可序列化的对象转换为可序列化的形式
        from multiprocessing import Manager
        manager = Manager()
        shared_risk_cache = manager.dict(risk_cache)
        
        with ProcessPoolExecutor(max_workers=n_threads) as executor:
            futures = {
                executor.submit(
                    process_single_sample,
                    sample_idx,
                    all_actions,
                    initial_beliefs,
                    baseline_risk,
                    num_agents,
                    num_steps,
                    config_params,
                    target_timestep,
                    sample_seeds[sample_idx],
                    {},  # 每个进程使用独立的缓存
                    original_actions,  # 传入原始actions
                    verbose,
                ): sample_idx
                for sample_idx in range(n_samples)
            }
            
            completed_samples = 0
            for future in tqdm(as_completed(futures), total=n_samples, desc="MC采样进度", file=sys.stderr, ncols=100):
                sample_idx = futures[future]
                try:
                    marginal_contribs = future.result()
                    
                    # 检查返回的数据
                    if not marginal_contribs:
                        if verbose:
                            print(f"警告: 采样 {sample_idx} 返回空字典", file=sys.stderr)
                        completed_samples += 1  # 即使返回空字典，也算作完成了一次采样
                        continue
                    
                    # 累加Shapley值
                    for action, contrib in marginal_contribs.items():
                        shapley_values[action] += contrib
                    
                    completed_samples += 1
                except Exception as e:
                    # 即使非verbose模式，也输出错误信息（重要错误）
                    print(f"错误: 采样 {sample_idx} 失败: {e}", file=sys.stderr)
                    if verbose:
                        import traceback
                        traceback.print_exc()
                    continue
    else:
        # 单线程模式
        if verbose:
            print("串行计算...")
        
        completed_samples = 0
        for sample_idx in tqdm(range(n_samples), desc="MC采样进度", file=sys.stderr, ncols=100):
            marginal_contribs = process_single_sample(
                sample_idx,
                all_actions,
                initial_beliefs,
                baseline_risk,
                num_agents,
                num_steps,
                config_params,
                target_timestep,
                sample_seeds[sample_idx],
                risk_cache,
                original_actions,  # 传入原始actions
                verbose,
            )
            
            # 检查返回的数据
            if not marginal_contribs:
                if verbose:
                    print(f"警告: 采样 {sample_idx} 返回空字典", file=sys.stderr)
                continue
            
            # 累加Shapley值
            for action, contrib in marginal_contribs.items():
                shapley_values[action] += contrib
    
    # 检查是否有成功的采样
    if not shapley_values:
        print("错误: 所有采样都失败了，没有计算出任何Shapley值", file=sys.stderr)
        if n_threads > 1:
            print(f"成功完成的采样数: {completed_samples}", file=sys.stderr)
        raise RuntimeError("所有采样都失败了，无法计算Shapley值")
    
    # 平均化（使用实际成功的采样数）
    successful_samples = completed_samples if n_threads > 1 else n_samples
    if successful_samples == 0:
        raise RuntimeError("没有成功完成任何采样，无法计算Shapley值")
    
    for action in shapley_values:
        shapley_values[action] /= successful_samples
    
    return shapley_values, baseline_risk, real_risk


def main():
    """主函数"""
    args = parse_arguments()
    
    # 确定项目根目录
    if args.base_path:
        project_root = Path(args.base_path)
    else:
        project_root = Path(__file__).parent.parent.parent
    
    result_dir = Path(args.result_dir)
    if not result_dir.is_absolute():
        result_dir = project_root / result_dir
    
    if not result_dir.exists():
        print(f"错误: 结果目录不存在: {result_dir}", file=sys.stderr)
        sys.exit(1)
    
    # 加载results.json
    results_file = result_dir / "results.json"
    if not results_file.exists():
        print(f"错误: results.json不存在: {results_file}", file=sys.stderr)
        sys.exit(1)
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
    
    num_agents = results_data.get("num_agents", 20)
    num_steps = results_data.get("num_steps", 30)
    max_risk = results_data.get("max_risk", 0.0)
    max_risk_timestep = results_data.get("max_risk_timestep", 0)
    initial_risk = results_data.get("initial_risk", 0.0)
    
    print(f"\n加载结果数据:")
    print(f"  Agents: {num_agents}")
    print(f"  Timesteps: {num_steps}")
    print(f"  最高风险: {max_risk:.6f} (timestep {max_risk_timestep})")
    print(f"  初始风险: {initial_risk:.6f}")
    
    # 加载初始belief值
    initial_beliefs = results_data.get("initial_beliefs", [])
    if not initial_beliefs:
        # 尝试从timestep_results中提取
        timestep_results = results_data.get("timestep_results", [])
        if timestep_results and len(timestep_results) > 0:
            initial_beliefs = timestep_results[0].get("beliefs", [])
    
    if not initial_beliefs or len(initial_beliefs) != num_agents:
        print(f"错误: 无法加载初始belief值", file=sys.stderr)
        sys.exit(1)
    
    print(f"  初始belief值已加载: {len(initial_beliefs)} 个agents")
    
    # 加载原始actions（用于反事实模拟，必需）
    actions_file = result_dir / "actions.json"
    if not actions_file.exists():
        print(f"错误: actions.json不存在: {actions_file}", file=sys.stderr)
        sys.exit(1)
    
    try:
        original_actions = load_actions(str(actions_file))
        print(f"  原始actions已加载: {len(original_actions)} 个actions")
    except Exception as e:
        print(f"错误: 加载actions.json失败: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 加载配置参数（从config.yaml）
    possible_config_paths = [
        result_dir.parent.parent / "config" / "config.yaml",
        project_root / "config" / "config.yaml",
    ]
    
    config_params = None
    for config_file in possible_config_paths:
        if config_file.exists():
            try:
                import yaml
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                indicators_config = config.get("indicators", {})
                init_config = config.get("initialization", {})
                
                config_params = {
                    "seed": args.seed,
                    "base_post_preference": indicators_config.get("base_post_preference", 0.5),
                    "base_interaction_preference": indicators_config.get("base_interaction_preference", 0.4),
                    "base_sensitivity": indicators_config.get("base_sensitivity", 0.5),
                    "alpha": indicators_config.get("alpha", 0.5),
                    "beta": indicators_config.get("beta", 0.5),
                    "gamma": indicators_config.get("gamma", 0.3),
                    "update_magnitude": indicators_config.get("update_magnitude", 0.1),
                    "reinforcement_coefficient": indicators_config.get("reinforcement_coefficient", 0.5),
                    "no_post_multiplier": indicators_config.get("no_post_multiplier", 1.2),
                    "min_view_ratio": config.get("simulation", {}).get("min_view_ratio", 0.3),
                    "max_view_ratio": config.get("simulation", {}).get("max_view_ratio", 1.0),
                    "belief_range": tuple(init_config.get("belief_range", [-0.5, 0.5])),
                }
                break
            except Exception as e:
                if args.verbose:
                    print(f"警告: 加载配置文件失败 {config_file}: {e}", file=sys.stderr)
                continue
    
    if config_params is None:
        print("警告: 配置文件不存在或加载失败，使用默认参数", file=sys.stderr)
        config_params = {
            "seed": args.seed,
            "base_post_preference": 0.5,
            "base_interaction_preference": 0.4,
            "base_sensitivity": 0.5,
            "alpha": 0.5,
            "beta": 0.5,
            "gamma": 0.3,
            "update_magnitude": 0.1,
            "reinforcement_coefficient": 0.5,
            "no_post_multiplier": 1.2,
            "belief_range": (-0.5, 0.5),
        }
    
    # 生成所有(agent_id, timestep)组合（只到target_timestep）
    # 注意：max_risk_timestep是0-indexed，所以如果max_risk_timestep=11，应该包含timestep 0-11
    # 但是，我们只计算到max_risk_timestep之前的actions（timestep 0到max_risk_timestep-1）
    # 因为target_timestep是max_risk_timestep，我们需要计算的是影响这个时间步风险的所有之前的actions
    all_actions = []
    for agent_id in range(num_agents):
        for timestep in range(max_risk_timestep):  # 只到max_risk_timestep-1，因为target_timestep是max_risk_timestep
            all_actions.append((agent_id, timestep))
    
    print(f"\n开始计算Shapley value...")
    print(f"  采样次数: {args.n_samples}")
    print(f"  计算范围: timestep 0 到 {max_risk_timestep - 1} (影响timestep {max_risk_timestep}的风险)")
    print(f"  总actions数: {len(all_actions)}")
    
    # 计算Shapley值
    n_threads = get_num_threads(args.n_threads)
    if n_threads > 1:
        print(f"  并行线程数: {n_threads}")
    
    shapley_values, baseline_risk, real_risk = calculate_shapley_values(
        all_actions=all_actions,
        initial_beliefs=initial_beliefs,
        num_agents=num_agents,
        num_steps=num_steps,
        config_params=config_params,
        target_timestep=max_risk_timestep,
        n_samples=args.n_samples,
        seed=args.seed,
        n_threads=n_threads,
        original_actions=original_actions,  # 传入原始actions（必需）
        max_risk=max_risk,  # 传入原始模拟的max_risk作为real_risk
        verbose=args.verbose,
    )
    
    # 创建DataFrame
    if not shapley_values:
        print("错误: 没有计算出任何Shapley值", file=sys.stderr)
        print(f"shapley_values类型: {type(shapley_values)}, 长度: {len(shapley_values) if hasattr(shapley_values, '__len__') else 'N/A'}", file=sys.stderr)
        sys.exit(1)
    
    rows = []
    for key, shapley_value in shapley_values.items():
        # 检查key的格式
        if isinstance(key, tuple) and len(key) == 2:
            agent_id, timestep = key
        else:
            print(f"警告: shapley_values的key格式不对: {key} (类型: {type(key)})", file=sys.stderr)
            continue
        
        rows.append({
            'agent_id': agent_id,
            'timestep': timestep,
            'shapley_value': shapley_value,
        })
    
    if not rows:
        print("错误: 没有生成任何数据行", file=sys.stderr)
        print(f"shapley_values的keys示例: {list(shapley_values.keys())[:5] if shapley_values else 'None'}", file=sys.stderr)
        sys.exit(1)
    
    df = pd.DataFrame(rows)
    
    # 检查DataFrame是否有必要的列
    if df.empty:
        print("错误: DataFrame为空", file=sys.stderr)
        sys.exit(1)
    
    if 'agent_id' not in df.columns or 'timestep' not in df.columns:
        print(f"错误: DataFrame缺少必要的列。实际列: {df.columns.tolist()}", file=sys.stderr)
        print(f"DataFrame内容:\n{df.head()}", file=sys.stderr)
        print(f"rows示例: {rows[:3] if rows else 'None'}", file=sys.stderr)
        sys.exit(1)
    
    df = df.sort_values(['agent_id', 'timestep'])
    
    # 验证Shapley值总和
    total_shapley = df['shapley_value'].sum()
    # 注意：Shapley值总和应该等于 real_risk - baseline_risk
    # 而不是 max_risk - initial_risk
    expected_total = real_risk - baseline_risk
    
    print(f"\nShapley值验证:")
    print(f"  实际总和: {total_shapley:.6f}")
    print(f"  期望总和 (real_risk - baseline_risk): {expected_total:.6f}")
    print(f"  baseline_risk: {baseline_risk:.6f}")
    print(f"  real_risk: {real_risk:.6f}")
    print(f"  差异: {abs(total_shapley - expected_total):.6f}")
    
    if abs(total_shapley - expected_total) > 0.01:
        print(f"  警告: Shapley值总和与期望值差异较大！", file=sys.stderr)
    
    # 保存结果
    output_dir = result_dir / "shapley"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存CSV
    csv_file = output_dir / "shapley_attribution_timeseries.csv"
    df.to_csv(csv_file, index=False)
    print(f"\n结果已保存到: {csv_file}")
    
    # 保存统计信息
    stats = {
        "method": "monte_carlo_shapley",
        "num_agents": num_agents,
        "num_steps": num_steps,
        "max_risk_timestep": max_risk_timestep,
        "max_risk": max_risk,
        "initial_risk": initial_risk,
        "baseline_risk": float(baseline_risk),  # 保存到顶层，方便绘图脚本读取
        "real_risk": float(real_risk),  # 保存到顶层，方便绘图脚本读取
        "n_samples": args.n_samples,
        "seed": args.seed,
        "n_threads": n_threads,
        "total_combinations": len(all_actions),
        "validation": {
            "total_shapley": float(total_shapley),
            "expected_total": float(expected_total),
            "difference": float(abs(total_shapley - expected_total)),
            "baseline_risk": float(baseline_risk),
            "real_risk": float(real_risk),
        },
        "shapley_stats": {
            "mean": float(df['shapley_value'].mean()),
            "std": float(df['shapley_value'].std()),
            "min": float(df['shapley_value'].min()),
            "max": float(df['shapley_value'].max()),
            "sum": float(total_shapley),
        },
    }
    
    stats_file = output_dir / "shapley_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"统计信息已保存到: {stats_file}")
    print(f"\n✅ Shapley Value计算完成！")


if __name__ == '__main__':
    main()
