"""
主运行脚本
集成所有模块，运行社交媒体模拟
"""
import os
import argparse
import yaml
import json
from typing import Dict, Any
from model import SocialLLMModel
from counterfactual import run_counterfactual
from utils import calculate_polarization_risk, save_actions, save_random_states


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def run_simulation(config: Dict[str, Any], output_dir: str = "results"):
    """
    运行完整模拟
    
    Args:
        config: 配置字典
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取配置参数
    sim_config = config["simulation"]
    indicators_config = config["indicators"]
    llm_config = config["llm"]
    init_config = config["initialization"]
    
    # 创建模型
    model = SocialLLMModel(
        num_agents=sim_config["num_agents"],
        num_steps=sim_config["num_steps"],
        seed=sim_config.get("seed"),
        base_post_preference=indicators_config["base_post_preference"],
        base_interaction_preference=indicators_config["base_interaction_preference"],
        base_sensitivity=indicators_config["base_sensitivity"],
        alpha=indicators_config["alpha"],
        beta=indicators_config["beta"],
        gamma=indicators_config["gamma"],
        update_magnitude=indicators_config["update_magnitude"],
        reinforcement_coefficient=indicators_config["reinforcement_coefficient"],
        no_post_multiplier=indicators_config["no_post_multiplier"],
        min_view_ratio=indicators_config.get("min_view_ratio", 0.3),
        max_view_ratio=indicators_config.get("max_view_ratio", 0.9),
        belief_range=tuple(init_config["belief_range"]),
        llm_config_path=llm_config["config_path"],
        llm_temperature=llm_config["temperature"],
        llm_max_tokens=llm_config["max_tokens"],
    )
    
    print(f"开始运行模拟: {sim_config['num_agents']} agents, {sim_config['num_steps']} steps")
    
    # 运行模拟
    model.run(show_progress=True)
    
    # 保存结果
    actions_filepath = os.path.join(output_dir, "actions.json")
    random_states_filepath = os.path.join(output_dir, "random_states.json")
    model.save(actions_filepath, random_states_filepath)
    
    # 获取每个时间步的belief和风险
    # belief_history 包含 num_steps + 1 个时间步（包括初始状态 timestep 0）
    # risk_history 也包含 num_steps + 1 个时间步（包括初始状态 timestep 0）
    final_beliefs = model.get_all_belief_values()
    
    # 找到最高风险值和对应的时间步
    max_risk = max(model.risk_history)
    max_risk_timestep = model.risk_history.index(max_risk)
    
    # 计算整体的平均发帖数和看帖数（不包括初始状态 timestep 0）
    total_posts = sum(model.post_count_history[1:])  # 跳过初始状态
    total_views = sum(model.view_count_history[1:])  # 跳过初始状态
    avg_posts_per_agent_per_step = total_posts / (sim_config["num_agents"] * sim_config["num_steps"]) if sim_config["num_steps"] > 0 else 0.0
    avg_views_per_agent_per_step = total_views / (sim_config["num_agents"] * sim_config["num_steps"]) if sim_config["num_steps"] > 0 else 0.0
    
    # 构建按时间步组织的结果列表
    timestep_results = []
    for timestep in range(len(model.risk_history)):
        # 计算该时间步的平均发帖数和看帖数
        posts_this_step = model.post_count_history[timestep]
        views_this_step = model.view_count_history[timestep]
        avg_posts_this_step = posts_this_step / sim_config["num_agents"] if sim_config["num_agents"] > 0 else 0.0
        avg_views_this_step = views_this_step / sim_config["num_agents"] if sim_config["num_agents"] > 0 else 0.0
        
        timestep_results.append({
            "timestep": timestep,
            "beliefs": model.belief_history[timestep],  # 该时间步所有agent的belief值列表
            "risk": model.risk_history[timestep],  # 该时间步的极化风险值
            "avg_posts": avg_posts_this_step,  # 该时间步平均每个agent的发帖数
            "avg_views": avg_views_this_step  # 该时间步平均每个agent的看帖数
        })
    
    # 保存完整结果（包括每个时间步的belief和风险）
    results = {
        "num_agents": sim_config["num_agents"],
        "num_steps": sim_config["num_steps"],
        "avg_posts_per_agent_per_step": avg_posts_per_agent_per_step,  # 整体平均每个agent每步发帖数
        "avg_views_per_agent_per_step": avg_views_per_agent_per_step,  # 整体平均每个agent每步看帖数
        "final_beliefs": final_beliefs,
        "initial_risk": model.risk_history[0],  # 初始风险值（timestep 0）
        "max_risk": max_risk,  # 最高风险值
        "max_risk_timestep": max_risk_timestep,  # 最高风险值对应的时间步
        "timestep_results": timestep_results  # 每个时间步的详细结果
    }
    results_filepath = os.path.join(output_dir, "results.json")
    with open(results_filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n模拟完成!")
    print(f"初始风险值: {model.risk_history[0]:.4f} (timestep 0)")
    print(f"最高风险值: {max_risk:.4f} (timestep {max_risk_timestep})")
    print(f"最终风险值: {model.risk_history[-1]:.4f} (timestep {len(model.risk_history)-1})")
    print(f"平均发帖数: {avg_posts_per_agent_per_step:.4f} 帖/agent/步")
    print(f"平均看帖数: {avg_views_per_agent_per_step:.4f} 帖/agent/步")
    print(f"结果已保存到: {output_dir}")
    print(f"  - actions.json: 动作历史")
    print(f"  - random_states.json: 随机状态")
    print(f"  - results.json: 完整结果（包含每个时间步的belief和风险）")
    
    return model, results


def run_counterfactual_simulation(
    config: Dict[str, Any],
    actions_filepath: str,
    random_states_filepath: str,
    masked_agent_id: int,
    masked_timestep: int,
    output_dir: str = "results/counterfactual",
):
    """
    运行反事实模拟
    
    Args:
        config: 配置字典
        actions_filepath: 原始动作历史文件路径
        random_states_filepath: 原始随机状态文件路径
        masked_agent_id: 被遮挡的agent ID
        masked_timestep: 被遮挡的时间步
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取配置参数
    sim_config = config["simulation"]
    indicators_config = config["indicators"]
    init_config = config["initialization"]
    
    config_params = {
        "seed": sim_config.get("seed"),
        "base_post_preference": indicators_config["base_post_preference"],
        "base_interaction_preference": indicators_config["base_interaction_preference"],
        "base_sensitivity": indicators_config["base_sensitivity"],
        "alpha": indicators_config["alpha"],
        "beta": indicators_config["beta"],
        "gamma": indicators_config["gamma"],
        "update_magnitude": indicators_config["update_magnitude"],
        "reinforcement_coefficient": indicators_config["reinforcement_coefficient"],
        "no_post_multiplier": indicators_config["no_post_multiplier"],
        "belief_range": init_config["belief_range"],
    }
    
    print(f"运行反事实模拟: 遮挡 Agent {masked_agent_id} 在时间步 {masked_timestep} 的动作")
    
    # 运行反事实模拟
    final_beliefs = run_counterfactual(
        actions_filepath=actions_filepath,
        random_states_filepath=random_states_filepath,
        masked_agent_id=masked_agent_id,
        masked_timestep=masked_timestep,
        num_agents=sim_config["num_agents"],
        num_steps=sim_config["num_steps"],
        config_params=config_params,
    )
    
    # 计算最终极化风险
    final_risk = calculate_polarization_risk(final_beliefs)
    
    # 保存结果
    results = {
        "final_beliefs": final_beliefs,
        "final_polarization_risk": final_risk,
        "masked_agent_id": masked_agent_id,
        "masked_timestep": masked_timestep,
    }
    results_filepath = os.path.join(output_dir, f"counterfactual_agent_{masked_agent_id}_step_{masked_timestep}.json")
    with open(results_filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"反事实模拟完成!")
    print(f"最终极化风险: {final_risk:.4f}")
    print(f"结果已保存到: {results_filepath}")
    
    return final_beliefs, final_risk


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SocialLLM 社交媒体模拟")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="输出目录",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["simulate", "counterfactual"],
        default="simulate",
        help="运行模式: simulate (正常模拟) 或 counterfactual (反事实模拟)",
    )
    parser.add_argument(
        "--actions_file",
        type=str,
        help="反事实模拟: 原始动作历史文件路径",
    )
    parser.add_argument(
        "--random_states_file",
        type=str,
        help="反事实模拟: 原始随机状态文件路径",
    )
    parser.add_argument(
        "--masked_agent",
        type=int,
        help="反事实模拟: 被遮挡的agent ID",
    )
    parser.add_argument(
        "--masked_timestep",
        type=int,
        help="反事实模拟: 被遮挡的时间步",
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    if args.mode == "simulate":
        # 正常运行模拟
        run_simulation(config, args.output)
    elif args.mode == "counterfactual":
        # 运行反事实模拟
        if not all([args.actions_file, args.random_states_file, args.masked_agent is not None, args.masked_timestep is not None]):
            print("错误: 反事实模拟需要提供 --actions_file, --random_states_file, --masked_agent, --masked_timestep")
            return
        
        run_counterfactual_simulation(
            config,
            args.actions_file,
            args.random_states_file,
            args.masked_agent,
            args.masked_timestep,
            args.output,
        )


if __name__ == "__main__":
    main()
