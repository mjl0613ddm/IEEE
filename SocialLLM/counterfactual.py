"""
反事实模拟模块
实现rule-based的反事实模拟（不使用LLM）
"""
import random
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from agents import SocialMediaAgent
from utils import load_actions, load_random_states, clamp


class CounterfactualModel:
    """反事实模拟模型（不使用LLM，rule-based）"""
    
    def __init__(
        self,
        num_agents: int,
        num_steps: int,
        seed: Optional[int] = None,
        base_post_preference: float = 0.3,
        base_interaction_preference: float = 0.4,
        base_sensitivity: float = 0.5,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.3,
        update_magnitude: float = 0.1,
        reinforcement_coefficient: float = 0.5,
        no_post_multiplier: float = 1.2,
        belief_range: Tuple[float, float] = (-0.5, 0.5),
        actions_history: Optional[Dict] = None,
        random_states: Optional[Dict] = None,
    ):
        """
        初始化反事实模型
        
        Args:
            num_agents: Agent数量
            num_steps: 模拟步数
            seed: 随机种子
            base_post_preference: 基础发帖偏好
            base_interaction_preference: 基础互动偏好
            base_sensitivity: 基础敏感度
            alpha: 敏感度系数
            beta: 发帖偏好系数
            gamma: 互动偏好系数
            update_magnitude: 信念更新幅度
            reinforcement_coefficient: 反馈强化系数
            no_post_multiplier: 没人发帖时的发帖偏好乘数
            belief_range: 初始信念值范围
            actions_history: 原始动作历史（如果提供，用于反事实模拟）
            random_states: 原始随机状态（如果提供，用于反事实模拟）
        """
        self.num_agents = num_agents
        self.num_steps = num_steps
        self.seed = seed
        
        # 初始化随机数生成器（用于反事实模拟中仍需要的随机性）
        self.random = random.Random(seed)
        np.random.seed(seed)
        
        # 保存参数
        self.base_post_preference = base_post_preference
        self.base_interaction_preference = base_interaction_preference
        self.base_sensitivity = base_sensitivity
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.update_magnitude = update_magnitude
        self.reinforcement_coefficient = reinforcement_coefficient
        self.no_post_multiplier = no_post_multiplier
        
        # 加载或创建agents（初始信念值需要从原始模拟中获取）
        # 这里我们假设初始信念值范围
        self.agents = []
        for i in range(num_agents):
            initial_belief = self.random.uniform(belief_range[0], belief_range[1])
            agent = SocialMediaAgent(
                agent_id=i,
                initial_belief_value=initial_belief,
                base_post_preference=base_post_preference,
                base_interaction_preference=base_interaction_preference,
                base_sensitivity=base_sensitivity,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                update_magnitude=update_magnitude,
                reinforcement_coefficient=reinforcement_coefficient,
            )
            self.agents.append(agent)
        
        # 原始动作历史和随机状态
        self.original_actions = actions_history if actions_history is not None else {}
        self.original_random_states = random_states if random_states is not None else {}
        
        # 被遮挡的动作（用于反事实模拟）
        self.masked_actions: Set[Tuple[int, int]] = set()  # {(agent_id, timestep)}
        
        self.timestep = 0
        self.current_posts = []
        self.post_interactions = {}
    
    def mask_action(self, agent_id: int, timestep: int):
        """
        遮挡某个agent在某个时间步的动作
        
        Args:
            agent_id: Agent ID
            timestep: 时间步
        """
        self.masked_actions.add((agent_id, timestep))
    
    def reset(self):
        """重置模型到初始状态"""
        for agent in self.agents:
            agent.reset_to_initial()
        self.timestep = 0
        self.current_posts = []
        self.post_interactions = {}
        self.masked_actions.clear()
    
    def rule_based_decide_post(self, agent: SocialMediaAgent) -> bool:
        """
        Rule-based发帖决策（基于发帖偏好）
        
        Args:
            agent: Agent
        
        Returns:
            True表示发帖，False表示不发帖
        """
        return self.random.random() < agent.post_preference
    
    def rule_based_decide_interaction(
        self,
        agent: SocialMediaAgent,
        post_belief: float,
    ) -> Optional[str]:
        """
        Rule-based互动决策
        
        Args:
            agent: Agent
            post_belief: 帖子信念值
        
        Returns:
            "like", "dislike", 或 None
        """
        # 先决定是否互动
        if self.random.random() >= agent.interaction_preference:
            return None
        
        # 决定点赞还是点踩（基于信念值差异）
        belief_diff = abs(agent.belief_value - post_belief)
        if belief_diff < 0.3:  # 信念相近，点赞
            return "like"
        else:  # 信念差异大，点踩
            return "dislike"
    
    def step(self):
        """执行一个时间步（反事实模拟）"""
        self.current_posts = []
        self.post_interactions = {}
        
        # 阶段1: 所有agent决定发帖/不发帖
        for agent in self.agents:
            action_key = (agent.agent_id, self.timestep)
            
            # 如果动作被遮挡，强制不发帖
            if action_key in self.masked_actions:
                should_post = False
                post_belief = None
            elif self.original_actions and action_key in self.original_actions:
                # 如果提供了原始actions，使用原始的发帖决策
                original_action = self.original_actions[action_key]
                post_value = original_action.get("post")
                should_post = (post_value is not None)
                post_belief = post_value if should_post else None
            else:
                # 如果没有原始actions，使用rule-based决策
                should_post = self.rule_based_decide_post(agent)
                post_belief = agent.belief_value if should_post else None
            
            if should_post and post_belief is not None:
                self.current_posts.append({
                    "author_id": agent.agent_id,
                    "belief_value": post_belief,
                })
                agent.record_post(self.timestep, post_belief)
                self.post_interactions[agent.agent_id] = {
                    "views": 0,
                    "likes": 0,
                    "dislikes": 0,
                }
        
        # 如果所有agent都不发帖
        if len(self.current_posts) == 0:
            # 跳过看帖子步骤，所有agent的发帖偏好乘以乘数
            for agent in self.agents:
                agent.post_preference = clamp(
                    agent.post_preference * self.no_post_multiplier,
                    0.0, 1.0
                )
            self.timestep += 1
            return
        
        # 阶段2: 生成随机看帖子序列（不依赖原始数据）
        available_post_ids = [post["author_id"] for post in self.current_posts]
        
        # 获取当前时间步被masked的agent集合（这些agent发的帖子需要被移除）
        masked_agents_this_step = set()
        for agent_id in range(self.num_agents):
            if (agent_id, self.timestep) in self.masked_actions:
                masked_agents_this_step.add(agent_id)
        
        # 从available_post_ids中移除被masked agent发的帖子
        available_post_ids = [pid for pid in available_post_ids if pid not in masked_agents_this_step]
        
        # 阶段3: 每个agent看帖子并决定互动
        for agent in self.agents:
            state_key = (agent.agent_id, self.timestep)
            
            # 如果动作被遮挡，跳过互动
            if state_key in self.masked_actions:
                continue
            
            # 如果提供了原始actions，使用原始的看帖子和互动决策
            if self.original_actions and state_key in self.original_actions:
                original_action = self.original_actions[state_key]
                interactions = original_action.get("interactions", [])
                
                # 按照原始actions中的interactions顺序处理
                for interaction in interactions:
                    post_id = interaction.get("post_id")
                    action = interaction.get("action")
                    
                    # 检查帖子是否存在且没有被mask
                    if post_id not in available_post_ids or post_id in masked_agents_this_step:
                        continue
                    
                    # 找到对应的帖子
                    post = None
                    for p in self.current_posts:
                        if p["author_id"] == post_id:
                            post = p
                            break
                    
                    if post is None:
                        continue
                    
                    # 获取帖子当前的互动情况
                    post_stats = self.post_interactions.get(post_id, {"views": 0, "likes": 0, "dislikes": 0})
                    
                    # 增加浏览量
                    self.post_interactions[post_id]["views"] = post_stats["views"] + 1
                    
                    # 使用原始的互动决策
                    if action == "like":
                        self.post_interactions[post_id]["likes"] += 1
                    elif action == "dislike":
                        self.post_interactions[post_id]["dislikes"] += 1
                    
                    # 根据互动更新信念值
                    if action in ["like", "dislike"]:
                        agent.update_belief_from_interaction(post["belief_value"], action)
            else:
                # 如果没有原始actions，使用rule-based决策
                # 使用与原始模拟相同的逻辑：看30%-100%的帖子
                min_view_ratio = 0.3
                max_view_ratio = 1.0
                num_posts = len(available_post_ids)
                if num_posts > 0:
                    min_posts = max(1, int(math.ceil(num_posts * min_view_ratio)))
                    max_posts = min(num_posts, int(math.ceil(num_posts * max_view_ratio)))
                    num_to_see = self.random.randint(min_posts, max_posts)
                    post_ids_to_see = self.random.sample(available_post_ids, num_to_see)
                    view_order = post_ids_to_see.copy()
                    self.random.shuffle(view_order)
                else:
                    post_ids_to_see = []
                    view_order = []
                
                # 过滤出实际存在的帖子，并移除被masked agent发的帖子
                actual_post_ids = [pid for pid in post_ids_to_see if pid in available_post_ids and pid not in masked_agents_this_step]
                
                # 按照view_order的顺序看帖子
                for post_id in view_order:
                    if post_id not in actual_post_ids:
                        continue  # 跳过不存在的帖子
                    
                    # 找到对应的帖子
                    post = None
                    for p in self.current_posts:
                        if p["author_id"] == post_id:
                            post = p
                            break
                    
                    if post is None:
                        continue
                    
                    # 获取帖子当前的互动情况
                    post_stats = self.post_interactions.get(post_id, {"views": 0, "likes": 0, "dislikes": 0})
                    
                    # 增加浏览量
                    self.post_interactions[post_id]["views"] = post_stats["views"] + 1
                    
                    # Rule-based决策互动
                    action = self.rule_based_decide_interaction(agent, post["belief_value"])
                    
                    # 更新帖子互动统计
                    if action == "like":
                        self.post_interactions[post_id]["likes"] += 1
                    elif action == "dislike":
                        self.post_interactions[post_id]["dislikes"] += 1
                    
                    # 根据互动更新信念值
                    if action in ["like", "dislike"]:
                        agent.update_belief_from_interaction(post["belief_value"], action)
        
        # 阶段4: 根据自己发的帖子的反馈更新信念值
        for agent in self.agents:
            if agent.agent_id in self.post_interactions:
                stats = self.post_interactions[agent.agent_id]
                agent.update_post_feedback(
                    self.timestep,
                    stats["views"],
                    stats["likes"],
                    stats["dislikes"],
                )
                agent.update_belief_from_feedback(self.timestep)
        
        self.timestep += 1
    
    def run(self):
        """运行完整反事实模拟"""
        for _ in range(self.num_steps):
            self.step()
    
    def get_all_belief_values(self) -> List[float]:
        """获取所有agent的当前信念值"""
        return [agent.belief_value for agent in self.agents]


def run_counterfactual(
    actions_filepath: str,
    random_states_filepath: str,
    masked_agent_id: int,
    masked_timestep: int,
    num_agents: int,
    num_steps: int,
    config_params: Dict,
) -> List[float]:
    """
    运行反事实模拟
    
    Args:
        actions_filepath: 原始动作历史文件路径
        random_states_filepath: 原始随机状态文件路径
        masked_agent_id: 被遮挡的agent ID
        masked_timestep: 被遮挡的时间步
        num_agents: Agent数量
        num_steps: 模拟步数
        config_params: 配置参数字典
    
    Returns:
        最终所有agent的信念值列表
    """
    # 加载原始数据
    original_actions = load_actions(actions_filepath)
    original_random_states = load_random_states(random_states_filepath)
    
    # 创建反事实模型
    counterfactual_model = CounterfactualModel(
        num_agents=num_agents,
        num_steps=num_steps,
        seed=config_params.get("seed"),
        base_post_preference=config_params.get("base_post_preference", 0.3),
        base_interaction_preference=config_params.get("base_interaction_preference", 0.4),
        base_sensitivity=config_params.get("base_sensitivity", 0.5),
        alpha=config_params.get("alpha", 0.5),
        beta=config_params.get("beta", 0.3),
        gamma=config_params.get("gamma", 0.3),
        update_magnitude=config_params.get("update_magnitude", 0.1),
        reinforcement_coefficient=config_params.get("reinforcement_coefficient", 0.5),
        no_post_multiplier=config_params.get("no_post_multiplier", 1.2),
        belief_range=tuple(config_params.get("belief_range", (-0.5, 0.5))),
        actions_history=original_actions,
        random_states=original_random_states,
    )
    
    # 遮挡指定动作
    counterfactual_model.mask_action(masked_agent_id, masked_timestep)
    
    # 运行模拟
    counterfactual_model.run()
    
    # 返回最终信念值
    return counterfactual_model.get_all_belief_values()


def run_counterfactual_with_masked_set(
    actions_filepath: str,
    random_states_filepath: str,
    masked_actions: Set[Tuple[int, int]],
    num_agents: int,
    num_steps: int,
    config_params: Dict,
    target_timestep: Optional[int] = None,
) -> Tuple[List[float], float]:
    """
    运行反事实模拟，支持mask多个agent在多个时间步的动作
    
    Args:
        actions_filepath: 原始动作历史文件路径
        random_states_filepath: 原始随机状态文件路径
        masked_actions: 被遮挡的动作集合 {(agent_id, timestep)}
        num_agents: Agent数量
        num_steps: 模拟步数
        config_params: 配置参数字典
        target_timestep: 目标时间步（如果指定，返回该时间步的风险值；否则返回最终信念值）
    
    Returns:
        (最终所有agent的信念值列表, 目标时间步的风险值)
    """
    from utils import calculate_polarization_risk
    
    # 加载原始数据
    original_actions = load_actions(actions_filepath)
    original_random_states = load_random_states(random_states_filepath)
    
    # 创建反事实模型
    counterfactual_model = CounterfactualModel(
        num_agents=num_agents,
        num_steps=num_steps,
        seed=config_params.get("seed"),
        base_post_preference=config_params.get("base_post_preference", 0.3),
        base_interaction_preference=config_params.get("base_interaction_preference", 0.4),
        base_sensitivity=config_params.get("base_sensitivity", 0.5),
        alpha=config_params.get("alpha", 0.5),
        beta=config_params.get("beta", 0.3),
        gamma=config_params.get("gamma", 0.3),
        update_magnitude=config_params.get("update_magnitude", 0.1),
        reinforcement_coefficient=config_params.get("reinforcement_coefficient", 0.5),
        no_post_multiplier=config_params.get("no_post_multiplier", 1.2),
        belief_range=tuple(config_params.get("belief_range", (-0.5, 0.5))),
        actions_history=original_actions,
        random_states=original_random_states,
    )
    
    # 遮挡所有指定的动作
    for agent_id, timestep in masked_actions:
        counterfactual_model.mask_action(agent_id, timestep)
    
    # 运行模拟并记录每个时间步的belief和风险
    belief_history = []
    risk_history = []
    
    # 记录初始状态
    initial_beliefs = counterfactual_model.get_all_belief_values()
    initial_risk = calculate_polarization_risk(initial_beliefs)
    belief_history.append(initial_beliefs.copy())
    risk_history.append(initial_risk)
    
    # 优化：如果指定了target_timestep，只需要运行到该时间步即可
    # 这样可以节省大量计算时间（特别是当target_timestep < num_steps时）
    # 注意：target_timestep是0-indexed，所以如果target_timestep=11，需要运行12步（包括初始状态）
    # risk_history[0]是初始状态，risk_history[1]是timestep 0之后，risk_history[target_timestep+1]是target_timestep之后
    steps_to_run = num_steps
    if target_timestep is not None and target_timestep < num_steps:
        # target_timestep是0-indexed，所以需要运行target_timestep+1步才能到达target_timestep
        # 例如：target_timestep=11，需要运行12步（timestep 0-11）
        steps_to_run = target_timestep + 1
    
    # 运行模拟
    for _ in range(steps_to_run):
        counterfactual_model.step()
        current_beliefs = counterfactual_model.get_all_belief_values()
        current_risk = calculate_polarization_risk(current_beliefs)
        belief_history.append(current_beliefs.copy())
        risk_history.append(current_risk)
    
    # 如果指定了目标时间步，返回该时间步的风险值
    # risk_history[0]是初始状态，risk_history[target_timestep+1]是target_timestep之后的状态
    if target_timestep is not None:
        target_index = target_timestep + 1
        if target_index < len(risk_history):
            target_risk = risk_history[target_index]
        else:
            target_risk = risk_history[-1] if risk_history else 0.0
    else:
        target_risk = risk_history[-1] if risk_history else 0.0
    
    return counterfactual_model.get_all_belief_values(), target_risk


def run_counterfactual_with_masked_set_optimized(
    initial_beliefs: List[float],
    masked_actions: Set[Tuple[int, int]],
    num_agents: int,
    num_steps: int,
    config_params: Dict,
    target_timestep: Optional[int] = None,
    original_actions: Optional[Dict] = None,  # 原始模拟的actions，如果提供则使用它们
) -> Tuple[List[float], float]:
    """
    运行反事实模拟，支持mask多个agent在多个时间步的动作（优化版本，直接运行模拟，不依赖原始数据）
    
    Args:
        initial_beliefs: 初始belief值列表（从results.json的timestep 0读取）
        masked_actions: 被遮挡的动作集合 {(agent_id, timestep)}
        num_agents: Agent数量
        num_steps: 模拟步数
        config_params: 配置参数字典
        target_timestep: 目标时间步（如果指定，返回该时间步的风险值；否则返回最终信念值）
    
    Returns:
        (最终所有agent的信念值列表, 目标时间步的风险值)
    """
    from utils import calculate_polarization_risk
    
    # 创建反事实模型（完全独立运行，不依赖原始数据）
    counterfactual_model = CounterfactualModel(
        num_agents=num_agents,
        num_steps=num_steps,
        seed=config_params.get("seed"),
        base_post_preference=config_params.get("base_post_preference", 0.3),
        base_interaction_preference=config_params.get("base_interaction_preference", 0.4),
        base_sensitivity=config_params.get("base_sensitivity", 0.5),
        alpha=config_params.get("alpha", 0.5),
        beta=config_params.get("beta", 0.3),
        gamma=config_params.get("gamma", 0.3),
        update_magnitude=config_params.get("update_magnitude", 0.1),
        reinforcement_coefficient=config_params.get("reinforcement_coefficient", 0.5),
        no_post_multiplier=config_params.get("no_post_multiplier", 1.2),
        belief_range=tuple(config_params.get("belief_range", (-0.5, 0.5))),
        actions_history=None,  # 不使用原始动作
        random_states=None,  # 不使用原始随机状态
    )
    
    # 设置初始belief值（从results.json读取）
    for i, agent in enumerate(counterfactual_model.agents):
        if i < len(initial_beliefs):
            agent.belief_value = initial_beliefs[i]
            agent.initial_belief_value = initial_beliefs[i]
    
    # 遮挡所有指定的动作
    for agent_id, timestep in masked_actions:
        counterfactual_model.mask_action(agent_id, timestep)
    
    # 运行模拟并记录每个时间步的belief和风险
    belief_history = []
    risk_history = []
    
    # 记录初始状态
    initial_beliefs = counterfactual_model.get_all_belief_values()
    initial_risk = calculate_polarization_risk(initial_beliefs)
    belief_history.append(initial_beliefs.copy())
    risk_history.append(initial_risk)
    
    # 优化：如果指定了target_timestep，只需要运行到该时间步即可
    # 这样可以节省大量计算时间（特别是当target_timestep < num_steps时）
    # 注意：target_timestep是0-indexed，所以如果target_timestep=11，需要运行12步（包括初始状态）
    # risk_history[0]是初始状态，risk_history[1]是timestep 0之后，risk_history[target_timestep+1]是target_timestep之后
    steps_to_run = num_steps
    if target_timestep is not None and target_timestep < num_steps:
        # target_timestep是0-indexed，所以需要运行target_timestep+1步才能到达target_timestep
        # 例如：target_timestep=11，需要运行12步（timestep 0-11）
        steps_to_run = target_timestep + 1
    
    # 运行模拟
    for _ in range(steps_to_run):
        counterfactual_model.step()
        current_beliefs = counterfactual_model.get_all_belief_values()
        current_risk = calculate_polarization_risk(current_beliefs)
        belief_history.append(current_beliefs.copy())
        risk_history.append(current_risk)
    
    # 如果指定了目标时间步，返回该时间步的风险值
    # risk_history[0]是初始状态，risk_history[target_timestep+1]是target_timestep之后的状态
    if target_timestep is not None:
        target_index = target_timestep + 1
        if target_index < len(risk_history):
            target_risk = risk_history[target_index]
        else:
            target_risk = risk_history[-1] if risk_history else 0.0
    else:
        target_risk = risk_history[-1] if risk_history else 0.0
    
    return counterfactual_model.get_all_belief_values(), target_risk
