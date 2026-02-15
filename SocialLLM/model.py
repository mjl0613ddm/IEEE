"""
Model类实现
管理整个社交媒体模拟流程
"""
import random
import math
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm
from agents import SocialMediaAgent
from llm_decision import LLMDecisionMaker
from utils import save_actions, load_actions, save_random_states, load_random_states, clamp, calculate_polarization_risk


class SocialLLMModel:
    """社交媒体LLM模拟模型"""
    
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
        min_view_ratio: float = 0.3,
        max_view_ratio: float = 0.9,
        belief_range: Tuple[float, float] = (-0.5, 0.5),
        llm_config_path: str = "config/api.yaml",
        llm_temperature: float = 0.7,
        llm_max_tokens: int = 500,
    ):
        """
        初始化模型
        
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
            min_view_ratio: 最少看帖比例（至少看多少比例的帖子）
            max_view_ratio: 最多看帖比例（最多看多少比例的帖子）
            belief_range: 初始信念值范围
            llm_config_path: LLM配置文件路径
            llm_temperature: LLM温度参数
            llm_max_tokens: LLM最大token数
        """
        self.num_agents = num_agents
        self.num_steps = num_steps
        self.seed = seed
        
        # 初始化随机数生成器
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
        self.min_view_ratio = min_view_ratio
        self.max_view_ratio = max_view_ratio
        
        # 创建LLM决策制定器
        self.llm_decision_maker = LLMDecisionMaker(
            config_path=llm_config_path,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
        )
        
        # 创建Agents
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
        
        # 当前时间步
        self.timestep = 0
        
        # 动作历史
        self.actions_history = {}  # {(agent_id, timestep): {...}}
        
        # 随机状态历史
        self.random_states = {}  # {(agent_id, timestep): {...}}
        
        # 当前时间步的帖子
        self.current_posts = []  # [{"author_id": int, "belief_value": float}]
        
        # 当前时间步的帖子互动统计
        self.post_interactions = {}  # {author_id: {"views": int, "likes": int, "dislikes": int}}
        
        # 每个时间步的belief和风险记录
        self.belief_history = []  # [List[float]] - 每个时间步所有agent的belief值
        self.risk_history = []  # [float] - 每个时间步的极化风险值
        # 每个时间步的发帖和看帖统计
        self.post_count_history = []  # [int] - 每个时间步的发帖数
        self.view_count_history = []  # [int] - 每个时间步的总看帖数
    
    def reset(self):
        """重置模型到初始状态"""
        for agent in self.agents:
            agent.reset_to_initial()
        self.timestep = 0
        self.actions_history = {}
        self.random_states = {}
        self.current_posts = []
        self.post_interactions = {}
        self.belief_history = []
        self.risk_history = []
        self.post_count_history = []
        self.view_count_history = []
    
    def step(self, show_progress: bool = False):
        """
        执行一个时间步
        
        Args:
            show_progress: 是否显示agent进度（用于内部调用）
        """
        self.current_posts = []
        self.post_interactions = {}
        
        # 阶段1: 所有agent决定发帖/不发帖
        agent_iterator = tqdm(self.agents, desc="发帖决策", leave=False, unit="agent") if show_progress else self.agents
        for agent in agent_iterator:
            should_post = self.llm_decision_maker.decide_post(
                agent.post_preference,
                agent.belief_value
            )
            
            post_belief = None
            if should_post:
                post_belief = agent.belief_value
                self.current_posts.append({
                    "author_id": agent.agent_id,
                    "belief_value": post_belief,
                })
                agent.record_post(self.timestep, post_belief)
                # 初始化帖子互动统计
                self.post_interactions[agent.agent_id] = {
                    "views": 0,
                    "likes": 0,
                    "dislikes": 0,
                }
            
            # 记录动作
            action_key = (agent.agent_id, self.timestep)
            self.actions_history[action_key] = {
                "post": post_belief,
                "interactions": [],
            }
        
        # 记录该时间步的发帖数
        num_posts_this_step = len(self.current_posts)
        
        # 如果所有agent都不发帖
        if len(self.current_posts) == 0:
            # 跳过看帖子步骤，所有agent的发帖偏好乘以乘数
            for agent in self.agents:
                agent.post_preference = clamp(
                    agent.post_preference * self.no_post_multiplier,
                    0.0, 1.0
                )
            # 记录该时间步的统计（发帖数0，看帖数0）
            self.post_count_history.append(0)
            self.view_count_history.append(0)
            # 记录当前时间步的belief和风险（没有人发帖时，belief不应该变化）
            current_beliefs = self.get_all_belief_values()
            current_risk = calculate_polarization_risk(current_beliefs)
            self.belief_history.append(current_beliefs.copy())
            self.risk_history.append(current_risk)
            self.timestep += 1
            return
        
        # 阶段2: 生成随机看帖子序列（保存随机状态）
        m = len(self.current_posts)
        available_post_ids = [post["author_id"] for post in self.current_posts]
        
        # 统计该时间步的总看帖数
        total_views_this_step = 0
        
        for agent in self.agents:
            # 使用更积极的看帖策略：至少看min_view_ratio比例的帖子，最多看max_view_ratio比例的帖子，向上取整
            if m == 0:
                num_posts_to_see = 0
            else:
                min_posts = max(1, math.ceil(m * self.min_view_ratio))  # 至少看1个或min_view_ratio比例的帖子（向上取整）
                max_posts = min(m, math.ceil(m * self.max_view_ratio))  # 最多看m个或max_view_ratio比例的帖子（向上取整）
                if min_posts <= max_posts:
                    num_posts_to_see = self.random.randint(min_posts, max_posts)
                else:
                    # 如果min_posts > max_posts（很少见），则看所有帖子
                    num_posts_to_see = m
            total_views_this_step += num_posts_to_see
            
            # 随机选择看哪些帖子
            if num_posts_to_see > 0:
                post_ids_to_see = self.random.sample(available_post_ids, num_posts_to_see)
            else:
                post_ids_to_see = []
            
            # 随机决定看的顺序
            view_order = post_ids_to_see.copy()
            self.random.shuffle(view_order)
            
            # 保存随机状态
            state_key = (agent.agent_id, self.timestep)
            self.random_states[state_key] = {
                "num_posts_to_see": num_posts_to_see,
                "post_ids_to_see": post_ids_to_see,
                "view_order": view_order,
            }
        
        # 阶段3: 每个agent看帖子并决定互动
        agent_iterator = tqdm(self.agents, desc="互动决策", leave=False, unit="agent") if show_progress else self.agents
        for agent in agent_iterator:
            state_key = (agent.agent_id, self.timestep)
            random_state = self.random_states[state_key]
            
            # 获取实际看到的帖子（根据保存的随机状态）
            post_ids_to_see = random_state["post_ids_to_see"]
            view_order = random_state["view_order"]
            
            # 过滤出实际存在的帖子
            actual_post_ids = [pid for pid in post_ids_to_see if pid in available_post_ids]
            
            # 按照view_order的顺序看帖子
            interactions = []
            for post_id in view_order:
                if post_id not in actual_post_ids:
                    continue  # 跳过不存在的帖子（反事实模拟中使用）
                
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
                
                # LLM决策互动
                action = self.llm_decision_maker.decide_interaction(
                    agent.interaction_preference,
                    agent.belief_value,
                    post["belief_value"],
                    post_stats,  # 使用更新前的统计
                )
                
                # 记录互动
                interactions.append({
                    "post_id": post_id,
                    "action": action,
                })
                
                # 更新帖子互动统计
                if action == "like":
                    self.post_interactions[post_id]["likes"] += 1
                elif action == "dislike":
                    self.post_interactions[post_id]["dislikes"] += 1
                
                # 根据互动更新信念值
                if action in ["like", "dislike"]:
                    agent.update_belief_from_interaction(post["belief_value"], action)
            
            # 更新动作历史
            action_key = (agent.agent_id, self.timestep)
            self.actions_history[action_key]["interactions"] = interactions
        
        # 阶段4: 根据自己发的帖子的反馈更新信念值
        for agent in self.agents:
            # 更新帖子反馈统计
            if agent.agent_id in self.post_interactions:
                stats = self.post_interactions[agent.agent_id]
                agent.update_post_feedback(
                    self.timestep,
                    stats["views"],
                    stats["likes"],
                    stats["dislikes"],
                )
                # 根据反馈更新信念值
                agent.update_belief_from_feedback(self.timestep)
        
        # 记录当前时间步的belief和风险
        current_beliefs = self.get_all_belief_values()
        current_risk = calculate_polarization_risk(current_beliefs)
        self.belief_history.append(current_beliefs.copy())
        self.risk_history.append(current_risk)
        
        # 记录当前时间步的发帖数和看帖数
        self.post_count_history.append(num_posts_this_step)
        self.view_count_history.append(total_views_this_step)
        
        self.timestep += 1
    
    def run(self, show_progress: bool = True):
        """
        运行完整模拟
        
        Args:
            show_progress: 是否显示进度条
        """
        # 记录初始状态的belief和风险
        initial_beliefs = self.get_all_belief_values()
        initial_risk = calculate_polarization_risk(initial_beliefs)
        self.belief_history.append(initial_beliefs.copy())
        self.risk_history.append(initial_risk)
        # 初始状态没有发帖和看帖
        self.post_count_history.append(0)
        self.view_count_history.append(0)
        
        if show_progress:
            # 使用tqdm显示进度条
            step_iterator = tqdm(range(self.num_steps), desc="模拟进度", unit="步")
        else:
            step_iterator = range(self.num_steps)
        
        for step in step_iterator:
            self.step()
            
            if show_progress:
                # 更新进度条信息
                current_risk = self.risk_history[-1] if self.risk_history else 0.0
                step_iterator.set_postfix({
                    "步数": f"{self.timestep}/{self.num_steps}",
                    "风险": f"{current_risk:.4f}"
                })
    
    def get_all_belief_values(self) -> List[float]:
        """获取所有agent的当前信念值"""
        return [agent.belief_value for agent in self.agents]
    
    def save(self, actions_filepath: str, random_states_filepath: str):
        """保存动作历史和随机状态"""
        save_actions(self.actions_history, actions_filepath)
        save_random_states(self.random_states, random_states_filepath)
    
    def load(self, actions_filepath: str, random_states_filepath: str):
        """加载动作历史和随机状态"""
        self.actions_history = load_actions(actions_filepath)
        self.random_states = load_random_states(random_states_filepath)
