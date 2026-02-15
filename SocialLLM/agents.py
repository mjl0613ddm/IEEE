"""
Agent类实现
管理每个社交媒体用户的指标和信念值
"""
from typing import Optional
from utils import calculate_indicators, clamp


class SocialMediaAgent:
    """社交媒体Agent"""
    
    def __init__(
        self,
        agent_id: int,
        initial_belief_value: float,
        base_post_preference: float,
        base_interaction_preference: float,
        base_sensitivity: float,
        alpha: float,
        beta: float,
        gamma: float,
        update_magnitude: float,
        reinforcement_coefficient: float,
    ):
        """
        初始化Agent
        
        Args:
            agent_id: Agent ID
            initial_belief_value: 初始信念值 [-1, 1]
            base_post_preference: 基础发帖偏好
            base_interaction_preference: 基础互动偏好
            base_sensitivity: 基础敏感度
            alpha: 敏感度系数
            beta: 发帖偏好系数
            gamma: 互动偏好系数
            update_magnitude: 信念更新幅度
            reinforcement_coefficient: 反馈强化系数
        """
        self.agent_id = agent_id
        self.belief_value = initial_belief_value
        self.initial_belief_value = initial_belief_value
        
        # 保存参数
        self.base_post_preference = base_post_preference
        self.base_interaction_preference = base_interaction_preference
        self.base_sensitivity = base_sensitivity
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.update_magnitude = update_magnitude
        self.reinforcement_coefficient = reinforcement_coefficient
        
        # 初始化指标
        self.initialize_indicators()
        
        # 记录自己发的帖子（用于反馈更新）
        self.my_posts = []  # 格式: [{"timestep": int, "belief_value": float, "views": int, "likes": int, "dislikes": int}]
    
    def initialize_indicators(self):
        """根据当前信念值初始化/更新指标"""
        self.post_preference, self.interaction_preference, self.belief_update_sensitivity = calculate_indicators(
            self.belief_value,
            self.base_post_preference,
            self.base_interaction_preference,
            self.base_sensitivity,
            self.alpha,
            self.beta,
            self.gamma,
        )
    
    def update_indicators_from_belief(self):
        """根据当前信念值更新指标（信念值变化后调用）"""
        self.initialize_indicators()
    
    def reset_to_initial(self):
        """重置到初始状态"""
        self.belief_value = self.initial_belief_value
        self.initialize_indicators()
        self.my_posts = []
    
    def update_belief_from_interaction(
        self,
        post_belief: float,
        action: str,  # "like" or "dislike"
    ):
        """
        根据互动（点赞/点踩）更新信念值
        
        Args:
            post_belief: 帖子的信念值
            action: 互动类型 ("like" 或 "dislike")
        """
        if action not in ["like", "dislike"]:
            return
        
        belief_diff = abs(post_belief - self.belief_value)
        
        if action == "like":
            # 点赞：向帖子方向移动
            if post_belief > self.belief_value:
                self.belief_value += self.belief_update_sensitivity * self.update_magnitude * belief_diff
            else:
                self.belief_value -= self.belief_update_sensitivity * self.update_magnitude * belief_diff
        elif action == "dislike":
            # 点踩：远离帖子方向移动
            if post_belief > self.belief_value:
                self.belief_value -= self.belief_update_sensitivity * self.update_magnitude * belief_diff
            else:
                self.belief_value += self.belief_update_sensitivity * self.update_magnitude * belief_diff
        
        # 限制在[-1, 1]范围内
        self.belief_value = clamp(self.belief_value, -1.0, 1.0)
        
        # 更新指标
        self.update_indicators_from_belief()
    
    def update_belief_from_feedback(
        self,
        timestep: int,
    ):
        """
        根据自己发的帖子的反馈更新信念值
        
        Args:
            timestep: 当前时间步
        """
        # 找到当前时间步的帖子
        my_post = None
        for post in self.my_posts:
            if post["timestep"] == timestep:
                my_post = post
                break
        
        if my_post is None:
            return
        
        views = my_post.get("views", 0)
        likes = my_post.get("likes", 0)
        dislikes = my_post.get("dislikes", 0)
        
        # 如果帖子没人看，不更新信念值
        if views == 0:
            return
        
        # 计算净点赞率
        net_like_rate = (likes - dislikes) / views
        
        # 更新信念值
        # 新信念值 = 原信念值 * (1 + 净点赞率 * 强化系数)
        old_belief = self.belief_value
        self.belief_value = old_belief * (1 + net_like_rate * self.reinforcement_coefficient)
        
        # 限制在[-1, 1]范围内
        self.belief_value = clamp(self.belief_value, -1.0, 1.0)
        
        # 更新指标
        self.update_indicators_from_belief()
    
    def record_post(self, timestep: int, belief_value: float):
        """记录自己发的帖子"""
        self.my_posts.append({
            "timestep": timestep,
            "belief_value": belief_value,
            "views": 0,
            "likes": 0,
            "dislikes": 0,
        })
    
    def update_post_feedback(self, timestep: int, views: int, likes: int, dislikes: int):
        """更新帖子的反馈统计"""
        for post in self.my_posts:
            if post["timestep"] == timestep:
                post["views"] = views
                post["likes"] = likes
                post["dislikes"] = dislikes
                break
