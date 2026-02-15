"""
LLM决策模块
用于Agent的发帖和互动决策
"""
import json
import os
import sys
import random
import yaml
from typing import Optional, Dict, Any
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, wait_random


class LLMDecisionMaker:
    """LLM驱动的决策制定器"""
    
    def __init__(self, config_path: str = "config/api.yaml", temperature: float = 0.7, max_tokens: int = 500):
        """
        初始化LLM决策制定器
        
        Args:
            config_path: API配置文件路径
            temperature: LLM温度参数
            max_tokens: 最大token数
        """
        # 获取配置文件的绝对路径（基于 SocialLLM 目录）
        current_dir = os.path.dirname(os.path.abspath(__file__))  # SocialLLM 目录
        # 如果 config_path 是相对路径，则基于当前目录解析
        if not os.path.isabs(config_path):
            config_path = os.path.join(current_dir, config_path)
        
        # 加载配置文件
        with open(config_path, "r") as config_file:
            self.config = yaml.load(config_file, Loader=yaml.FullLoader)
        
        self.api_keys = self.config["api_key"]
        self.model_name = self.config["model_name"]
        self.base_url = self.config["base_url"]
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=random.choice(self.api_keys),
            base_url=self.base_url,
            timeout=600.0,
        )
    
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10) + wait_random(0, 2),
        stop=stop_after_attempt(10)
    )
    def _call_api(self, messages, **kwargs):
        """调用LLM API"""
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        # 确保类型正确
        temperature = float(temperature)
        max_tokens = int(max_tokens)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            # 添加详细的错误信息
            print(f"[DEBUG] API调用失败详情:", file=sys.stderr, flush=True)
            print(f"  - Model: {self.model_name}", file=sys.stderr, flush=True)
            print(f"  - Base URL: {self.base_url}", file=sys.stderr, flush=True)
            print(f"  - API Key: {self.client.api_key[:20]}..." if self.client.api_key else "  - API Key: None", file=sys.stderr, flush=True)
            print(f"  - Error: {type(e).__name__}: {str(e)}", file=sys.stderr, flush=True)
            raise  # 重新抛出异常以便 retry 机制处理
    
    def decide_post(self, post_preference: float, belief_value: float) -> bool:
        """
        LLM决策是否发帖
        
        Args:
            post_preference: 发帖偏好 [0, 1]
            belief_value: 信念值 [-1, 1]
        
        Returns:
            True表示发帖，False表示不发帖
        """
        # 将信念值转换为描述性文本
        if belief_value < -0.7:
            belief_desc = "strongly left-leaning (very negative)"
        elif belief_value < -0.3:
            belief_desc = "left-leaning (negative)"
        elif belief_value < 0.3:
            belief_desc = "neutral (centrist)"
        elif belief_value < 0.7:
            belief_desc = "right-leaning (positive)"
        else:
            belief_desc = "strongly right-leaning (very positive)"
        
        prompt = f"""You are a social media user deciding whether to post.

Your posting preference (likelihood to post): {post_preference:.2f} (0 = never post, 1 = always post)
Your current belief/opinion: {belief_value:.2f} ({belief_desc})

Based on your posting preference and current belief, decide whether to post.

Respond with ONLY a JSON object in this format:
{{"post": true or false}}

Do not include any other text."""
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that makes posting decisions. Always respond with valid JSON only."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self._call_api(messages)
            # 尝试解析JSON
            # 移除可能的markdown代码块标记
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            result = json.loads(response)
            return result.get("post", False)
        except Exception as e:
            print(f"[警告] LLM发帖决策失败: {e}, 使用默认决策（基于偏好随机）", file=sys.stderr, flush=True)
            # 降级到基于偏好的随机决策
            return random.random() < post_preference
    
    def decide_interaction(
        self,
        interaction_preference: float,
        belief_value: float,
        post_belief: float,
        post_interactions: Optional[Dict[str, int]] = None
    ) -> Optional[str]:
        """
        LLM决策对帖子的互动
        
        Args:
            interaction_preference: 互动偏好 [0, 1]
            belief_value: 当前信念值 [-1, 1]
            post_belief: 帖子信念值 [-1, 1]
            post_interactions: 帖子当前的互动情况 {"likes": int, "dislikes": int, "views": int} (可选)
        
        Returns:
            "like" 表示点赞, "dislike" 表示点踩, None 表示不互动
        """
        # 描述信念值
        def describe_belief(belief):
            if belief < -0.7:
                return "strongly left-leaning (very negative)"
            elif belief < -0.3:
                return "left-leaning (negative)"
            elif belief < 0.3:
                return "neutral (centrist)"
            elif belief < 0.7:
                return "right-leaning (positive)"
            else:
                return "strongly right-leaning (very positive)"
        
        belief_desc = describe_belief(belief_value)
        post_desc = describe_belief(post_belief)
        
        interaction_info = ""
        if post_interactions:
            interaction_info = f"""
Post engagement statistics:
- Views: {post_interactions.get('views', 0)}
- Likes: {post_interactions.get('likes', 0)}
- Dislikes: {post_interactions.get('dislikes', 0)}
"""
        
        prompt = f"""You are a social media user deciding how to interact with a post.

Your interaction preference (likelihood to interact): {interaction_preference:.2f} (0 = never interact, 1 = always interact)
Your current belief/opinion: {belief_value:.2f} ({belief_desc})
Post's belief/opinion: {post_belief:.2f} ({post_desc})
{interaction_info}
Based on your interaction preference and the alignment between your belief and the post's belief, decide:
1. If you interact: "like" if you agree with the post, "dislike" if you disagree
2. If you don't interact: choose null

Respond with ONLY a JSON object in this format:
{{"interact": true or false, "action": "like" or "dislike" or null}}

If interact is false, action must be null. If interact is true, action must be "like" or "dislike"."""

        messages = [
            {"role": "system", "content": "You are a helpful assistant that makes interaction decisions. Always respond with valid JSON only."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self._call_api(messages)
            # 移除可能的markdown代码块标记
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            result = json.loads(response)
            if result.get("interact", False):
                action = result.get("action")
                if action in ["like", "dislike"]:
                    return action
                else:
                    return None
            else:
                return None
        except Exception as e:
            print(f"[警告] LLM互动决策失败: {e}, 使用默认决策（基于偏好随机）", file=sys.stderr, flush=True)
            # 降级到基于偏好的随机决策
            if random.random() < interaction_preference:
                # 根据信念值差异决定点赞或点踩
                belief_diff = abs(belief_value - post_belief)
                if belief_diff < 0.3:  # 信念相近，点赞
                    return "like"
                else:  # 信念差异大，点踩
                    return "dislike"
            else:
                return None
