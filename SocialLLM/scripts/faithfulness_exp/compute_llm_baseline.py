#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM as a Judge Baseline方法：使用LLM分析社交媒体模拟轨迹，为每个agent-timestep对分配风险归因分数
用于faithfulness实验的baseline对比，支持分块输入避免token过多

使用方法:
    python scripts/faithfulness_exp/compute_llm_baseline.py --result_dir results/gpt-4o-mini/gpt-4o-mini_42
    python scripts/faithfulness_exp/compute_llm_baseline.py --result_dir results/gpt-4o-mini/gpt-4o-mini_42 --config config/api.yaml
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import re
import time
import signal
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# 全局中断标志
_interrupted = False

def signal_handler(signum, frame):
    """处理中断信号"""
    global _interrupted
    _interrupted = True
    print("\n收到中断信号，正在停止...", file=sys.stderr)
    sys.exit(130)

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# 尝试导入yaml库
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

def simple_yaml_load(file_path):
    """简单的YAML解析器，用于处理简单的配置文件"""
    config = {}
    current_key = None
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            if not line or line.startswith('#'):
                continue
            
            stripped = line.lstrip()
            if stripped.startswith('-'):
                if current_key is not None:
                    if current_key not in config:
                        config[current_key] = []
                    list_value = stripped[1:].strip()
                    config[current_key].append(list_value)
                continue
            
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                if value:
                    config[key] = value
                    current_key = None
                else:
                    current_key = key
                    config[key] = []
    return config

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 尝试导入openai库（支持新旧版本）
OPENAI_AVAILABLE = False
OPENAI_CLIENT_CLASS = None
OPENAI_LEGACY = False

try:
    from openai import OpenAI
    OPENAI_CLIENT_CLASS = OpenAI
    OPENAI_AVAILABLE = True
except (ImportError, ModuleNotFoundError, AttributeError):
    try:
        import openai
        if hasattr(openai, 'ChatCompletion'):
            OPENAI_LEGACY = True
            OPENAI_AVAILABLE = True
        else:
            OPENAI_AVAILABLE = False
    except (ImportError, ModuleNotFoundError):
        OPENAI_AVAILABLE = False


def load_config(config_path=None):
    """加载配置文件或使用默认配置（支持JSON和YAML格式）"""
    default_config = {
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "temperature": 0.7,
        "timeout": int(os.getenv("OPENAI_TIMEOUT", "300")),
        "rows_per_batch": 100
    }
    
    # 如果提供了config_path，使用指定的配置文件
    if config_path and Path(config_path).exists():
        config_file = Path(config_path)
        with open(config_file, 'r', encoding='utf-8') as f:
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                if YAML_AVAILABLE:
                    file_config = yaml.safe_load(f)
                else:
                    file_config = simple_yaml_load(config_file)
            else:
                file_config = json.load(f)
            
            if isinstance(file_config.get("api_key"), list):
                file_config["api_key"] = file_config["api_key"][0] if file_config["api_key"] else ""
            
            if "model_name" in file_config and "model" not in file_config:
                file_config["model"] = file_config.pop("model_name")
            
            default_config.update(file_config)
    else:
        # 如果没有指定config_path，优先尝试从统一配置路径加载
        unified_config_file = project_root / "config" / "api.yaml"
        if unified_config_file.exists():
            try:
                with open(unified_config_file, 'r', encoding='utf-8') as f:
                    if YAML_AVAILABLE:
                        unified_config = yaml.safe_load(f)
                    else:
                        unified_config = simple_yaml_load(unified_config_file)
                    
                    # 处理配置项
                    if "api_key" in unified_config:
                        if isinstance(unified_config["api_key"], list):
                            unified_config["api_key"] = unified_config["api_key"][0] if unified_config["api_key"] else ""
                        default_config["api_key"] = unified_config["api_key"]
                    
                    if "model" in unified_config:
                        default_config["model"] = unified_config["model"]
                    elif "model_name" in unified_config:
                        default_config["model"] = unified_config["model_name"]
                    
                    if "base_url" in unified_config:
                        default_config["base_url"] = unified_config["base_url"]
                    
                    if "temperature" in unified_config:
                        default_config["temperature"] = unified_config["temperature"]
            except Exception as e:
                print(f"警告: 无法加载统一配置文件: {e}")
    
    return default_config


def create_llm_client(config: Dict):
    """创建LLM客户端"""
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI库未安装，请先安装: pip install openai")
    
    api_key = config.get("api_key", "")
    base_url = config.get("base_url", "https://api.openai.com/v1")
    
    if not api_key:
        raise ValueError("API key未设置，请通过配置文件或环境变量OPENAI_API_KEY设置")
    
    if OPENAI_LEGACY:
        # 旧版本
        openai.api_key = api_key
        if base_url != "https://api.openai.com/v1":
            openai.api_base = base_url
        return None
    else:
        # 新版本
        return OPENAI_CLIENT_CLASS(api_key=api_key, base_url=base_url)


def call_llm_api(client, model: str, prompt: str, temperature: float = 0.7, timeout: int = 300, max_retries: int = 3):
    """调用LLM API"""
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    last_error = None
    for attempt in range(max_retries):
        try:
            if OPENAI_LEGACY:
                # 旧版本（<1.0）使用openai.ChatCompletion.create
                chat_response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature
                )
            else:
                # 新版本（>=1.0）使用client.chat.completions.create
                chat_response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    timeout=timeout
                )
            
            if chat_response.choices:
                completion_text = chat_response.choices[0].message.content
            else:
                completion_text = None
            
            return completion_text
            
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # 指数退避：2秒、4秒、6秒
                print(f"  错误 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                print(f"  等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                raise Exception(f"LLM API调用失败（已重试{max_retries}次）: {str(last_error)}")
    
    raise Exception(f"LLM API调用失败: {str(last_error)}")


def load_action_table(action_table_file: Path) -> pd.DataFrame:
    """加载action_table数据"""
    if not action_table_file.exists():
        raise FileNotFoundError(f"Action table文件不存在: {action_table_file}。请先运行extract_action_table.py")
    
    df = pd.read_csv(action_table_file)
    return df


def split_action_df_by_rows(action_df: pd.DataFrame, rows_per_batch: int = 100) -> List[Dict]:
    """按行数分段，每段最多rows_per_batch行"""
    batches = []
    total_rows = len(action_df)
    total_batches = (total_rows + rows_per_batch - 1) // rows_per_batch
    
    for i in range(0, total_rows, rows_per_batch):
        batch_df = action_df.iloc[i:i+rows_per_batch].copy()
        # 计算该段的timestep范围
        timesteps = sorted(batch_df['timestep'].unique())
        batch_info = {
            'batch_id': len(batches) + 1,
            'start_timestep': timesteps[0] if timesteps else None,
            'end_timestep': timesteps[-1] if timesteps else None,
            'total_batches': total_batches,
            'df': batch_df,
            'start_row': i,
            'end_row': min(i + rows_per_batch, total_rows)
        }
        batches.append(batch_info)
    
    return batches


def format_action_table_for_llm(action_df: pd.DataFrame) -> str:
    """将action_table格式化为LLM可理解的JSON格式"""
    log_entries = []
    
    for _, row in action_df.iterrows():
        entry = {
            "agent_id": int(row["agent_id"]),
            "timestep": int(row["timestep"]),
            "posted": int(row.get("posted", 0)),
            "view_count": int(row.get("view_count", 0)),
            "like_count": int(row.get("like_count", 0)),
            "dislike_count": int(row.get("dislike_count", 0))
        }
        log_entries.append(entry)
    
    # 按timestep和agent_id排序
    log_entries.sort(key=lambda x: (x["timestep"], x["agent_id"]))
    
    return json.dumps(log_entries, indent=2)


def build_llm_judge_prompt(action_log: str, baseline_risk: float, real_risk: float, 
                           n_agents: int, n_timesteps: int, target_timestep: int,
                           include_reason: bool = True) -> str:
    """构建LLM as a Judge的prompt"""
    
    risk_diff = abs(real_risk - baseline_risk)
    
    if include_reason:
        output_format = (
            "[\n"
            '  {\n'
            '    "agent_id": <agent_id>,\n'
            '    "timestep": <timestep>,\n'
            '    "risk_score": <float between 0 and 1>,\n'
            '    "reason": "<brief explanation>"\n'
            '  },\n'
            "  ...\n"
            "]\n"
        )
        output_note = "- The explanations are concise and grounded in social media dynamics.\n"
    else:
        output_format = (
            "[\n"
            '  {\n'
            '    "agent_id": <agent_id>,\n'
            '    "timestep": <timestep>,\n'
            '    "risk_score": <float between 0 and 1>\n'
            '  },\n'
            "  ...\n"
            "]\n"
        )
        output_note = ""
    
    prompt = (
        "You are an AI assistant tasked with analyzing a social media simulation trajectory.\n"
        "\n"
        "The simulation consists of {} agents interacting over {} timesteps.\n"
        "At each timestep, agents can post content and interact with others' posts (view, like, dislike),\n"
        "which jointly influence the polarization dynamics and systemic risk.\n"
        "\n"
        "The problem is:\n"
        "The social media system exhibits a polarization risk that is significantly higher than a baseline system. "
        "The current system risk is {:.6f}, while the baseline risk is {:.6f}. "
        "The risk difference is {:.6f}.\n"
        "\n"
        "The polarization risk measures the variance and extremism of agent beliefs,\n"
        "reflecting how polarized the social media ecosystem has become.\n"
        "\n"
        "Risk computation:\n"
        "1. Belief values range from -1 (extreme negative) to +1 (extreme positive)\n"
        "2. Polarization risk = variance of beliefs + extremism penalty\n"
        "3. Extremism penalty increases when beliefs are close to -1 or +1\n"
        "\n"
        "Ground truth:\n"
        "The baseline system risk trajectory is provided as the reference.\n"
        "The current system produces a much higher risk level than the baseline.\n"
        "\n"
        "Your task:\n"
        "Analyze the system trajectory and attribute the excessive risk to individual agents' actions\n"
        "across time.\n"
        "\n"
        "You are given the system execution trace in chronological order.\n"
        "Each entry represents one agent's activity at a specific timestep, with the following fields:\n"
        "- agent_id: The identifier of the agent\n"
        "- timestep: The timestep (0-indexed)\n"
        "- posted: Whether the agent posted content (1) or not (0)\n"
        "- view_count: Number of posts the agent viewed\n"
        "- like_count: Number of posts the agent liked\n"
        "- dislike_count: Number of posts the agent disliked\n"
        "\n"
        "Here is the system trajectory:\n"
        "{}\n"
        "\n"
        "Based on this trajectory, please predict the following:\n"
        "\n"
        "For EACH agent at EACH timestep, assign a risk attribution score in the range [0, 1],\n"
        "indicating how much that specific agent's actions at that timestep contribute to\n"
        "reaching the peak risk value ({:.6f}) at the target timestep ({}).\n"
        "\n"
        "Scoring principles:\n"
        "- A higher score indicates a stronger causal contribution to increased polarization\n"
        "  and higher risk.\n"
        "- Scores should reflect both direct and indirect effects of agents' actions on system instability.\n"
        "- If an action is largely irrelevant to the increased risk, assign a score close to 0.\n"
        "- If an action is a major driver of risk escalation, assign a score close to 1.\n"
        "- Scores should be comparable across agents and timesteps.\n"
        "- Consider temporal causality: earlier actions can influence later risk, so actions at earlier timesteps\n"
        "  may have higher scores if they trigger cascading effects.\n"
        "\n"
        "Please output your result in the following JSON format:\n"
        "{}\n"
        "\n"
        "Ensure that:\n"
        "- All agent-timestep pairs in the trajectory are covered.\n"
        "- The attribution reflects temporal causality (earlier actions can influence later risk).\n"
        "{}"
        "- Output ONLY the JSON array, no additional text before or after.\n"
    ).format(
        n_agents,
        n_timesteps,
        target_timestep,
        target_timestep,
        real_risk,
        baseline_risk,
        risk_diff,
        target_timestep,
        target_timestep,
        action_log,
        real_risk,
        target_timestep,
        output_format,
        output_note
    )
    
    return prompt


def build_llm_judge_prompt_batch(action_log: str, baseline_risk: float, real_risk: float, 
                                n_agents: int, n_timesteps: int, batch_id: int, total_batches: int,
                                start_timestep: int, end_timestep: int, target_timestep: int,
                                include_reason: bool = True) -> str:
    """构建LLM as a Judge的prompt（分段版本）"""
    
    risk_diff = abs(real_risk - baseline_risk)
    
    if include_reason:
        output_format = (
            "[\n"
            '  {\n'
            '    "agent_id": <agent_id>,\n'
            '    "timestep": <timestep>,\n'
            '    "risk_score": <float between 0 and 1>,\n'
            '    "reason": "<brief explanation>"\n'
            '  },\n'
            "  ...\n"
            "]\n"
        )
        output_note = "- The explanations are concise and grounded in social media dynamics.\n"
    else:
        output_format = (
            "[\n"
            '  {\n'
            '    "agent_id": <agent_id>,\n'
            '    "timestep": <timestep>,\n'
            '    "risk_score": <float between 0 and 1>\n'
            '  },\n'
            "  ...\n"
            "]\n"
        )
        output_note = ""
    
    prompt = (
        "You are an AI assistant tasked with analyzing a social media simulation trajectory.\n"
        "\n"
        "IMPORTANT: This is batch {} of {}.\n"
        "Timestep range for this batch: {} to {}.\n"
        "Target risk timestep: {}.\n"
        "\n"
        "The simulation consists of {} agents interacting over {} timesteps.\n"
        "At each timestep, agents can post content and interact with others' posts (view, like, dislike),\n"
        "which jointly influence the polarization dynamics and systemic risk.\n"
        "\n"
        "The problem is:\n"
        "The social media system exhibits a polarization risk that reaches its PEAK value at timestep {} (the target timestep). "
        "The PEAK risk value at timestep {} is {:.6f}, while the initial baseline risk is {:.6f}. "
        "The risk increase from baseline to peak is {:.6f}.\n"
        "\n"
        "The polarization risk measures the variance and extremism of agent beliefs,\n"
        "reflecting how polarized the social media ecosystem has become.\n"
        "\n"
        "Risk computation:\n"
        "1. Belief values range from -1 (extreme negative) to +1 (extreme positive)\n"
        "2. Polarization risk = variance of beliefs + extremism penalty\n"
        "3. Extremism penalty increases when beliefs are close to -1 or +1\n"
        "\n"
        "Ground truth:\n"
        "The baseline system risk trajectory is provided as the reference.\n"
        "The current system produces a much higher risk level than the baseline, reaching its PEAK at timestep {}.\n"
        "\n"
        "Your task:\n"
        "Analyze the system trajectory and attribute the PEAK risk value (at timestep {}) to individual agents' actions\n"
        "across time. Your goal is to identify which agents' actions at which timesteps contributed most to reaching\n"
        "this PEAK risk value.\n"
        "\n"
        "You are given a PARTIAL system execution trace (batch {} of {}) in chronological order.\n"
        "Each entry represents one agent's activity at a specific timestep, with the following fields:\n"
        "- agent_id: The identifier of the agent\n"
        "- timestep: The timestep (0-indexed)\n"
        "- posted: Whether the agent posted content (1) or not (0)\n"
        "- view_count: Number of posts the agent viewed\n"
        "- like_count: Number of posts the agent liked\n"
        "- dislike_count: Number of posts the agent disliked\n"
        "\n"
        "Here is the system trajectory for this batch:\n"
        "{}\n"
        "\n"
        "Based on this trajectory, please predict the following:\n"
        "\n"
        "For EACH agent at EACH timestep in THIS BATCH, assign a risk attribution score in the range [0, 1],\n"
        "indicating how much that specific agent's actions at that timestep contribute to\n"
        "reaching the PEAK risk value ({:.6f}) at the target timestep ({}).\n"
        "\n"
        "Scoring principles:\n"
        "- A higher score indicates a stronger causal contribution to increased polarization\n"
        "  and higher risk.\n"
        "- Scores should reflect both direct and indirect effects of agents' actions on system instability.\n"
        "- If an action is largely irrelevant to the increased risk, assign a score close to 0.\n"
        "- If an action is a major driver of risk escalation, assign a score close to 1.\n"
        "- Scores should be comparable across agents and timesteps.\n"
        "- Consider temporal causality: earlier actions can influence later risk, so actions at earlier timesteps\n"
        "  may have higher scores if they trigger cascading effects.\n"
        "- IMPORTANT: Only score the (agent_id, timestep) pairs that appear in the trajectory above.\n"
        "\n"
        "Please output your result in the following JSON format:\n"
        "{}\n"
        "\n"
        "Ensure that:\n"
        "- All agent-timestep pairs in THIS BATCH are covered.\n"
        "- The attribution reflects temporal causality (earlier actions can influence later risk).\n"
        "{}"
        "- Output ONLY the JSON array, no additional text before or after.\n"
    ).format(
        batch_id,
        total_batches,
        start_timestep,
        end_timestep,
        target_timestep,
        n_agents,
        n_timesteps,
        target_timestep,
        target_timestep,
        real_risk,
        baseline_risk,
        risk_diff,
        target_timestep,
        target_timestep,
        batch_id,
        total_batches,
        action_log,
        real_risk,
        target_timestep,
        output_format,
        output_note
    )
    
    return prompt


def parse_llm_response(response_text: str) -> List[Dict]:
    """解析LLM响应，提取分数"""
    try:
        # 尝试直接解析JSON
        response_data = json.loads(response_text)
        
        # 处理不同的响应格式
        if isinstance(response_data, list):
            return response_data
        elif isinstance(response_data, dict) and "scores" in response_data:
            return response_data["scores"]
        else:
            raise ValueError(f"无法解析响应格式: {type(response_data)}")
    
    except json.JSONDecodeError:
        # 如果直接解析失败，尝试提取JSON部分
        # 查找第一个[或{到最后一个]或}
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            try:
                response_data = json.loads(json_match.group())
                if isinstance(response_data, list):
                    return response_data
                elif isinstance(response_data, dict) and "scores" in response_data:
                    return response_data["scores"]
            except:
                pass
        
        # 如果还是失败，尝试查找所有JSON对象
        json_objects = re.findall(r'\{[^{}]*\}', response_text)
        if json_objects:
            results = []
            for obj_str in json_objects:
                try:
                    obj = json.loads(obj_str)
                    if "agent_id" in obj and "timestep" in obj and "risk_score" in obj:
                        results.append(obj)
                except:
                    continue
            if results:
                return results
        
        raise ValueError(f"无法解析LLM响应: {response_text[:200]}...")


def process_single_batch(batch_info: Dict, baseline_risk: float, real_risk: float,
                        n_agents: int, n_timesteps: int, target_timestep: int, config: Dict) -> Tuple[int, pd.DataFrame]:
    """处理单个batch（用于并发处理）"""
    batch_id = batch_info['batch_id']
    total_batches = batch_info['total_batches']
    start_timestep = batch_info['start_timestep']
    end_timestep = batch_info['end_timestep']
    batch_df = batch_info['df']
    
    # 为每个线程创建独立的client（避免线程安全问题）
    client = create_llm_client(config)
    
    try:
        # 格式化该段的action数据
        action_log = format_action_table_for_llm(batch_df)
        
        # 构建prompt
        prompt = build_llm_judge_prompt_batch(
            action_log=action_log,
            baseline_risk=baseline_risk,
            real_risk=real_risk,
            n_agents=n_agents,
            n_timesteps=n_timesteps,
            batch_id=batch_id,
            total_batches=total_batches,
            start_timestep=start_timestep,
            end_timestep=end_timestep,
            target_timestep=target_timestep,
            include_reason=False
        )
        
        # 调用LLM API
        response_text = call_llm_api(
            client=client,
            model=config["model"],
            prompt=prompt,
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 300)
        )
        
        # 解析响应
        scores = parse_llm_response(response_text)
        
        # 转换为DataFrame
        scores_df = pd.DataFrame(scores)
        
        # 验证列名
        if "risk_score" in scores_df.columns:
            scores_df = scores_df.rename(columns={"risk_score": "llm_value"})
        
        # 确保有agent_id和timestep列
        if "agent_id" not in scores_df.columns or "timestep" not in scores_df.columns:
            raise ValueError("LLM响应缺少agent_id或timestep字段")
        
        # 确保llm_value列存在
        if "llm_value" not in scores_df.columns:
            raise ValueError("LLM响应缺少risk_score/llm_value字段")
        
        return batch_id, scores_df
    
    except Exception as e:
        print(f"  ✗ 批次 {batch_id}/{total_batches} 处理失败: {e}")
        if config.get("verbose", False):
            import traceback
            traceback.print_exc()
        raise


def process_llm_batches(batches: List[Dict], baseline_risk: float, real_risk: float,
                       n_agents: int, n_timesteps: int, target_timestep: int, config: Dict) -> List[pd.DataFrame]:
    """批量处理所有段，返回每段的分数DataFrame列表（支持并发）"""
    all_batch_scores = []
    max_workers = config.get("max_workers", 1)  # 默认串行，1表示不并发
    
    if max_workers <= 1:
        # 串行处理（原有逻辑）
        client = create_llm_client(config)
        
        for batch_info in batches:
            if _interrupted:
                print("\n收到中断信号，停止处理...")
                break
            
            batch_id = batch_info['batch_id']
            total_batches = batch_info['total_batches']
            start_timestep = batch_info['start_timestep']
            end_timestep = batch_info['end_timestep']
            batch_df = batch_info['df']
            
            print(f"\n处理批次 {batch_id}/{total_batches} (Timestep: {start_timestep} 到 {end_timestep}, {len(batch_df)} 行)...")
            
            try:
                _, scores_df = process_single_batch(
                    batch_info, baseline_risk, real_risk, n_agents, n_timesteps, target_timestep, config
                )
                all_batch_scores.append((batch_id, scores_df))
                print(f"  ✓ 成功处理批次 {batch_id}/{total_batches}")
            
            except Exception as e:
                print(f"  ✗ 批次 {batch_id}/{total_batches} 处理失败: {e}")
                if config.get("verbose", False):
                    import traceback
                    traceback.print_exc()
                # 继续处理下一批次
                continue
    else:
        # 并发处理（使用ThreadPoolExecutor，因为API调用是I/O密集型）
        print(f"\n使用 {max_workers} 个线程并发处理 {len(batches)} 个批次...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_batch = {
                executor.submit(
                    process_single_batch,
                    batch_info, baseline_risk, real_risk, n_agents, n_timesteps, target_timestep, config
                ): batch_info
                for batch_info in batches
            }
            
            # 收集结果
            for future in as_completed(future_to_batch):
                batch_info = future_to_batch[future]
                batch_id = batch_info['batch_id']
                total_batches = batch_info['total_batches']
                
                if _interrupted:
                    print("\n收到中断信号，停止处理...")
                    break
                
                try:
                    batch_id, scores_df = future.result()
                    all_batch_scores.append((batch_id, scores_df))
                    print(f"  ✓ 成功处理批次 {batch_id}/{total_batches}")
                except Exception as e:
                    print(f"  ✗ 批次 {batch_id}/{total_batches} 处理失败: {e}")
                    if config.get("verbose", False):
                        import traceback
                        traceback.print_exc()
    
    # 按batch_id排序，确保顺序
    all_batch_scores.sort(key=lambda x: x[0])
    return [scores_df for _, scores_df in all_batch_scores]


def compute_llm_baseline(result_dir: Path, config: Dict) -> pd.DataFrame:
    """
    使用LLM方法计算baseline分数
    
    Args:
        result_dir: 结果目录路径
        config: LLM配置字典
    
    Returns:
        DataFrame with columns: agent_id, timestep, llm_value
    """
    results_file = result_dir / "results.json"
    action_table_file = result_dir / "action_table" / "action_table.csv"
    
    # 加载数据
    with open(results_file, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
    
    action_df = load_action_table(action_table_file)
    
    # 获取参数
    num_agents = results_data.get('num_agents', 20)
    max_risk_timestep = results_data.get('max_risk_timestep')
    if max_risk_timestep is None:
        max_risk_timestep = results_data.get('num_steps', 30)
    
    baseline_risk = results_data.get('initial_risk', 0.0)
    real_risk = results_data.get('max_risk', 0.0)
    
    print(f"处理: {result_dir.name}")
    print("="*60)
    print(f"  Num agents: {num_agents}")
    print(f"  Max timestep: {max_risk_timestep}")
    print(f"  Baseline risk: {baseline_risk:.6f}")
    print(f"  Real risk: {real_risk:.6f}")
    
    # 分段处理
    rows_per_batch = config.get("rows_per_batch", 100)
    batches = split_action_df_by_rows(action_df, rows_per_batch)
    
    print(f"\n将数据分为 {len(batches)} 个批次处理（每批最多 {rows_per_batch} 行）...")
    
    # 批量处理
    all_batch_scores = process_llm_batches(
        batches=batches,
        baseline_risk=baseline_risk,
        real_risk=real_risk,
        n_agents=num_agents,
        n_timesteps=max_risk_timestep,
        target_timestep=max_risk_timestep,
        config=config
    )
    
    if not all_batch_scores:
        raise ValueError("所有批次处理失败，无法生成结果")
    
    # 合并所有批次的结果
    print("\n合并所有批次的结果...")
    combined_df = pd.concat(all_batch_scores, ignore_index=True)
    
    # 确保所有(agent_id, timestep)对都有分数
    # 创建完整的(agent_id, timestep)组合
    all_combinations = []
    for agent_id in range(num_agents):
        for timestep in range(max_risk_timestep):
            all_combinations.append({'agent_id': agent_id, 'timestep': timestep})
    
    full_df = pd.DataFrame(all_combinations)
    
    # 合并，保留LLM分数
    merged_df = full_df.merge(combined_df, on=['agent_id', 'timestep'], how='left')
    
    # 填充缺失值（如果LLM没有为某些组合打分，使用0）
    merged_df['llm_value'] = merged_df['llm_value'].fillna(0.0)
    
    # 确保分数在[0, 1]范围内
    merged_df['llm_value'] = merged_df['llm_value'].clip(0.0, 1.0)
    
    # 只保留需要的列
    result_df = merged_df[['agent_id', 'timestep', 'llm_value']].copy()
    
    # 计算统计信息
    score_stats = {
        "mean": float(result_df['llm_value'].mean()),
        "std": float(result_df['llm_value'].std()),
        "min": float(result_df['llm_value'].min()),
        "max": float(result_df['llm_value'].max()),
        "sum": float(result_df['llm_value'].sum()),
        "median": float(result_df['llm_value'].median())
    }
    
    print(f"\nLLM分数统计:")
    print(f"    Mean: {score_stats['mean']:.6f}")
    print(f"    Std: {score_stats['std']:.6f}")
    print(f"    Min: {score_stats['min']:.6f}")
    print(f"    Max: {score_stats['max']:.6f}")
    print(f"    Median: {score_stats['median']:.6f}")
    
    return result_df, score_stats


def main():
    parser = argparse.ArgumentParser(
        description='使用LLM as a Judge方法计算baseline分数',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--result_dir',
        type=str,
        required=True,
        help='结果目录路径（包含results.json和action_table/action_table.csv）'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='LLM配置文件路径（可选，默认从config/api.yaml加载）'
    )
    
    parser.add_argument(
        '--rows-per-batch',
        type=int,
        default=100,
        help='每批处理的行数（可选，默认：100）'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=1,
        help='并发线程数（可选，默认：1表示串行。API调用是I/O密集型，可以设置较大值如10-20来加速）'
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='如果结果文件已存在，则跳过'
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
    output_dir = result_dir / "faithfulness_exp" / "llm"
    output_file = output_dir / "llm_attribution_timeseries.csv"
    stats_file = output_dir / "llm_stats.json"
    
    if args.skip_existing and output_file.exists():
        print(f"跳过已存在的文件: {output_file}")
        return
    
    # 加载配置
    config = load_config(args.config)
    config["rows_per_batch"] = args.rows_per_batch
    config["max_workers"] = args.max_workers
    config["verbose"] = args.verbose
    
    # 计算LLM baseline
    try:
        df, score_stats = compute_llm_baseline(result_dir, config)
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存CSV文件
        df.to_csv(output_file, index=False)
        print(f"\n✓ CSV文件已保存到: {output_file}")
        
        # 保存统计信息
        with open(result_dir / "results.json", 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        
        stats = {
            "method": "llm",
            "num_agents": results_data.get('num_agents', 20),
            "max_risk_timestep": results_data.get('max_risk_timestep'),
            "max_risk": float(results_data.get('max_risk', 0.0)),
            "initial_risk": float(results_data.get('initial_risk', 0.0)),
            "score_stats": score_stats,
            "model": config.get("model", "unknown"),
            "rows_per_batch": args.rows_per_batch
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 统计信息已保存到: {stats_file}")
    
    except Exception as e:
        print(f"错误: 计算LLM baseline失败: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
