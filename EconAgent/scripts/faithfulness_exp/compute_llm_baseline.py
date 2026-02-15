#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM as a Judge Baseline方法：使用LLM分析宏观经济模拟轨迹，为每个agent-timestep对分配风险归因分数
用于faithfulness实验的baseline对比，支持分块输入避免token过多

使用方法:
    python scripts/faithfulness_exp/compute_llm_baseline.py gpt/gpt_42
    python scripts/faithfulness_exp/compute_llm_baseline.py gpt/gpt_42 --config llm_config.json
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import re
import signal
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_ROOT = PROJECT_ROOT / "datas"
sys.path.insert(0, str(PROJECT_ROOT))

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
    """加载配置文件或使用默认配置（支持JSON和YAML格式）
    
    配置文件加载优先级：
    1. 如果提供了config_path，使用指定的配置文件
    2. 否则，尝试从 scripts/faithfulness_exp/config.yaml 加载（统一配置路径）
    3. 再尝试从项目根目录的config.yaml加载（llm部分）
    4. 最后，使用环境变量作为默认值
    """
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
        unified_config_file = SCRIPT_DIR / "config.yaml"
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
                        default_config["temperature"] = float(unified_config["temperature"])
                    
                    if "timeout" in unified_config:
                        default_config["timeout"] = int(unified_config["timeout"])
                    
                    if "rows_per_batch" in unified_config:
                        default_config["rows_per_batch"] = int(unified_config["rows_per_batch"])
                    
                    if "include_reason" in unified_config:
                        default_config["include_reason"] = bool(unified_config["include_reason"])
                    
                    print(f"  已从统一配置文件加载: {unified_config_file}")
            except Exception as e:
                print(f"  警告: 无法从统一配置文件 {unified_config_file} 加载配置: {str(e)}")
        
        # 如果统一配置文件不存在或加载失败，尝试从项目根目录的config.yaml加载
        if not default_config["api_key"]:
            project_config_file = PROJECT_ROOT / "config.yaml"
            if project_config_file.exists():
                try:
                    with open(project_config_file, 'r', encoding='utf-8') as f:
                        if YAML_AVAILABLE:
                            project_config = yaml.safe_load(f)
                        else:
                            project_config = simple_yaml_load(project_config_file)
                        
                        # 从llm部分提取配置
                        if "llm" in project_config:
                            llm_config = project_config["llm"]
                            
                            # 提取api_key（优先使用api_key，否则从api_key_env环境变量读取）
                            if "api_key" in llm_config and llm_config["api_key"]:
                                default_config["api_key"] = llm_config["api_key"]
                            elif "api_key_env" in llm_config:
                                env_key = llm_config["api_key_env"]
                                env_value = os.getenv(env_key, "")
                                if env_value:
                                    default_config["api_key"] = env_value
                            
                            # 提取base_url
                            if "api_base" in llm_config:
                                default_config["base_url"] = llm_config["api_base"]
                            
                            # 提取model
                            if "model" in llm_config:
                                default_config["model"] = llm_config["model"]
                            
                            # 提取其他可选配置
                            if "temperature" in llm_config:
                                default_config["temperature"] = float(llm_config["temperature"])
                            if "timeout" in llm_config:
                                default_config["timeout"] = int(llm_config["timeout"])
                        
                        print(f"  已从项目根目录配置文件加载: {project_config_file}")
                except Exception as e:
                    print(f"  警告: 无法从项目根目录配置文件 {project_config_file} 加载配置: {str(e)}")
    
    # 处理列表类型的api_key
    if isinstance(default_config.get("api_key"), list):
        default_config["api_key"] = default_config["api_key"][0] if default_config["api_key"] else ""
    
    # 最终检查api_key
    if not default_config["api_key"]:
        raise ValueError(
            "API key not found. Please:\n"
            "  1. Set OPENAI_API_KEY environment variable, OR\n"
            "  2. Add 'api_key' to scripts/faithfulness_exp/config.yaml, OR\n"
            "  3. Add 'api_key' to 'llm' section in project root config.yaml, OR\n"
            "  4. Provide --config parameter with a config file path"
        )
    
    return default_config


def create_llm_client(config):
    """创建LLM客户端（支持新旧版本openai库）"""
    if not OPENAI_AVAILABLE:
        raise ImportError("openai library is required. Install with: pip install openai")
    
    if OPENAI_CLIENT_CLASS:
        client = OPENAI_CLIENT_CLASS(
            api_key=config["api_key"],
            base_url=config.get("base_url", "https://api.openai.com/v1")
        )
        return client
    elif OPENAI_LEGACY:
        import openai
        openai.api_key = config["api_key"]
        openai.api_base = config.get("base_url", "https://api.openai.com/v1")
        return openai
    else:
        raise ImportError("openai library is not properly installed or version is incompatible")


def chat_completion_request_openai(client, prompt, model, temperature=1.0, timeout=300, max_retries=3):
    """调用LLM API（支持新旧版本）"""
    messages = [{"role": "user", "content": prompt}]
    
    import time
    last_error = None
    
    for attempt in range(max_retries):
        try:
            if OPENAI_LEGACY:
                import openai
                try:
                    chat_response = openai.ChatCompletion.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        timeout=timeout
                    )
                except TypeError:
                    chat_response = openai.ChatCompletion.create(
                        model=model,
                        messages=messages,
                        temperature=temperature
                    )
            else:
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
                wait_time = (attempt + 1) * 2
                print(f"  错误 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                print(f"  等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                raise Exception(f"LLM API调用失败（已重试{max_retries}次）: {str(last_error)}")
    
    raise Exception(f"LLM API调用失败: {str(last_error)}")


def load_shapley_stats(model_path: Path) -> Dict:
    """加载shapley stats获取实验参数"""
    shapley_dir = model_path / "shapley"
    if not shapley_dir.exists():
        raise FileNotFoundError(f"Shapley目录不存在: {shapley_dir}")
    
    stats_file = shapley_dir / "shapley_stats.json"
    if not stats_file.exists():
        raise FileNotFoundError(f"Shapley stats文件不存在: {stats_file}")
    
    with open(stats_file, 'r') as f:
        shapley_stats = json.load(f)
    
    return shapley_stats


def load_action_table(model_path: Path) -> pd.DataFrame:
    """加载action_table数据"""
    action_table_file = model_path / "action_table" / "action_table.csv"
    if not action_table_file.exists():
        raise FileNotFoundError(f"action_table CSV文件不存在: {action_table_file}。请先运行extract_action_table.py")
    
    df = pd.read_csv(action_table_file)
    return df


def split_action_df_by_rows(action_df: pd.DataFrame, rows_per_batch: int = 100) -> List[Dict]:
    """按行数分段，每段最多rows_per_batch行"""
    batches = []
    total_rows = len(action_df)
    total_batches = (total_rows + rows_per_batch - 1) // rows_per_batch
    
    for i in range(0, total_rows, rows_per_batch):
        batch_df = action_df.iloc[i:i+rows_per_batch].copy()
        timesteps = sorted(batch_df['timestep'].unique())
        batch_info = {
            'batch_id': len(batches) + 1,
            'start_timestep': int(timesteps[0]) if timesteps else None,
            'end_timestep': int(timesteps[-1]) if timesteps else None,
            'total_batches': total_batches,
            'df': batch_df,
            'start_row': i,
            'end_row': min(i + rows_per_batch, total_rows)
        }
        batches.append(batch_info)
    
    return batches


def normalize_scores(df_scores: pd.DataFrame) -> pd.DataFrame:
    """归一化分数到[0,1]范围"""
    if len(df_scores) == 0:
        return df_scores
    min_score = df_scores['score'].min()
    max_score = df_scores['score'].max()
    if max_score > min_score:
        df_scores = df_scores.copy()
        df_scores['score'] = (df_scores['score'] - min_score) / (max_score - min_score)
    else:
        df_scores = df_scores.copy()
        df_scores['score'] = 0.0
    return df_scores


def format_action_table_for_llm(action_df: pd.DataFrame) -> str:
    """将action_table格式化为LLM可理解的JSON格式"""
    log_entries = []
    
    for _, row in action_df.iterrows():
        entry = {
            "agent_id": int(row["agent_id"]),
            "timestep": int(row["timestep"]),
            "wealth": float(row["wealth"]),
            "income": float(row["income"]),
            "consumption_rate": float(row.get("endogenous_Consumption Rate", 0.0)),
            "job": str(row.get("endogenous_job", ""))
        }
        log_entries.append(entry)
    
    # 按timestep和agent_id排序
    log_entries.sort(key=lambda x: (x["timestep"], x["agent_id"]))
    
    return json.dumps(log_entries, indent=2)


def build_llm_judge_prompt_batch(action_log: str, baseline_risk: float, real_risk: float,
                                num_agents: int, max_timestep: int, batch_id: int, total_batches: int,
                                start_timestep: int, end_timestep: int, target_timesteps: List[int],
                                risk_lambda: float = 0.94, include_reason: bool = True) -> str:
    """构建LLM as a Judge的prompt（分段版本）"""
    
    risk_diff = abs(real_risk - baseline_risk)
    target_t_str = ", ".join(map(str, target_timesteps)) if target_timesteps else str(max_timestep)
    
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
        output_note = "- The explanations are concise and grounded in economic reasoning.\n"
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
        "You are an AI assistant tasked with analyzing a multi-agent macroeconomic simulation trajectory.\n"
        "\n"
        "IMPORTANT: This is batch {} of {}.\n"
        "Timestep range for this batch: {} to {}.\n"
        "Target risk timesteps: {}.\n"
        "\n"
        "The simulation consists of {} agents interacting over {} time steps.\n"
        "At each time step, each agent makes economic decisions such as consumption and labor participation,\n"
        "which jointly influence the macroeconomic price level and inflation dynamics.\n"
        "\n"
        "The problem is:\n"
        "The economic system exhibits a risk indicator that is significantly higher than a baseline system. "
        "The current system risk is {:.6f}, while the baseline risk is {:.6f}. "
        "The risk difference is {:.6f}.\n"
        "\n"
        "The risk indicator is defined based on Engle (1982) and Bollerslev (1986) and measures the conditional variance\n"
        "of inflation forecast errors, reflecting systemic economic risk.\n"
        "\n"
        "Risk computation:\n"
        "1. Inflation rate: π_t = log(P_t) - log(P_{{t-1}})\n"
        "2. Expected inflation (naive expectation): E_{{t-1}}[π_t] = π_{{t-1}}\n"
        "3. Forecast error: e_t = π_t - E_{{t-1}}[π_t]\n"
        "4. Risk indicator: h_t = λ · h_{{t-1}} + (1 - λ) · e_{{t-1}}², where λ = {}\n"
        "\n"
        "Ground truth:\n"
        "The baseline system risk trajectory h_t^baseline is provided as the reference.\n"
        "The current system produces a much higher risk level than the baseline.\n"
        "\n"
        "Your task:\n"
        "Analyze the system trajectory and attribute the excessive risk to individual agents' actions\n"
        "across time.\n"
        "\n"
        "You are given a PARTIAL system execution trace (batch {} of {}) in chronological order.\n"
        "Each entry represents one agent's action at a specific timestep, with the following fields:\n"
        "- agent_id: The identifier of the agent (0 to {})\n"
        "- timestep: The time step (1 to {})\n"
        "- wealth: The agent's wealth at this timestep\n"
        "- income: The agent's income at this timestep\n"
        "- consumption_rate: The agent's consumption rate (0-1)\n"
        "- job: The agent's job type\n"
        "\n"
        "Here is the system trajectory for this batch:\n"
        "{}\n"
        "\n"
        "Based on this trajectory, please predict the following:\n"
        "\n"
        "For EACH agent at EACH timestep in THIS BATCH ONLY, assign a risk attribution score in the range [0, 1],\n"
        "indicating how much that specific agent's action at that timestep contributes to\n"
        "the elevated system risk relative to the baseline.\n"
        "\n"
        "Scoring principles:\n"
        "- A higher score indicates a stronger causal contribution to increased inflation volatility\n"
        "  and higher conditional variance h_t.\n"
        "- Scores should reflect both direct and indirect effects of agents' actions on macroeconomic instability.\n"
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
        target_t_str,
        num_agents,
        max_timestep,
        real_risk,
        baseline_risk,
        risk_diff,
        risk_lambda,
        batch_id,
        total_batches,
        num_agents - 1,
        max_timestep,
        action_log,
        output_format,
        output_note
    )
    
    return prompt


def process_llm_batches(batches: List[Dict], baseline_risk: float, real_risk: float,
                       num_agents: int, max_timestep: int, target_timesteps: List[int],
                       risk_lambda: float, config: Dict, include_reason: bool = True) -> List[pd.DataFrame]:
    """批量处理所有段，返回每段的分数DataFrame列表"""
    all_batch_scores = []
    client = create_llm_client(config)
    
    for batch_info in batches:
        batch_id = batch_info['batch_id']
        total_batches = batch_info['total_batches']
        start_timestep = batch_info['start_timestep']
        end_timestep = batch_info['end_timestep']
        batch_df = batch_info['df']
        
        print(f"\n处理批次 {batch_id}/{total_batches} (timestep: {start_timestep} 到 {end_timestep}, {len(batch_df)} 行)...")
        
        try:
            # 格式化该段的action数据
            action_log = format_action_table_for_llm(batch_df)
            
            # 构建prompt
            prompt = build_llm_judge_prompt_batch(
                action_log=action_log,
                baseline_risk=baseline_risk,
                real_risk=real_risk,
                num_agents=num_agents,
                max_timestep=max_timestep,
                batch_id=batch_id,
                total_batches=total_batches,
                start_timestep=start_timestep,
                end_timestep=end_timestep,
                target_timesteps=target_timesteps,
                risk_lambda=risk_lambda,
                include_reason=include_reason
            )
            
            # 根据prompt大小调整超时时间
            prompt_size = len(prompt)
            prompt_size_mb = prompt_size / (1024 * 1024)
            estimated_timeout = max(60, int(prompt_size_mb * 60) + 30)
            actual_timeout = config.get('timeout', estimated_timeout)
            
            # 调用LLM
            print(f"  调用LLM API (超时: {actual_timeout}秒)...")
            response_text = chat_completion_request_openai(
                client=client,
                prompt=prompt,
                model=config['model'],
                temperature=config.get('temperature', 0.7),
                timeout=actual_timeout
            )
            
            if not response_text:
                raise ValueError("LLM返回空响应")
            
            print(f"  响应长度: {len(response_text):,} 字符")
            
            # 解析响应
            print(f"  解析LLM响应...")
            parsed_scores = parse_llm_response(response_text)
            print(f"  解析到 {len(parsed_scores)} 个分数")
            
            # 转换为DataFrame
            scores_dict = {}
            for item in parsed_scores:
                agent_id = int(item.get("agent_id", -1))
                timestep = int(item.get("timestep", -1))
                score = float(item.get("risk_score", item.get("score", 0.0)))
                # 确保分数在[0, 1]范围内
                score = max(0.0, min(1.0, score))
                scores_dict[(agent_id, timestep)] = score
            
            # 创建DataFrame（确保包含该段的所有action）
            data = []
            for _, row in batch_df.iterrows():
                agent_id = int(row["agent_id"])
                timestep = int(row["timestep"])
                key = (agent_id, timestep)
                if key in scores_dict:
                    score = scores_dict[key]
                else:
                    # 如果LLM没有返回该action的分数，使用0.0
                    print(f"    警告: LLM未返回 (agent_id={agent_id}, timestep={timestep}) 的分数，使用0.0")
                    score = 0.0
                data.append({
                    'agent_id': agent_id,
                    'timestep': timestep,
                    'score': score
                })
            
            batch_scores_df = pd.DataFrame(data)
            
            # 归一化该段的分数
            batch_scores_df = normalize_scores(batch_scores_df)
            
            all_batch_scores.append(batch_scores_df)
            print(f"  ✓ 批次 {batch_id} 完成，获得 {len(batch_scores_df)} 个分数")
            
        except Exception as e:
            print(f"  ✗ 批次 {batch_id} 处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            # 即使失败，也创建一个包含0分的DataFrame，确保后续处理不会中断
            data = []
            for _, row in batch_df.iterrows():
                data.append({
                    'agent_id': int(row["agent_id"]),
                    'timestep': int(row["timestep"]),
                    'score': 0.0
                })
            all_batch_scores.append(pd.DataFrame(data))
    
    return all_batch_scores


def parse_llm_response(response_text: str) -> List[Dict]:
    """解析LLM返回的JSON响应"""
    if not response_text or len(response_text.strip()) == 0:
        raise ValueError("LLM response is empty")
    
    # 尝试提取JSON部分（可能包含markdown代码块）
    response_text = response_text.strip()
    
    # 移除可能的markdown代码块标记
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    elif response_text.startswith("```"):
        response_text = response_text[3:]
    
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    
    response_text = response_text.strip()
    
    # 尝试解析JSON
    try:
        parsed_data = json.loads(response_text)
        if isinstance(parsed_data, list):
            return parsed_data
        elif isinstance(parsed_data, dict) and "scores" in parsed_data:
            return parsed_data["scores"]
        else:
            raise ValueError(f"Unexpected JSON structure: {type(parsed_data)}")
    except json.JSONDecodeError as e:
        # 尝试修复常见的JSON错误
        print(f"  警告: JSON解析失败，尝试修复...")
        print(f"  错误: {str(e)}")
        
        # 尝试提取JSON数组部分
        match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if match:
            try:
                parsed_data = json.loads(match.group(0))
                return parsed_data if isinstance(parsed_data, list) else [parsed_data]
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"无法解析LLM响应为JSON: {str(e)}\n响应内容:\n{response_text[:500]}")


def scores_to_matrix(df_scores: pd.DataFrame, num_agents: int, max_timestep: int) -> np.ndarray:
    """将分数DataFrame转换为矩阵格式"""
    score_matrix = np.zeros((num_agents, max_timestep))
    
    for _, row in df_scores.iterrows():
        agent_id = int(row['agent_id'])
        timestep = int(row['timestep'])
        score = float(row['score'])
        # timestep从1开始，矩阵索引从0开始
        if 0 <= agent_id < num_agents and 0 < timestep <= max_timestep:
            score_matrix[agent_id, timestep - 1] = score
    
    return score_matrix


def compute_llm_baseline(model_path: str, config: Dict = None, config_path: str = None) -> Tuple[np.ndarray, Dict]:
    """使用LLM as a Judge方法计算baseline分数"""
    model_path_obj = DATA_ROOT / model_path
    print(f"处理: {model_path}")
    print("="*60)
    
    # 加载配置
    if config is None:
        config = load_config(config_path)
    
    print(f"使用模型: {config['model']}")
    
    # 加载数据
    print("\n加载数据...")
    shapley_stats = load_shapley_stats(model_path_obj)
    action_df = load_action_table(model_path_obj)
    
    num_agents = shapley_stats['num_agents']
    episode_length = shapley_stats['episode_length']
    target_timesteps = shapley_stats.get('target_timesteps', None)
    baseline_risk = shapley_stats.get('baseline_risk', 0.0)
    real_risk = shapley_stats.get('real_risk', 0.0)
    risk_lambda = shapley_stats.get('risk_lambda', 0.94)
    
    if target_timesteps is None or len(target_timesteps) == 0:
        max_timestep = episode_length
    else:
        max_timestep = max(target_timesteps)
    
    print(f"  日期范围: timestep 1 到 {max_timestep}")
    print(f"  Baseline risk: {baseline_risk:.6f}")
    print(f"  Real risk: {real_risk:.6f}")
    print(f"  Action table rows: {len(action_df)}")
    
    # 检查是否包含reason字段
    include_reason = config.get('include_reason', True)
    if not include_reason:
        print("  模式: 仅输出分数（不包含reason字段，以减少token消耗）")
    else:
        print("  模式: 输出分数和reason字段")
    
    # 决定是否使用分段处理
    rows_per_batch = config.get('rows_per_batch', 100)
    use_batch = len(action_df) > rows_per_batch
    
    if use_batch:
        print(f"\n使用分段处理模式 (每段 {rows_per_batch} 行)...")
        # 分段处理
        batches = split_action_df_by_rows(action_df, rows_per_batch=rows_per_batch)
        print(f"  总共分为 {len(batches)} 段")
        
        # 批量处理所有段
        all_batch_scores = process_llm_batches(
            batches=batches,
            baseline_risk=baseline_risk,
            real_risk=real_risk,
            num_agents=num_agents,
            max_timestep=max_timestep,
            target_timesteps=target_timesteps if target_timesteps else [max_timestep],
            risk_lambda=risk_lambda,
            config=config,
            include_reason=include_reason
        )
        
        # 合并所有段的分数
        print("\n合并所有段的分数...")
        df_scores = pd.concat(all_batch_scores, ignore_index=True)
        
        # 检查分数完整性
        expected_count = len(action_df)
        actual_count = len(df_scores)
        if actual_count < expected_count * 0.5:
            print(f"  警告: 只获得了 {actual_count}/{expected_count} 个分数（{actual_count/expected_count*100:.1f}%）")
        else:
            print(f"  成功获得 {actual_count}/{expected_count} 个分数（{actual_count/expected_count*100:.1f}%）")
        
        # 最终归一化（对所有分数一起归一化）
        print("  对合并后的分数进行最终归一化...")
        df_scores = normalize_scores(df_scores)
        
        n_batches = len(batches)
        batch_size = rows_per_batch
    else:
        print("\n使用单次处理模式（数据量较小）...")
        # 单次处理逻辑（作为fallback，但这里主要使用分块方式）
        action_log = format_action_table_for_llm(action_df)
        log_size = len(action_log)
        log_size_kb = log_size / 1024
        log_size_mb = log_size_kb / 1024
        print(f"  系统轨迹长度: {log_size:,} 字符 ({log_size_kb:.1f} KB, {log_size_mb:.2f} MB)")
        print("  警告: 数据量较小但建议使用分块处理以确保稳定性")
        # 使用单批次处理
        batches = [{
            'batch_id': 1,
            'start_timestep': 1,
            'end_timestep': max_timestep,
            'total_batches': 1,
            'df': action_df
        }]
        all_batch_scores = process_llm_batches(
            batches=batches,
            baseline_risk=baseline_risk,
            real_risk=real_risk,
            num_agents=num_agents,
            max_timestep=max_timestep,
            target_timesteps=target_timesteps if target_timesteps else [max_timestep],
            risk_lambda=risk_lambda,
            config=config,
            include_reason=include_reason
        )
        df_scores = all_batch_scores[0]
        df_scores = normalize_scores(df_scores)
        n_batches = 1
        batch_size = len(action_df)
    
    # 转换为矩阵格式
    print("\n转换为分数矩阵...")
    score_matrix = scores_to_matrix(df_scores, num_agents, max_timestep)
    
    # 计算统计信息
    score_values = score_matrix.flatten()
    score_stats = {
        "mean": float(np.mean(score_values)),
        "std": float(np.std(score_values)),
        "min": float(np.min(score_values)),
        "max": float(np.max(score_values)),
        "sum": float(np.sum(score_values)),
        "median": float(np.median(score_values)),
        "n_batches": n_batches,
        "batch_size": batch_size,
        "use_batch_processing": use_batch
    }
    
    print(f"\n  分数统计:")
    print(f"    Mean: {score_stats['mean']:.6f}")
    print(f"    Std: {score_stats['std']:.6f}")
    print(f"    Min: {score_stats['min']:.6f}")
    print(f"    Max: {score_stats['max']:.6f}")
    print(f"    Median: {score_stats['median']:.6f}")
    if use_batch:
        print(f"    分段信息: {n_batches} 段, 每段 {batch_size} 行")
    
    # 保存结果
    output_dir = model_path_obj / "faithfulness_exp" / "llm"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存numpy数组
    scores_file = output_dir / "llm_scores.npy"
    np.save(scores_file, score_matrix)
    print(f"\n分数矩阵已保存到: {scores_file}")
    
    # 保存统计信息
    stats = {
        "method": "llm",
        "baseline_risk": float(baseline_risk),
        "real_risk": float(real_risk),
        "num_agents": num_agents,
        "max_timestep": max_timestep,
        "episode_length": episode_length,
        "target_timesteps": target_timesteps if target_timesteps else [],
        "score_stats": score_stats,
        "model": config['model'],
        "temperature": config.get('temperature', 0.7)
    }
    
    # 从shapley_stats复制配置参数
    for key in ['metric_name', 'baseline_type', 'baseline_work', 'baseline_consumption',
                'risk_lambda', 'seed', 'inflation_threshold', 'use_metric_directly',
                'risk_aggregation', 'include_both_risks', 'use_probabilistic_baseline']:
        if key in shapley_stats:
            stats[key] = shapley_stats[key]
    
    stats_file = output_dir / "llm_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"统计信息已保存到: {stats_file}")
    
    print("\n" + "="*60)
    print("完成!")
    print("="*60)
    
    return score_matrix, stats


def main():
    parser = argparse.ArgumentParser(
        description='使用LLM as a Judge方法计算baseline分数',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 计算gpt/gpt_42的LLM baseline（使用环境变量中的API key）
  python scripts/faithfulness_exp/compute_llm_baseline.py gpt/gpt_42
  
  # 使用配置文件
  python scripts/faithfulness_exp/compute_llm_baseline.py gpt/gpt_42 --config llm_config.json
        """
    )
    
    parser.add_argument(
        'model_path',
        type=str,
        help='模型路径，格式为 "model/model_id"，如 "gpt/gpt_42"'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='配置文件路径（JSON格式）'
    )
    
    args = parser.parse_args()
    
    try:
        score_matrix, stats = compute_llm_baseline(args.model_path, config_path=args.config)
        return 0
    except Exception as e:
        print(f"\n错误: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
