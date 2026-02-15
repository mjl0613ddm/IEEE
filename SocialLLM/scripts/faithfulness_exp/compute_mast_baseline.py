#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAST Baseline方法：基于LLM-as-a-Judge为所有action生成分数矩阵
使用MAST定义和示例来指导LLM评分，支持分块输入避免token过多
用于faithfulness实验的baseline对比

使用方法:
    python scripts/faithfulness_exp/compute_mast_baseline.py --result_dir results/gpt-4o-mini/gpt-4o-mini_42
    python scripts/faithfulness_exp/compute_mast_baseline.py --result_dir results/gpt-4o-mini/gpt-4o-mini_42 --config config/api.yaml
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
        "timeout": int(os.getenv("OPENAI_TIMEOUT", "120")),
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


def format_action_table(action_df: pd.DataFrame) -> str:
    """将action_table格式化为Markdown表格格式（用于MAST prompt）"""
    # 选择关键列
    columns = ['agent_id', 'timestep', 'posted', 'view_count', 'like_count', 'dislike_count']
    
    # 确保所有列都存在
    available_columns = [col for col in columns if col in action_df.columns]
    df_subset = action_df[available_columns].copy()
    
    # 转换为Markdown表格
    markdown_lines = []
    
    # 表头
    header = "| " + " | ".join(available_columns) + " |"
    markdown_lines.append(header)
    
    # 分隔线
    separator = "| " + " | ".join(["---"] * len(available_columns)) + " |"
    markdown_lines.append(separator)
    
    # 数据行
    for _, row in df_subset.iterrows():
        row_data = [str(row[col]) for col in available_columns]
        row_line = "| " + " | ".join(row_data) + " |"
        markdown_lines.append(row_line)
    
    return "\n".join(markdown_lines)


def load_mast_definitions() -> tuple:
    """加载MAST定义和示例（基于MAST pipeline，适配社交媒体场景）"""
    definitions = """
Multi-Agent System Testing (MAST) Failure Modes (adapted for social media):

1. Coordination Failures: Agents fail to coordinate their actions, leading to suboptimal or unstable system behavior.
2. Resource Contention: Multiple agents compete for limited attention/resources, causing conflicts and inefficiencies.
3. Cascading Failures: A failure in one agent triggers failures in other agents, propagating through the system.
4. Deadlock: Agents wait for each other indefinitely, causing the system to freeze.
5. Race Conditions: The outcome depends on the timing of agent actions, leading to unpredictable behavior.
6. Information Asymmetry: Agents have different information, leading to misaligned decisions.
7. Echo Chamber Formation: Agents only interact with like-minded agents, amplifying polarization.
8. Herding Behavior: Agents follow the actions of others without independent analysis, amplifying polarization.
9. Polarization Cascades: Extreme content triggers cascading polarization effects across the network.
10. Belief Extremism: Agents' beliefs become increasingly extreme due to reinforcement from interactions.
"""
    
    examples = """
Examples of failure modes in social media simulations:

1. Echo Chamber Formation: Agents only view and like posts from agents with similar beliefs, creating isolated communities.
2. Polarization Cascades: An extreme post triggers a chain reaction of extreme responses, rapidly polarizing the network.
3. Herding Behavior: Many agents simultaneously like/dislike the same post, amplifying its impact.
4. Belief Extremism: Agents' beliefs drift to extremes (-1 or +1) due to repeated reinforcement from interactions.
5. Information Asymmetry: Some agents post more frequently or have more influence, skewing the information ecosystem.
"""
    
    return definitions, examples


def mast_action_scorer(action_table: str, baseline_risk: float, real_risk: float,
                       n_agents: int, n_timesteps: int, target_timestep: int,
                       definitions: str = "", examples: str = "") -> str:
    """MAST风格的action评分函数"""
    risk_diff = abs(real_risk - baseline_risk)
    
    prompt = (
        "You are analyzing a social media simulation system. Below is a table of agent actions at each timestep.\n"
        "Analyze each action's contribution to the polarization risk value.\n"
        "\n"
        "Here is the action table:\n"
        f"{action_table}\n"
        "\n"
        "Context information:\n"
        f"- PEAK risk value at target timestep {target_timestep}: {real_risk:.6f}\n"
        f"- Initial baseline risk value: {baseline_risk:.6f}\n"
        f"- Risk increase from baseline to peak: {risk_diff:.6f}\n"
        f"- Number of agents: {n_agents}\n"
        f"- Number of timesteps: {n_timesteps}\n"
        f"- Target timestep (where peak risk occurs): {target_timestep}\n"
        "\n"
        "Task: Score each action's contribution to reaching the PEAK risk value on a scale of 0 to 1, where:\n"
        "- 0 means the action has minimal/no contribution to reaching the peak risk\n"
        "- 1 means the action has maximum contribution to reaching the peak risk\n"
        "- Scores should reflect how much each action contributes to moving the risk from baseline to the PEAK risk value at timestep {target_timestep}\n"
        "- Consider the action's impact on polarization, echo chambers, and potential failure modes\n"
        "\n"
    )
    
    # 添加MAST的definitions和examples作为参考
    if definitions:
        prompt += (
            "To help you understand multi-agent system behaviors and potential issues, "
            "here are definitions of common failure modes in multi-agent systems:\n"
            f"{definitions}\n"
            "\n"
        )
    
    if examples:
        prompt += (
            "Here are some examples of failure modes in social media simulations for reference:\n"
            f"{examples}\n"
            "\n"
        )
    
    prompt += (
        "Output format: You MUST respond with a valid JSON object in the following format:\n"
        "{\n"
        '  "scores": [\n'
        '    {"agent_id": 0, "timestep": 0, "score": 0.5},\n'
        '    {"agent_id": 1, "timestep": 0, "score": 0.3},\n'
        "    ...\n"
        "  ]\n"
        "}\n"
        "\n"
        "Important:\n"
        "- Include ALL (agent_id, timestep) pairs from the table above\n"
        "- Each score must be a number between 0 and 1\n"
        "- Output ONLY the JSON object, no additional text before or after\n"
    )
    
    return prompt


def mast_action_scorer_batch(action_table: str, baseline_risk: float, real_risk: float,
                             n_agents: int, n_timesteps: int, batch_id: int, total_batches: int,
                             start_timestep: int, end_timestep: int, target_timestep: int,
                             definitions: str = "", examples: str = "") -> str:
    """MAST风格的action评分函数（分段版本）"""
    risk_diff = abs(real_risk - baseline_risk)
    
    prompt = (
        "You are analyzing a social media simulation system. Below is a PARTIAL table of agent actions.\n"
        "\n"
        "IMPORTANT: This is batch {} of {}.\n"
        "Timestep range for this batch: {} to {}.\n"
        "Target risk timestep: {}.\n"
        "\n"
        "Analyze each action's contribution to the polarization risk value.\n"
        "\n"
        "Here is the action table for this batch:\n"
        "{}\n"
        "\n"
        "Context information:\n"
        "- PEAK risk value at target timestep {}: {:.6f}\n"
        "- Initial baseline risk value: {:.6f}\n"
        "- Risk increase from baseline to peak: {:.6f}\n"
        "- Number of agents: {}\n"
        "- Number of timesteps: {}\n"
        "- Target timestep (where peak risk occurs): {}\n"
        "\n"
        "Task: Score each action's contribution to reaching the PEAK risk value on a scale of 0 to 1, where:\n"
        "- 0 means the action has minimal/no contribution to reaching the peak risk\n"
        "- 1 means the action has maximum contribution to reaching the peak risk\n"
        "- Scores should reflect how much each action contributes to moving the risk from baseline to the PEAK risk value at timestep {}\n"
        "- Consider the action's impact on polarization, echo chambers, and potential failure modes\n"
        "- IMPORTANT: Only score the (agent_id, timestep) pairs that appear in the table above.\n"
        "\n"
    ).format(
        batch_id,
        total_batches,
        start_timestep,
        end_timestep,
        target_timestep,
        action_table,
        target_timestep,
        real_risk,
        baseline_risk,
        risk_diff,
        n_agents,
        n_timesteps,
        target_timestep,
        target_timestep
    )
    
    # 添加MAST的definitions和examples作为参考
    if definitions:
        prompt += (
            "To help you understand multi-agent system behaviors and potential issues, "
            "here are definitions of common failure modes in multi-agent systems:\n"
            f"{definitions}\n"
            "\n"
        )
    
    if examples:
        prompt += (
            "Here are some examples of failure modes in social media simulations for reference:\n"
            f"{examples}\n"
            "\n"
        )
    
    prompt += (
        "Please output your result in the following JSON format:\n"
        "{\n"
        '  "scores": [\n'
        '    {\n'
        '      "agent_id": <agent_id>,\n'
        '      "timestep": <timestep>,\n'
        '      "score": <float between 0 and 1>\n'
        '    },\n'
        "    ...\n"
        "  ]\n"
        "}\n"
        "\n"
        "Ensure that:\n"
        "- All agent-timestep pairs in THIS BATCH are covered.\n"
        "- Output ONLY the JSON object, no additional text before or after.\n"
    )
    
    return prompt


def parse_mast_response(response_text: str) -> List[Dict]:
    """解析MAST响应，提取分数"""
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
        json_match = re.search(r'\{.*"scores".*\}', response_text, re.DOTALL)
        if json_match:
            try:
                response_data = json.loads(json_match.group())
                if isinstance(response_data, dict) and "scores" in response_data:
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
                    if "agent_id" in obj and "timestep" in obj and "score" in obj:
                        results.append(obj)
                except:
                    continue
            if results:
                return results
        
        raise ValueError(f"无法解析MAST响应: {response_text[:200]}...")


def process_single_mast_batch(batch_info: Dict, baseline_risk: float, real_risk: float,
                              n_agents: int, n_timesteps: int, target_timestep: int, config: Dict,
                              definitions: str, examples: str) -> Tuple[int, pd.DataFrame]:
    """处理单个MAST batch（用于并发处理）"""
    batch_id = batch_info['batch_id']
    total_batches = batch_info['total_batches']
    start_timestep = batch_info['start_timestep']
    end_timestep = batch_info['end_timestep']
    batch_df = batch_info['df']
    
    # 为每个线程创建独立的client（避免线程安全问题）
    client = create_llm_client(config)
    
    try:
        # 格式化该段的action数据
        action_table = format_action_table(batch_df)
        
        # 构建prompt
        prompt = mast_action_scorer_batch(
            action_table=action_table,
            baseline_risk=baseline_risk,
            real_risk=real_risk,
            n_agents=n_agents,
            n_timesteps=n_timesteps,
            batch_id=batch_id,
            total_batches=total_batches,
            start_timestep=start_timestep,
            end_timestep=end_timestep,
            target_timestep=target_timestep,
            definitions=definitions,
            examples=examples
        )
        
        # 调用LLM API
        response_text = call_llm_api(
            client=client,
            model=config["model"],
            prompt=prompt,
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 120)
        )
        
        # 解析响应
        scores = parse_mast_response(response_text)
        
        # 转换为DataFrame
        scores_df = pd.DataFrame(scores)
        
        # 验证列名
        if "score" in scores_df.columns:
            scores_df = scores_df.rename(columns={"score": "mast_value"})
        
        # 确保有agent_id和timestep列
        if "agent_id" not in scores_df.columns or "timestep" not in scores_df.columns:
            raise ValueError("MAST响应缺少agent_id或timestep字段")
        
        # 确保mast_value列存在
        if "mast_value" not in scores_df.columns:
            raise ValueError("MAST响应缺少score/mast_value字段")
        
        return batch_id, scores_df
    
    except Exception as e:
        print(f"  ✗ 批次 {batch_id}/{total_batches} 处理失败: {e}")
        if config.get("verbose", False):
            import traceback
            traceback.print_exc()
        raise


def process_mast_batches(batches: List[Dict], baseline_risk: float, real_risk: float,
                        n_agents: int, n_timesteps: int, target_timestep: int, config: Dict,
                        definitions: str, examples: str) -> List[pd.DataFrame]:
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
                _, scores_df = process_single_mast_batch(
                    batch_info, baseline_risk, real_risk, n_agents, n_timesteps, target_timestep, config,
                    definitions, examples
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
                    process_single_mast_batch,
                    batch_info, baseline_risk, real_risk, n_agents, n_timesteps, target_timestep, config,
                    definitions, examples
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


def compute_mast_baseline(result_dir: Path, config: Dict) -> pd.DataFrame:
    """
    使用MAST方法计算baseline分数
    
    Args:
        result_dir: 结果目录路径
        config: LLM配置字典
    
    Returns:
        DataFrame with columns: agent_id, timestep, mast_value
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
    
    # 加载MAST定义和示例
    definitions, examples = load_mast_definitions()
    
    # 分段处理
    rows_per_batch = config.get("rows_per_batch", 100)
    batches = split_action_df_by_rows(action_df, rows_per_batch)
    
    print(f"\n将数据分为 {len(batches)} 个批次处理（每批最多 {rows_per_batch} 行）...")
    
    # 批量处理
    all_batch_scores = process_mast_batches(
        batches=batches,
        baseline_risk=baseline_risk,
        real_risk=real_risk,
        n_agents=num_agents,
        n_timesteps=max_risk_timestep,
        target_timestep=max_risk_timestep,
        config=config,
        definitions=definitions,
        examples=examples
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
    
    # 合并，保留MAST分数
    merged_df = full_df.merge(combined_df, on=['agent_id', 'timestep'], how='left')
    
    # 填充缺失值（如果MAST没有为某些组合打分，使用0）
    merged_df['mast_value'] = merged_df['mast_value'].fillna(0.0)
    
    # 确保分数在[0, 1]范围内
    merged_df['mast_value'] = merged_df['mast_value'].clip(0.0, 1.0)
    
    # 只保留需要的列
    result_df = merged_df[['agent_id', 'timestep', 'mast_value']].copy()
    
    # 计算统计信息
    score_stats = {
        "mean": float(result_df['mast_value'].mean()),
        "std": float(result_df['mast_value'].std()),
        "min": float(result_df['mast_value'].min()),
        "max": float(result_df['mast_value'].max()),
        "sum": float(result_df['mast_value'].sum()),
        "median": float(result_df['mast_value'].median())
    }
    
    print(f"\nMAST分数统计:")
    print(f"    Mean: {score_stats['mean']:.6f}")
    print(f"    Std: {score_stats['std']:.6f}")
    print(f"    Min: {score_stats['min']:.6f}")
    print(f"    Max: {score_stats['max']:.6f}")
    print(f"    Median: {score_stats['median']:.6f}")
    
    return result_df, score_stats


def main():
    parser = argparse.ArgumentParser(
        description='使用MAST方法计算baseline分数',
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
    output_dir = result_dir / "faithfulness_exp" / "mast"
    output_file = output_dir / "mast_attribution_timeseries.csv"
    stats_file = output_dir / "mast_stats.json"
    
    if args.skip_existing and output_file.exists():
        print(f"跳过已存在的文件: {output_file}")
        return
    
    # 加载配置
    config = load_config(args.config)
    config["rows_per_batch"] = args.rows_per_batch
    config["max_workers"] = args.max_workers
    config["verbose"] = args.verbose
    
    # 计算MAST baseline
    try:
        df, score_stats = compute_mast_baseline(result_dir, config)
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存CSV文件
        df.to_csv(output_file, index=False)
        print(f"\n✓ CSV文件已保存到: {output_file}")
        
        # 保存统计信息
        with open(result_dir / "results.json", 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        
        stats = {
            "method": "mast",
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
        print(f"错误: 计算MAST baseline失败: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
