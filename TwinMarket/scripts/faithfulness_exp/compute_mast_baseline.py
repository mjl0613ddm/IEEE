#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAST Baseline方法：基于LLM-as-a-Judge为所有action生成分数矩阵
使用MAST定义和示例来指导LLM评分
用于faithfulness实验的baseline对比

使用方法:
    python scripts/faithfulness_exp/compute_mast_baseline.py --log_dir gpt-4o-mini_42
    python scripts/faithfulness_exp/compute_mast_baseline.py --log_dir gpt-4o-mini_42 --config mast_config.json
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
            original_line = line
            line = line.rstrip()
            if not line or line.startswith('#'):
                continue
            
            # 检查是否是列表项（以-开头，可能有缩进）
            stripped = line.lstrip()
            if stripped.startswith('-'):
                # 这是一个列表项
                if current_key is not None:
                    if current_key not in config:
                        config[current_key] = []
                    list_value = stripped[1:].strip()
                    config[current_key].append(list_value)
                continue
            
            # 检查是否是键值对
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                if value:
                    # 有值的键值对
                    config[key] = value
                    current_key = None
                else:
                    # 只有键，值可能是列表（下一行开始）
                    current_key = key
                    config[key] = []  # 初始化为空列表
    return config

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 尝试导入openai库（支持新旧版本）
OPENAI_AVAILABLE = False
OPENAI_CLIENT_CLASS = None
OPENAI_LEGACY = False

try:
    # 尝试新版本（>=1.0）的导入方式
    from openai import OpenAI
    OPENAI_CLIENT_CLASS = OpenAI
    OPENAI_AVAILABLE = True
except (ImportError, ModuleNotFoundError, AttributeError):
    try:
        # 尝试旧版本（<1.0）的导入方式
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
        "timeout": int(os.getenv("OPENAI_TIMEOUT", "120"))  # 默认120秒超时
    }
    
    if config_path and Path(config_path).exists():
        config_file = Path(config_path)
        with open(config_file, 'r', encoding='utf-8') as f:
            # 根据文件扩展名判断格式
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                if YAML_AVAILABLE:
                    file_config = yaml.safe_load(f)
                else:
                    # 使用简单的YAML解析器
                    file_config = simple_yaml_load(config_file)
            else:
                # 默认尝试JSON格式
                file_config = json.load(f)
            
            # 处理YAML中api_key可能是列表的情况
            if isinstance(file_config.get("api_key"), list):
                file_config["api_key"] = file_config["api_key"][0] if file_config["api_key"] else ""
            
            # 处理model_name字段（YAML中可能使用model_name而不是model）
            if "model_name" in file_config and "model" not in file_config:
                file_config["model"] = file_config.pop("model_name")
            
            default_config.update(file_config)
    
    # 如果api_key是列表，取第一个元素
    if isinstance(default_config.get("api_key"), list):
        default_config["api_key"] = default_config["api_key"][0] if default_config["api_key"] else ""
    
    if not default_config["api_key"]:
        raise ValueError("API key not found. Please set OPENAI_API_KEY environment variable or provide config file.")
    
    return default_config


def create_llm_client(config):
    """创建LLM客户端（支持新旧版本openai库）"""
    if not OPENAI_AVAILABLE:
        raise ImportError("openai library is required. Install with: pip install openai")
    
    if OPENAI_CLIENT_CLASS:
        # 新版本（>=1.0）使用OpenAI类
        client = OPENAI_CLIENT_CLASS(
            api_key=config["api_key"],
            base_url=config.get("base_url", "https://api.openai.com/v1")
        )
        return client
    elif OPENAI_LEGACY:
        # 旧版本（<1.0）直接设置api_key和api_base
        import openai
        openai.api_key = config["api_key"]
        openai.api_base = config.get("base_url", "https://api.openai.com/v1")
        return openai
    else:
        raise ImportError("openai library is not properly installed or version is incompatible")


def chat_completion_request_openai(client, prompt, model, temperature=1.0, timeout=120, max_retries=3):
    """调用LLM API（支持新旧版本）"""
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    import time
    last_error = None
    
    for attempt in range(max_retries):
        try:
            if OPENAI_LEGACY:
                # 旧版本（<1.0）使用ChatCompletion.create
                import openai
                try:
                    chat_response = openai.ChatCompletion.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        timeout=timeout
                    )
                except TypeError:
                    # 如果不支持timeout参数，则不传递
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


def load_shapley_stats(log_dir: Path) -> Dict:
    """加载shapley stats获取实验参数"""
    shapley_dir = log_dir / "shapley"
    if not shapley_dir.exists():
        raise FileNotFoundError(f"Shapley目录不存在: {shapley_dir}")
    
    stats_files = list(shapley_dir.glob("shapley_stats_*.json"))
    if not stats_files:
        raise FileNotFoundError(f"Shapley stats文件不存在: {shapley_dir}")
    
    stats_file = sorted(stats_files)[-1]
    with open(stats_file, 'r') as f:
        shapley_stats = json.load(f)
    
    # 从date_range提取日期
    date_range = shapley_stats.get('date_range', '')
    if '_' in date_range:
        start_date, target_date = date_range.split('_', 1)
        shapley_stats['start_date'] = start_date
        shapley_stats['target_date'] = target_date
    
    return shapley_stats


def load_action_table(log_dir: Path) -> pd.DataFrame:
    """加载action_table数据"""
    action_table_dir = log_dir / "action_table"
    if not action_table_dir.exists():
        raise FileNotFoundError(f"action_table目录不存在: {action_table_dir}。请先运行extract_action_features.py")
    
    # 查找CSV文件
    csv_files = list(action_table_dir.glob("action_table_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"action_table CSV文件不存在: {action_table_dir}")
    
    csv_file = sorted(csv_files)[-1]
    df = pd.read_csv(csv_file)
    
    return df


def split_action_df_by_rows(action_df: pd.DataFrame, rows_per_batch: int = 100) -> List[Dict]:
    """按行数分段，每段最多rows_per_batch行"""
    batches = []
    total_rows = len(action_df)
    total_batches = (total_rows + rows_per_batch - 1) // rows_per_batch
    
    for i in range(0, total_rows, rows_per_batch):
        batch_df = action_df.iloc[i:i+rows_per_batch].copy()
        # 计算该段的日期范围
        dates = sorted(batch_df['date'].unique())
        batch_info = {
            'batch_id': len(batches) + 1,
            'start_date': dates[0] if dates else None,
            'end_date': dates[-1] if dates else None,
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


def format_action_table(action_df: pd.DataFrame) -> str:
    """将action_table格式化为Markdown表格格式（用于MAST prompt）"""
    # 选择关键列
    columns = ['user_id', 'date', 'n_transactions', 'n_buy', 'n_sell', 
               'buy_amount', 'sell_amount', 'n_buy_stocks', 'n_sell_stocks', 'target_date']
    
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
    """加载MAST定义和示例（基于MAST pipeline）"""
    definitions = """
Multi-Agent System Testing (MAST) Failure Modes:

1. Coordination Failures: Agents fail to coordinate their actions, leading to suboptimal or unstable system behavior.
2. Resource Contention: Multiple agents compete for limited resources, causing conflicts and inefficiencies.
3. Cascading Failures: A failure in one agent triggers failures in other agents, propagating through the system.
4. Deadlock: Agents wait for each other indefinitely, causing the system to freeze.
5. Race Conditions: The outcome depends on the timing of agent actions, leading to unpredictable behavior.
6. Information Asymmetry: Agents have different information, leading to misaligned decisions.
7. Market Manipulation: Agents exploit system mechanisms for personal gain at the expense of system stability.
8. Herding Behavior: Agents follow the actions of others without independent analysis, amplifying market volatility.
9. Liquidity Crises: Sudden withdrawal of trading activity causes price instability.
10. Price Bubbles: Unsustainable price increases driven by speculative behavior rather than fundamentals.
"""
    
    examples = """
Examples of failure modes in stock market simulations:

1. Herding Behavior: Many users simultaneously buy the same stock, causing price to spike unrealistically.
2. Market Manipulation: A user places large buy orders to inflate price, then sells at profit.
3. Cascading Failures: A price drop triggers stop-loss orders, causing further price drops.
4. Liquidity Crisis: Most users stop trading, causing large price swings from small transactions.
5. Information Asymmetry: Some users have early access to information, making profitable trades before others.
"""
    
    return definitions, examples


def mast_action_scorer(action_table: str, baseline_risk: float, real_risk: float,
                       n_users: int, n_dates: int, definitions: str = "", examples: str = "") -> str:
    """MAST风格的action评分函数"""
    risk_diff = abs(real_risk - baseline_risk)
    
    prompt = (
        "You are analyzing a stock market simulation system. Below is a table of user trading actions at each date.\n"
        "Analyze each action's contribution to the risk value.\n"
        "\n"
        "Here is the action table:\n"
        f"{action_table}\n"
        "\n"
        "Context information:\n"
        f"- Target date risk value: {real_risk:.6f}\n"
        f"- Baseline risk value: {baseline_risk:.6f}\n"
        f"- Risk difference: {risk_diff:.6f}\n"
        f"- Number of users: {n_users}\n"
        f"- Number of dates: {n_dates}\n"
        "\n"
        "Task: Score each action's contribution to the risk value on a scale of 0 to 1, where:\n"
        "- 0 means the action has minimal/no contribution to the risk\n"
        "- 1 means the action has maximum contribution to the risk\n"
        "- Scores should reflect how much each action contributes to moving the risk from baseline to the target risk value\n"
        "- Consider the action's impact on market behavior, coordination, and potential failure modes\n"
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
            "Here are some examples of failure modes in stock market simulations for reference:\n"
            f"{examples}\n"
            "\n"
        )
    
    prompt += (
        "Output format: You MUST respond with a valid JSON object in the following format:\n"
        "{\n"
        '  "scores": [\n'
        '    {"user_id": "0", "date": "2023-06-19", "score": 0.5},\n'
        '    {"user_id": "1", "date": "2023-06-19", "score": 0.3},\n'
        "    ...\n"
        "  ]\n"
        "}\n"
        "\n"
        "Important:\n"
        "- Include ALL (user_id, date) pairs from the table above\n"
        "- Each score must be a number between 0 and 1\n"
        "- Output ONLY the JSON object, no additional text before or after\n"
    )
    
    return prompt


def mast_action_scorer_batch(action_table: str, baseline_risk: float, real_risk: float,
                             n_users: int, n_dates: int, batch_id: int, total_batches: int,
                             start_date: str, end_date: str, target_date: str,
                             definitions: str = "", examples: str = "") -> str:
    """MAST风格的action评分函数（分段版本）"""
    risk_diff = abs(real_risk - baseline_risk)
    
    prompt = (
        "You are analyzing a stock market simulation system. Below is a PARTIAL table of user trading actions.\n"
        "\n"
        "IMPORTANT: This is batch {} of {}.\n"
        "Date range for this batch: {} to {}.\n"
        "Target risk date: {}.\n"
        "\n"
        "Analyze each action's contribution to the risk value.\n"
        "\n"
        "Here is the action table for this batch:\n"
        "{}\n"
        "\n"
        "Context information:\n"
        "- Target date risk value: {:.6f}\n"
        "- Baseline risk value: {:.6f}\n"
        "- Risk difference: {:.6f}\n"
        "- Number of users: {}\n"
        "- Number of dates: {}\n"
        "\n"
        "Task: Score each action's contribution to the risk value on a scale of 0 to 1, where:\n"
        "- 0 means the action has minimal/no contribution to the risk\n"
        "- 1 means the action has maximum contribution to the risk\n"
        "- Scores should reflect how much each action contributes to moving the risk from baseline to the target risk value\n"
        "- Consider the action's impact on market behavior, coordination, and potential failure modes\n"
        "- IMPORTANT: Only score the (user_id, date) pairs that appear in the table above.\n"
        "\n"
    ).format(
        batch_id,
        total_batches,
        start_date,
        end_date,
        target_date,
        action_table,
        real_risk,
        baseline_risk,
        risk_diff,
        n_users,
        n_dates
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
            "Here are some examples of failure modes in stock market simulations for reference:\n"
            f"{examples}\n"
            "\n"
        )
    
    prompt += (
        "Please output your result in the following JSON format:\n"
        "{\n"
        '  "scores": [\n'
        '    {\n'
        '      "user_id": "<user_id>",\n'
        '      "date": "<date>",\n'
        '      "score": <float between 0 and 1>\n'
        '    },\n'
        "    ...\n"
        "  ]\n"
        "}\n"
        "\n"
        "Ensure that:\n"
        "- All user-date pairs in THIS BATCH are covered.\n"
        "- Output ONLY the JSON object, no additional text before or after.\n"
    )
    
    return prompt


def process_mast_batches(batches: List[Dict], baseline_risk: float, real_risk: float,
                        n_users: int, n_dates: int, target_date: str, config: Dict,
                        definitions: str, examples: str) -> List[pd.DataFrame]:
    """批量处理所有段，返回每段的分数DataFrame列表"""
    all_batch_scores = []
    client = create_llm_client(config)
    
    for batch_info in batches:
        batch_id = batch_info['batch_id']
        total_batches = batch_info['total_batches']
        start_date = batch_info['start_date']
        end_date = batch_info['end_date']
        batch_df = batch_info['df']
        
        print(f"\n处理批次 {batch_id}/{total_batches} (日期: {start_date} 到 {end_date}, {len(batch_df)} 行)...")
        
        try:
            # 格式化该段的action数据
            action_table = format_action_table(batch_df)
            
            # 构建prompt
            prompt = mast_action_scorer_batch(
                action_table=action_table,
                baseline_risk=baseline_risk,
                real_risk=real_risk,
                n_users=n_users,
                n_dates=n_dates,
                batch_id=batch_id,
                total_batches=total_batches,
                start_date=start_date,
                end_date=end_date,
                target_date=target_date,
                definitions=definitions,
                examples=examples
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
            parsed_data = parse_llm_response(response_text)
            
            # 提取分数
            scores_dict = {}
            if "scores" in parsed_data:
                for item in parsed_data["scores"]:
                    user_id = str(item.get("user_id", ""))
                    date = str(item.get("date", ""))
                    score = float(item.get("score", 0.0))
                    # 确保分数在[0, 1]范围内
                    score = max(0.0, min(1.0, score))
                    scores_dict[(user_id, date)] = score
            
            print(f"  解析到 {len(scores_dict)} 个分数")
            
            # 创建DataFrame（确保包含该段的所有action）
            data = []
            for _, row in batch_df.iterrows():
                user_id = str(row["user_id"])
                date = str(row["date"])
                key = (user_id, date)
                if key in scores_dict:
                    score = scores_dict[key]
                else:
                    # 如果LLM没有返回该action的分数，使用0.0
                    print(f"    警告: LLM未返回 ({user_id}, {date}) 的分数，使用0.0")
                    score = 0.0
                data.append({
                    'user_id': user_id,
                    'date': date,
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
                    'user_id': str(row["user_id"]),
                    'date': str(row["date"]),
                    'score': 0.0
                })
            all_batch_scores.append(pd.DataFrame(data))
    
    return all_batch_scores


def parse_llm_response(response_text: str) -> Dict:
    """解析LLM返回的JSON响应"""
    # 尝试提取JSON（可能包含markdown代码块）
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # 尝试直接查找JSON对象
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            raise ValueError("Could not find JSON in LLM response")
    
    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Attempted to parse: {json_str[:500]}")
        raise


def compute_mast_baseline(log_dir: Path, config: Dict = None, config_path: str = None) -> Tuple[pd.DataFrame, Dict]:
    """使用MAST方法计算baseline分数"""
    print(f"处理: {log_dir.name}")
    print("="*60)
    
    # 加载配置
    if config is None:
        config = load_config(config_path)
    
    print(f"使用模型: {config['model']}")
    
    # 加载数据
    print("\n加载数据...")
    shapley_stats = load_shapley_stats(log_dir)
    action_df = load_action_table(log_dir)
    
    start_date = shapley_stats.get('start_date')
    target_date = shapley_stats.get('target_date')
    baseline_risk = shapley_stats.get('baseline_metric', 0.0)
    real_risk = shapley_stats.get('real_metric', 0.0)
    n_users = shapley_stats.get('n_users', 20)
    n_dates = shapley_stats.get('n_dates', 32)
    
    if not start_date or not target_date:
        raise ValueError("shapley_stats中缺少日期范围信息")
    
    print(f"  日期范围: {start_date} 到 {target_date}")
    print(f"  Baseline risk: {baseline_risk:.6f}")
    print(f"  Real risk: {real_risk:.6f}")
    print(f"  Action table rows: {len(action_df)}")
    
    # 加载MAST的definitions和examples
    print("\n加载MAST评估表（definitions和examples）...")
    definitions, examples = load_mast_definitions()
    if definitions:
        print(f"  已加载definitions ({len(definitions)} chars)")
    if examples:
        print(f"  已加载examples ({len(examples)} chars)")
    
    # 决定是否使用分段处理
    rows_per_batch = config.get('rows_per_batch', 100)
    use_batch = len(action_df) > rows_per_batch
    
    if use_batch:
        print(f"\n使用分段处理模式 (每段 {rows_per_batch} 行)...")
        # 分段处理
        batches = split_action_df_by_rows(action_df, rows_per_batch=rows_per_batch)
        print(f"  总共分为 {len(batches)} 段")
        
        # 批量处理所有段
        all_batch_scores = process_mast_batches(
            batches=batches,
            baseline_risk=baseline_risk,
            real_risk=real_risk,
            n_users=n_users,
            n_dates=n_dates,
            target_date=target_date,
            config=config,
            definitions=definitions,
            examples=examples
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
        # 原有的一次性处理逻辑（作为fallback）
        action_table = format_action_table(action_df)
        table_size = len(action_table)
        table_size_kb = table_size / 1024
        print(f"  表格大小: {table_size:,} 字符 ({table_size_kb:.1f} KB)")
        
        # 构建prompt（使用MAST风格的evaluator函数）
        print("构建LLM prompt（基于MAST pipeline，包含评估表）...")
        prompt = mast_action_scorer(action_table, baseline_risk, real_risk, n_users, n_dates, definitions, examples)
        
        prompt_size = len(prompt)
        prompt_size_kb = prompt_size / 1024
        prompt_size_mb = prompt_size_kb / 1024
        print(f"  Prompt大小: {prompt_size:,} 字符 ({prompt_size_kb:.1f} KB, {prompt_size_mb:.2f} MB)")
        
        # 根据prompt大小调整超时时间
        estimated_timeout = max(60, int(prompt_size_mb * 60) + 30)
        actual_timeout = config.get('timeout', estimated_timeout)
        if actual_timeout < estimated_timeout:
            print(f"  警告: 建议超时时间至少 {estimated_timeout}秒（基于prompt大小）")
            print(f"  当前超时设置: {actual_timeout}秒，可能不够")
        
        # 调用LLM
        print("\n调用LLM API...")
        print(f"  超时设置: {actual_timeout}秒")
        
        client = create_llm_client(config)
        
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
        print("\n解析LLM响应...")
        parsed_data = parse_llm_response(response_text)
        
        # 提取分数
        scores_dict = {}
        if "scores" in parsed_data:
            for item in parsed_data["scores"]:
                user_id = str(item.get("user_id", ""))
                date = str(item.get("date", ""))
                score = float(item.get("score", 0.0))
                # 确保分数在[0, 1]范围内
                score = max(0.0, min(1.0, score))
                scores_dict[(user_id, date)] = score
        
        print(f"  解析到 {len(scores_dict)} 个分数")
        
        # 创建DataFrame（确保包含所有action）
        data = []
        for _, row in action_df.iterrows():
            user_id = str(row["user_id"])
            date = str(row["date"])
            key = (user_id, date)
            if key in scores_dict:
                score = scores_dict[key]
            else:
                # 如果LLM没有返回该action的分数，使用0.0
                print(f"  警告: LLM未返回 ({user_id}, {date}) 的分数，使用0.0")
                score = 0.0
            data.append({
                'user_id': user_id,
                'date': date,
                'score': score
            })
        
        df_scores = pd.DataFrame(data)
        n_batches = 1
        batch_size = len(action_df)
    
    # 计算统计信息
    score_values = df_scores['score'].values
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
    output_dir = log_dir / "faithfulness_exp" / "mast"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存CSV文件
    csv_file = output_dir / f"mast_attribution_timeseries_{start_date}_{target_date}.csv"
    df_scores.to_csv(csv_file, index=False)
    print(f"\nCSV文件已保存到: {csv_file}")
    
    # 保存统计信息
    stats = {
        "method": "mast",
        "date_range": f"{start_date}_{target_date}",
        "start_date": start_date,
        "target_date": target_date,
        "baseline_metric": float(baseline_risk),
        "real_metric": float(real_risk),
        "n_actions": len(df_scores),
        "score_stats": score_stats,
        "model": config['model'],
        "temperature": config.get('temperature', 0.7)
    }
    
    # 从shapley_stats复制配置参数
    for key in ['metric_name', 'baseline_type', 'n_users', 'n_dates']:
        if key in shapley_stats:
            stats[key] = shapley_stats[key]
    
    stats_file = output_dir / f"mast_stats_{start_date}_{target_date}.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"统计信息已保存到: {stats_file}")
    
    print("\n" + "="*60)
    print("完成!")
    print("="*60)
    
    return df_scores, stats
        

def main():
    parser = argparse.ArgumentParser(
        description='使用MAST方法计算baseline分数',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 计算gpt-4o-mini_42的MAST baseline（使用环境变量中的API key）
  python scripts/faithfulness_exp/compute_mast_baseline.py --log_dir gpt-4o-mini_42
  
  # 使用配置文件
  python scripts/faithfulness_exp/compute_mast_baseline.py --log_dir gpt-4o-mini_42 --config mast_config.json
        """
    )
    
    parser.add_argument(
        '--log_dir',
        type=str,
        required=True,
        help='结果目录名称（如gpt-4o-mini_42）'
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        default=None,
        help='结果根目录路径（默认：项目根目录/results）'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='配置文件路径（JSON格式）'
    )
    
    args = parser.parse_args()
    
    # 确定结果目录
    if args.results_dir:
        results_root = Path(args.results_dir)
    else:
        results_root = project_root / "results"
    
    # 查找log_dir（可能在多个模型文件夹下）
    log_dir = None
    for model_dir in results_root.iterdir():
        if model_dir.is_dir():
            potential_log_dir = model_dir / args.log_dir
            if potential_log_dir.exists():
                log_dir = potential_log_dir
                break
    
    if log_dir is None:
        raise FileNotFoundError(f"找不到结果目录: {args.log_dir}")
    
    try:
        df_scores, stats = compute_mast_baseline(log_dir, config_path=args.config)
        return 0
    except Exception as e:
        print(f"\n错误: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
