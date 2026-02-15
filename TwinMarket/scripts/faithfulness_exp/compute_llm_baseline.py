#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM as a Judge Baseline方法：使用LLM分析TwinMarket交易数据，为每个(user_id, date)对分配风险归因分数
用于faithfulness实验的baseline对比

使用方法:
    python scripts/faithfulness_exp/compute_llm_baseline.py --log_dir gpt-4o-mini_42
    python scripts/faithfulness_exp/compute_llm_baseline.py --log_dir gpt-4o-mini_42 --config llm_config.json
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
    sys.exit(130)  # 130 是 SIGINT (Ctrl+C) 的标准退出码

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
        "timeout": int(os.getenv("OPENAI_TIMEOUT", "300"))  # 默认300秒超时
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


def chat_completion_request_openai(client, prompt, model, temperature=1.0, timeout=300, max_retries=3):
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


def format_action_table_for_llm(action_df: pd.DataFrame) -> str:
    """将action_table格式化为LLM可理解的JSON格式"""
    log_entries = []
    
    for _, row in action_df.iterrows():
        entry = {
            "user_id": str(row["user_id"]),
            "date": str(row["date"]),
            "n_transactions": int(row.get("n_transactions", 0)),
            "n_buy": int(row.get("n_buy", 0)),
            "n_sell": int(row.get("n_sell", 0)),
            "buy_amount": float(row.get("buy_amount", 0.0)),
            "sell_amount": float(row.get("sell_amount", 0.0)),
            "n_buy_stocks": int(row.get("n_buy_stocks", 0)),
            "n_sell_stocks": int(row.get("n_sell_stocks", 0)),
            "target_date": str(row.get("target_date", ""))
        }
        log_entries.append(entry)
    
    # 按date和user_id排序
    log_entries.sort(key=lambda x: (x["date"], int(x["user_id"])))
    
    return json.dumps(log_entries, indent=2)


def build_llm_judge_prompt(action_log: str, baseline_risk: float, real_risk: float, 
                           n_users: int, n_dates: int, include_reason: bool = True) -> str:
    """构建LLM as a Judge的prompt"""
    
    risk_diff = abs(real_risk - baseline_risk)
    
    if include_reason:
        output_format = (
            "[\n"
            '  {\n'
            '    "user_id": "<user_id>",\n'
            '    "date": "<date>",\n'
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
            '    "user_id": "<user_id>",\n'
            '    "date": "<date>",\n'
            '    "risk_score": <float between 0 and 1>\n'
            '  },\n'
            "  ...\n"
            "]\n"
        )
        output_note = ""
    
    prompt = (
        "You are an AI assistant tasked with analyzing a stock market simulation trajectory.\n"
        "\n"
        "The simulation consists of {} users trading stocks over {} days.\n"
        "At each day, each user makes trading decisions (buy/sell orders),\n"
        "which jointly influence the market price dynamics and systemic risk.\n"
        "\n"
        "The problem is:\n"
        "The market system exhibits a risk indicator that is significantly higher than a baseline system. "
        "The current system risk is {:.6f}, while the baseline risk is {:.6f}. "
        "The risk difference is {:.6f}.\n"
        "\n"
        "The risk indicator measures the conditional variance of inflation forecast errors,\n"
        "reflecting systemic economic risk based on market price volatility.\n"
        "\n"
        "Risk computation:\n"
        "1. Market average price is calculated from stock prices\n"
        "2. Log return: π_t = log(P_t) - log(P_{{t-1}})\n"
        "3. Expected return (naive expectation): E_{{t-1}}[π_t] = π_{{t-1}}\n"
        "4. Forecast error: e_t = π_t - E_{{t-1}}[π_t]\n"
        "5. Risk indicator: h_t = λ · h_{{t-1}} + (1 - λ) · e_{{t-1}}², where λ = 0.94\n"
        "\n"
        "Ground truth:\n"
        "The baseline system risk trajectory h_t^baseline is provided as the reference.\n"
        "The current system produces a much higher risk level than the baseline.\n"
        "\n"
        "Your task:\n"
        "Analyze the system trajectory and attribute the excessive risk to individual users' actions\n"
        "across time.\n"
        "\n"
        "You are given the system execution trace in chronological order.\n"
        "Each entry represents one user's trading activity at a specific date, with the following fields:\n"
        "- user_id: The identifier of the user\n"
        "- date: The trading date (YYYY-MM-DD)\n"
        "- n_transactions: Total number of buy and sell sub-orders\n"
        "- n_buy: Number of buy sub-orders\n"
        "- n_sell: Number of sell sub-orders\n"
        "- buy_amount: Total buy quantity * price\n"
        "- sell_amount: Total sell quantity * price\n"
        "- n_buy_stocks: Number of unique stocks bought\n"
        "- n_sell_stocks: Number of unique stocks sold\n"
        "- target_date: The date when risk occurred\n"
        "\n"
        "Here is the system trajectory:\n"
        "{}\n"
        "\n"
        "Based on this trajectory, please predict the following:\n"
        "\n"
        "For EACH user at EACH date, assign a risk attribution score in the range [0, 1],\n"
        "indicating how much that specific user's trading actions at that date contribute to\n"
        "the elevated system risk relative to the baseline.\n"
        "\n"
        "Scoring principles:\n"
        "- A higher score indicates a stronger causal contribution to increased price volatility\n"
        "  and higher conditional variance h_t.\n"
        "- Scores should reflect both direct and indirect effects of users' trading actions on market instability.\n"
        "- If an action is largely irrelevant to the increased risk, assign a score close to 0.\n"
        "- If an action is a major driver of risk escalation, assign a score close to 1.\n"
        "- Scores should be comparable across users and dates.\n"
        "- Consider temporal causality: earlier actions can influence later risk, so actions at earlier dates\n"
        "  may have higher scores if they trigger cascading effects.\n"
        "\n"
        "Please output your result in the following JSON format:\n"
        "{}\n"
        "\n"
        "Ensure that:\n"
        "- All user-date pairs in the trajectory are covered.\n"
        "- The attribution reflects temporal causality (earlier actions can influence later risk).\n"
        "{}"
        "- Output ONLY the JSON array, no additional text before or after.\n"
    ).format(
        n_users,
        n_dates,
        real_risk,
        baseline_risk,
        risk_diff,
        action_log,
        output_format,
        output_note
    )
    
    return prompt


def build_llm_judge_prompt_batch(action_log: str, baseline_risk: float, real_risk: float, 
                                n_users: int, n_dates: int, batch_id: int, total_batches: int,
                                start_date: str, end_date: str, target_date: str,
                                include_reason: bool = True) -> str:
    """构建LLM as a Judge的prompt（分段版本）"""
    
    risk_diff = abs(real_risk - baseline_risk)
    
    if include_reason:
        output_format = (
            "[\n"
            '  {\n'
            '    "user_id": "<user_id>",\n'
            '    "date": "<date>",\n'
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
            '    "user_id": "<user_id>",\n'
            '    "date": "<date>",\n'
            '    "risk_score": <float between 0 and 1>\n'
            '  },\n'
            "  ...\n"
            "]\n"
        )
        output_note = ""
    
    prompt = (
        "You are an AI assistant tasked with analyzing a stock market simulation trajectory.\n"
        "\n"
        "IMPORTANT: This is batch {} of {}.\n"
        "Date range for this batch: {} to {}.\n"
        "Target risk date: {}.\n"
        "\n"
        "The simulation consists of {} users trading stocks over {} days.\n"
        "At each day, each user makes trading decisions (buy/sell orders),\n"
        "which jointly influence the market price dynamics and systemic risk.\n"
        "\n"
        "The problem is:\n"
        "The market system exhibits a risk indicator that is significantly higher than a baseline system. "
        "The current system risk is {:.6f}, while the baseline risk is {:.6f}. "
        "The risk difference is {:.6f}.\n"
        "\n"
        "The risk indicator measures the conditional variance of inflation forecast errors,\n"
        "reflecting systemic economic risk based on market price volatility.\n"
        "\n"
        "Risk computation:\n"
        "1. Market average price is calculated from stock prices\n"
        "2. Log return: π_t = log(P_t) - log(P_{{t-1}})\n"
        "3. Expected return (naive expectation): E_{{t-1}}[π_t] = π_{{t-1}}\n"
        "4. Forecast error: e_t = π_t - E_{{t-1}}[π_t]\n"
        "5. Risk indicator: h_t = λ · h_{{t-1}} + (1 - λ) · e_{{t-1}}², where λ = 0.94\n"
        "\n"
        "Ground truth:\n"
        "The baseline system risk trajectory h_t^baseline is provided as the reference.\n"
        "The current system produces a much higher risk level than the baseline.\n"
        "\n"
        "Your task:\n"
        "Analyze the system trajectory and attribute the excessive risk to individual users' actions\n"
        "across time.\n"
        "\n"
        "You are given a PARTIAL system execution trace (batch {} of {}) in chronological order.\n"
        "Each entry represents one user's trading activity at a specific date, with the following fields:\n"
        "- user_id: The identifier of the user\n"
        "- date: The trading date (YYYY-MM-DD)\n"
        "- n_transactions: Total number of buy and sell sub-orders\n"
        "- n_buy: Number of buy sub-orders\n"
        "- n_sell: Number of sell sub-orders\n"
        "- buy_amount: Total buy quantity * price\n"
        "- sell_amount: Total sell quantity * price\n"
        "- n_buy_stocks: Number of unique stocks bought\n"
        "- n_sell_stocks: Number of unique stocks sold\n"
        "- target_date: The date when risk occurred\n"
        "\n"
        "Here is the system trajectory for this batch:\n"
        "{}\n"
        "\n"
        "Based on this trajectory, please predict the following:\n"
        "\n"
        "For EACH user at EACH date in THIS BATCH ONLY, assign a risk attribution score in the range [0, 1],\n"
        "indicating how much that specific user's trading actions at that date contribute to\n"
        "the elevated system risk relative to the baseline.\n"
        "\n"
        "Scoring principles:\n"
        "- A higher score indicates a stronger causal contribution to increased price volatility\n"
        "  and higher conditional variance h_t.\n"
        "- Scores should reflect both direct and indirect effects of users' trading actions on market instability.\n"
        "- If an action is largely irrelevant to the increased risk, assign a score close to 0.\n"
        "- If an action is a major driver of risk escalation, assign a score close to 1.\n"
        "- Scores should be comparable across users and dates.\n"
        "- Consider temporal causality: earlier actions can influence later risk, so actions at earlier dates\n"
        "  may have higher scores if they trigger cascading effects.\n"
        "- IMPORTANT: Only score the (user_id, date) pairs that appear in the trajectory above.\n"
        "\n"
        "Please output your result in the following JSON format:\n"
        "{}\n"
        "\n"
        "Ensure that:\n"
        "- All user-date pairs in THIS BATCH are covered.\n"
        "- The attribution reflects temporal causality (earlier actions can influence later risk).\n"
        "{}"
        "- Output ONLY the JSON array, no additional text before or after.\n"
    ).format(
        batch_id,
        total_batches,
        start_date,
        end_date,
        target_date,
        n_users,
        n_dates,
        real_risk,
        baseline_risk,
        risk_diff,
        batch_id,
        total_batches,
        action_log,
        output_format,
        output_note
    )
    
    return prompt


def process_llm_batches(batches: List[Dict], baseline_risk: float, real_risk: float,
                       n_users: int, n_dates: int, target_date: str, config: Dict,
                       include_reason: bool = True) -> List[pd.DataFrame]:
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
            action_log = format_action_table_for_llm(batch_df)
            
            # 构建prompt
            prompt = build_llm_judge_prompt_batch(
                action_log=action_log,
                baseline_risk=baseline_risk,
                real_risk=real_risk,
                n_users=n_users,
                n_dates=n_dates,
                batch_id=batch_id,
                total_batches=total_batches,
                start_date=start_date,
                end_date=end_date,
                target_date=target_date,
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
                user_id = str(item.get("user_id", ""))
                date = str(item.get("date", ""))
                score = float(item.get("risk_score", item.get("score", 0.0)))
                # 确保分数在[0, 1]范围内
                score = max(0.0, min(1.0, score))
                scores_dict[(user_id, date)] = score
            
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


def compute_llm_baseline(log_dir: Path, config: Dict = None, config_path: str = None) -> Tuple[pd.DataFrame, Dict]:
    """使用LLM as a Judge方法计算baseline分数"""
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
            n_users=n_users,
            n_dates=n_dates,
            target_date=target_date,
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
        # 原有的一次性处理逻辑（作为fallback）
        action_log = format_action_table_for_llm(action_df)
        log_size = len(action_log)
        log_size_kb = log_size / 1024
        log_size_mb = log_size_kb / 1024
        print(f"  系统轨迹长度: {log_size:,} 字符 ({log_size_kb:.1f} KB, {log_size_mb:.2f} MB)")
        
        # 构建prompt
        print("构建LLM prompt...")
        prompt = build_llm_judge_prompt(action_log, baseline_risk, real_risk, n_users, n_dates, include_reason=include_reason)
        
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
        parsed_scores = parse_llm_response(response_text)
        print(f"  解析到 {len(parsed_scores)} 个分数")
        
        # 转换为DataFrame
        scores_dict = {}
        for item in parsed_scores:
            user_id = str(item.get("user_id", ""))
            date = str(item.get("date", ""))
            score = float(item.get("risk_score", item.get("score", 0.0)))
            # 确保分数在[0, 1]范围内
            score = max(0.0, min(1.0, score))
            scores_dict[(user_id, date)] = score
        
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
    output_dir = log_dir / "faithfulness_exp" / "llm"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存CSV文件
    csv_file = output_dir / f"llm_attribution_timeseries_{start_date}_{target_date}.csv"
    df_scores.to_csv(csv_file, index=False)
    print(f"\nCSV文件已保存到: {csv_file}")
    
    # 保存统计信息
    stats = {
        "method": "llm",
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
    
    stats_file = output_dir / f"llm_stats_{start_date}_{target_date}.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"统计信息已保存到: {stats_file}")
    
    print("\n" + "="*60)
    print("完成!")
    print("="*60)
    
    return df_scores, stats


def main():
    parser = argparse.ArgumentParser(
        description='使用LLM as a Judge方法计算baseline分数',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 计算gpt-4o-mini_42的LLM baseline（使用环境变量中的API key）
  python scripts/faithfulness_exp/compute_llm_baseline.py --log_dir gpt-4o-mini_42
  
  # 使用配置文件
  python scripts/faithfulness_exp/compute_llm_baseline.py --log_dir gpt-4o-mini_42 --config llm_config.json
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
        df_scores, stats = compute_llm_baseline(log_dir, config_path=args.config)
        return 0
    except Exception as e:
        print(f"\n错误: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
