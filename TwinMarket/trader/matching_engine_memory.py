"""
内存版本的交易撮合引擎（不依赖数据库）

该模块实现了不依赖数据库的快速撮合系统，用于Shapley Value计算。
保持与原始撮合系统完全一致的逻辑，但使用内存状态管理替代数据库操作。

核心功能：
1. 使用内存状态（last_prices）替代数据库查询
2. 生成必要的CSV文件供RiskMetricsCalculator使用
3. 完全保持撮合逻辑一致，确保价格计算无误差
"""

import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd

# 从原始matching_engine导入核心函数（这些函数不依赖数据库）
from trader.matching_engine import (
    Order,
    calculate_closing_price,
    create_orders_from_decisions,
    process_daily_orders,
    read_json,
    save_daily_results,
    validate_order_timestamps,
)


def process_trading_day_memory(
    decisions: List[Dict],
    last_prices: Dict[str, float],
    current_date: str,
    output_dir: str,
    json_file_path: str = None,
    verbose: bool = False,
) -> Dict:
    """
    处理单个交易日的完整交易流程（内存版本，不依赖数据库）
    
    该函数是交易日处理的核心协调器，负责将用户决策转换为订单，
    执行撮合交易，并生成必要的CSV文件。
    
    完整处理流程：
    1. 决策转换：将用户交易决策转换为标准订单格式
    2. 时间戳验证：确保所有订单时间戳的唯一性
    3. 订单撮合：执行多股票并行撮合处理
    4. 结果保存：保存交易结果到CSV文件
    
    Args:
        decisions: 用户交易决策列表（已转换格式）
        last_prices: 上一交易日各股票收盘价，格式{stock_code: price}
        current_date: 当前交易日期，格式'YYYY-MM-DD'
        output_dir: 结果输出目录
        json_file_path: 原始决策JSON文件路径（可选，用于生成orders文件）
        
    Returns:
        dict: 每支股票的交易结果，包含收盘价、成交量、交易明细等
        
    Note:
        - 支持多股票并行处理
        - 包含完整的数据验证和异常处理
        - 会自动生成daily_summary和transactions CSV文件
        - 不依赖数据库，所有状态在内存中维护
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 转换决策为订单，并分配随机时间戳
    orders = create_orders_from_decisions(decisions, current_date)
    
    # 2. 验证时间戳是否有重复
    assert validate_order_timestamps(orders), "存在重复时间戳"
    
    # 3. 处理订单（使用内存状态，不依赖数据库）
    results = process_daily_orders(
        orders, last_prices, current_date, output_dir, json_file_path, verbose=verbose
    )
    
    # 4. 保存结果（生成CSV文件）
    save_daily_results(results, current_date, output_dir, verbose=verbose)
    
    return results


def generate_stock_data_memory(
    decisions: List[Dict],
    last_prices: Dict[str, float],
) -> Dict[str, float]:
    """
    从决策和last_prices生成股票数据（内存版本）
    
    Args:
        decisions: 用户交易决策列表
        last_prices: 上一交易日各股票收盘价
        
    Returns:
        股票数据字典 {stock_code: price}
    """
    # 提取唯一的stock_code列表
    stock_codes = {decision["stock_code"] for decision in decisions}
    
    # 从last_prices中提取数据
    stock_data = {}
    for stock_code in stock_codes:
        if stock_code in last_prices:
            stock_data[stock_code] = last_prices[stock_code]
        else:
            # 如果last_prices中没有，使用默认值（应该不会发生）
            stock_data[stock_code] = 100.0
    
    return stock_data


def convert_decisions_from_dict(decisions_dict: Dict) -> List[Dict]:
    """
    将决策字典转换为决策列表（内存版本，不依赖文件）
    
    该函数与read_json的逻辑完全一致，但直接从内存中的字典转换，避免文件I/O。
    
    Args:
        decisions_dict: 决策字典，格式 {user_id: {stock_decisions: {stock_code: {action, sub_orders: [...]}}}}
        
    Returns:
        决策列表，格式 [{user_id, stock_code, direction, amount, target_price}, ...]
    """
    converted_decisions = []
    
    for user_id, user_data in decisions_dict.items():
        # 检查 user_data 是否包含 stock_decisions
        if not user_data or "stock_decisions" not in user_data:
            continue
        
        stock_decisions = user_data["stock_decisions"]
        # 检查 stock_decisions 是否为空
        if not stock_decisions:
            continue
        
        # stock_decisions 是一个字典，键为 stock_code
        for stock_code, stock_info in stock_decisions.items():
            # 检查是否有 sub_orders
            if "sub_orders" not in stock_info or not stock_info["sub_orders"]:
                continue  # 如果没有 sub_orders，跳过
            
            # 遍历 sub_orders
            for sub_order in stock_info["sub_orders"]:
                decision = {
                    "user_id": f"{user_id}_{stock_code}",
                    "stock_code": stock_code,
                    "direction": stock_info["action"],
                    "amount": int(sub_order["quantity"]),
                    "target_price": round(sub_order["price"], 2),
                }
                converted_decisions.append(decision)
    
    return converted_decisions


def process_trading_day_memory_no_csv(
    decisions: List[Dict],
    last_prices: Dict[str, float],
    current_date: str,
    json_file_path: str = None,
    verbose: bool = False,
) -> Dict:
    """
    处理单个交易日的完整交易流程（完全内存版本，不写入任何文件）
    
    这是优化版本，不写入任何文件（包括CSV和JSON），直接在内存中返回结果，
    完全避免文件I/O瓶颈，类似EconAgent的实现方式。
    
    Args:
        decisions: 用户交易决策列表（已转换格式）
        last_prices: 上一交易日各股票收盘价，格式{stock_code: price}
        current_date: 当前交易日期，格式'YYYY-MM-DD'
        json_file_path: 原始决策JSON文件路径（已弃用，保留以兼容）
        verbose: 是否输出详细信息
        
    Returns:
        dict: 每支股票的交易结果，包含收盘价、成交量、交易明细等
    """
    # 1. 转换决策为订单，并分配随机时间戳
    orders = create_orders_from_decisions(decisions, current_date)
    
    # 2. 验证时间戳是否有重复
    assert validate_order_timestamps(orders), "存在重复时间戳"
    
    # 3. 处理订单（完全内存版本，不写入任何文件）
    # 传入None作为json_file_path，避免写入orders JSON文件
    results = process_daily_orders(
        orders, last_prices, current_date, "", None, verbose=verbose
    )
    
    # 不再调用save_daily_results，直接返回结果（避免文件I/O）
    # 这样可以在内存中直接计算风险指标，大幅提升性能
    
    return results


