#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TwinMarket Action特征提取脚本

从TwinMarket结果文件夹中提取每个(user_id, date)对的交易特征，包括：
- 交易笔数
- 买入卖出笔数
- 买入卖出金额
- 买入卖出股票数量
- 风险发生的目标日期

输出CSV和JSON两种格式到 action_table 文件夹。
"""

import os
import sys
import json
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 默认结果目录
DEFAULT_RESULTS_DIR = project_root / "results"


def load_shapley_stats(shapley_dir: Path) -> Optional[Dict]:
    """
    加载shapley_stats.json文件，获取日期范围
    
    Args:
        shapley_dir: shapley文件夹路径
        
    Returns:
        包含shapley统计信息的字典，如果文件不存在则返回None
    """
    # 查找所有shapley_stats文件
    stats_files = list(shapley_dir.glob("shapley_stats_*.json"))
    
    if not stats_files:
        return None
    
    # 如果有多个文件，使用最新的（按文件名排序，取最后一个）
    stats_file = sorted(stats_files)[-1]
    
    try:
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        
        # 从文件名或date_range字段提取日期范围
        if 'date_range' in stats:
            date_range = stats['date_range']
            if '_' in date_range:
                start_date, target_date = date_range.split('_', 1)
                stats['start_date'] = start_date
                stats['target_date'] = target_date
        else:
            # 从文件名提取
            filename = stats_file.stem
            if filename.startswith('shapley_stats_'):
                date_part = filename.replace('shapley_stats_', '')
                if '_' in date_part:
                    start_date, target_date = date_part.rsplit('_', 1)
                    stats['start_date'] = start_date
                    stats['target_date'] = target_date
        
        return stats
    except Exception as e:
        print(f"  错误: 无法加载shapley_stats文件 {stats_file}: {e}")
        return None


def load_trading_decisions(trading_records_dir: Path, date: str) -> Dict:
    """
    加载指定日期的交易决策文件
    
    Args:
        trading_records_dir: trading_records文件夹路径
        date: 日期字符串 (YYYY-MM-DD)
        
    Returns:
        交易决策字典
    """
    json_file = trading_records_dir / f"{date}.json"
    if not json_file.exists():
        return {}
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"  警告: 无法加载交易决策文件 {json_file}: {e}")
        return {}


def extract_user_date_features(
    user_id: str,
    date: str,
    user_data: Dict,
    target_date: str
) -> Optional[Dict]:
    """
    提取单个(user_id, date)的特征
    
    Args:
        user_id: 用户ID
        date: 日期
        user_data: 用户的交易决策数据
        target_date: 目标日期（风险发生的日期）
        
    Returns:
        特征字典，如果用户没有有效交易则返回None
    """
    if not isinstance(user_data, dict) or "stock_decisions" not in user_data:
        return None
    
    stock_decisions = user_data.get("stock_decisions", {})
    if not stock_decisions:
        return None
    
    # 初始化统计变量
    n_transactions = 0
    n_buy = 0
    n_sell = 0
    buy_amount = 0.0
    sell_amount = 0.0
    buy_stocks = set()
    sell_stocks = set()
    
    # 遍历所有股票决策
    for stock_code, stock_info in stock_decisions.items():
        action = stock_info.get("action", "")
        sub_orders = stock_info.get("sub_orders", [])
        
        if action in ["buy", "sell"] and sub_orders:
            # 统计交易笔数（sub_orders的数量）
            n_orders = len(sub_orders)
            n_transactions += n_orders
            
            # 计算金额
            for order in sub_orders:
                quantity = float(order.get("quantity", 0))
                price = float(order.get("price", 0))
                amount = quantity * price
                
                if action == "buy":
                    n_buy += 1
                    buy_amount += amount
                    buy_stocks.add(stock_code)
                elif action == "sell":
                    n_sell += 1
                    sell_amount += amount
                    sell_stocks.add(stock_code)
    
    # 如果没有交易，返回None
    if n_transactions == 0:
        return None
    
    return {
        "user_id": user_id,
        "date": date,
        "n_transactions": n_transactions,
        "n_buy": n_buy,
        "n_sell": n_sell,
        "buy_amount": buy_amount,
        "sell_amount": sell_amount,
        "n_buy_stocks": len(buy_stocks),
        "n_sell_stocks": len(sell_stocks),
        "target_date": target_date
    }


def get_all_dates(trading_records_dir: Path, start_date: str, target_date: str) -> List[str]:
    """
    获取从start_date到target_date的所有交易日
    
    Args:
        trading_records_dir: trading_records文件夹路径
        start_date: 开始日期
        target_date: 目标日期
        
    Returns:
        日期列表（按时间顺序）
    """
    if not trading_records_dir.exists():
        return []
    
    all_files = list(trading_records_dir.glob("*.json"))
    dates = []
    
    for file_path in all_files:
        date_str = file_path.stem  # 获取不带扩展名的文件名
        # 检查是否是日期格式（排除_orders.json文件）
        if len(date_str) == 10 and date_str.startswith('2023-'):
            try:
                # 尝试解析日期
                datetime.strptime(date_str, "%Y-%m-%d")
                if start_date <= date_str <= target_date:
                    dates.append(date_str)
            except (ValueError, TypeError):
                continue
    
    return sorted(dates)


def extract_action_features(simulation_dir: Path, output_dir: Path) -> bool:
    """
    提取单个simulation文件夹的action特征
    
    Args:
        simulation_dir: {model}_{seed}文件夹路径
        output_dir: 输出文件夹路径（action_table）
        
    Returns:
        是否成功提取
    """
    shapley_dir = simulation_dir / "shapley"
    trading_records_dir = simulation_dir / "trading_records"
    
    # 检查必要的文件夹是否存在
    if not shapley_dir.exists():
        print(f"  跳过: shapley文件夹不存在")
        return False
    
    if not trading_records_dir.exists():
        print(f"  跳过: trading_records文件夹不存在")
        return False
    
    # 加载shapley_stats获取日期范围
    stats = load_shapley_stats(shapley_dir)
    if not stats:
        print(f"  跳过: 无法找到或加载shapley_stats文件")
        return False
    
    start_date = stats.get('start_date')
    target_date = stats.get('target_date')
    
    if not start_date or not target_date:
        print(f"  跳过: shapley_stats中缺少日期范围信息")
        return False
    
    print(f"  日期范围: {start_date} 到 {target_date}")
    
    # 获取所有日期
    dates = get_all_dates(trading_records_dir, start_date, target_date)
    if not dates:
        print(f"  警告: 在日期范围内没有找到交易记录文件")
        return False
    
    print(f"  找到 {len(dates)} 个交易日")
    
    # 提取所有特征
    all_features = []
    
    for date in dates:
        decisions = load_trading_decisions(trading_records_dir, date)
        
        for user_id, user_data in decisions.items():
            # 跳过错误信息
            if isinstance(user_data, dict) and "error" in user_data:
                continue
            
            features = extract_user_date_features(
                user_id, date, user_data, target_date
            )
            
            if features:
                all_features.append(features)
    
    if not all_features:
        print(f"  警告: 没有提取到任何特征")
        return False
    
    print(f"  提取了 {len(all_features)} 个(user_id, date)对的特征")
    
    # 创建输出文件夹
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成输出文件名
    output_base = f"action_table_{start_date}_{target_date}"
    csv_file = output_dir / f"{output_base}.csv"
    json_file = output_dir / f"{output_base}.json"
    
    # 保存CSV格式
    if all_features:
        fieldnames = list(all_features[0].keys())
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_features)
        print(f"  CSV已保存: {csv_file}")
    
    # 保存JSON格式
    json_data = {
        "start_date": start_date,
        "target_date": target_date,
        "n_actions": len(all_features),
        "actions": all_features
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"  JSON已保存: {json_file}")
    
    return True


def process_simulation_dir(simulation_dir: Path) -> bool:
    """
    处理单个{model}_{seed}文件夹
    
    Args:
        simulation_dir: {model}_{seed}文件夹路径
        
    Returns:
        是否成功处理
    """
    output_dir = simulation_dir / "action_table"
    return extract_action_features(simulation_dir, output_dir)


def batch_extract(
    results_root: Path = DEFAULT_RESULTS_DIR,
    model_names: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, int]:
    """
    批量提取所有模型文件夹的action特征
    
    Args:
        results_root: 结果根目录路径
        model_names: 要处理的模型名称列表，如果为None则处理所有模型
        verbose: 是否输出详细信息
        
    Returns:
        统计信息字典，包含成功和失败的数量
    """
    if not results_root.exists():
        print(f"错误: 结果目录不存在: {results_root}")
        return {"success": 0, "failed": 0, "skipped": 0}
    
    stats = {"success": 0, "failed": 0, "skipped": 0}
    
    # 获取所有模型文件夹
    if model_names:
        model_dirs = [results_root / model for model in model_names if (results_root / model).exists()]
    else:
        # 获取所有子文件夹（排除一些特殊文件夹）
        exclude_dirs = {"accuracy", "shapley_error_aggregate"}
        model_dirs = [
            d for d in results_root.iterdir()
            if d.is_dir() and d.name not in exclude_dirs
        ]
    
    if not model_dirs:
        print("警告: 没有找到任何模型文件夹")
        return stats
    
    print(f"找到 {len(model_dirs)} 个模型文件夹")
    print("=" * 60)
    
    # 处理每个模型文件夹
    for model_dir in sorted(model_dirs):
        model_name = model_dir.name
        print(f"\n处理模型: {model_name}")
        
        # 获取所有{model}_{seed}子文件夹
        simulation_dirs = [
            d for d in model_dir.iterdir()
            if d.is_dir() and d.name.startswith(model_name + "_")
        ]
        
        if not simulation_dirs:
            print(f"  没有找到{model_name}_*子文件夹")
            stats["skipped"] += 1
            continue
        
        print(f"  找到 {len(simulation_dirs)} 个子文件夹")
        
        # 处理每个子文件夹
        for sim_dir in sorted(simulation_dirs):
            sim_name = sim_dir.name
            print(f"\n  处理: {sim_name}")
            
            try:
                success = process_simulation_dir(sim_dir)
                if success:
                    stats["success"] += 1
                    print(f"  ✓ 成功")
                else:
                    stats["failed"] += 1
                    print(f"  ✗ 失败")
            except Exception as e:
                stats["failed"] += 1
                print(f"  ✗ 错误: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("处理完成!")
    print(f"  成功: {stats['success']}")
    print(f"  失败: {stats['failed']}")
    print(f"  跳过: {stats['skipped']}")
    
    return stats


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='提取TwinMarket结果文件夹中的action特征',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理所有模型
  python scripts/faithfulness_exp/extract_action_features.py
  
  # 处理指定模型
  python scripts/faithfulness_exp/extract_action_features.py --models deepseek-v3.2
  
  # 处理指定结果目录
  python scripts/faithfulness_exp/extract_action_features.py --results_dir /path/to/results
        """
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        default=None,
        help=f'结果根目录路径（默认: {DEFAULT_RESULTS_DIR}）'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='要处理的模型名称列表（如果不指定则处理所有模型）'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='静默模式，减少输出'
    )
    
    args = parser.parse_args()
    
    # 确定结果目录
    if args.results_dir:
        results_root = Path(args.results_dir)
    else:
        results_root = DEFAULT_RESULTS_DIR
    
    # 执行批量提取
    stats = batch_extract(
        results_root=results_root,
        model_names=args.models,
        verbose=not args.quiet
    )
    
    # 返回退出码
    return 0 if stats["failed"] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
