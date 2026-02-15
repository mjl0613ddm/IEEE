#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据过滤脚本：根据risk_indicator_naive的风险最高点位置标记文件夹

该脚本处理 ACL24-EconAgent/datas 目录下的数据：
1. 计算每个子文件夹的risk_indicator_naive
2. 如果最高风险点在步骤14及之前（timestep <= 14），给文件夹添加_rm后缀
3. 汇总未被标记文件夹的统计信息
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


def calculate_risk_indicator(df, method='naive', lambda_param=0.94):
    """
    计算风险指标（基于Engle (1982)和Bollerslev (1986)）
    复用 plot_world_metrics.py 中的逻辑
    
    Args:
        df: 包含price列的DataFrame
        method: 'naive' 表示预期规则（固定使用naive方法）
        lambda_param: RiskMetrics参数λ，默认0.94
    
    Returns:
        (timestep, risk_values) 元组，如果出错返回None
    """
    if 'price' not in df.columns:
        print(f"  Warning: CSV文件中缺少 'price' 列，无法计算风险指标")
        return None
    
    # 计算通胀率 π_t = log P_t - log P_{t-1}
    prices = df['price'].values
    log_prices = np.log(prices)
    pi_t = np.diff(log_prices)  # π_t = log P_t - log P_{t-1}
    
    # 在开头插入NaN以保持长度一致（第一个时间步没有前一个价格）
    pi_t = np.insert(pi_t, 0, np.nan)
    
    n = len(pi_t)
    
    # 初始化数组
    E_pi = np.full(n, np.nan)  # 预期通胀率
    e_t = np.full(n, np.nan)   # 预测误差
    h_t = np.full(n, np.nan)   # 风险指标
    
    # 计算预期和误差（仅使用naive方法）
    for t in range(1, n):
        # naive forecast E_{t-1}[π_t] = π_{t-1}
        E_pi[t] = pi_t[t-1]
        
        # 计算预测误差 e_t = π_t - E_{t-1}[π_t]
        if not np.isnan(E_pi[t]) and not np.isnan(pi_t[t]):
            e_t[t] = pi_t[t] - E_pi[t]
    
    # 计算风险指标 h_t = λ * h_{t-1} + (1-λ) * e_{t-1}^2
    # 根据RiskMetrics标准：h_t 使用 e_{t-1}^2，所以需要找到第一个有效的 e_t
    first_valid_idx = None
    for i in range(1, n):
        if not np.isnan(e_t[i]):
            first_valid_idx = i
            break
    
    if first_valid_idx is not None:
        # 初始化：h_t[first_valid_idx] = e_t[first_valid_idx]^2
        h_t[first_valid_idx] = e_t[first_valid_idx] ** 2
        
        # 递归计算：h_t = λ * h_{t-1} + (1-λ) * e_{t-1}^2
        for t in range(first_valid_idx + 1, n):
            if not np.isnan(e_t[t-1]) and not np.isnan(h_t[t-1]):
                h_t[t] = lambda_param * h_t[t-1] + (1 - lambda_param) * (e_t[t-1] ** 2)
    
    timestep = df['timestep'].values
    return (timestep, h_t)


def process_subfolder(subfolder_path: Path) -> Optional[Dict]:
    """
    处理单个子文件夹
    
    Args:
        subfolder_path: 子文件夹路径，如 /path/to/claude/claude_42
    
    Returns:
        结果字典：{
            'should_rename': bool,
            'max_risk_step': int or None,
            'max_risk_value': float or None,
            'model': str,
            'subfolder_name': str,
            'original_path': str,
            'error': str or None
        }，如果出错返回None
    """
    csv_path = subfolder_path / "metrics_csv" / "world_metrics.csv"
    
    # 从路径中提取模型名称和子文件夹名称
    model = subfolder_path.parent.name
    subfolder_name = subfolder_path.name
    
    result = {
        'should_rename': False,
        'max_risk_step': None,
        'max_risk_value': None,
        'model': model,
        'subfolder_name': subfolder_name,
        'original_path': str(subfolder_path),
        'error': None
    }
    
    # 检查CSV文件是否存在
    if not csv_path.exists():
        result['error'] = f"CSV file not found: {csv_path}"
        return result
    
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        
        # 检查必要的列
        if 'price' not in df.columns or 'timestep' not in df.columns:
            result['error'] = f"Required columns (price, timestep) not found in CSV"
            return result
        
        # 计算风险指标
        risk_result = calculate_risk_indicator(df, method='naive')
        if risk_result is None:
            result['error'] = "Failed to calculate risk indicator"
            return result
        
        timestep, risk_values = risk_result
        
        # 过滤掉NaN值，只考虑有效的风险值
        valid_mask = ~np.isnan(risk_values)
        valid_timesteps = timestep[valid_mask]
        valid_risks = risk_values[valid_mask]
        
        if len(valid_risks) == 0:
            result['error'] = "No valid risk values found (all NaN)"
            return result
        
        # 找到风险最大值及其对应的timestep
        max_risk_idx = np.argmax(valid_risks)
        max_risk_value = float(valid_risks[max_risk_idx])
        max_risk_step = int(valid_timesteps[max_risk_idx])
        
        result['max_risk_step'] = max_risk_step
        result['max_risk_value'] = max_risk_value
        
        # 如果最高风险点在步骤14及之前（timestep <= 14），需要重命名
        if max_risk_step <= 14:
            result['should_rename'] = True
        
        return result
        
    except Exception as e:
        result['error'] = f"Error processing subfolder: {str(e)}"
        return result


def rename_folder(subfolder_path: Path, should_rename: bool) -> Tuple[Path, bool]:
    """
    重命名文件夹（如果需要）
    
    Args:
        subfolder_path: 原始文件夹路径
        should_rename: 是否需要重命名
    
    Returns:
        (new_path, was_renamed) 元组
    """
    if not should_rename:
        return subfolder_path, False
    
    # 检查是否已经包含_rm后缀
    if subfolder_path.name.endswith('_rm'):
        return subfolder_path, False
    
    # 构建新路径
    new_path = subfolder_path.parent / f"{subfolder_path.name}_rm"
    
    # 如果目标路径已存在，跳过
    if new_path.exists():
        print(f"  Warning: Target path already exists: {new_path}")
        return new_path, False
    
    try:
        # 重命名文件夹
        shutil.move(str(subfolder_path), str(new_path))
        return new_path, True
    except Exception as e:
        print(f"  Error renaming folder {subfolder_path} to {new_path}: {e}")
        return subfolder_path, False


def process_all_models(data_dir: Path) -> Dict:
    """
    批量处理所有模型文件夹
    
    Args:
        data_dir: 数据根目录路径
    
    Returns:
        包含所有处理结果的字典，按模型分组
    """
    models = ['claude', 'ds', 'gpt', 'llama', 'qwen']
    all_results = {model: [] for model in models}
    
    print("=" * 80)
    print("开始处理所有模型数据")
    print("=" * 80)
    
    for model in models:
        model_dir = data_dir / model
        
        if not model_dir.exists():
            print(f"\nWarning: Model directory not found: {model_dir}")
            continue
        
        print(f"\n处理模型: {model}")
        print("-" * 80)
        
        # 获取所有子文件夹（跳过已标记为_rm的文件夹和隐藏文件夹）
        subfolders = [
            d for d in model_dir.iterdir() 
            if d.is_dir() and not d.name.startswith('.') and not d.name.endswith('_rm')
        ]
        subfolders.sort()
        
        print(f"找到 {len(subfolders)} 个子文件夹（已跳过标记为_rm的文件夹）")
        
        for subfolder in subfolders:
            print(f"\n处理: {subfolder.name}")
            
            # 处理子文件夹
            result = process_subfolder(subfolder)
            
            if result is None:
                print(f"  Error: Failed to process subfolder")
                continue
            
            if result['error']:
                print(f"  Error: {result['error']}")
                all_results[model].append(result)
                continue
            
            # 重命名文件夹（如果需要）
            # 使用原始的subfolder路径，而不是result中的original_path（可能已经被重命名）
            new_path, was_renamed = rename_folder(subfolder, result['should_rename'])
            result['new_path'] = str(new_path)
            result['was_renamed'] = was_renamed
            
            if was_renamed:
                print(f"  ✓ 已重命名: {subfolder.name} -> {new_path.name}")
                print(f"    最大风险步数: {result['max_risk_step']}, 最大风险值: {result['max_risk_value']:.6f}")
            else:
                print(f"  - 无需重命名: {subfolder.name}")
                print(f"    最大风险步数: {result['max_risk_step']}, 最大风险值: {result['max_risk_value']:.6f}")
            
            all_results[model].append(result)
    
    return all_results


def generate_summary(all_results: Dict, output_dir: Path):
    """
    生成汇总统计信息
    
    Args:
        all_results: 所有处理结果，按模型分组
        output_dir: 输出目录路径
    """
    print("\n" + "=" * 80)
    print("生成汇总统计信息")
    print("=" * 80)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_stats = {}
    
    for model, results in all_results.items():
        # 过滤出未被重命名的文件夹（should_rename=False 且没有错误）
        valid_results = [
            r for r in results 
            if not r.get('should_rename', True) and r.get('error') is None and r.get('max_risk_step') is not None
        ]
        
        if len(valid_results) == 0:
            print(f"\n模型 {model}: 没有有效的未被标记的文件夹")
            summary_stats[model] = {
                'count': 0,
                'avg_max_risk_step': None,
                'median_max_risk_step': None,
                'std_max_risk_step': None,
                'max_risk_value': None,
                'min_risk_value': None,
                'avg_max_risk_value': None,
                'median_max_risk_value': None,
                'std_max_risk_value': None
            }
            continue
        
        # 提取统计信息
        max_risk_steps = np.array([r['max_risk_step'] for r in valid_results])
        max_risk_values = np.array([r['max_risk_value'] for r in valid_results])
        
        # 计算标准差（如果只有一个元素，返回0.0）
        std_max_risk_step = float(np.std(max_risk_steps)) if len(max_risk_steps) > 1 else 0.0
        std_max_risk_value = float(np.std(max_risk_values)) if len(max_risk_values) > 1 else 0.0
        
        summary_stats[model] = {
            'count': len(valid_results),
            'avg_max_risk_step': float(np.mean(max_risk_steps)),
            'median_max_risk_step': float(np.median(max_risk_steps)),
            'std_max_risk_step': std_max_risk_step,
            'max_risk_value': float(np.max(max_risk_values)),
            'min_risk_value': float(np.min(max_risk_values)),
            'avg_max_risk_value': float(np.mean(max_risk_values)),
            'median_max_risk_value': float(np.median(max_risk_values)),
            'std_max_risk_value': std_max_risk_value
        }
        
        print(f"\n模型 {model}:")
        print(f"  未被标记的文件夹数量: {summary_stats[model]['count']}")
        print(f"  平均最大风险步数: {summary_stats[model]['avg_max_risk_step']:.2f}")
        print(f"  最大风险步数中位数: {summary_stats[model]['median_max_risk_step']:.2f}")
        print(f"  最大风险步数标准差: {summary_stats[model]['std_max_risk_step']:.2f}")
        print(f"  最大风险值: {summary_stats[model]['max_risk_value']:.6f}")
        print(f"  最小风险值: {summary_stats[model]['min_risk_value']:.6f}")
        print(f"  平均最大风险值: {summary_stats[model]['avg_max_risk_value']:.6f}")
        print(f"  最大风险值中位数: {summary_stats[model]['median_max_risk_value']:.6f}")
        print(f"  最大风险值标准差: {summary_stats[model]['std_max_risk_value']:.6f}")
        
        # 保存每个模型的汇总文件
        model_output_dir = output_dir / model
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存JSON
        json_path = model_output_dir / "summary.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'model': model,
                'summary': summary_stats[model],
                'details': valid_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ JSON汇总已保存: {json_path}")
        
        # 保存CSV
        csv_data = []
        for r in valid_results:
            csv_data.append({
                'subfolder_name': r['subfolder_name'],
                'max_risk_step': r['max_risk_step'],
                'max_risk_value': r['max_risk_value']
            })
        
        df = pd.DataFrame(csv_data)
        csv_path = model_output_dir / "summary.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"  ✓ CSV汇总已保存: {csv_path}")
    
    # 保存总体汇总
    overall_summary_path = output_dir / "overall_summary.json"
    with open(overall_summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 总体汇总已保存: {overall_summary_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='根据risk_indicator_naive的风险最高点位置标记数据文件夹'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/mnt/shared-storage-user/meijilin/Economic_System_Attribution/ACL24-EconAgent/datas',
        help='数据根目录路径'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/mnt/shared-storage-user/meijilin/Economic_System_Attribution/ACL24-EconAgent/results/filter_results',
        help='输出目录路径'
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    print("=" * 80)
    print("数据过滤脚本：根据risk_indicator_naive标记文件夹")
    print("=" * 80)
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    print("=" * 80)
    
    # 处理所有模型
    all_results = process_all_models(data_dir)
    
    # 生成汇总
    generate_summary(all_results, output_dir)
    
    print("\n" + "=" * 80)
    print("处理完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()