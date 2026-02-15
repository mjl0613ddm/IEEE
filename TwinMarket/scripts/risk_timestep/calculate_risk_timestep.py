#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算TwinMarket中风险最高点的平均时间步

该脚本处理 TwinMarket/results 目录下的数据：
1. 计算每个子文件夹的risk_indicator_simple
2. 找到风险最高点对应的日期，转换为时间步
3. 汇总所有不带_rm后缀的结果
4. 计算平均风险最高点时间步
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def find_max_risk_timestep(result_dir: Path) -> Optional[Dict]:
    """
    从market_metrics.csv中找到risk_indicator_simple的最大值对应的时间步
    
    Args:
        result_dir: 结果目录路径，如 /path/to/claude-3-haiku-20240307/claude-3-haiku-20240307_42
    
    Returns:
        结果字典：{
            'max_risk_timestep': int or None,
            'max_risk_value': float or None,
            'max_risk_date': str or None,
            'total_timesteps': int or None,
            'error': str or None
        }，如果出错返回None
    """
    csv_path = result_dir / "analysis" / "market_metrics.csv"
    
    result = {
        'max_risk_timestep': None,
        'max_risk_value': None,
        'max_risk_date': None,
        'total_timesteps': None,
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
        if 'date' not in df.columns:
            result['error'] = f"Required column 'date' not found in CSV"
            return result
        
        if 'risk_indicator_simple' not in df.columns:
            result['error'] = f"Required column 'risk_indicator_simple' not found in CSV"
            return result
        
        # 过滤掉risk_indicator_simple为空值的行
        df_valid = df[df['risk_indicator_simple'].notna()].copy()
        
        if df_valid.empty:
            result['error'] = "No valid risk_indicator_simple values found (all NaN)"
            return result
        
        # 将日期转换为时间步（从0开始）
        df_valid = df_valid.sort_values('date').reset_index(drop=True)
        df_valid['timestep'] = df_valid.index
        
        # 找到风险最大值及其对应的timestep
        max_risk_idx = df_valid['risk_indicator_simple'].argmax()
        max_row = df_valid.iloc[max_risk_idx]
        
        max_risk_value = float(max_row['risk_indicator_simple'])
        max_risk_timestep = int(max_row['timestep'])
        max_risk_date = str(max_row['date'])
        total_timesteps = len(df_valid)
        
        result['max_risk_timestep'] = max_risk_timestep
        result['max_risk_value'] = max_risk_value
        result['max_risk_date'] = max_risk_date
        result['total_timesteps'] = total_timesteps
        
        return result
        
    except Exception as e:
        result['error'] = f"Error processing result directory: {str(e)}"
        return result


def process_model(model_dir: Path, model_name: str) -> List[Dict]:
    """
    处理单个模型的所有结果文件夹
    
    Args:
        model_dir: 模型目录路径，如 /path/to/results/claude-3-haiku-20240307
        model_name: 模型名称，如 'claude-3-haiku-20240307'
    
    Returns:
        结果列表，每个元素包含一个子文件夹的处理结果
    """
    results = []
    
    if not model_dir.exists():
        print(f"\nWarning: Model directory not found: {model_dir}")
        return results
    
    print(f"\n处理模型: {model_name}")
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
        result = find_max_risk_timestep(subfolder)
        
        if result is None:
            print(f"  Error: Failed to process subfolder")
            continue
        
        if result['error']:
            print(f"  Error: {result['error']}")
            results.append({
                'model': model_name,
                'subfolder_name': subfolder.name,
                'original_path': str(subfolder),
                **result
            })
            continue
        
        print(f"  ✓ 最大风险时间步: {result['max_risk_timestep']}, "
              f"最大风险值: {result['max_risk_value']:.6f}, "
              f"日期: {result['max_risk_date']}")
        
        results.append({
            'model': model_name,
            'subfolder_name': subfolder.name,
            'original_path': str(subfolder),
            **result
        })
    
    return results


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
        # 过滤出有效的未被标记的文件夹（没有错误且有max_risk_timestep）
        valid_results = [
            r for r in results 
            if r.get('error') is None and r.get('max_risk_timestep') is not None
        ]
        
        if len(valid_results) == 0:
            print(f"\n模型 {model}: 没有有效的未被标记的文件夹")
            summary_stats[model] = {
                'count': 0,
                'avg_max_risk_timestep': None,
                'median_max_risk_timestep': None,
                'std_max_risk_timestep': None,
                'max_risk_value': None,
                'min_risk_value': None,
                'avg_max_risk_value': None,
                'median_max_risk_value': None,
                'std_max_risk_value': None
            }
            continue
        
        # 提取统计信息
        max_risk_timesteps = np.array([r['max_risk_timestep'] for r in valid_results])
        max_risk_values = np.array([r['max_risk_value'] for r in valid_results])
        
        # 计算标准差（如果只有一个元素，返回0.0）
        std_max_risk_timestep = float(np.std(max_risk_timesteps)) if len(max_risk_timesteps) > 1 else 0.0
        std_max_risk_value = float(np.std(max_risk_values)) if len(max_risk_values) > 1 else 0.0
        
        summary_stats[model] = {
            'count': len(valid_results),
            'avg_max_risk_timestep': float(np.mean(max_risk_timesteps)),
            'median_max_risk_timestep': float(np.median(max_risk_timesteps)),
            'std_max_risk_timestep': std_max_risk_timestep,
            'max_risk_value': float(np.max(max_risk_values)),
            'min_risk_value': float(np.min(max_risk_values)),
            'avg_max_risk_value': float(np.mean(max_risk_values)),
            'median_max_risk_value': float(np.median(max_risk_values)),
            'std_max_risk_value': std_max_risk_value
        }
        
        print(f"\n模型 {model}:")
        print(f"  未被标记的文件夹数量: {summary_stats[model]['count']}")
        print(f"  平均最大风险时间步: {summary_stats[model]['avg_max_risk_timestep']:.2f}")
        print(f"  最大风险时间步中位数: {summary_stats[model]['median_max_risk_timestep']:.2f}")
        print(f"  最大风险时间步标准差: {summary_stats[model]['std_max_risk_timestep']:.2f}")
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
                'max_risk_timestep': r['max_risk_timestep'],
                'max_risk_value': r['max_risk_value'],
                'max_risk_date': r['max_risk_date'],
                'total_timesteps': r['total_timesteps']
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
        description='计算TwinMarket中风险最高点的平均时间步'
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        default='/mnt/shared-storage-user/meijilin/Economic_System_Attribution/TwinMarket/results',
        help='结果根目录路径'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/mnt/shared-storage-user/meijilin/Economic_System_Attribution/TwinMarket/results/risk_timestep_summary',
        help='输出目录路径'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['claude-3-haiku-20240307', 'deepseek-v3.2', 'gpt-4o-mini', 'llama-3.1-70b-instruct', 'qwen-plus'],
        help='要处理的模型列表（默认：所有指定模型）'
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    print("=" * 80)
    print("计算TwinMarket风险最高点平均时间步")
    print("=" * 80)
    print(f"结果目录: {results_dir}")
    print(f"输出目录: {output_dir}")
    print(f"处理模型: {', '.join(args.models)}")
    print("=" * 80)
    
    all_results = {}
    
    # 处理每个模型
    for model_name in args.models:
        model_dir = results_dir / model_name
        
        if not model_dir.exists():
            print(f"\nWarning: Model directory not found: {model_dir}")
            continue
        
        results = process_model(model_dir, model_name)
        all_results[model_name] = results
    
    # 生成汇总
    generate_summary(all_results, output_dir)
    
    print("\n" + "=" * 80)
    print("处理完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
