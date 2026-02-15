#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SocialLLM 结果筛选脚本：根据风险指标筛选可以进行归因的模拟结果

该脚本处理 SocialLLM/results 目录下的数据：
1. 读取每个结果目录的 results.json
2. 应用筛选标准：
   - 最高风险时刻 >= 10
   - 最高风险比初始风险高150%以上（max_risk >= initial_risk * 2.5）
3. 不符合标准的文件夹添加 _rm 后缀
4. 生成汇总统计信息
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


def process_result_dir(result_dir_path: Path) -> Dict:
    """
    处理单个结果目录
    
    Args:
        result_dir_path: 结果目录路径，如 /path/to/gpt-4o-mini/gpt-4o-mini_42
    
    Returns:
        结果字典：{
            'should_rename': bool,
            'max_risk_timestep': int or None,
            'max_risk': float or None,
            'initial_risk': float or None,
            'model': str,
            'result_dir_name': str,
            'original_path': str,
            'error': str or None
        }
    """
    results_file = result_dir_path / "results.json"
    
    # 从路径中提取模型名称和结果目录名称
    model = result_dir_path.parent.name
    result_dir_name = result_dir_path.name
    
    result = {
        'should_rename': False,
        'max_risk_timestep': None,
        'max_risk': None,
        'initial_risk': None,
        'model': model,
        'result_dir_name': result_dir_name,
        'original_path': str(result_dir_path),
        'error': None
    }
    
    # 检查results.json文件是否存在
    if not results_file.exists():
        result['error'] = f"results.json file not found: {results_file}"
        return result
    
    try:
        # 读取results.json文件
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查必要的字段
        required_fields = ['max_risk_timestep', 'max_risk', 'initial_risk']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            result['error'] = f"Required fields missing in results.json: {missing_fields}"
            return result
        
        # 提取风险指标
        max_risk_timestep = data.get('max_risk_timestep')
        max_risk = data.get('max_risk')
        initial_risk = data.get('initial_risk')
        
        result['max_risk_timestep'] = max_risk_timestep
        result['max_risk'] = float(max_risk)
        result['initial_risk'] = float(initial_risk)
        
        # 应用筛选标准
        # 标准1: 最高风险时刻 >= 10
        if max_risk_timestep is None or max_risk_timestep < 10:
            result['should_rename'] = True
            return result
        
        # 标准2: 最高风险比初始风险高150%以上（即 max_risk >= initial_risk * 2.5）
        if initial_risk is None or initial_risk <= 0:
            result['error'] = f"Invalid initial_risk value: {initial_risk}"
            return result
        
        risk_threshold = initial_risk * 2.5
        if max_risk < risk_threshold:
            result['should_rename'] = True
            return result
        
        # 两个标准都满足，不需要重命名
        result['should_rename'] = False
        return result
        
    except json.JSONDecodeError as e:
        result['error'] = f"Failed to parse JSON file: {str(e)}"
        return result
    except Exception as e:
        result['error'] = f"Error processing result directory: {str(e)}"
        return result


def rename_folder(result_dir_path: Path, should_rename: bool, dry_run: bool = False) -> Tuple[Path, bool]:
    """
    重命名文件夹（如果需要）
    
    Args:
        result_dir_path: 原始文件夹路径
        should_rename: 是否需要重命名
        dry_run: 是否为干运行模式（只检查不实际重命名）
    
    Returns:
        (new_path, was_renamed) 元组
    """
    if not should_rename:
        return result_dir_path, False
    
    # 检查是否已经包含_rm后缀
    if result_dir_path.name.endswith('_rm'):
        return result_dir_path, False
    
    # 构建新路径
    new_path = result_dir_path.parent / f"{result_dir_path.name}_rm"
    
    # 如果目标路径已存在，跳过
    if new_path.exists():
        print(f"  Warning: Target path already exists: {new_path}")
        return new_path, False
    
    if dry_run:
        print(f"  [DRY RUN] Would rename: {result_dir_path.name} -> {new_path.name}")
        return new_path, False
    
    try:
        # 重命名文件夹
        shutil.move(str(result_dir_path), str(new_path))
        return new_path, True
    except Exception as e:
        print(f"  Error renaming folder {result_dir_path} to {new_path}: {e}")
        return result_dir_path, False


def process_all_models(results_dir: Path, dry_run: bool = False) -> Dict:
    """
    批量处理所有模型文件夹
    
    Args:
        results_dir: 结果根目录路径
        dry_run: 是否为干运行模式
    
    Returns:
        包含所有处理结果的字典，按模型分组
    """
    all_results = {}
    
    print("=" * 80)
    print("开始处理所有模型数据")
    print("=" * 80)
    
    # 获取所有模型目录
    model_dirs = [
        d for d in results_dir.iterdir() 
        if d.is_dir() and not d.name.startswith('.') and d.name != 'filter_results'
    ]
    model_dirs.sort()
    
    if len(model_dirs) == 0:
        print("Warning: No model directories found in results directory")
        return all_results
    
    print(f"找到 {len(model_dirs)} 个模型目录")
    
    for model_dir in model_dirs:
        model_name = model_dir.name
        all_results[model_name] = []
        
        print(f"\n处理模型: {model_name}")
        print("-" * 80)
        
        # 获取所有结果目录（跳过已标记为_rm的文件夹和隐藏文件夹）
        result_dirs = [
            d for d in model_dir.iterdir() 
            if d.is_dir() and not d.name.startswith('.') and not d.name.endswith('_rm')
        ]
        result_dirs.sort()
        
        if len(result_dirs) == 0:
            print(f"  没有找到结果目录（已跳过标记为_rm的文件夹）")
            continue
        
        print(f"  找到 {len(result_dirs)} 个结果目录")
        
        for result_dir in result_dirs:
            print(f"\n处理: {result_dir.name}")
            
            # 处理结果目录
            result = process_result_dir(result_dir)
            
            if result['error']:
                print(f"  Error: {result['error']}")
                all_results[model_name].append(result)
                continue
            
            # 显示风险指标
            print(f"    最高风险时刻: {result['max_risk_timestep']}")
            print(f"    最高风险值: {result['max_risk']:.6f}")
            print(f"    初始风险值: {result['initial_risk']:.6f}")
            print(f"    风险阈值: {result['initial_risk'] * 2.5:.6f}")
            
            # 重命名文件夹（如果需要）
            # 使用原始的result_dir路径，而不是result中的original_path（可能已经被重命名）
            new_path, was_renamed = rename_folder(result_dir, result['should_rename'], dry_run)
            result['new_path'] = str(new_path)
            result['was_renamed'] = was_renamed
            
            if was_renamed:
                print(f"  ✓ 已重命名: {result_dir.name} -> {new_path.name}")
                print(f"    原因: 不符合筛选标准")
            else:
                print(f"  - 无需重命名: {result_dir.name}")
                print(f"    原因: 符合筛选标准")
            
            all_results[model_name].append(result)
    
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
            if not r.get('should_rename', True) and r.get('error') is None 
            and r.get('max_risk_timestep') is not None
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
                'std_max_risk_value': None,
                'avg_initial_risk': None,
                'median_initial_risk': None
            }
            continue
        
        # 提取统计信息
        max_risk_timesteps = np.array([r['max_risk_timestep'] for r in valid_results])
        max_risk_values = np.array([r['max_risk'] for r in valid_results])
        initial_risk_values = np.array([r['initial_risk'] for r in valid_results])
        
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
            'std_max_risk_value': std_max_risk_value,
            'avg_initial_risk': float(np.mean(initial_risk_values)),
            'median_initial_risk': float(np.median(initial_risk_values))
        }
        
        print(f"\n模型 {model}:")
        print(f"  未被标记的文件夹数量: {summary_stats[model]['count']}")
        print(f"  平均最大风险步数: {summary_stats[model]['avg_max_risk_timestep']:.2f}")
        print(f"  最大风险步数中位数: {summary_stats[model]['median_max_risk_timestep']:.2f}")
        print(f"  最大风险步数标准差: {summary_stats[model]['std_max_risk_timestep']:.2f}")
        print(f"  最大风险值范围: [{summary_stats[model]['min_risk_value']:.6f}, {summary_stats[model]['max_risk_value']:.6f}]")
        print(f"  平均最大风险值: {summary_stats[model]['avg_max_risk_value']:.6f}")
        print(f"  最大风险值中位数: {summary_stats[model]['median_max_risk_value']:.6f}")
        print(f"  最大风险值标准差: {summary_stats[model]['std_max_risk_value']:.6f}")
        print(f"  平均初始风险值: {summary_stats[model]['avg_initial_risk']:.6f}")
        
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
                'result_dir_name': r['result_dir_name'],
                'max_risk_timestep': r['max_risk_timestep'],
                'max_risk': r['max_risk'],
                'initial_risk': r['initial_risk'],
                'risk_ratio': r['max_risk'] / r['initial_risk'] if r['initial_risk'] > 0 else None
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
        description='根据风险指标筛选SocialLLM模拟结果，不符合标准的文件夹添加_rm后缀'
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        default=None,
        help='结果根目录路径（可选，默认：项目根目录下的results/）'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='输出目录路径（可选，默认：results/filter_results/）'
    )
    
    parser.add_argument(
        '--base_path',
        type=str,
        default=None,
        help='项目根目录路径（可选，默认：自动检测）'
    )
    
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='干运行模式（只检查不实际重命名）'
    )
    
    args = parser.parse_args()
    
    # 确定项目根目录
    if args.base_path:
        base_path = Path(args.base_path)
    else:
        # 自动检测：假设脚本在 scripts/filter/ 目录下
        script_dir = Path(__file__).parent
        base_path = script_dir.parent.parent
    
    base_path = base_path.resolve()
    
    # 确定结果目录
    if args.results_dir:
        results_dir = Path(args.results_dir)
        if not results_dir.is_absolute():
            results_dir = base_path / results_dir
    else:
        results_dir = base_path / "results"
    
    results_dir = results_dir.resolve()
    
    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = base_path / output_dir
    else:
        output_dir = results_dir / "filter_results"
    
    output_dir = output_dir.resolve()
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    print("=" * 80)
    print("SocialLLM 结果筛选脚本")
    print("=" * 80)
    print(f"结果目录: {results_dir}")
    print(f"输出目录: {output_dir}")
    if args.dry_run:
        print("模式: 干运行（只检查不重命名）")
    else:
        print("模式: 实际执行（会重命名文件夹）")
    print("=" * 80)
    
    # 处理所有模型
    all_results = process_all_models(results_dir, dry_run=args.dry_run)
    
    # 生成汇总
    generate_summary(all_results, output_dir)
    
    # 统计总体信息
    total_processed = sum(len(results) for results in all_results.values())
    total_renamed = sum(
        sum(1 for r in results if r.get('was_renamed', False))
        for results in all_results.values()
    )
    total_valid = sum(
        sum(1 for r in results if not r.get('should_rename', True) and r.get('error') is None)
        for results in all_results.values()
    )
    
    print("\n" + "=" * 80)
    print("处理完成！")
    print("=" * 80)
    print(f"总处理数: {total_processed}")
    print(f"已重命名: {total_renamed}")
    print(f"有效结果: {total_valid}")
    print("=" * 80)


if __name__ == "__main__":
    main()
