#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总Faithfulness实验结果脚本

汇总所有模型的faithfulness实验结果，按模型分组，生成汇总CSV文件。

使用方法:
    python scripts/faithfulness_exp/aggregate_faithfulness_results.py
    python scripts/faithfulness_exp/aggregate_faithfulness_results.py --models gpt-4o-mini qwen-plus
"""

import os
import sys
import json
import argparse
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def scan_model_results(results_dir: Path, model_name: str, methods: List[str] = None) -> Tuple[List[Dict], Set[str]]:
    """
    扫描单个模型的所有faithfulness结果
    
    Args:
        results_dir: results目录路径
        model_name: 模型名称
        methods: 要处理的方法列表（None表示处理所有方法）
    
    Returns:
        (结果列表, 所有出现的metric_type集合)
        结果列表的每个元素包含model, subfolder, method, 以及所有metric_type作为键
    """
    model_dir = results_dir / model_name
    if not model_dir.exists():
        print(f"  警告: 模型目录不存在: {model_dir}")
        return [], set()
    
    results_dict = {}  # {(subfolder, method): result_dict}
    all_metric_types: Set[str] = set()
    
    # 查找所有子文件夹（排除_rm结尾的）
    subfolders = []
    for item in model_dir.iterdir():
        if item.is_dir() and not item.name.endswith('_rm'):
            subfolders.append(item.name)
    
    if not subfolders:
        print(f"  警告: 模型 {model_name} 下没有找到子文件夹")
        return [], set()
    
    print(f"  找到 {len(subfolders)} 个子文件夹")
    
    # 第一遍：收集所有出现的metric_type
    for subfolder in sorted(subfolders):
        subfolder_path = model_dir / subfolder
        faithfulness_exp_dir = subfolder_path / "faithfulness_exp"
        
        if not faithfulness_exp_dir.exists():
            continue
        
        # 查找所有faithfulness结果JSON文件
        json_files = list(faithfulness_exp_dir.glob("faithfulness_results_*.json"))
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                metric_type = result.get('metric_type', '')
                if metric_type:
                    all_metric_types.add(metric_type)
            except Exception as e:
                print(f"    警告: 无法读取 {json_file}: {e}")
                continue
    
    # 按类型和n值排序metric_type
    def sort_metric_type(mt: str) -> tuple:
        """返回排序键：(类型前缀, n值)"""
        if mt.startswith('deletion_top_'):
            try:
                n = int(mt.replace('deletion_top_', ''))
                return (0, n)
            except:
                return (0, 999)
        elif mt.startswith('insertion_top_'):
            try:
                n = int(mt.replace('insertion_top_', ''))
                return (1, n)
            except:
                return (1, 999)
        elif mt.startswith('deletion_low_'):
            try:
                n = int(mt.replace('deletion_low_', ''))
                return (2, n)
            except:
                return (2, 999)
        else:
            return (3, 0)
    
    sorted_metric_types = sorted(all_metric_types, key=sort_metric_type)
    
    # 第二遍：收集所有结果
    for subfolder in sorted(subfolders):
        subfolder_path = model_dir / subfolder
        faithfulness_exp_dir = subfolder_path / "faithfulness_exp"
        
        if not faithfulness_exp_dir.exists():
            continue
        
        # 查找所有faithfulness结果JSON文件
        json_files = list(faithfulness_exp_dir.glob("faithfulness_results_*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                method = data.get('method', 'unknown')
                metric_type = data.get('metric_type', '')
                
                # 如果指定了methods，只处理指定的方法
                if methods and method not in methods:
                    continue
                
                if not metric_type or metric_type not in sorted_metric_types:
                    continue
                
                # 为每个(subfolder, method)组合创建或更新结果字典
                key = (subfolder, method)
                if key not in results_dict:
                    result_dict = {
                        'model': model_name,
                        'subfolder': subfolder,
                        'method': method,
                        'real_risk': None,
                        'baseline_risk': None  # 使用initial_risk（第一个时间步的极化风险）
                    }
                    # 为所有metric_type初始化None值
                    for mt in sorted_metric_types:
                        result_dict[mt] = None
                    results_dict[key] = result_dict
                else:
                    result_dict = results_dict[key]
                
                # 提取公共字段
                if result_dict['real_risk'] is None:
                    result_dict['real_risk'] = data.get('max_risk', None)
                if result_dict['baseline_risk'] is None:
                    result_dict['baseline_risk'] = data.get('initial_risk', 0.0)  # 使用initial_risk作为baseline_risk
                
                # 根据metric_type提取相应的指标值
                if metric_type.startswith('deletion_top_'):
                    risk_decrease = data.get('risk_decrease_relative', None)
                    result_dict[metric_type] = risk_decrease
                elif metric_type.startswith('insertion_top_'):
                    risk_increase = data.get('risk_increase_relative', None)
                    result_dict[metric_type] = risk_increase
                elif metric_type.startswith('deletion_low_'):
                    risk_increase = data.get('risk_increase_relative', None)
                    result_dict[metric_type] = risk_increase
                
            except Exception as e:
                print(f"    警告: 无法读取 {json_file}: {e}")
                continue
    
    # 转换为列表，只保留至少有一个指标值不为None的结果
    results = []
    for result_dict in results_dict.values():
        has_any_metric = any(result_dict.get(mt) is not None for mt in sorted_metric_types)
        if has_any_metric:
            results.append(result_dict)
    
    return results, all_metric_types


def generate_summary_csv(results_dir: Path, output_dir: Path, model_name: str, methods: List[str] = None, force_overwrite: bool = False):
    """
    为指定模型生成汇总CSV文件（TwinMarket格式）
    
    Args:
        results_dir: results目录路径
        output_dir: 输出目录路径（results/faithfulness_exp）
        model_name: 模型名称
        methods: 要处理的方法列表（None表示处理所有方法）
        force_overwrite: 是否强制覆盖已存在的文件
    """
    print(f"\n处理模型: {model_name}")
    print("=" * 60)
    
    # 扫描结果
    results, all_metric_types = scan_model_results(results_dir, model_name, methods)
    
    if not results:
        print(f"  没有找到任何结果，跳过")
        return False
    
    print(f"  找到 {len(results)} 个结果 (subfolder, method) 组合")
    print(f"  包含的metric_type: {sorted(all_metric_types)}")
    
    # 按类型和n值排序metric_type
    def sort_metric_type(mt: str) -> tuple:
        """返回排序键：(类型前缀, n值)"""
        if mt.startswith('deletion_top_'):
            try:
                n = int(mt.replace('deletion_top_', ''))
                return (0, n)
            except:
                return (0, 999)
        elif mt.startswith('insertion_top_'):
            try:
                n = int(mt.replace('insertion_top_', ''))
                return (1, n)
            except:
                return (1, 999)
        elif mt.startswith('deletion_low_'):
            try:
                n = int(mt.replace('deletion_low_', ''))
                return (2, n)
            except:
                return (2, 999)
        else:
            return (3, 0)
    
    sorted_metric_types = sorted(all_metric_types, key=sort_metric_type)
    
    # 创建DataFrame
    # 列顺序：model, subfolder, method, 然后所有metric_type, real_risk, baseline_risk
    csv_columns = ['model', 'subfolder', 'method'] + sorted_metric_types + ['real_risk', 'baseline_risk']
    
    df = pd.DataFrame(results)
    
    # 重新排序列，确保顺序正确，缺失的列填充为None
    for col in csv_columns:
        if col not in df.columns:
            df[col] = None
    df = df[csv_columns]
    
    # 按照指定的方法顺序排序：random, mast, llm, loo, shapley
    method_order = ['random', 'mast', 'llm', 'loo', 'shapley']
    method_order_dict = {method: idx for idx, method in enumerate(method_order)}
    
    # 创建排序键：先按subfolder排序，然后按method的指定顺序排序
    df['_method_order'] = df['method'].map(lambda x: method_order_dict.get(x, 999))
    df = df.sort_values(['subfolder', '_method_order'])
    df = df.drop('_method_order', axis=1)
    
    # 保存CSV文件
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / model_name / "faithfulness_summary.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 检查文件是否已存在
    if output_file.exists():
        if force_overwrite:
            print(f"  汇总文件已存在，但配置为覆盖模式，将重新生成: {output_file}")
        else:
            print(f"  汇总文件已存在，跳过: {output_file} (如需覆盖，请使用 --force 选项)")
            return True
    
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"  ✓ 汇总文件已保存到: {output_file}")
    print(f"  包含 {len(df)} 行, {len(df.columns)} 列")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='汇总Faithfulness实验结果',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--models',
        type=str,
        nargs='*',
        default=None,
        help='要处理的模型列表（可选，默认：自动识别所有模型）'
    )
    
    parser.add_argument(
        '--methods',
        type=str,
        nargs='*',
        default=None,
        help='要处理的方法列表（可选，默认：处理所有方法）'
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        default=None,
        help='结果目录路径（可选，默认：项目根目录/results）'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='输出目录路径（可选，默认：项目根目录/results/faithfulness_exp）'
    )
    
    parser.add_argument(
        '--force',
        '--overwrite',
        dest='force_overwrite',
        action='store_true',
        help='强制覆盖已存在的汇总文件（默认：跳过已存在的文件）'
    )
    
    args = parser.parse_args()
    
    # 确定路径
    if args.results_dir:
        results_dir = Path(args.results_dir).resolve()
    else:
        results_dir = project_root / "results"
    
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = project_root / "results" / "faithfulness_exp"
    
    print("=" * 60)
    print("汇总Faithfulness实验结果")
    print("=" * 60)
    print(f"结果目录: {results_dir}")
    print(f"输出目录: {output_dir}")
    print(f"方法列表: {args.methods if args.methods else '所有方法'}")
    print(f"覆盖模式: {args.force_overwrite} (True=覆盖已存在文件, False=跳过已存在文件)")
    print()
    
    if not results_dir.exists():
        print(f"错误: 结果目录不存在: {results_dir}", file=sys.stderr)
        sys.exit(1)
    
    # 确定要处理的模型列表
    if args.models:
        model_names = args.models
    else:
        # 自动识别所有模型目录
        model_names = []
        for item in results_dir.iterdir():
            if item.is_dir() and not item.name.endswith('_rm'):
                # 检查是否有非_rm的子文件夹
                has_valid_subfolder = False
                for subfolder in item.iterdir():
                    if subfolder.is_dir() and not subfolder.name.endswith('_rm'):
                        has_valid_subfolder = True
                        break
                if has_valid_subfolder:
                    model_names.append(item.name)
        
        if not model_names:
            print("错误: 未找到任何模型目录", file=sys.stderr)
            sys.exit(1)
        
        print(f"自动识别到 {len(model_names)} 个模型: {', '.join(model_names)}")
    
    # 处理每个模型
    success_count = 0
    fail_count = 0
    
    for model_name in sorted(model_names):
        try:
            if generate_summary_csv(results_dir, output_dir, model_name, args.methods, args.force_overwrite):
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"  错误: 处理模型 {model_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1
    
    print()
    print("=" * 60)
    print("汇总完成")
    print("=" * 60)
    print(f"成功: {success_count} 个模型")
    print(f"失败: {fail_count} 个模型")
    
    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
