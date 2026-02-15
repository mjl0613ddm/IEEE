#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总Faithfulness实验结果脚本

汇总所有模型的faithfulness实验结果，按模型分组，计算并保存汇总统计信息。

使用方法:
    python scripts/faithfulness_exp/aggregate_faithfulness_results.py
    python scripts/faithfulness_exp/aggregate_faithfulness_results.py --models gpt-4o-mini qwen-plus
"""

import os
import sys
import json
import argparse
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def calculate_faithfulness_score(deletion_curve: List[float], real_risk: float, step: int = 10) -> Optional[float]:
    """
    计算faithfulness分数
    
    Args:
        deletion_curve: 删除曲线，列表形式
        real_risk: 真实风险值
        step: 要使用的步数（默认10，即索引10）
    
    Returns:
        faithfulness分数（下降百分比），如果计算失败返回None
        正值表示下降百分比（越高越好），负值表示上升百分比（风险增加）
    """
    # 检查数组长度
    if len(deletion_curve) <= step:
        return None
    
    # 检查real_risk是否为0（避免除零）
    if real_risk == 0:
        return None
    
    # 获取第10步的risk值（索引10）
    deletion_step10_risk = deletion_curve[step]
    
    # 计算faithfulness分数：下降的risk占real_risk的百分比
    # 公式：(real_risk - deletion_curve[10]) / real_risk
    # 正值：下降百分比（越高越好，可以超过100%）
    # 负值：上升百分比（风险增加）
    faithfulness_score = (real_risk - deletion_step10_risk) / real_risk
    
    return faithfulness_score


def scan_model_results(results_dir: Path, model_name: str, methods: List[str] = None) -> Tuple[List[Dict], Set[str]]:
    """
    扫描单个模型的所有faithfulness结果
    
    Args:
        results_dir: results目录路径
        model_name: 模型名称
        methods: 要处理的方法列表（None表示处理所有方法）
    
    Returns:
        (结果列表, 所有出现的metric_type集合)
        结果列表的每个元素包含subfolder, method, faithfulness_score等信息
    """
    model_dir = results_dir / model_name
    if not model_dir.exists():
        print(f"  警告: 模型目录不存在: {model_dir}")
        return [], set()
    
    results = []
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
        
        # 查找所有方法目录
        method_dirs = []
        for item in faithfulness_exp_dir.iterdir():
            if item.is_dir():
                method_dirs.append(item.name)
        
        if methods:
            method_dirs = [m for m in method_dirs if m in methods]
        
        for method in method_dirs:
            method_dir = faithfulness_exp_dir / method
            # 扫描所有JSON文件，收集metric_type
            json_files = list(method_dir.glob("faithfulness_results_*.json"))
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    metric_type = data.get('metric_type', '')
                    if metric_type:
                        all_metric_types.add(metric_type)
                except:
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
    
    # 第二遍：处理所有结果
    for subfolder in sorted(subfolders):
        subfolder_path = model_dir / subfolder
        faithfulness_exp_dir = subfolder_path / "faithfulness_exp"
        
        if not faithfulness_exp_dir.exists():
            print(f"    跳过 {subfolder}: faithfulness_exp目录不存在")
            continue
        
        # 查找所有方法目录
        method_dirs = []
        for item in faithfulness_exp_dir.iterdir():
            if item.is_dir():
                method_dirs.append(item.name)
        
        if methods:
            method_dirs = [m for m in method_dirs if m in methods]
        
        # 对每个方法处理，收集所有metric_type的结果
        for method in sorted(method_dirs):
            method_dir = faithfulness_exp_dir / method
            
            # 为每个(model, subfolder, method)组合创建结果字典
            result_dict = {
                'model': model_name,
                'subfolder': subfolder,
                'method': method,
                'real_risk': None,
                'baseline_risk': None
            }
            # 为所有metric_type初始化None值
            for mt in sorted_metric_types:
                result_dict[mt] = None
            
            # 扫描所有JSON文件
            json_files = list(method_dir.glob("faithfulness_results_*.json"))
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 提取metric_type
                    file_metric_type = data.get('metric_type', '')
                    if not file_metric_type:
                        # 兼容旧格式：尝试从文件名推断
                        filename = json_file.stem
                        # 尝试匹配 faithfulness_results_{metric_type}_{n}actions 或 faithfulness_results_{metric_type}
                        match = re.match(r'faithfulness_results_(.+?)(?:_\d+actions)?$', filename)
                        if match:
                            file_metric_type = match.group(1)
                    
                    if not file_metric_type or file_metric_type not in sorted_metric_types:
                        continue
                    
                    # 提取公共字段
                    if result_dict['real_risk'] is None:
                        result_dict['real_risk'] = data.get('real_risk', None)
                    if result_dict['baseline_risk'] is None:
                        result_dict['baseline_risk'] = data.get('baseline_risk', None)
                    
                    # 根据metric_type提取相应的指标值
                    if file_metric_type.startswith('deletion_top_'):
                        risk_decrease = data.get('risk_decrease_relative', None)
                        result_dict[file_metric_type] = risk_decrease
                    elif file_metric_type.startswith('insertion_top_'):
                        risk_increase = data.get('risk_increase_relative', None)
                        result_dict[file_metric_type] = risk_increase
                    elif file_metric_type.startswith('deletion_low_'):
                        risk_increase = data.get('risk_increase_relative', None)
                        result_dict[file_metric_type] = risk_increase
                    
                except Exception as e:
                    print(f"    警告: 读取 {json_file} 时出错: {str(e)}")
                    continue
            
            # 只有当至少有一个指标值不为None时才添加到结果中
            has_any_metric = any(result_dict.get(mt) is not None for mt in sorted_metric_types)
            if has_any_metric:
                results.append(result_dict)
                # 打印处理成功信息
                metrics_found = [mt for mt in sorted_metric_types if result_dict.get(mt) is not None]
                print(f"    ✓ 处理成功: {subfolder}/{method} (指标: {', '.join(metrics_found)})")
            else:
                print(f"    跳过 {subfolder}/{method}: 没有找到任何指标数据")
    
    return results, all_metric_types


def calculate_statistics(results: List[Dict], metric_types: List[str]) -> Dict:
    """
    计算统计信息
    
    Args:
        results: 结果列表
        metric_types: 要统计的metric_type列表
    
    Returns:
        统计信息字典
    """
    if not results:
        return {}
    
    # 按方法分组，为每个metric_type分别统计
    method_stats = {}
    overall_stats = {}
    
    # 为每个metric_type计算统计
    for metric_type in metric_types:
        # 按方法分组
        method_values = defaultdict(list)
        all_values = []
        
        for result in results:
            value = result.get(metric_type)
            if value is not None:
                method = result['method']
                method_values[method].append(value)
                all_values.append(value)
        
        # 如果这个metric_type有数据，计算统计
        if all_values:
            # 计算每个方法的统计
            if metric_type not in method_stats:
                method_stats[metric_type] = {}
            
            for method in method_values.keys():
                values_array = np.array(method_values[method])
                method_stats[metric_type][method] = {
                    'mean': float(np.mean(values_array)),
                    'std': float(np.std(values_array)),
                    'variance': float(np.var(values_array)),
                    'count': len(values_array),
                    'min': float(np.min(values_array)),
                    'max': float(np.max(values_array))
                }
            
            # 计算总体统计
            all_values_array = np.array(all_values)
            overall_stats[metric_type] = {
                'mean': float(np.mean(all_values_array)),
                'std': float(np.std(all_values_array)),
                'variance': float(np.var(all_values_array)),
                'count': len(all_values),
                'min': float(np.min(all_values_array)),
                'max': float(np.max(all_values_array))
            }
    
    return {
        'method_stats': method_stats,
        'overall_stats': overall_stats
    }


def aggregate_faithfulness_results(results_dir: Path, output_dir: Path, models: List[str] = None, methods: List[str] = None):
    """
    汇总所有模型的faithfulness结果
    
    Args:
        results_dir: results目录路径
        output_dir: 输出目录路径
        models: 要处理的模型列表（None表示处理所有模型）
        methods: 要处理的方法列表（None表示处理所有方法）
    """
    print("=" * 60)
    print("汇总Faithfulness实验结果")
    print("=" * 60)
    print(f"结果目录: {results_dir}")
    print(f"输出目录: {output_dir}")
    if models:
        print(f"指定模型: {', '.join(models)}")
    if methods:
        print(f"指定方法: {', '.join(methods)}")
    print("")
    
    # 如果指定了models，使用指定的；否则扫描所有模型
    if models:
        model_names = models
    else:
        model_names = []
        for item in results_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                model_names.append(item.name)
        model_names = sorted(model_names)
    
    if not model_names:
        print("错误: 没有找到任何模型目录")
        return
    
    print(f"找到 {len(model_names)} 个模型: {', '.join(model_names)}")
    print("")
    
    # 对每个模型处理
    all_model_stats = {}
    
    for model_name in model_names:
        print(f"处理模型: {model_name}")
        print("-" * 60)
        
        # 扫描该模型的所有结果
        model_results, model_metric_types = scan_model_results(results_dir, model_name, methods=methods)
        
        if not model_results:
            print(f"  模型 {model_name} 没有找到任何有效结果")
            print("")
            continue
        
        print(f"  找到 {len(model_results)} 个有效结果")
        
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
        
        sorted_metric_types = sorted(model_metric_types, key=sort_metric_type)
        
        # 计算统计信息
        stats = calculate_statistics(model_results, sorted_metric_types)
        
        # 保存该模型的输出文件
        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存CSV文件
        csv_file = model_output_dir / "faithfulness_summary.csv"
        # 确保列顺序：基础列 + 指标列 + 公共列
        csv_columns = ['model', 'subfolder', 'method'] + sorted_metric_types + ['real_risk', 'baseline_risk']
        df = pd.DataFrame(model_results)
        # 重新排序列，确保顺序正确，缺失的列填充为None
        for col in csv_columns:
            if col not in df.columns:
                df[col] = None
        df = df[csv_columns]
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"  保存CSV文件: {csv_file}")
        
        # 保存JSON统计文件
        json_file = model_output_dir / "faithfulness_stats.json"
        # 动态构建detailed_results
        detailed_results = []
        for r in model_results:
            result_dict = {
                'subfolder': r['subfolder'],
                'method': r['method'],
                'real_risk': r.get('real_risk'),
                'baseline_risk': r.get('baseline_risk')
            }
            # 添加所有metric_type的值
            for mt in sorted_metric_types:
                result_dict[mt] = r.get(mt)
            detailed_results.append(result_dict)
        
        output_data = {
            'model': model_name,
            'summary': stats.get('method_stats', {}),
            'overall': stats.get('overall_stats', {}),
            'detailed_results': detailed_results
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"  保存JSON文件: {json_file}")
        
        # 打印统计信息（为每个metric_type显示）
        if stats.get('overall_stats'):
            overall = stats['overall_stats']
            
            # 动态生成metric_type显示名称
            def get_metric_type_name(mt: str) -> str:
                if mt.startswith('deletion_top_'):
                    n = mt.replace('deletion_top_', '')
                    return f'Deletion Top {n} (风险下降)'
                elif mt.startswith('insertion_top_'):
                    n = mt.replace('insertion_top_', '')
                    return f'Insertion Top {n} (风险上升)'
                elif mt.startswith('deletion_low_'):
                    n = mt.replace('deletion_low_', '')
                    return f'Deletion Low {n} (风险上升)'
                else:
                    return mt
            
            for metric_type in sorted_metric_types:
                if metric_type in overall:
                    metric_stats = overall[metric_type]
                    mean_val = metric_stats.get('mean', 0)
                    mean_pct = mean_val * 100
                    std_val = metric_stats.get('std', 0)
                    std_pct = std_val * 100
                    metric_name = get_metric_type_name(metric_type)
                    
                    if metric_type.startswith('deletion_top_'):
                        # deletion_top类型：正值表示风险下降（好）
                        if mean_val >= 0:
                            print(f"  总体统计 ({metric_name}): 均值={mean_val:.4f} ({mean_pct:.2f}%下降), 标准差={std_val:.4f} ({std_pct:.2f}%), 方差={metric_stats.get('variance', 0):.6f}, 数量={metric_stats.get('count', 0)}")
                        else:
                            print(f"  总体统计 ({metric_name}): 均值={mean_val:.4f} ({abs(mean_pct):.2f}%上升), 标准差={std_val:.4f} ({abs(std_pct):.2f}%), 方差={metric_stats.get('variance', 0):.6f}, 数量={metric_stats.get('count', 0)}")
                    else:
                        # insertion和deletion_low类型：正值表示风险上升
                        if mean_val >= 0:
                            print(f"  总体统计 ({metric_name}): 均值={mean_val:.4f} ({mean_pct:.2f}%上升), 标准差={std_val:.4f} ({std_pct:.2f}%), 方差={metric_stats.get('variance', 0):.6f}, 数量={metric_stats.get('count', 0)}")
                        else:
                            print(f"  总体统计 ({metric_name}): 均值={mean_val:.4f} ({abs(mean_pct):.2f}%下降), 标准差={std_val:.4f} ({abs(std_pct):.2f}%), 方差={metric_stats.get('variance', 0):.6f}, 数量={metric_stats.get('count', 0)}")
        
        all_model_stats[model_name] = stats
        print("")
    
    print("=" * 60)
    print("汇总完成！")
    print("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='汇总Faithfulness实验结果',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理所有模型
  python scripts/faithfulness_exp/aggregate_faithfulness_results.py
  
  # 只处理指定模型
  python scripts/faithfulness_exp/aggregate_faithfulness_results.py \\
    --models gpt-4o-mini qwen-plus
  
  # 只处理指定方法
  python scripts/faithfulness_exp/aggregate_faithfulness_results.py \\
    --methods shapley llm mast
        """
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        default=None,
        help='results目录路径（默认: 项目根目录/results）'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='输出目录路径（默认: results/faithfulness_exp）'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='要处理的模型列表（默认: 处理所有模型）'
    )
    
    parser.add_argument(
        '--methods',
        type=str,
        nargs='+',
        default=None,
        help='要处理的方法列表（默认: 处理所有方法）'
    )
    
    args = parser.parse_args()
    
    # 确定路径
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = project_root / "results"
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_dir / "faithfulness_exp"
    
    # 检查results目录是否存在
    if not results_dir.exists():
        print(f"错误: results目录不存在: {results_dir}")
        sys.exit(1)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 执行汇总
    try:
        aggregate_faithfulness_results(
            results_dir=results_dir,
            output_dir=output_dir,
            models=args.models,
            methods=args.methods
        )
        sys.exit(0)
    except Exception as e:
        print(f"\n错误: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
