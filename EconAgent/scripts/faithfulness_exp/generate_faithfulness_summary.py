#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成faithfulness汇总CSV文件

从所有模型的faithfulness结果JSON文件中汇总数据，生成CSV文件。

使用方法:
    python scripts/faithfulness_exp/generate_faithfulness_summary.py gpt
"""

import json
import csv
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Set

# 添加项目路径
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"


def generate_summary(model_name: str) -> bool:
    """
    为指定模型生成汇总CSV文件
    
    Args:
        model_name: 模型名称，如 "gpt"
    
    Returns:
        是否成功生成
    """
    model_dir = RESULTS_ROOT / "faithfulness_exp" / model_name
    
    if not model_dir.exists():
        print(f"错误: 模型目录不存在: {model_dir}")
        return False
    
    # 收集所有结果和所有出现的metric_type
    all_results = []
    all_metric_types: Set[str] = set()
    
    # 第一遍：收集所有出现的metric_type
    for subfolder_dir in sorted(model_dir.iterdir()):
        if not subfolder_dir.is_dir():
            continue
        
        json_files = list(subfolder_dir.glob("faithfulness_results_*.json"))
        for json_file in json_files:
            with open(json_file, 'r') as f:
                result = json.load(f)
            metric_type = result.get('metric_type', '')
            if metric_type:
                all_metric_types.add(metric_type)
    
    # 按类型和n值排序metric_type（deletion_top_1, deletion_top_3, deletion_top_5, ...）
    def sort_metric_type(mt: str) -> tuple:
        """返回排序键：(类型前缀, n值)"""
        if mt.startswith('deletion_top_'):
            n = int(mt.replace('deletion_top_', ''))
            return (0, n)  # deletion_top 排在前面
        elif mt.startswith('insertion_top_'):
            n = int(mt.replace('insertion_top_', ''))
            return (1, n)  # insertion_top 排在后面
        else:
            return (2, 0)  # 其他类型
    
    sorted_metric_types = sorted(all_metric_types, key=sort_metric_type)
    
    # 遍历所有子文件夹（模型ID）
    for subfolder_dir in sorted(model_dir.iterdir()):
        if not subfolder_dir.is_dir():
            continue
        
        subfolder = subfolder_dir.name
        
        # 查找所有结果JSON文件
        json_files = list(subfolder_dir.glob("faithfulness_results_*.json"))
        
        if not json_files:
            continue
        
        # 按方法分组结果
        method_results = {}
        for json_file in json_files:
            with open(json_file, 'r') as f:
                result = json.load(f)
            
            method = result.get('method', 'unknown')
            metric_type = result.get('metric_type', '')
            
            if method not in method_results:
                method_results[method] = {
                    'real_risk': result.get('real_risk', 0.0),
                    'baseline_risk': result.get('baseline_risk', 0.0),
                }
                # 为所有metric_type初始化None值
                for mt in sorted_metric_types:
                    method_results[method][mt] = None
            
            # 根据metric_type填充数据
            if metric_type.startswith('deletion_top_'):
                method_results[method][metric_type] = result.get('risk_decrease_relative', None)
            elif metric_type.startswith('insertion_top_'):
                method_results[method][metric_type] = result.get('risk_increase_relative', None)
        
        # 为每个方法创建一行
        for method, data in method_results.items():
            row = {
                'model': model_name,
                'subfolder': subfolder,
                'method': method,
            }
            # 添加所有metric_type列
            for mt in sorted_metric_types:
                row[mt] = data.get(mt, None)
            # 添加real_risk和baseline_risk
            row['real_risk'] = data['real_risk']
            row['baseline_risk'] = data['baseline_risk']
            all_results.append(row)
    
    if not all_results:
        print(f"警告: 未找到任何结果文件")
        return False
    
    # 排序结果：按subfolder和method排序
    all_results.sort(key=lambda x: (x['subfolder'], x['method']))
    
    # 写入CSV文件
    csv_file = model_dir / "faithfulness_summary.csv"
    
    # 动态构建fieldnames：基础列 + 所有metric_type列 + 风险列
    fieldnames = ['model', 'subfolder', 'method'] + sorted_metric_types + ['real_risk', 'baseline_risk']
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"汇总文件已生成: {csv_file}")
    print(f"共 {len(all_results)} 条记录")
    
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='生成faithfulness汇总CSV文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 为gpt模型生成汇总文件
  python scripts/faithfulness_exp/generate_faithfulness_summary.py gpt
  
  # 为所有模型生成汇总文件
  python scripts/faithfulness_exp/generate_faithfulness_summary.py --all
        """
    )
    
    parser.add_argument(
        'model_name',
        nargs='?',
        type=str,
        help='模型名称，如 "gpt"（如果指定--all则忽略）'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='为所有模型生成汇总文件'
    )
    
    args = parser.parse_args()
    
    if args.all:
        # 为所有模型生成汇总文件
        faithfulness_exp_dir = RESULTS_ROOT / "faithfulness_exp"
        if not faithfulness_exp_dir.exists():
            print(f"错误: 结果目录不存在: {faithfulness_exp_dir}")
            return 1
        
        success_count = 0
        fail_count = 0
        
        for model_dir in sorted(faithfulness_exp_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            
            model_name = model_dir.name
            print(f"\n处理模型: {model_name}")
            if generate_summary(model_name):
                success_count += 1
            else:
                fail_count += 1
        
        print(f"\n完成: 成功 {success_count} 个模型, 失败 {fail_count} 个模型")
        return 0 if fail_count == 0 else 1
    else:
        if not args.model_name:
            parser.print_help()
            return 1
        
        if generate_summary(args.model_name):
            return 0
        else:
            return 1


if __name__ == '__main__':
    sys.exit(main())
