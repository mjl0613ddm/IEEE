#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从agent_metrics.csv中提取指定列，生成action_table.csv文件用于LLM和MAST方法打分
使用方法:
    python scripts/faithfulness_exp/extract_action_table.py  # 自动处理所有模型
    python scripts/faithfulness_exp/extract_action_table.py gpt/gpt_42 claude/claude_42  # 处理指定模型
    python scripts/faithfulness_exp/extract_action_table.py --skip-existing  # 跳过已存在的文件
"""

import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path

# 设置项目根目录
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_ROOT = PROJECT_ROOT / "datas"

# 需要提取的列（用于LLM和MAST方法）
REQUIRED_COLUMNS = [
    'agent_id',
    'timestep',
    'wealth',
    'income',
    'endogenous_Consumption Rate',
    'endogenous_job'
]


def extract_action_table(model_path):
    """
    从指定目录的agent_metrics.csv中提取列，生成action_table.csv
    
    Args:
        model_path: 模型路径，格式为 "model/model_id"，如 "gpt/gpt_42"
    """
    sim_dir = DATA_ROOT / model_path
    input_file = sim_dir / "metrics_csv" / "agent_metrics.csv"
    output_dir = sim_dir / "action_table"
    output_file = output_dir / "action_table.csv"
    shapley_stats_file = sim_dir / "shapley" / "shapley_stats.json"
    
    # 检查输入文件是否存在
    if not input_file.exists():
        return {
            'model_path': model_path,
            'status': 'error',
            'message': f'输入文件不存在: {input_file}'
        }
    
    try:
        # 读取CSV文件
        df = pd.read_csv(input_file)
        
        # 检查必需的列是否存在
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            return {
                'model_path': model_path,
                'status': 'error',
                'message': f'缺少必需的列: {missing_columns}'
            }
        
        # 读取shapley_stats.json获取target_timesteps
        max_target_timestep = None
        if shapley_stats_file.exists():
            try:
                with open(shapley_stats_file, 'r') as f:
                    shapley_stats = json.load(f)
                    target_timesteps = shapley_stats.get('target_timesteps', None)
                    if target_timesteps and isinstance(target_timesteps, list) and len(target_timesteps) > 0:
                        max_target_timestep = max(target_timesteps)
                        print(f"  找到target_timesteps: {target_timesteps}, 最大值为: {max_target_timestep}")
            except Exception as e:
                print(f"  警告: 无法读取shapley_stats.json: {str(e)}")
        else:
            print(f"  警告: shapley_stats.json不存在，将保留所有timestep的数据")
        
        # 提取指定列
        action_df = df[REQUIRED_COLUMNS].copy()
        
        # 根据max_target_timestep过滤数据
        if max_target_timestep is not None:
            original_rows = len(action_df)
            action_df = action_df[action_df['timestep'] <= max_target_timestep].copy()
            filtered_rows = len(action_df)
            print(f"  已过滤: {original_rows} -> {filtered_rows} 行 (保留timestep <= {max_target_timestep})")
        
        # 确保输出目录存在
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存为CSV文件
        action_df.to_csv(output_file, index=False)
        
        return {
            'model_path': model_path,
            'status': 'success',
            'message': f'成功提取 {len(action_df)} 行数据',
            'output_file': str(output_file)
        }
        
    except Exception as e:
        return {
            'model_path': model_path,
            'status': 'error',
            'message': f'处理失败: {str(e)}'
        }


def find_model_directories():
    """查找所有不带_rm后缀的模型目录"""
    model_paths = []
    
    if not DATA_ROOT.exists():
        print(f"警告: 数据目录不存在: {DATA_ROOT}")
        return model_paths
    
    # 遍历所有模型文件夹（gpt, claude, llama, qwen, ds等）
    for model_dir in DATA_ROOT.iterdir():
        if not model_dir.is_dir():
            continue
        
        # 遍历该模型下的所有实验目录
        for exp_dir in model_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            # 跳过带_rm后缀的目录
            if exp_dir.name.endswith('_rm'):
                continue
            
            # 检查是否有shapley目录（确保是有效的实验目录）
            shapley_dir = exp_dir / "shapley"
            if shapley_dir.exists() and (shapley_dir / "shapley_stats.json").exists():
                model_path = f"{model_dir.name}/{exp_dir.name}"
                model_paths.append(model_path)
    
    return sorted(model_paths)


def main():
    parser = argparse.ArgumentParser(
        description='从agent_metrics.csv中提取指定列，生成action_table.csv文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动处理所有模型（推荐）
  python scripts/faithfulness_exp/extract_action_table.py
  
  # 提取指定的模型
  python scripts/faithfulness_exp/extract_action_table.py gpt/gpt_42 claude/claude_42
  
  # 提取单个模型
  python scripts/faithfulness_exp/extract_action_table.py gpt/gpt_42
  
  # 跳过已存在的文件
  python scripts/faithfulness_exp/extract_action_table.py --skip-existing
        """
    )
    
    parser.add_argument(
        'model_paths',
        nargs='*',
        help='要处理的模型路径列表，格式为 "model/model_id"，如 "gpt/gpt_42"。如果不提供，自动查找所有不带_rm后缀的模型目录'
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='跳过已存在的action_table.csv文件'
    )
    
    args = parser.parse_args()
    
    # 如果没有提供model_paths，自动查找所有模型
    if not args.model_paths:
        print("自动查找模型目录...")
        model_paths = find_model_directories()
        print(f"找到 {len(model_paths)} 个模型目录")
    else:
        model_paths = args.model_paths
    
    if len(model_paths) == 0:
        print("错误: 没有找到有效的模型目录")
        return 1
    
    print("="*60)
    print("提取Action Table CSV数据")
    print("="*60)
    print(f"要处理的模型数量: {len(model_paths)}")
    print(f"提取的列: {', '.join(REQUIRED_COLUMNS)}")
    if args.skip_existing:
        print("模式: 跳过已存在的文件")
    print("="*60)
    
    results = []
    success_count = 0
    skipped_count = 0
    error_count = 0
    
    for idx, model_path in enumerate(model_paths, 1):
        print(f"\n[{idx}/{len(model_paths)}] 处理: {model_path}")
        
        # 检查是否已存在且需要跳过
        if args.skip_existing:
            sim_dir = DATA_ROOT / model_path
            output_file = sim_dir / "action_table" / "action_table.csv"
            if output_file.exists():
                print(f"  ⏭ 已存在，跳过")
                results.append({
                    'model_path': model_path,
                    'status': 'skipped',
                    'message': 'already exists'
                })
                skipped_count += 1
                continue
        
        result = extract_action_table(model_path)
        results.append(result)
        
        if result['status'] == 'success':
            print(f"  ✓ {result['message']}")
            print(f"  → {result['output_file']}")
            success_count += 1
        else:
            print(f"  ✗ {result['message']}")
            error_count += 1
    
    # 打印汇总信息
    print("\n" + "="*60)
    print("处理完成")
    print("="*60)
    print(f"成功: {success_count}")
    if skipped_count > 0:
        print(f"跳过: {skipped_count}")
    print(f"失败: {error_count}")
    print("="*60)
    
    # 如果有错误，返回非零退出码
    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
