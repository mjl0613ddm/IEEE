#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量运行Faithfulness实验的Baseline方法

自动发现datas/下所有不带_rm后缀的模型目录，按顺序执行：
1. extract_action_table
2. compute_random_baseline
3. compute_loo_baseline
4. compute_llm_baseline
5. compute_mast_baseline

使用方法:
    python scripts/faithfulness_exp/batch_compute_baselines.py
    python scripts/faithfulness_exp/batch_compute_baselines.py --methods random loo
    python scripts/faithfulness_exp/batch_compute_baselines.py --skip-action-table
"""

import os
import sys
import json
import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

# 添加项目路径
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_ROOT = PROJECT_ROOT / "datas"
sys.path.insert(0, str(PROJECT_ROOT))

# 导入各个模块的函数
from scripts.faithfulness_exp.extract_action_table import extract_action_table
from scripts.faithfulness_exp.compute_random_baseline import compute_random_baseline
from scripts.faithfulness_exp.compute_loo_baseline import compute_loo_baseline
from scripts.faithfulness_exp.compute_llm_baseline import compute_llm_baseline
from scripts.faithfulness_exp.compute_mast_baseline import compute_mast_baseline

# 所有支持的方法
ALL_METHODS = ['action_table', 'random', 'loo', 'llm', 'mast']


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


def check_file_exists(model_path, file_path):
    """检查文件是否存在"""
    full_path = DATA_ROOT / model_path / file_path
    return full_path.exists()


def run_action_table_extraction(model_path, skip_if_exists=True):
    """运行action table提取"""
    if skip_if_exists:
        if check_file_exists(model_path, "action_table/action_table.csv"):
            return {'status': 'skipped', 'reason': 'already exists'}
    
    try:
        result = extract_action_table(model_path)
        if result['status'] == 'success':
            return {'status': 'success'}
        else:
            return {'status': 'error', 'message': result['message']}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


def run_baseline_method(model_path, method, config_path=None, skip_if_exists=True):
    """运行单个baseline方法"""
    start_time = time.time()
    
    try:
        # 检查是否已存在
        if skip_if_exists:
            if method == 'random':
                if check_file_exists(model_path, "faithfulness_exp/random/random_scores.npy"):
                    return {'status': 'skipped', 'reason': 'already exists', 'time': f"{time.time() - start_time:.1f}s"}
            elif method == 'loo':
                if check_file_exists(model_path, "faithfulness_exp/loo/loo_scores.npy"):
                    return {'status': 'skipped', 'reason': 'already exists', 'time': f"{time.time() - start_time:.1f}s"}
            elif method == 'llm':
                if check_file_exists(model_path, "faithfulness_exp/llm/llm_scores.npy"):
                    return {'status': 'skipped', 'reason': 'already exists', 'time': f"{time.time() - start_time:.1f}s"}
            elif method == 'mast':
                if check_file_exists(model_path, "faithfulness_exp/mast/mast_scores.npy"):
                    return {'status': 'skipped', 'reason': 'already exists', 'time': f"{time.time() - start_time:.1f}s"}
        
        # 运行方法
        if method == 'random':
            score_matrix, stats = compute_random_baseline(model_path)
        elif method == 'loo':
            score_matrix, stats = compute_loo_baseline(model_path)
        elif method == 'llm':
            score_matrix, stats = compute_llm_baseline(model_path, config_path=config_path)
        elif method == 'mast':
            score_matrix, stats = compute_mast_baseline(model_path, config_path=config_path)
        else:
            return {'status': 'error', 'message': f'Unknown method: {method}'}
        
        elapsed = time.time() - start_time
        return {'status': 'success', 'time': f"{elapsed:.1f}s"}
    
    except Exception as e:
        elapsed = time.time() - start_time
        return {'status': 'error', 'message': str(e), 'time': f"{elapsed:.1f}s"}


def find_default_config():
    """查找默认配置文件"""
    # 按优先级查找配置文件
    possible_configs = [
        SCRIPT_DIR / "config.yaml",  # 统一配置路径（最高优先级）
        SCRIPT_DIR / "config.yml",
        PROJECT_ROOT / "mast_config.json",
        PROJECT_ROOT / "llm_config.json",
        PROJECT_ROOT / "config.json",
        Path.home() / ".econagent_config.json"
    ]
    
    for config_path in possible_configs:
        if config_path.exists():
            return str(config_path)
    return None


def batch_run_experiments(methods=None, config_path=None, skip_action_table=False, 
                          skip_if_exists=True, model_paths=None):
    """批量运行所有实验"""
    if methods is None:
        methods = ALL_METHODS
    
    # 如果没有指定config_path，尝试查找默认配置
    if config_path is None:
        default_config = find_default_config()
        if default_config:
            config_path = default_config
            print(f"找到默认配置文件: {config_path}")
    
    # 如果没有指定model_paths，自动查找
    if model_paths is None:
        print("自动查找模型目录...")
        model_paths = find_model_directories()
        print(f"找到 {len(model_paths)} 个模型目录")
    
    if len(model_paths) == 0:
        print("错误: 没有找到有效的模型目录")
        return None
    
    print("="*60)
    print("批量运行Faithfulness Baseline实验")
    print("="*60)
    print(f"模型数量: {len(model_paths)}")
    print(f"方法: {', '.join(methods)}")
    if config_path:
        print(f"配置文件: {config_path}")
    print("="*60)
    
    start_time = datetime.now()
    report = {
        'start_time': start_time.isoformat(),
        'model_paths': model_paths,
        'methods': methods,
        'config_path': config_path,
        'results': {}
    }
    
    total_tasks = len(model_paths) * len(methods)
    completed_tasks = 0
    success_count = 0
    skipped_count = 0
    error_count = 0
    
    for model_idx, model_path in enumerate(model_paths, 1):
        print(f"\n[{model_idx}/{len(model_paths)}] 处理模型: {model_path}")
        report['results'][model_path] = {}
        
        # 步骤1: 提取action table
        if 'action_table' in methods and not skip_action_table:
            print(f"  [1/{len(methods)}] 提取action table...")
            result = run_action_table_extraction(model_path, skip_if_exists=skip_if_exists)
            report['results'][model_path]['action_table'] = result
            
            if result['status'] == 'success':
                print(f"    ✓ 成功")
                success_count += 1
            elif result['status'] == 'skipped':
                print(f"    ⏭ 跳过: {result.get('reason', 'N/A')}")
                skipped_count += 1
            else:
                print(f"    ✗ 失败: {result.get('message', 'Unknown error')}")
                error_count += 1
            completed_tasks += 1
        
        # 步骤2-5: 运行baseline方法
        baseline_methods = [m for m in methods if m != 'action_table']
        for method_idx, method in enumerate(baseline_methods, 1):
            step_num = method_idx + (1 if 'action_table' in methods and not skip_action_table else 0)
            print(f"  [{step_num}/{len(methods)}] 运行 {method.upper()} baseline...")
            result = run_baseline_method(model_path, method, config_path, skip_if_exists=skip_if_exists)
            report['results'][model_path][method] = result
            
            if result['status'] == 'success':
                time_str = result.get('time', 'N/A')
                print(f"    ✓ 成功 ({time_str})")
                success_count += 1
            elif result['status'] == 'skipped':
                print(f"    ⏭ 跳过: {result.get('reason', 'N/A')}")
                skipped_count += 1
            else:
                print(f"    ✗ 失败: {result.get('message', 'Unknown error')}")
                error_count += 1
            completed_tasks += 1
    
    end_time = datetime.now()
    report['end_time'] = end_time.isoformat()
    report['duration'] = str(end_time - start_time)
    report['summary'] = {
        'total_tasks': completed_tasks,
        'success': success_count,
        'skipped': skipped_count,
        'error': error_count
    }
    
    # 打印汇总
    print("\n" + "="*60)
    print("执行完成")
    print("="*60)
    print(f"开始时间: {report['start_time']}")
    print(f"结束时间: {report['end_time']}")
    print(f"总耗时: {report['duration']}")
    print(f"\n汇总统计:")
    print(f"  总任务: {completed_tasks}")
    print(f"  成功: {success_count}")
    print(f"  跳过: {skipped_count}")
    print(f"  失败: {error_count}")
    print("="*60)
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description='批量运行Faithfulness实验的Baseline方法',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行所有方法，处理所有模型
  python scripts/faithfulness_exp/batch_compute_baselines.py
  
  # 只运行部分方法
  python scripts/faithfulness_exp/batch_compute_baselines.py --methods random loo
  
  # 跳过已存在的文件
  python scripts/faithfulness_exp/batch_compute_baselines.py --no-skip
  
  # 跳过action table提取
  python scripts/faithfulness_exp/batch_compute_baselines.py --skip-action-table
  
  # 只处理指定的模型
  python scripts/faithfulness_exp/batch_compute_baselines.py --model-paths gpt/gpt_42 claude/claude_42
  
  # 使用配置文件（用于LLM和MAST方法）
  python scripts/faithfulness_exp/batch_compute_baselines.py --config llm_config.json
        """
    )
    
    parser.add_argument(
        '--methods',
        nargs='+',
        choices=ALL_METHODS,
        default=None,
        help=f'要运行的方法列表（默认：所有方法 {ALL_METHODS}）'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='LLM方法的配置文件路径（JSON格式）'
    )
    
    parser.add_argument(
        '--skip-action-table',
        action='store_true',
        help='跳过action table提取步骤'
    )
    
    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='不跳过已存在的文件（重新计算）'
    )
    
    parser.add_argument(
        '--model-paths',
        nargs='+',
        default=None,
        help='要处理的模型路径列表，格式为 "model/model_id"。如果不指定，自动查找所有不带_rm后缀的模型目录'
    )
    
    parser.add_argument(
        '--output-report',
        type=str,
        default=None,
        help='保存执行报告到JSON文件（可选）'
    )
    
    args = parser.parse_args()
    
    try:
        report = batch_run_experiments(
            methods=args.methods,
            config_path=args.config,
            skip_action_table=args.skip_action_table,
            skip_if_exists=not args.no_skip,
            model_paths=args.model_paths
        )
        
        if report is None:
            return 1
        
        # 保存报告
        if args.output_report:
            report_file = Path(args.output_report)
            report_file.parent.mkdir(parents=True, exist_ok=True)
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n执行报告已保存到: {report_file}")
        
        # 如果有错误，返回非零退出码
        return 0 if report['summary']['error'] == 0 else 1
        
    except KeyboardInterrupt:
        print("\n\n用户中断执行")
        return 130
    except Exception as e:
        print(f"\n\n错误: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
