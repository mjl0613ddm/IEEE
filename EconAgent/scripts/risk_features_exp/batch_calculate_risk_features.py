#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行批量计算所有模型的risk features
使用方法:
    python batch_calculate_risk_features.py --max_workers 64
    python batch_calculate_risk_features.py --max_workers 64 --models claude gpt
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional
import time

# 设置项目根目录
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # ACL24-EconAgent
DATAS_ROOT = PROJECT_ROOT / "datas"
RESULTS_ROOT = PROJECT_ROOT / "results"
CALCULATE_SCRIPT = SCRIPT_DIR / "calculate_risk_features.py"


def discover_tasks(datas_root: Path, models: List[str] = None) -> List[Tuple[str, str, Path]]:
    """
    自动发现datas目录下所有包含shapley文件的模型/seed目录
    
    Args:
        datas_root: datas根目录路径
        models: 要处理的模型列表（部分匹配），如果为None则处理所有模型
        
    Returns:
        [(model_name, seed_name, data_dir), ...] 列表
    """
    if models is None:
        models = []
    
    discovered_tasks = []
    
    # 遍历所有模型目录
    for model_dir in datas_root.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        # 跳过一些非模型目录
        if model_name.startswith('.'):
            continue
        
        # 如果指定了模型过滤，检查是否匹配
        if models:
            matched = False
            for filter_model in models:
                if filter_model.lower() in model_name.lower():
                    matched = True
                    break
            if not matched:
                continue
        
        # 遍历该模型下的所有子目录（如 claude_42, claude_44等）
        for seed_dir in model_dir.iterdir():
            if not seed_dir.is_dir():
                continue
            
            seed_name = seed_dir.name
            
            # 排除带_rm后缀的目录（如claude_42_rm）
            if seed_name.endswith('_rm'):
                continue
            
            shapley_dir = seed_dir / "shapley"
            
            # 检查是否存在shapley目录和文件
            if not shapley_dir.exists() or not shapley_dir.is_dir():
                continue
            
            shapley_file = shapley_dir / "shapley_values.npy"
            if not shapley_file.exists():
                continue
            
            discovered_tasks.append((model_name, seed_name, seed_dir))
    
    return discovered_tasks


def run_risk_feature_calculation(model_name: str, seed_name: str, data_dir: Path, 
                                  calculate_script: Path, project_root: Path) -> dict:
    """
    运行单个risk feature计算任务
    
    Args:
        model_name: 模型名称
        seed_name: seed名称
        data_dir: 数据目录路径
        calculate_script: 计算脚本路径
        project_root: 项目根目录
        
    Returns:
        包含任务执行结果的字典
    """
    # 设置输出目录
    output_dir = project_root / "results" / "risk_features_exp" / model_name / seed_name
    
    # 构建命令
    cmd = [
        sys.executable,
        str(calculate_script.absolute()),
        '--data_dir', str(data_dir.absolute()),
        '--output_dir', str(output_dir.absolute()),
    ]
    
    # 运行命令
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=3600  # 1小时超时
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            return {
                'model': model_name,
                'seed': seed_name,
                'status': 'success',
                'elapsed_time': elapsed_time,
                'message': f'Completed in {elapsed_time:.1f}s'
            }
        else:
            return {
                'model': model_name,
                'seed': seed_name,
                'status': 'error',
                'elapsed_time': elapsed_time,
                'message': f'Failed with return code {result.returncode}',
                'stderr': result.stderr[-500:] if result.stderr else ''  # 只保留最后500字符
            }
    except subprocess.TimeoutExpired:
        return {
            'model': model_name,
            'seed': seed_name,
            'status': 'timeout',
            'elapsed_time': 3600,
            'message': 'Timeout after 3600s'
        }
    except Exception as e:
        return {
            'model': model_name,
            'seed': seed_name,
            'status': 'error',
            'message': f'Exception: {str(e)}'
        }


def main():
    parser = argparse.ArgumentParser(
        description='并行批量计算所有模型的risk features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动发现所有模型，使用64个CPU并行
  python batch_calculate_risk_features.py --max_workers 64
  
  # 只处理指定模型
  python batch_calculate_risk_features.py --max_workers 64 --models claude gpt
        """
    )
    
    parser.add_argument(
        '--datas_root',
        type=str,
        default=None,
        help='Datas根目录（默认: PROJECT_ROOT/datas）'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='要处理的模型列表（部分匹配，例如: claude gpt）'
    )
    
    parser.add_argument(
        '--max_workers',
        '-j',
        type=int,
        default=1,
        help='最大并行任务数（默认: 1）'
    )
    
    args = parser.parse_args()
    
    # 解析路径
    if args.datas_root:
        datas_root = Path(args.datas_root)
    else:
        datas_root = DATAS_ROOT
    
    if not datas_root.exists():
        print(f"Error: Datas root directory not found: {datas_root}")
        sys.exit(1)
    
    if not CALCULATE_SCRIPT.exists():
        print(f"Error: Calculate script not found: {CALCULATE_SCRIPT}")
        sys.exit(1)
    
    print("=" * 60)
    print("EconAgent 风险特征批量计算")
    print("=" * 60)
    print(f"Datas root: {datas_root}")
    print(f"Calculate script: {CALCULATE_SCRIPT}")
    print(f"Max workers: {args.max_workers}")
    if args.models:
        print(f"Models filter: {args.models}")
    print("=" * 60 + "\n")
    
    # 发现所有任务
    print("Discovering tasks...")
    tasks = discover_tasks(datas_root, args.models)
    print(f"Found {len(tasks)} tasks\n")
    
    if len(tasks) == 0:
        print("No tasks found. Exiting.")
        sys.exit(0)
    
    # 显示任务列表
    print("Tasks to process:")
    for model_name, seed_name, data_dir in tasks:
        print(f"  - {model_name}/{seed_name}")
    print()
    
    # 创建日志目录
    log_dir = SCRIPT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成日志文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"batch_run_{timestamp}.log"
    
    print(f"Log file: {log_file}\n")
    
    # 执行任务
    results = []
    success_count = 0
    error_count = 0
    
    with open(log_file, 'w', encoding='utf-8') as log_f:
        log_f.write(f"EconAgent Risk Feature Batch Calculation\n")
        log_f.write(f"Timestamp: {timestamp}\n")
        log_f.write(f"Total tasks: {len(tasks)}\n")
        log_f.write(f"Max workers: {args.max_workers}\n")
        if args.models:
            log_f.write(f"Models filter: {args.models}\n")
        log_f.write("=" * 60 + "\n\n")
        
        if args.max_workers == 1:
            # 串行执行
            print("Running tasks sequentially...\n")
            for i, (model_name, seed_name, data_dir) in enumerate(tasks, 1):
                print(f"[{i}/{len(tasks)}] Processing {model_name}/{seed_name}...")
                result = run_risk_feature_calculation(
                    model_name, seed_name, data_dir, CALCULATE_SCRIPT, PROJECT_ROOT
                )
                results.append(result)
                
                status = result['status']
                if status == 'success':
                    success_count += 1
                    print(f"  ✓ Success: {result['message']}")
                else:
                    error_count += 1
                    print(f"  ✗ Error: {result['message']}")
                    if 'stderr' in result:
                        print(f"    {result['stderr'][:200]}")
                
                log_f.write(f"{model_name}/{seed_name}: {status} - {result['message']}\n")
                if 'stderr' in result and result['stderr']:
                    log_f.write(f"  {result['stderr']}\n")
                log_f.flush()
                print()
        else:
            # 并行执行
            print(f"Running tasks in parallel (max_workers={args.max_workers})...\n")
            with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
                # 提交所有任务
                future_to_task = {
                    executor.submit(
                        run_risk_feature_calculation,
                        model_name, seed_name, data_dir, CALCULATE_SCRIPT, PROJECT_ROOT
                    ): (model_name, seed_name)
                    for model_name, seed_name, data_dir in tasks
                }
                
                # 处理完成的任务
                completed = 0
                for future in as_completed(future_to_task):
                    model_name, seed_name = future_to_task[future]
                    completed += 1
                    
                    try:
                        result = future.result()
                        results.append(result)
                        
                        status = result['status']
                        if status == 'success':
                            success_count += 1
                            print(f"[{completed}/{len(tasks)}] ✓ {model_name}/{seed_name}: {result['message']}")
                        else:
                            error_count += 1
                            print(f"[{completed}/{len(tasks)}] ✗ {model_name}/{seed_name}: {result['message']}")
                            if 'stderr' in result:
                                print(f"  {result['stderr'][:200]}")
                        
                        log_f.write(f"{model_name}/{seed_name}: {status} - {result['message']}\n")
                        if 'stderr' in result and result['stderr']:
                            log_f.write(f"  {result['stderr']}\n")
                        log_f.flush()
                    except Exception as e:
                        error_count += 1
                        error_msg = f"Exception: {str(e)}"
                        print(f"[{completed}/{len(tasks)}] ✗ {model_name}/{seed_name}: {error_msg}")
                        results.append({
                            'model': model_name,
                            'seed': seed_name,
                            'status': 'error',
                            'message': error_msg
                        })
                        log_f.write(f"{model_name}/{seed_name}: error - {error_msg}\n")
                        log_f.flush()
        
        # 写入汇总信息
        log_f.write("\n" + "=" * 60 + "\n")
        log_f.write(f"Summary:\n")
        log_f.write(f"  Total: {len(tasks)}\n")
        log_f.write(f"  Success: {success_count}\n")
        log_f.write(f"  Error: {error_count}\n")
        log_f.write("=" * 60 + "\n")
    
    # 打印汇总
    print("=" * 60)
    print("Summary:")
    print(f"  Total: {len(tasks)}")
    print(f"  Success: {success_count}")
    print(f"  Error: {error_count}")
    print("=" * 60)
    print(f"\nLog file: {log_file}")
    
    if error_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
