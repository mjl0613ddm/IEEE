#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行批量计算所有模型的risk features
使用方法:
    python batch_calculate_risk_features.py --max_workers 64
    python batch_calculate_risk_features.py --max_workers 64 --models deepseek claude
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
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # TwinMarket
RESULTS_ROOT = PROJECT_ROOT / "results"
CALCULATE_SCRIPT = SCRIPT_DIR / "calculate_risk_features.py"


def discover_tasks(results_root: Path, models: List[str] = None) -> List[Tuple[str, str, Path]]:
    """
    自动发现results目录下所有包含shapley文件的模型/seed目录
    
    Args:
        results_root: results根目录路径
        models: 要处理的模型列表（部分匹配），如果为None则处理所有模型
        
    Returns:
        [(model_name, seed_name, results_dir), ...] 列表
    """
    if models is None:
        models = []
    
    discovered_tasks = []
    
    # 遍历所有模型目录
    for model_dir in results_root.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        # 跳过一些非模型目录
        if model_name in ['accuracy', 'faithfulness_exp', 'shapley_error_aggregate', 'risk_feature']:
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
        
        # 遍历该模型下的所有子目录（如 model_42, model_43等）
        for seed_dir in model_dir.iterdir():
            if not seed_dir.is_dir():
                continue
            
            seed_name = seed_dir.name
            shapley_dir = seed_dir / "shapley"
            
            # 检查是否存在shapley目录和文件
            if not shapley_dir.exists() or not shapley_dir.is_dir():
                continue
            
            shapley_matrix_files = list(shapley_dir.glob("shapley_matrix_*.npy"))
            if not shapley_matrix_files:
                continue
            
            discovered_tasks.append((model_name, seed_name, seed_dir))
    
    return discovered_tasks


def run_risk_feature_calculation(model_name: str, seed_name: str, results_dir: Path, 
                                  calculate_script: Path, project_root: Path) -> dict:
    """
    运行单个risk feature计算任务
    
    Args:
        model_name: 模型名称
        seed_name: seed名称
        results_dir: 结果目录路径
        calculate_script: 计算脚本路径
        project_root: 项目根目录
        
    Returns:
        包含任务执行结果的字典
    """
    # 设置输出目录
    output_dir = project_root / "results" / "risk_feature" / model_name / seed_name
    
    # 构建命令
    cmd = [
        sys.executable,
        str(calculate_script.absolute()),
        '--results_dir', str(results_dir.absolute()),
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
  python batch_calculate_risk_features.py --max_workers 64 --models deepseek claude
        """
    )
    
    parser.add_argument(
        '--results_root',
        type=str,
        default=None,
        help='Results根目录（默认: PROJECT_ROOT/results）'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        help='要处理的模型列表（部分匹配，例如: deepseek claude），默认处理所有模型'
    )
    
    parser.add_argument(
        '--max_workers', '-j',
        type=int,
        default=1,
        help='最大并行任务数（默认: 1）'
    )
    
    args = parser.parse_args()
    
    # 设置results根目录
    if args.results_root:
        results_root = Path(args.results_root)
    else:
        results_root = RESULTS_ROOT
    
    # 验证results目录是否存在
    if not results_root.exists():
        print(f"❌ 错误: Results目录不存在: {results_root}")
        return 1
    
    # 验证计算脚本是否存在
    if not CALCULATE_SCRIPT.exists():
        print(f"❌ 错误: 计算脚本不存在: {CALCULATE_SCRIPT}")
        return 1
    
    # 发现所有任务
    print(f"\n自动发现任务（从 {results_root}）...")
    tasks = discover_tasks(results_root, args.models)
    
    if not tasks:
        print("❌ 错误: 没有找到有效的任务")
        return 1
    
    print(f"找到 {len(tasks)} 个任务:")
    for model_name, seed_name, _ in tasks[:10]:  # 只显示前10个
        print(f"  {model_name}/{seed_name}")
    if len(tasks) > 10:
        print(f"  ... 还有 {len(tasks) - 10} 个任务")
    
    print("\n" + "="*80)
    print("并行批量计算Risk Features")
    print("="*80)
    print(f"任务数量: {len(tasks)}")
    print(f"并行任务数: {args.max_workers}")
    print(f"Results目录: {results_root}")
    print("="*80)
    print()
    
    # 并行执行
    start_time = time.time()
    results = []
    
    if args.max_workers == 1:
        # 串行执行
        print("串行执行模式...\n")
        for model_name, seed_name, results_dir in tasks:
            result = run_risk_feature_calculation(
                model_name, seed_name, results_dir,
                CALCULATE_SCRIPT, PROJECT_ROOT
            )
            results.append(result)
            
            if result['status'] == 'success':
                print(f"✅ [{len(results)}/{len(tasks)}] {model_name}/{seed_name}: {result['message']}")
            elif result['status'] == 'timeout':
                print(f"⏱️  [{len(results)}/{len(tasks)}] {model_name}/{seed_name}: {result['message']}")
            else:
                print(f"❌ [{len(results)}/{len(tasks)}] {model_name}/{seed_name}: {result['message']}")
                if 'stderr' in result and result['stderr']:
                    print(f"   错误信息: {result['stderr'][:200]}")
    else:
        # 并行执行
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            # 提交所有任务
            print("正在提交任务...")
            future_to_task = {
                executor.submit(
                    run_risk_feature_calculation,
                    model_name, seed_name, results_dir,
                    CALCULATE_SCRIPT, PROJECT_ROOT
                ): (model_name, seed_name)
                for model_name, seed_name, results_dir in tasks
            }
            print(f"✅ 已提交 {len(future_to_task)} 个任务，正在并行执行中...")
            print("   (任务完成后会显示结果，请耐心等待)\n")
            
            # 处理完成的任务
            completed = 0
            for future in as_completed(future_to_task):
                model_name, seed_name = future_to_task[future]
                completed += 1
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['status'] == 'success':
                        print(f"✅ [{completed}/{len(tasks)}] {model_name}/{seed_name}: {result['message']}")
                    elif result['status'] == 'timeout':
                        print(f"⏱️  [{completed}/{len(tasks)}] {model_name}/{seed_name}: {result['message']}")
                    else:
                        print(f"❌ [{completed}/{len(tasks)}] {model_name}/{seed_name}: {result['message']}")
                        if 'stderr' in result and result['stderr']:
                            print(f"   错误信息: {result['stderr'][:200]}")
                except Exception as e:
                    print(f"❌ [{completed}/{len(tasks)}] {model_name}/{seed_name}: 异常 - {str(e)}")
                    results.append({
                        'model': model_name,
                        'seed': seed_name,
                        'status': 'error',
                        'message': f'Exception: {str(e)}'
                    })
    
    total_time = time.time() - start_time
    
    # 汇总结果
    print()
    print("="*80)
    print("执行汇总")
    print("="*80)
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = sum(1 for r in results if r['status'] == 'error')
    timeout_count = sum(1 for r in results if r['status'] == 'timeout')
    
    print(f"总任务数: {len(results)}")
    print(f"成功: {success_count}")
    print(f"失败: {error_count}")
    print(f"超时: {timeout_count}")
    print(f"总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    
    if success_count > 0:
        avg_time = sum(r.get('elapsed_time', 0) for r in results if r['status'] == 'success') / success_count
        print(f"平均每个任务耗时: {avg_time:.1f}秒")
    
    if error_count > 0:
        print()
        print("失败的任务:")
        for r in results:
            if r['status'] == 'error':
                print(f"  - {r.get('model', 'unknown')}/{r.get('seed', 'unknown')}: {r['message']}")
    
    # 保存结果到日志文件
    log_dir = SCRIPT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"batch_run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    with open(log_file, 'w') as f:
        f.write(f"Batch Run Results\n")
        f.write(f"Total tasks: {len(results)}\n")
        f.write(f"Success: {success_count}\n")
        f.write(f"Error: {error_count}\n")
        f.write(f"Timeout: {timeout_count}\n")
        f.write(f"Total time: {total_time:.1f}s\n\n")
        for r in results:
            f.write(f"{r['model']}/{r['seed']}: {r['status']} - {r['message']}\n")
    print(f"\n详细日志已保存到: {log_file}")
    
    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
