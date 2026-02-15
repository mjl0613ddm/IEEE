#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行批量计算多个模拟结果的Shapley值
使用方法:
    # 自动发现所有数据文件夹
    python scripts/shapley_culculate/batch_run_shapley.py --auto_discover --max_workers 64
    
    # 指定特定模型
    python scripts/shapley_culculate/batch_run_shapley.py --auto_discover --models claude gpt --max_workers 64
"""

import os
import sys
import json
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
DATA_ROOT = PROJECT_ROOT / "data"
SHAPLEY_SCRIPT = PROJECT_ROOT / "process" / "shapley.py"


def compute_baseline_fixed_values(baseline_actions_json_path: Path) -> Tuple[float, float]:
    """
    从baseline-0.8的all_actions.json中计算固定的work和consumption值
    
    Args:
        baseline_actions_json_path: baseline actions JSON文件路径
        
    Returns:
        (baseline_work, baseline_consumption) 元组
    """
    if not baseline_actions_json_path.exists():
        raise FileNotFoundError(f"Baseline actions file not found: {baseline_actions_json_path}")
    
    with open(baseline_actions_json_path, 'r') as f:
        all_actions = json.load(f)
    
    work_values = []
    consumption_values = []
    
    # 遍历所有时间步
    for step_key, step_actions in all_actions.items():
        if not step_key.startswith('step_'):
            continue
        
        # 遍历所有agent
        for agent_id_str, action in step_actions.items():
            if isinstance(action, list) and len(action) >= 2:
                work = float(action[0])  # 0或1
                consumption = float(action[1]) * 0.02  # 从量化值(0-50)转换为原始值(0-1)
                work_values.append(work)
                consumption_values.append(consumption)
    
    if not work_values or not consumption_values:
        # 如果没有找到actions，返回默认值
        print(f"Warning: No valid actions found in baseline file, using defaults")
        return 1.0, 0.8
    
    baseline_work = sum(work_values) / len(work_values)
    baseline_consumption = sum(consumption_values) / len(consumption_values)
    
    return baseline_work, baseline_consumption


def discover_data_folders(datas_dir: Path, models: List[str] = None) -> List[Tuple[str, str, Path]]:
    """
    自动发现datas目录下所有不带_rm后缀的子文件夹
    
    Args:
        datas_dir: datas根目录路径
        models: 要处理的模型列表，如果为None则处理所有模型
        
    Returns:
        [(model, seed, folder_path), ...] 列表
    """
    if models is None:
        models = ['claude', 'ds', 'gpt', 'llama', 'qwen']
    
    discovered_folders = []
    
    for model in models:
        model_dir = datas_dir / model
        if not model_dir.exists():
            continue
        
        # 查找所有子文件夹，排除带_rm后缀的
        for subfolder in model_dir.iterdir():
            if not subfolder.is_dir():
                continue
            if subfolder.name.startswith('.'):
                continue
            if subfolder.name.endswith('_rm'):
                continue
            
            # 提取seed（从文件夹名称中，如 claude_42 -> 42）
            folder_name = subfolder.name
            if '_' in folder_name:
                seed = folder_name.split('_', 1)[1] if folder_name.count('_') >= 1 else folder_name
            else:
                seed = folder_name
            
            # 检查是否包含actions_json
            actions_json = subfolder / "actions_json" / "all_actions.json"
            if actions_json.exists():
                discovered_folders.append((model, seed, subfolder))
    
    return discovered_folders


def run_shapley(model: str, seed: str, folder_path: Path, args, baseline_work: float, baseline_consumption: float, baseline_metrics_csv: Path):
    """
    运行单个Shapley计算任务
    
    Args:
        model: 模型名称（如 'claude'）
        seed: seed值（如 '42'）
        folder_path: 数据文件夹路径
        args: 命令行参数
        baseline_work: baseline work值
        baseline_consumption: baseline consumption值
        baseline_metrics_csv: baseline metrics CSV路径
    """
    actions_json = folder_path / "actions_json" / "all_actions.json"
    if not actions_json.exists():
        return {
            'model': model,
            'seed': seed,
            'status': 'error',
            'message': f'Could not find all_actions.json for {model}/{seed}'
        }
    
    # 设置输出目录
    output_dir = folder_path / "shapley"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 自动查找real_metrics_csv
    real_metrics_csv = folder_path / "metrics_csv" / "world_metrics.csv"
    
    # 构建命令（使用绝对路径）
    cmd = [
        sys.executable,
        str(SHAPLEY_SCRIPT.absolute()),
        '--actions_json', str(actions_json.absolute()),
        '--output_dir', str(output_dir.absolute()),
        '--num_agents', str(args.num_agents),
        '--episode_length', str(args.episode_length),
        '--n_samples', str(args.n_samples),
        '--seed', str(args.seed),
        '--metric_name', args.metric_name,
        '--baseline_type', 'fixed',  # 统一使用fixed baseline
        '--baseline_work', str(baseline_work),
        '--baseline_consumption', str(baseline_consumption),
        '--baseline_metrics_csv', str(baseline_metrics_csv.absolute()),
    ]
    
    # 添加real_metrics_csv（如果存在）
    if real_metrics_csv.exists():
        cmd.extend(['--real_metrics_csv', str(real_metrics_csv.absolute())])
    
    # 添加可选参数
    if args.inflation_threshold is not None:
        cmd.extend(['--inflation_threshold', str(args.inflation_threshold)])
    
    if args.risk_lambda is not None:
        cmd.extend(['--risk_lambda', str(args.risk_lambda)])
    
    if args.use_metric_directly:
        cmd.append('--use_metric_directly')
    
    if args.use_probabilistic_baseline:
        cmd.append('--use_probabilistic_baseline')
    
    # 运行命令
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=args.timeout if args.timeout else None
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            return {
                'model': model,
                'seed': seed,
                'status': 'success',
                'elapsed_time': elapsed_time,
                'message': f'Completed in {elapsed_time:.1f}s'
            }
        else:
            return {
                'model': model,
                'seed': seed,
                'status': 'error',
                'elapsed_time': elapsed_time,
                'message': f'Failed with return code {result.returncode}',
                'stderr': result.stderr[-500:] if result.stderr else ''  # 只保留最后500字符
            }
    except subprocess.TimeoutExpired:
        return {
            'model': model,
            'seed': seed,
            'status': 'timeout',
            'elapsed_time': args.timeout if args.timeout else 0,
            'message': f'Timeout after {args.timeout}s'
        }
    except Exception as e:
        return {
            'model': model,
            'seed': seed,
            'status': 'error',
            'message': f'Exception: {str(e)}'
        }


def main():
    parser = argparse.ArgumentParser(
        description='并行批量计算多个模拟结果的Shapley值',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动发现所有数据文件夹，使用64个CPU并行
  python scripts/shapley_culculate/batch_run_shapley.py --auto_discover --max_workers 64
  
  # 自动发现指定模型的数据文件夹
  python scripts/shapley_culculate/batch_run_shapley.py --auto_discover --models claude gpt --max_workers 64
  
  # 手动指定数据文件夹（格式：model/seed或model_seed）
  python scripts/shapley_culculate/batch_run_shapley.py --simulations claude/42 claude/43 gpt/42 --max_workers 64
        """
    )
    
    parser.add_argument(
        '--auto_discover',
        action='store_true',
        help='自动发现datas目录下所有不带_rm后缀的文件夹（如果未指定--simulations，则默认启用）'
    )
    
    parser.add_argument(
        '--simulations', '-s',
        nargs='+',
        required=False,
        help='手动指定的模拟结果列表（格式: model/seed 或 model_seed，例如: claude/42 gpt/43）'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        choices=['claude', 'ds', 'gpt', 'llama', 'qwen'],
        help='要处理的模型列表（仅在auto_discover模式下使用，默认: 所有模型）'
    )
    
    parser.add_argument(
        '--max_workers', '-j',
        type=int,
        default=64,
        help='最大并行任务数（默认: 64）'
    )
    
    parser.add_argument(
        '--baseline_dir',
        type=str,
        default=None,
        help='Baseline目录路径（默认: data/baseline-0.8）'
    )
    
    # Shapley计算参数
    parser.add_argument('--num_agents', type=int, default=10, help='Agent数量')
    parser.add_argument('--episode_length', type=int, default=50, help='Episode长度')
    parser.add_argument('--n_samples', type=int, default=5000, help='MC采样次数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子（用于Shapley计算的随机性，不是数据seed）')
    parser.add_argument('--metric_name', type=str, default='risk_indicator_naive',
                       choices=['price_inflation_rate', 'price', 'interest_rate', 'unemployment_rate', 'risk_indicator_naive'],
                       help='使用的指标名称')
    parser.add_argument('--inflation_threshold', type=float, default=None,
                       help='通胀阈值')
    parser.add_argument('--risk_lambda', type=float, default=None,
                       help='RiskMetrics lambda参数（默认: 0.94）')
    parser.add_argument('--use_metric_directly', action='store_true',
                       help='直接使用指标值而不是基于阈值的风险计算')
    parser.add_argument('--use_probabilistic_baseline', action='store_true',
                       help='使用概率性baseline')
    parser.add_argument('--timeout', type=int, default=None,
                       help='每个任务的最大执行时间（秒），超时则终止')
    
    args = parser.parse_args()
    
    # 设置baseline目录
    if args.baseline_dir is None:
        baseline_dir = DATA_ROOT / "baseline-0.8"
    else:
        baseline_dir = Path(args.baseline_dir)
    
    baseline_actions_json = baseline_dir / "actions_json" / "all_actions.json"
    baseline_metrics_csv = baseline_dir / "metrics_csv" / "world_metrics.csv"
    
    # 验证baseline文件是否存在
    if not baseline_actions_json.exists():
        print(f"❌ 错误: Baseline actions文件不存在: {baseline_actions_json}")
        return 1
    
    if not baseline_metrics_csv.exists():
        print(f"❌ 错误: Baseline metrics CSV文件不存在: {baseline_metrics_csv}")
        return 1
    
    # 计算baseline固定值
    print("计算baseline固定值...")
    try:
        baseline_work, baseline_consumption = compute_baseline_fixed_values(baseline_actions_json)
        print(f"  Baseline work: {baseline_work:.4f}")
        print(f"  Baseline consumption: {baseline_consumption:.4f}")
    except Exception as e:
        print(f"❌ 错误: 计算baseline固定值失败: {e}")
        return 1
    
    # 获取要处理的数据文件夹列表
    # 如果没有指定--simulations，则默认使用自动发现模式
    use_auto_discover = args.auto_discover or (args.simulations is None)
    tasks = []
    
    if use_auto_discover:
        # 自动发现模式
        print(f"\n自动发现数据文件夹（从 {DATAS_ROOT}）...")
        discovered_folders = discover_data_folders(DATAS_ROOT, args.models)
        
        if not discovered_folders:
            print("❌ 错误: 没有找到有效的数据文件夹")
            return 1
        
        print(f"找到 {len(discovered_folders)} 个数据文件夹:")
        for model, seed, folder_path in discovered_folders[:10]:  # 只显示前10个
            print(f"  {model}/{seed}")
        if len(discovered_folders) > 10:
            print(f"  ... 还有 {len(discovered_folders) - 10} 个文件夹")
        
        tasks = discovered_folders
    else:
        # 手动指定模式
        print(f"\n处理手动指定的数据文件夹...")
        for sim in args.simulations:
            # 支持格式: model/seed 或 model_seed
            if '/' in sim:
                model, seed = sim.split('/', 1)
            elif '_' in sim:
                parts = sim.split('_', 1)
                model = parts[0]
                seed = parts[1]
            else:
                print(f"⚠️  警告: 跳过 {sim}，格式不正确（应为 model/seed 或 model_seed）")
                continue
            
            folder_path = DATAS_ROOT / model / f"{model}_{seed}"
            actions_json = folder_path / "actions_json" / "all_actions.json"
            
            if not actions_json.exists():
                print(f"⚠️  警告: 跳过 {sim}，未找到 all_actions.json")
                continue
            
            tasks.append((model, seed, folder_path))
        
        if not tasks:
            print("❌ 错误: 没有有效的模拟结果")
            return 1
    
    print("\n" + "="*80)
    print("并行批量计算Shapley值")
    print("="*80)
    print(f"数据文件夹数量: {len(tasks)}")
    print(f"并行任务数: {args.max_workers}")
    print(f"Baseline目录: {baseline_dir}")
    print(f"参数: num_agents={args.num_agents}, episode_length={args.episode_length}")
    print(f"      n_samples={args.n_samples}, metric_name={args.metric_name}")
    print(f"      baseline_work={baseline_work:.4f}, baseline_consumption={baseline_consumption:.4f}")
    print("="*80)
    print()
    
    # 并行执行
    start_time = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # 提交所有任务
        print("正在提交任务...")
        future_to_task = {
            executor.submit(run_shapley, model, seed, folder_path, args, baseline_work, baseline_consumption, baseline_metrics_csv): (model, seed)
            for model, seed, folder_path in tasks
        }
        print(f"✅ 已提交 {len(future_to_task)} 个任务，正在并行执行中...")
        print("   (任务完成后会显示结果，请耐心等待)\n")
        
        # 处理完成的任务
        completed = 0
        for future in as_completed(future_to_task):
            model, seed = future_to_task[future]
            completed += 1
            try:
                result = future.result()
                results.append(result)
                
                if result['status'] == 'success':
                    print(f"✅ [{completed}/{len(tasks)}] {model}/{seed}: {result['message']}")
                elif result['status'] == 'timeout':
                    print(f"⏱️  [{completed}/{len(tasks)}] {model}/{seed}: {result['message']}")
                else:
                    print(f"❌ [{completed}/{len(tasks)}] {model}/{seed}: {result['message']}")
                    if 'stderr' in result and result['stderr']:
                        print(f"   错误信息: {result['stderr'][:200]}")
            except Exception as e:
                print(f"❌ [{completed}/{len(tasks)}] {model}/{seed}: 异常 - {str(e)}")
                results.append({
                    'model': model,
                    'seed': seed,
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
    
    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())