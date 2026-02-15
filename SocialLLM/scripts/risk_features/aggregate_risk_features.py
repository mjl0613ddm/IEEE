#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总所有模型的risk features计算结果
生成按模型汇总的结果和全模型对比结果

SocialLLM的结果结构：
results/{model_name}/{seed_name}/risk_feature/risk_features.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# 设置项目根目录
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # SocialLLM
RESULTS_ROOT = PROJECT_ROOT / "results"
RISK_FEATURE_OUTPUT_ROOT = RESULTS_ROOT / "risk_feature"

# 5个风险特征指标（按顺序：L_tm, G_ag, C_ag, Z_ag, G_be）
METRIC_NAMES = ['L_tm', 'G_ag', 'C_ag', 'Z_ag', 'G_be']
METRIC_DISPLAY_NAMES = {
    'L_tm': 'L_tm (Relative Risk Latency)',
    'G_ag': 'G_ag (Agent Risk Concentration)',
    'C_ag': 'C_ag (Risk-Instability Correlation)',
    'Z_ag': 'Z_ag (Agent Risk Synchronization)',
    'G_be': 'G_be (Behavioral Risk Concentration)'
}


def load_risk_features_json(json_path: Path) -> Dict:
    """
    加载单个risk_features.json文件
    
    Args:
        json_path: JSON文件路径
        
    Returns:
        包含指标和元数据的字典，如果文件不存在或格式错误则返回None
    """
    if not json_path.exists():
        return None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Warning: Failed to load {json_path}: {e}")
        return None


def discover_risk_feature_results(results_root: Path) -> List[Tuple[str, str, Path]]:
    """
    发现所有risk feature计算结果
    
    SocialLLM的结果结构：
    results/{model_name}/{seed_name}/risk_feature/risk_features.json
    
    Args:
        results_root: results根目录
        
    Returns:
        [(model_name, seed_name, json_path), ...] 列表
    """
    if not results_root.exists():
        print(f"Warning: Results root directory not found: {results_root}")
        return []
    
    discovered_results = []
    
    # 遍历所有模型目录
    for model_dir in results_root.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        # 跳过汇总文件目录
        if model_name == "risk_feature":
            continue
        
        # 遍历该模型下的所有seed目录
        for seed_dir in model_dir.iterdir():
            if not seed_dir.is_dir():
                continue
            
            seed_name = seed_dir.name
            
            # 跳过_rm结尾的文件夹
            if seed_name.endswith('_rm'):
                continue
            
            # 检查是否存在risk_feature/risk_features.json
            json_path = seed_dir / "risk_feature" / "risk_features.json"
            if json_path.exists():
                discovered_results.append((model_name, seed_name, json_path))
    
    return discovered_results


def aggregate_model_results(model_name: str, results: List[Dict]) -> Dict:
    """
    汇总单个模型的所有seed结果
    
    Args:
        model_name: 模型名称
        results: 该模型的所有结果列表（每个元素是一个seed的结果）
        
    Returns:
        包含汇总统计信息的字典
    """
    if not results:
        return None
    
    # 提取所有指标值
    metrics_data = {metric: [] for metric in METRIC_NAMES}
    metadata_list = []
    
    for result in results:
        for metric in METRIC_NAMES:
            if metric in result:
                value = result[metric]
                if value is not None:
                    metrics_data[metric].append(value)
        
        if 'metadata' in result:
            metadata_list.append(result['metadata'])
    
    # 计算统计信息
    summary = {
        'model': model_name,
        'num_seeds': len(results),
        'metrics': {}
    }
    
    for metric in METRIC_NAMES:
        values = metrics_data[metric]
        if not values:
            summary['metrics'][metric] = {
                'mean': None,
                'std': None,
                'min': None,
                'max': None,
                'median': None,
                'count': 0
            }
        else:
            values_array = np.array(values)
            summary['metrics'][metric] = {
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)) if len(values_array) > 1 else 0.0,
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'median': float(np.median(values_array)),
                'count': len(values_array)
            }
    
    # 保存元数据信息（从第一个结果中获取）
    if metadata_list:
        summary['metadata'] = {
            'num_agents': metadata_list[0].get('num_agents', 0),
            'num_timesteps': metadata_list[0].get('num_timesteps', 0),
        }
    
    return summary


def create_model_summary_csv(model_summary: Dict, output_path: Path):
    """
    创建单个模型的汇总CSV文件
    
    Args:
        model_summary: 模型汇总字典
        output_path: 输出CSV文件路径
    """
    rows = []
    
    for metric in METRIC_NAMES:
        if metric in model_summary['metrics']:
            stats = model_summary['metrics'][metric]
            rows.append({
                'Metric': METRIC_DISPLAY_NAMES.get(metric, metric),
                'Code': metric,
                'Mean': stats['mean'],
                'Std': stats['std'],
                'Min': stats['min'],
                'Max': stats['max'],
                'Median': stats['median'],
                'Count': stats['count']
            })
    
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format='%.6f')
    print(f"Saved model summary CSV to: {output_path}")


def create_all_models_comparison(all_model_summaries: List[Dict], output_dir: Path):
    """
    创建所有模型的对比CSV和JSON
    
    Args:
        all_model_summaries: 所有模型的汇总列表
        output_dir: 输出目录
    """
    # 创建对比CSV（每个指标一行，每列一个模型）
    comparison_data = {}
    
    for metric in METRIC_NAMES:
        comparison_data[metric] = {}
        for model_summary in all_model_summaries:
            model_name = model_summary['model']
            if metric in model_summary['metrics']:
                stats = model_summary['metrics'][metric]
                comparison_data[metric][model_name] = {
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'min': stats['min'],
                    'max': stats['max'],
                    'median': stats['median'],
                    'count': stats['count']
                }
    
    # 创建CSV（行：指标，列：模型，值：均值）
    csv_rows = []
    for metric in METRIC_NAMES:
        row = {
            'Metric': METRIC_DISPLAY_NAMES.get(metric, metric),
            'Code': metric
        }
        for model_summary in all_model_summaries:
            model_name = model_summary['model']
            if metric in model_summary['metrics']:
                row[f"{model_name}_mean"] = model_summary['metrics'][metric]['mean']
                row[f"{model_name}_std"] = model_summary['metrics'][metric]['std']
                row[f"{model_name}_count"] = model_summary['metrics'][metric]['count']
            else:
                row[f"{model_name}_mean"] = None
                row[f"{model_name}_std"] = None
                row[f"{model_name}_count"] = 0
        csv_rows.append(row)
    
    df = pd.DataFrame(csv_rows)
    csv_path = output_dir / "all_models_summary.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"Saved all models comparison CSV to: {csv_path}")
    
    # 创建JSON（包含完整统计信息）
    json_data = {
        'summary': {
            'num_models': len(all_model_summaries),
            'metrics': METRIC_NAMES,
            'generated_at': pd.Timestamp.now().isoformat()
        },
        'models': all_model_summaries
    }
    
    json_path = output_dir / "all_models_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"Saved all models comparison JSON to: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description='汇总所有模型的risk features计算结果'
    )
    
    parser.add_argument(
        '--results_root',
        type=str,
        default=None,
        help='Results根目录（默认: PROJECT_ROOT/results）'
    )
    
    parser.add_argument(
        '--output_root',
        type=str,
        default=None,
        help='汇总结果输出根目录（默认: PROJECT_ROOT/results/risk_feature）'
    )
    
    args = parser.parse_args()
    
    # 设置results根目录
    if args.results_root:
        results_root = Path(args.results_root)
    else:
        results_root = RESULTS_ROOT
    
    # 设置输出根目录
    if args.output_root:
        output_root = Path(args.output_root)
    else:
        output_root = RISK_FEATURE_OUTPUT_ROOT
    
    print("=" * 60)
    print("汇总Risk Features结果")
    print("=" * 60)
    print(f"Results root: {results_root}")
    print(f"Output root: {output_root}")
    print("=" * 60 + "\n")
    
    # 1. 发现所有结果
    print("Step 1: Discovering risk feature results...")
    all_results = discover_risk_feature_results(results_root)
    
    if not all_results:
        print("❌ 错误: 没有找到任何risk feature结果")
        return 1
    
    print(f"Found {len(all_results)} results from {len(set(r[0] for r in all_results))} models")
    print()
    
    # 2. 按模型分组加载结果
    print("Step 2: Loading results by model...")
    model_results = {}
    
    for model_name, seed_name, json_path in all_results:
        result = load_risk_features_json(json_path)
        if result is None:
            continue
        
        if model_name not in model_results:
            model_results[model_name] = []
        model_results[model_name].append(result)
    
    print(f"Loaded results from {len(model_results)} models:")
    for model_name, results in model_results.items():
        print(f"  {model_name}: {len(results)} seeds")
    print()
    
    # 3. 按模型汇总
    print("Step 3: Aggregating results by model...")
    model_summaries = []
    
    for model_name, results in sorted(model_results.items()):
        summary = aggregate_model_results(model_name, results)
        if summary is None:
            continue
        
        model_summaries.append(summary)
        
        # 保存单个模型的汇总JSON
        model_output_dir = output_root / model_name
        json_path = model_output_dir / f"{model_name}_summary.json"
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved {model_name} summary JSON to: {json_path}")
        
        # 保存单个模型的汇总CSV
        csv_path = model_output_dir / f"{model_name}_summary.csv"
        create_model_summary_csv(summary, csv_path)
        print()
    
    # 4. 创建全模型对比
    print("Step 4: Creating all models comparison...")
    create_all_models_comparison(model_summaries, output_root)
    print()
    
    # 5. 打印汇总统计
    print("=" * 60)
    print("汇总统计")
    print("=" * 60)
    print(f"总模型数: {len(model_summaries)}")
    print(f"总结果数: {len(all_results)}")
    
    print("\n各模型指标均值:")
    print(f"{'Model':<30} {'L_tm':<12} {'G_ag':<12} {'C_ag':<12} {'Z_ag':<12} {'G_be':<12}")
    print("-" * 90)
    
    for summary in sorted(model_summaries, key=lambda x: x['model']):
        model_name = summary['model'][:28]  # 截断长名称
        metrics_str = " ".join([
            f"{summary['metrics'][m]['mean']:>11.6f}" if summary['metrics'][m]['mean'] is not None else "      N/A  "
            for m in METRIC_NAMES
        ])
        print(f"{model_name:<30} {metrics_str}")
    
    print()
    print("=" * 60)
    print("汇总完成！")
    print("=" * 60)
    print(f"模型汇总保存在: {output_root}/{{model}}/{{model}}_summary.json/csv")
    print(f"全模型对比保存在: {output_root}/all_models_summary.json/csv")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
