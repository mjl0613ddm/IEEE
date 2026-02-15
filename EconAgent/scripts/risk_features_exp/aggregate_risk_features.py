#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总所有模型的risk features计算结果
生成按模型汇总的结果和全模型对比结果
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# 设置项目根目录
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # ACL24-EconAgent
RESULTS_ROOT = PROJECT_ROOT / "results"
RISK_FEATURE_ROOT = RESULTS_ROOT / "risk_features_exp"

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


def discover_risk_feature_results(risk_feature_root: Path) -> List[Tuple[str, str, Path]]:
    """
    发现所有risk feature计算结果
    
    Args:
        risk_feature_root: risk_feature根目录
        
    Returns:
        [(model_name, seed_name, json_path), ...] 列表
    """
    if not risk_feature_root.exists():
        print(f"Warning: Risk feature root directory not found: {risk_feature_root}")
        return []
    
    discovered_results = []
    
    # 遍历所有模型目录
    for model_dir in risk_feature_root.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        # 跳过汇总文件
        if model_name.startswith('all_models') or model_name.endswith('_summary'):
            continue
        
        # 遍历该模型下的所有seed目录
        for seed_dir in model_dir.iterdir():
            if not seed_dir.is_dir():
                continue
            
            seed_name = seed_dir.name
            
            # 检查是否存在risk_features.json
            json_path = seed_dir / "risk_features.json"
            if json_path.exists():
                discovered_results.append((model_name, seed_name, json_path))
    
    return discovered_results


def aggregate_model_results(model_name: str, results: List[Dict]) -> Dict:
    """
    汇总单个模型的所有seed结果
    
    Args:
        model_name: 模型名称
        results: 该模型的所有结果列表
        
    Returns:
        包含汇总统计的字典
    """
    if not results:
        return None
    
    # 提取所有指标值
    metric_values = {metric: [] for metric in METRIC_NAMES}
    
    for result in results:
        for metric in METRIC_NAMES:
            value = result.get(metric)
            if value is not None:
                metric_values[metric].append(value)
    
    # 计算统计量
    summary = {
        'model': model_name,
        'num_seeds': len(results),
        'metrics': {}
    }
    
    for metric in METRIC_NAMES:
        values = metric_values[metric]
        if values:
            summary['metrics'][metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'count': len(values)
            }
        else:
            summary['metrics'][metric] = {
                'mean': None,
                'std': None,
                'min': None,
                'max': None,
                'median': None,
                'count': 0
            }
    
    return summary


def save_model_summary(model_name: str, summary: Dict, risk_feature_root: Path):
    """
    保存单模型汇总结果
    
    Args:
        model_name: 模型名称
        summary: 汇总字典
        risk_feature_root: risk_feature根目录
    """
    model_dir = risk_feature_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存JSON
    json_path = model_dir / f"{model_name}_summary.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 保存CSV
    csv_path = model_dir / f"{model_name}_summary.csv"
    rows = []
    for metric in METRIC_NAMES:
        metric_data = summary['metrics'].get(metric, {})
        rows.append({
            'Metric': METRIC_DISPLAY_NAMES.get(metric, metric),
            'Mean': metric_data.get('mean'),
            'Std': metric_data.get('std'),
            'Min': metric_data.get('min'),
            'Max': metric_data.get('max'),
            'Median': metric_data.get('median'),
            'Count': metric_data.get('count')
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    
    print(f"Saved {model_name} summary: {json_path}, {csv_path}")


def aggregate_all_models(all_results: Dict[str, List[Dict]], risk_feature_root: Path) -> Dict:
    """
    汇总所有模型的结果
    
    Args:
        all_results: {model_name: [results, ...]} 字典
        risk_feature_root: risk_feature根目录
        
    Returns:
        包含所有模型汇总的字典
    """
    model_summaries = {}
    
    # 计算每个模型的汇总
    for model_name, results in all_results.items():
        summary = aggregate_model_results(model_name, results)
        if summary:
            model_summaries[model_name] = summary
            save_model_summary(model_name, summary, risk_feature_root)
    
    # 计算所有模型的总体汇总（按模型均值的均值）
    all_models_summary = {
        'num_models': len(model_summaries),
        'models': {}
    }
    
    for metric in METRIC_NAMES:
        model_means = []
        for model_name, summary in model_summaries.items():
            metric_data = summary['metrics'].get(metric, {})
            mean_value = metric_data.get('mean')
            if mean_value is not None:
                model_means.append(mean_value)
        
        if model_means:
            all_models_summary['models'][metric] = {
                'mean': float(np.mean(model_means)),
                'std': float(np.std(model_means)),
                'min': float(np.min(model_means)),
                'max': float(np.max(model_means)),
                'median': float(np.median(model_means)),
                'count': len(model_means)
            }
        else:
            all_models_summary['models'][metric] = {
                'mean': None,
                'std': None,
                'min': None,
                'max': None,
                'median': None,
                'count': 0
            }
    
    # 保存所有模型汇总
    json_path = risk_feature_root / "all_models_summary.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_models_summary, f, indent=2, ensure_ascii=False)
    
    # 保存CSV（按模型汇总）
    csv_path = risk_feature_root / "all_models_summary.csv"
    rows = []
    for model_name, summary in model_summaries.items():
        for metric in METRIC_NAMES:
            metric_data = summary['metrics'].get(metric, {})
            rows.append({
                'Model': model_name,
                'Metric': METRIC_DISPLAY_NAMES.get(metric, metric),
                'Mean': metric_data.get('mean'),
                'Std': metric_data.get('std'),
                'Min': metric_data.get('min'),
                'Max': metric_data.get('max'),
                'Median': metric_data.get('median'),
                'Count': metric_data.get('count')
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    
    print(f"Saved all models summary: {json_path}, {csv_path}")
    
    return all_models_summary


def main():
    parser = argparse.ArgumentParser(
        description='汇总所有模型的risk features计算结果'
    )
    
    parser.add_argument(
        '--risk_feature_root',
        type=str,
        default=None,
        help='Risk feature根目录（默认: PROJECT_ROOT/results/risk_features_exp）'
    )
    
    args = parser.parse_args()
    
    # 解析路径
    if args.risk_feature_root:
        risk_feature_root = Path(args.risk_feature_root)
    else:
        risk_feature_root = RISK_FEATURE_ROOT
    
    print("=" * 60)
    print("EconAgent 风险特征汇总")
    print("=" * 60)
    print(f"Risk feature root: {risk_feature_root}")
    print("=" * 60 + "\n")
    
    # 发现所有结果
    print("Discovering risk feature results...")
    results = discover_risk_feature_results(risk_feature_root)
    print(f"Found {len(results)} results\n")
    
    if len(results) == 0:
        print("No results found. Exiting.")
        return
    
    # 按模型分组
    all_results = {}
    for model_name, seed_name, json_path in results:
        data = load_risk_features_json(json_path)
        if data is None:
            continue
        
        if model_name not in all_results:
            all_results[model_name] = []
        all_results[model_name].append(data)
    
    print(f"Found {len(all_results)} models:\n")
    for model_name, model_results in all_results.items():
        print(f"  {model_name}: {len(model_results)} seeds")
    print()
    
    # 汇总所有模型
    print("Aggregating results...\n")
    all_models_summary = aggregate_all_models(all_results, risk_feature_root)
    
    print("\n" + "=" * 60)
    print("Aggregation completed!")
    print("=" * 60)
    print(f"Total models: {len(all_results)}")
    print(f"Total results: {len(results)}")
    print("=" * 60)


if __name__ == '__main__':
    main()
