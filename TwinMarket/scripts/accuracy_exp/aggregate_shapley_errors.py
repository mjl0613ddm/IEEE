#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总多个模型的Shapley误差分析结果
"""

import json
import pandas as pd
from pathlib import Path
import sys
import argparse

def aggregate_results(base_dir, models, n_samples_list, seeds):
    """汇总所有模型的结果"""
    results = []
    all_model_results = {}
    
    for model in models:
        # 将斜杠替换为下划线，确保目录名称安全
        model_safe = model.replace('/', '_')
        output_dir = base_dir / "results" / f"{model_safe}_shapley_error_analyse"
        error_json = output_dir / "error_analysis_results.json"
        
        if not error_json.exists():
            print(f"⚠️  警告: 模型 {model} 的误差分析结果不存在: {error_json}")
            continue
        
        with open(error_json, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        all_model_results[model] = model_data
        
        # 提取每个采样次数的统计信息
        for result in model_data['results']:
            n_samples = result['n_samples']
            results.append({
                'model': model,
                'n_samples': n_samples,
                'mean_error': result['mean_error'],
                'std_error': result['std_error'],
                'min_error': result['min_error'],
                'max_error': result['max_error'],
                'num_seeds': len(result['errors'])
            })
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 保存详细结果
    detailed_csv = base_dir / "results" / "shapley_error_aggregate" / "detailed_errors.csv"
    df.to_csv(detailed_csv, index=False)
    print(f"✅ 详细结果已保存到: {detailed_csv}")
    
    # 按采样次数汇总
    summary_data = []
    for n_samples in n_samples_list:
        subset = df[df['n_samples'] == n_samples]
        if len(subset) > 0:
            summary_data.append({
                'n_samples': n_samples,
                'mean_error_across_models': subset['mean_error'].mean(),
                'std_error_across_models': subset['mean_error'].std(),
                'min_error_across_models': subset['mean_error'].min(),
                'max_error_across_models': subset['mean_error'].max(),
                'num_models': len(subset)
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv = base_dir / "results" / "shapley_error_aggregate" / "summary_by_n_samples.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"✅ 汇总结果已保存到: {summary_csv}")
    
    # 按模型汇总
    model_summary_data = []
    for model in models:
        model_subset = df[df['model'] == model]
        if len(model_subset) > 0:
            model_summary_data.append({
                'model': model,
                'mean_error_across_samples': model_subset['mean_error'].mean(),
                'std_error_across_samples': model_subset['mean_error'].std(),
                'min_error_across_samples': model_subset['mean_error'].min(),
                'max_error_across_samples': model_subset['mean_error'].max(),
                'num_n_samples': len(model_subset)
            })
    
    model_summary_df = pd.DataFrame(model_summary_data)
    model_summary_csv = base_dir / "results" / "shapley_error_aggregate" / "summary_by_model.csv"
    model_summary_df.to_csv(model_summary_csv, index=False)
    print(f"✅ 按模型汇总结果已保存到: {model_summary_csv}")
    
    # 保存完整的JSON结果
    aggregate_json = {
        'models': models,
        'n_samples_list': n_samples_list,
        'seeds': seeds,
        'model_results': all_model_results,
        'summary_by_n_samples': summary_data,
        'summary_by_model': model_summary_data
    }
    
    aggregate_json_file = base_dir / "results" / "shapley_error_aggregate" / "aggregate_results.json"
    with open(aggregate_json_file, 'w', encoding='utf-8') as f:
        json.dump(aggregate_json, f, indent=2, ensure_ascii=False)
    print(f"✅ 完整汇总JSON已保存到: {aggregate_json_file}")
    
    # 打印摘要
    print("\n" + "="*60)
    print("汇总结果摘要")
    print("="*60)
    print("\n按采样次数汇总:")
    print(summary_df.to_string(index=False))
    print("\n按模型汇总:")
    print(model_summary_df.to_string(index=False))
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='汇总多个模型的Shapley误差分析结果')
    parser.add_argument('--base_dir', type=str, required=True, help='项目根目录')
    parser.add_argument('--models', type=str, nargs='+', required=True, help='模型列表')
    parser.add_argument('--n_samples_list', type=int, nargs='+', default=[10, 100, 1000, 10000], help='MC采样次数列表')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 401, 45, 46, 613, 48, 49, 50, 51], help='随机种子列表')
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    aggregate_results(base_dir, args.models, args.n_samples_list, args.seeds)
