#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析风险阈值脚本
分析每个模型下不带rm后缀的risk indicator naive最高值是第一个值的百分之多少
按模型分类汇总均值最大最小值，然后再汇总所有模型的结果
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import json


def calculate_risk_indicator_naive(df, lambda_param=0.94):
    """
    计算风险指标（基于Engle (1982)和Bollerslev (1986)）
    使用naive forecast方法：E_{t-1}[π_t] = π_{t-1}
    
    Args:
        df: 包含price列的DataFrame
        lambda_param: RiskMetrics参数λ，默认0.94
    
    Returns:
        risk_values数组
    """
    if 'price' not in df.columns:
        raise ValueError("CSV文件中缺少 'price' 列，无法计算风险指标")
    
    # 计算通胀率 π_t = log P_t - log P_{t-1}
    prices = df['price'].values
    log_prices = np.log(prices)
    pi_t = np.diff(log_prices)  # π_t = log P_t - log P_{t-1}
    
    # 在开头插入NaN以保持长度一致（第一个时间步没有前一个价格）
    pi_t = np.insert(pi_t, 0, np.nan)
    
    n = len(pi_t)
    
    # 初始化数组
    E_pi = np.full(n, np.nan)  # 预期通胀率
    e_t = np.full(n, np.nan)   # 预测误差
    h_t = np.full(n, np.nan)   # 风险指标
    
    # 计算预期和误差（naive forecast方法）
    for t in range(1, n):
        # Naive forecast: E_{t-1}[π_t] = π_{t-1}
        E_pi[t] = pi_t[t-1]
        
        # 计算预测误差 e_t = π_t - E_{t-1}[π_t]
        if not np.isnan(E_pi[t]) and not np.isnan(pi_t[t]):
            e_t[t] = pi_t[t] - E_pi[t]
    
    # 计算风险指标 h_t = λ * h_{t-1} + (1-λ) * e_{t-1}^2
    # 找到第一个有效的 e_t
    first_valid_idx = None
    for i in range(1, n):
        if not np.isnan(e_t[i]):
            first_valid_idx = i
            break
    
    if first_valid_idx is not None:
        # 初始化：h_t[first_valid_idx] = e_t[first_valid_idx]^2
        h_t[first_valid_idx] = e_t[first_valid_idx] ** 2
        
        # 递归计算：h_t = λ * h_{t-1} + (1-λ) * e_{t-1}^2
        for t in range(first_valid_idx + 1, n):
            if not np.isnan(e_t[t-1]) and not np.isnan(h_t[t-1]):
                h_t[t] = lambda_param * h_t[t-1] + (1 - lambda_param) * (e_t[t-1] ** 2)
    
    return h_t


def process_seed(seed_dir):
    """
    处理单个种子目录
    
    Args:
        seed_dir: 种子目录路径
        
    Returns:
        字典包含：
            - ratio: 最高值/第一个值 * 100
            - first_value: 第一个有效值
            - max_value: 最高值
            - max_timestep: 最高值对应的时间步
            如果出错返回None
    """
    world_metrics_path = os.path.join(seed_dir, 'metrics_csv', 'world_metrics.csv')
    
    if not os.path.exists(world_metrics_path):
        print(f"  Warning: world_metrics.csv not found in {seed_dir}")
        return None
    
    try:
        # 读取CSV文件
        df = pd.read_csv(world_metrics_path)
        
        # 计算risk_indicator_naive
        if 'risk_indicator_naive' in df.columns:
            risk_values = df['risk_indicator_naive'].values
        else:
            risk_values = calculate_risk_indicator_naive(df, lambda_param=0.94)
        
        # 找到第一个有效值（非NaN）
        first_valid_idx = None
        first_value = None
        for i in range(len(risk_values)):
            if not np.isnan(risk_values[i]) and risk_values[i] > 0:
                first_valid_idx = i
                first_value = risk_values[i]
                break
        
        if first_valid_idx is None or first_value is None:
            print(f"  Warning: No valid risk indicator value found in {seed_dir}")
            return None
        
        # 找到最高值
        valid_risk_values = risk_values[first_valid_idx:]
        valid_risk_values_clean = valid_risk_values[~np.isnan(valid_risk_values)]
        
        if len(valid_risk_values_clean) == 0:
            print(f"  Warning: No valid risk indicator values after first valid in {seed_dir}")
            return None
        
        max_value = np.max(valid_risk_values_clean)
        max_relative_idx = np.argmax(valid_risk_values_clean)
        max_absolute_idx = first_valid_idx + max_relative_idx
        max_timestep = int(df.iloc[max_absolute_idx]['timestep']) if max_absolute_idx < len(df) else None
        
        # 计算比率（最高值相对于第一个值的百分比）
        ratio = (max_value / first_value) * 100 if first_value > 0 else None
        
        if ratio is None:
            print(f"  Warning: Cannot calculate ratio (first_value is 0) in {seed_dir}")
            return None
        
        # 过滤掉比率超过10000%的异常数据
        if ratio > 10000:
            print(f"  Warning: Ratio {ratio:.2f}% exceeds 10000%, filtering out in {seed_dir}")
            return None
        
        return {
            'ratio': ratio,
            'first_value': first_value,
            'max_value': max_value,
            'max_timestep': max_timestep
        }
        
    except Exception as e:
        print(f"  Error processing {seed_dir}: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_all_models(datas_dir):
    """
    分析所有模型的数据
    
    Args:
        datas_dir: datas目录路径
        
    Returns:
        字典包含：
            - by_model: 按模型分类的结果
            - overall: 所有模型的汇总结果
    """
    results_by_model = defaultdict(list)  # {model_name: [ratios]}
    detailed_results = defaultdict(list)  # {model_name: [{seed, ratio, ...}]}
    
    # 遍历所有模型目录
    if not os.path.exists(datas_dir):
        print(f"Error: Datas directory not found: {datas_dir}")
        return None
    
    models = [d for d in os.listdir(datas_dir) 
              if os.path.isdir(os.path.join(datas_dir, d))]
    
    print(f"Found models: {models}")
    print(f"Processing seeds (excluding *_rm)...")
    
    # 遍历每个模型
    for model_name in sorted(models):
        model_dir = os.path.join(datas_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
        
        # 获取所有不带rm后缀的种子目录
        seeds = [d for d in os.listdir(model_dir) 
                 if os.path.isdir(os.path.join(model_dir, d)) and not d.endswith('_rm')]
        
        print(f"\n  Processing model: {model_name} ({len(seeds)} seeds)")
        
        # 遍历每个种子
        for seed_name in sorted(seeds):
            seed_dir = os.path.join(model_dir, seed_name)
            print(f"    Processing {seed_name}...", end=' ')
            
            result = process_seed(seed_dir)
            if result is not None:
                ratio = result['ratio']
                results_by_model[model_name].append(ratio)
                detailed_results[model_name].append({
                    'seed': seed_name,
                    'ratio': ratio,
                    'first_value': result['first_value'],
                    'max_value': result['max_value'],
                    'max_timestep': result['max_timestep']
                })
                print(f"✓ ratio: {ratio:.2f}%")
            else:
                print("✗ failed")
    
    # 按模型汇总统计
    model_summary = {}
    for model_name, ratios in results_by_model.items():
        if len(ratios) > 0:
            model_summary[model_name] = {
                'count': len(ratios),
                'mean': float(np.mean(ratios)),
                'max': float(np.max(ratios)),
                'min': float(np.min(ratios)),
                'std': float(np.std(ratios)),
                'detailed': detailed_results[model_name]
            }
    
    # 汇总所有模型的结果
    all_ratios = []
    for ratios in results_by_model.values():
        all_ratios.extend(ratios)
    
    overall_summary = None
    if len(all_ratios) > 0:
        overall_summary = {
            'count': len(all_ratios),
            'mean': float(np.mean(all_ratios)),
            'max': float(np.max(all_ratios)),
            'min': float(np.min(all_ratios)),
            'std': float(np.std(all_ratios))
        }
    
    return {
        'by_model': model_summary,
        'overall': overall_summary
    }


def print_summary(results):
    """打印汇总结果"""
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # 按模型打印
    print("\n【By Model Summary】")
    print("-"*80)
    print(f"{'Model':<15} {'Count':<8} {'Mean (%)':<12} {'Max (%)':<12} {'Min (%)':<12} {'Std':<12}")
    print("-"*80)
    
    for model_name in sorted(results['by_model'].keys()):
        stats = results['by_model'][model_name]
        print(f"{model_name:<15} {stats['count']:<8} {stats['mean']:<12.2f} "
              f"{stats['max']:<12.2f} {stats['min']:<12.2f} {stats['std']:<12.2f}")
    
    # 总体汇总
    if results['overall']:
        print("\n【Overall Summary (All Models)】")
        print("-"*80)
        overall = results['overall']
        print(f"Total Count: {overall['count']}")
        print(f"Mean: {overall['mean']:.2f}%")
        print(f"Max: {overall['max']:.2f}%")
        print(f"Min: {overall['min']:.2f}%")
        print(f"Std: {overall['std']:.2f}")
    
    print("="*80)


def save_results(results, output_dir):
    """保存结果到文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存JSON格式的详细结果
    json_path = os.path.join(output_dir, 'risk_threshold_analysis.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {json_path}")
    
    # 保存CSV格式的汇总结果
    csv_path = os.path.join(output_dir, 'risk_threshold_summary.csv')
    rows = []
    
    # 按模型的数据
    for model_name in sorted(results['by_model'].keys()):
        stats = results['by_model'][model_name]
        rows.append({
            'Model': model_name,
            'Count': stats['count'],
            'Mean (%)': f"{stats['mean']:.2f}",
            'Max (%)': f"{stats['max']:.2f}",
            'Min (%)': f"{stats['min']:.2f}",
            'Std': f"{stats['std']:.2f}"
        })
    
    # 总体汇总
    if results['overall']:
        overall = results['overall']
        rows.append({
            'Model': 'OVERALL',
            'Count': overall['count'],
            'Mean (%)': f"{overall['mean']:.2f}",
            'Max (%)': f"{overall['max']:.2f}",
            'Min (%)': f"{overall['min']:.2f}",
            'Std': f"{overall['std']:.2f}"
        })
    
    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Summary CSV saved to: {csv_path}")
    
    # 保存详细的按模型数据到CSV
    detailed_csv_path = os.path.join(output_dir, 'risk_threshold_detailed.csv')
    detailed_rows = []
    
    for model_name in sorted(results['by_model'].keys()):
        for detail in results['by_model'][model_name]['detailed']:
            detailed_rows.append({
                'Model': model_name,
                'Seed': detail['seed'],
                'Ratio (%)': f"{detail['ratio']:.2f}",
                'First Value': f"{detail['first_value']:.6e}",
                'Max Value': f"{detail['max_value']:.6e}",
                'Max Timestep': detail['max_timestep']
            })
    
    df_detailed = pd.DataFrame(detailed_rows)
    df_detailed.to_csv(detailed_csv_path, index=False, encoding='utf-8-sig')
    print(f"Detailed CSV saved to: {detailed_csv_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='分析risk indicator naive的最高值相对于第一个值的百分比'
    )
    parser.add_argument(
        '--datas_dir',
        type=str,
        default='/mnt/shared-storage-user/meijilin/Economic_System_Attribution/ACL24-EconAgent/datas',
        help='datas目录路径'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/mnt/shared-storage-user/meijilin/Economic_System_Attribution/ACL24-EconAgent/results/analyse_risk_threshold',
        help='结果输出目录路径'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Risk Threshold Analysis")
    print("="*80)
    print(f"Datas directory: {args.datas_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # 分析所有模型
    results = analyze_all_models(args.datas_dir)
    
    if results is None or len(results['by_model']) == 0:
        print("\nError: No results found!")
        return
    
    # 打印汇总
    print_summary(results)
    
    # 保存结果
    save_results(results, args.output_dir)
    
    print("\nAnalysis completed!")


if __name__ == '__main__':
    main()
