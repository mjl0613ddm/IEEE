#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算Exact Shapley值和蒙特卡罗方法的余弦相似度
使用已有的结果文件计算余弦相似度作为误差度量
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
import glob
import re

# Try to import pandas, fallback to manual CSV writing if not available
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def compute_cosine_similarity(phi_mc, phi_exact):
    """
    计算余弦相似度：cos(θ) = (A·B) / (||A|| * ||B||)
    
    Args:
        phi_mc: 蒙特卡罗Shapley值 (num_agents, episode_length)
        phi_exact: 完全Shapley值 (num_agents, episode_length)
    
    Returns:
        cosine_similarity: float (范围: -1 to 1, 1表示完全相同)
    """
    # 展平为一维向量
    vec_mc = phi_mc.flatten()
    vec_exact = phi_exact.flatten()
    
    # 计算点积
    dot_product = np.dot(vec_mc, vec_exact)
    
    # 计算范数
    norm_mc = np.linalg.norm(vec_mc, ord=2)
    norm_exact = np.linalg.norm(vec_exact, ord=2)
    
    # 处理零向量的情况
    if norm_mc == 0 and norm_exact == 0:
        return 1.0  # 两个都是零向量，认为完全相似
    if norm_mc == 0 or norm_exact == 0:
        return 0.0  # 一个是零向量，另一个不是，相似度为0
    
    # 计算余弦相似度
    cosine_similarity = dot_product / (norm_mc * norm_exact)
    
    # 确保结果在有效范围内（由于浮点误差可能会有轻微超出）
    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
    
    return float(cosine_similarity)


def find_exact_shapley_file(results_dir):
    """
    查找exact Shapley值文件（fallback文件，用于没有seed-specific文件时）
    
    Args:
        results_dir: 结果目录路径
    
    Returns:
        exact_shapley_path: exact Shapley值文件路径，如果不存在则返回None
    """
    # 优先级：exact_shapley_values.npy > exact_shapley_values_mean.npy
    possible_files = [
        os.path.join(results_dir, 'exact_shapley_values.npy'),
        os.path.join(results_dir, 'exact_shapley_values_mean.npy'),
    ]
    
    for file_path in possible_files:
        if os.path.exists(file_path):
            return file_path
    
    return None


def find_exact_shapley_seed_files(results_dir, seeds):
    """
    查找每个seed对应的exact Shapley值文件
    
    Args:
        results_dir: 结果目录路径
        seeds: 种子列表
    
    Returns:
        exact_seed_files: dict of {seed: file_path}，如果某个seed没有对应的文件，则不在dict中
    """
    exact_seed_files = {}
    
    for seed in seeds:
        seed_file = os.path.join(results_dir, f'exact_shapley_seed{seed}.npy')
        if os.path.exists(seed_file):
            exact_seed_files[seed] = seed_file
    
    return exact_seed_files


def find_mc_shapley_files(results_dir):
    """
    查找所有MC Shapley值文件，并提取samples和seeds信息
    
    Args:
        results_dir: 结果目录路径
    
    Returns:
        mc_files_dict: dict of {n_samples: {seed: file_path}}
        seeds: set of all seeds found
        samples: sorted list of all sample sizes
    """
    mc_files_dict = {}  # {n_samples: {seed: file_path}}
    seeds_set = set()
    samples_set = set()
    
    # 匹配模式: mc_shapley_n{samples}_seed{seed}.npy
    pattern = os.path.join(results_dir, 'mc_shapley_n*_seed*.npy')
    files = glob.glob(pattern)
    
    for file_path in files:
        filename = os.path.basename(file_path)
        # 提取samples和seed
        match = re.match(r'mc_shapley_n(\d+)_seed(\d+)\.npy', filename)
        if match:
            n_samples = int(match.group(1))
            seed = int(match.group(2))
            
            if n_samples not in mc_files_dict:
                mc_files_dict[n_samples] = {}
            
            mc_files_dict[n_samples][seed] = file_path
            seeds_set.add(seed)
            samples_set.add(n_samples)
    
    seeds = sorted(list(seeds_set))
    samples = sorted(list(samples_set))
    
    return mc_files_dict, seeds, samples


def load_existing_error_analysis(results_dir):
    """
    加载现有的error_analysis_results.json（如果存在）以获取元数据
    
    Args:
        results_dir: 结果目录路径
    
    Returns:
        metadata: dict with metadata from error_analysis_results.json, or None
    """
    json_path = os.path.join(results_dir, 'error_analysis_results.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description='Compute cosine similarity between exact and MC Shapley values')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name (e.g., claude, qwen, gpt, llama, ds)')
    parser.add_argument('--results_dir', type=str, default=None,
                       help='Override default results directory pattern (default: results/{model}_shapley_error_analysis)')
    parser.add_argument('--mc_samples_list', type=int, nargs='+', default=None,
                       help='List of MC sample sizes to process (default: detect from existing files)')
    
    args = parser.parse_args()
    
    # 设置结果目录
    if args.results_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        args.results_dir = os.path.join(project_root, 'results', f'{args.model}_shapley_error_analysis')
    else:
        args.results_dir = os.path.abspath(args.results_dir)
    
    print("="*60)
    print("Cosine Similarity Analysis")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Results directory: {args.results_dir}")
    
    # 检查结果目录是否存在
    if not os.path.exists(args.results_dir):
        print(f"❌ Error: Results directory does not exist: {args.results_dir}")
        sys.exit(1)
    
    # 查找MC Shapley值文件（先找到seeds）
    print("\nFinding MC Shapley value files...")
    mc_files_dict, seeds, samples = find_mc_shapley_files(args.results_dir)
    
    if len(mc_files_dict) == 0:
        print(f"❌ Error: No MC Shapley value files found in {args.results_dir}")
        print(f"   Expected pattern: mc_shapley_n{{samples}}_seed{{seed}}.npy")
        sys.exit(1)
    
    # 查找exact Shapley值文件（每个seed对应的文件）
    print("\nFinding exact Shapley value files...")
    exact_seed_files = find_exact_shapley_seed_files(args.results_dir, seeds)
    exact_shapley_fallback = find_exact_shapley_file(args.results_dir)
    
    if len(exact_seed_files) == 0 and exact_shapley_fallback is None:
        print(f"❌ Error: Cannot find exact Shapley values files in {args.results_dir}")
        print(f"   Expected files: exact_shapley_seed{{seed}}.npy or exact_shapley_values.npy")
        sys.exit(1)
    
    if len(exact_seed_files) > 0:
        print(f"✅ Found {len(exact_seed_files)} seed-specific exact Shapley files")
        if exact_shapley_fallback:
            print(f"✅ Found fallback exact Shapley file: {os.path.basename(exact_shapley_fallback)}")
            print(f"   Will use seed-specific files when available, fallback for missing seeds")
        else:
            print(f"   Using seed-specific files only")
    else:
        print(f"✅ Using fallback exact Shapley file: {os.path.basename(exact_shapley_fallback)}")
        print(f"   Will use the same exact Shapley values for all seeds")
    
    # 加载fallback exact值（如果需要）
    phi_exact_fallback = None
    if exact_shapley_fallback:
        phi_exact_fallback = np.load(exact_shapley_fallback)
        print(f"   Fallback shape: {phi_exact_fallback.shape}, sum: {np.sum(phi_exact_fallback):.6f}")
    
    print(f"✅ Found MC files for {len(samples)} sample sizes: {samples}")
    print(f"✅ Found {len(seeds)} seeds: {seeds}")
    
    # 确定要处理的sample sizes
    if args.mc_samples_list is not None:
        mc_samples_list = [s for s in args.mc_samples_list if s in samples]
        if len(mc_samples_list) != len(args.mc_samples_list):
            missing = set(args.mc_samples_list) - set(mc_samples_list)
            print(f"⚠️  Warning: Some specified sample sizes not found: {missing}")
    else:
        mc_samples_list = samples
    
    if len(mc_samples_list) == 0:
        print(f"❌ Error: No valid sample sizes to process")
        sys.exit(1)
    
    print(f"\nProcessing {len(mc_samples_list)} sample sizes: {mc_samples_list}")
    
    # 加载现有的error_analysis_results.json以获取元数据
    metadata = load_existing_error_analysis(args.results_dir)
    
    # 计算余弦相似度
    print("\n" + "="*60)
    print("Computing Cosine Similarity")
    print("="*60)
    
    all_seed_results = {}  # {n_samples: [similarity1, similarity2, ...]}
    
    for n_samples in mc_samples_list:
        if n_samples not in mc_files_dict:
            print(f"⚠️  Warning: No MC files found for n_samples={n_samples}, skipping...")
            continue
        
        seed_files = mc_files_dict[n_samples]
        similarities_list = []
        
        for seed in seeds:
            if seed not in seed_files:
                print(f"⚠️  Warning: MC file not found for n_samples={n_samples}, seed={seed}, skipping...")
                continue
            
            mc_file_path = seed_files[seed]
            phi_mc = np.load(mc_file_path)
            
            # 加载对应seed的exact Shapley值
            if seed in exact_seed_files:
                phi_exact = np.load(exact_seed_files[seed])
                exact_source = f"exact_shapley_seed{seed}.npy"
            elif phi_exact_fallback is not None:
                phi_exact = phi_exact_fallback
                exact_source = os.path.basename(exact_shapley_fallback) if exact_shapley_fallback else "fallback"
            else:
                print(f"⚠️  Warning: No exact Shapley file found for seed={seed}, skipping...")
                continue
            
            # 计算余弦相似度（使用相同seed的exact和MC值）
            cosine_sim = compute_cosine_similarity(phi_mc, phi_exact)
            similarities_list.append(cosine_sim)
            
            print(f"  n_samples={n_samples}, seed={seed}: cosine_similarity={cosine_sim:.6f} (exact from {exact_source})")
        
        if len(similarities_list) > 0:
            all_seed_results[n_samples] = similarities_list
    
    # 汇总结果
    print("\n" + "="*60)
    print("Summary: Aggregating Results Across All Seeds")
    print("="*60)
    
    results = []
    
    for n_samples in mc_samples_list:
        similarities_list = all_seed_results.get(n_samples, [])
        
        if len(similarities_list) == 0:
            print(f"⚠️  Warning: No similarities found for n_samples={n_samples}, skipping...")
            continue
        
        mean_similarity = np.mean(similarities_list)
        std_similarity = np.std(similarities_list)
        
        results.append({
            'n_samples': n_samples,
            'mean_cosine_similarity': mean_similarity,
            'std_cosine_similarity': std_similarity,
            'min_cosine_similarity': np.min(similarities_list),
            'max_cosine_similarity': np.max(similarities_list),
            'cosine_similarities': similarities_list
        })
        
        print(f"n_samples={n_samples}: mean={mean_similarity:.6f} ± {std_similarity:.6f} "
              f"(range: [{np.min(similarities_list):.6f}, {np.max(similarities_list):.6f}], "
              f"n_seeds={len(similarities_list)})")
    
    # 构建结果摘要（使用第一个可用的exact值作为参考）
    reference_exact = None
    reference_exact_path = None
    if len(exact_seed_files) > 0:
        first_seed = seeds[0]
        if first_seed in exact_seed_files:
            reference_exact = np.load(exact_seed_files[first_seed])
            reference_exact_path = os.path.basename(exact_seed_files[first_seed])
    elif phi_exact_fallback is not None:
        reference_exact = phi_exact_fallback
        reference_exact_path = os.path.basename(exact_shapley_fallback)
    
    results_summary = {
        'exact_shapley_path': reference_exact_path,
        'exact_shapley_sum': float(np.sum(reference_exact)) if reference_exact is not None else None,
        'exact_shapley_mean': float(np.mean(reference_exact)) if reference_exact is not None else None,
        'exact_shapley_std': float(np.std(reference_exact)) if reference_exact is not None else None,
        'num_agents': reference_exact.shape[0] if reference_exact is not None and len(reference_exact.shape) >= 1 else None,
        'episode_length': reference_exact.shape[1] if reference_exact is not None and len(reference_exact.shape) >= 2 else None,
        'use_seed_specific_exact': len(exact_seed_files) > 0,
        'seeds': seeds,
        'metric_name': metadata.get('metric_name', None) if metadata else None,
        'risk_lambda': metadata.get('risk_lambda', None) if metadata else None,
        'computation_method': 'cosine_similarity',
        'results': [
            {
                'n_samples': r['n_samples'],
                'mean_cosine_similarity': r['mean_cosine_similarity'],
                'std_cosine_similarity': r['std_cosine_similarity'],
                'min_cosine_similarity': r['min_cosine_similarity'],
                'max_cosine_similarity': r['max_cosine_similarity'],
                'cosine_similarities': [float(s) for s in r['cosine_similarities']]
            }
            for r in results
        ]
    }
    
    # 保存JSON格式的结果
    results_json_path = os.path.join(args.results_dir, 'cosine_similarity_results.json')
    with open(results_json_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n✅ Saved cosine similarity results JSON to: {results_json_path}")
    
    # 保存CSV格式的结果
    cosine_csv_path = os.path.join(args.results_dir, 'cosine_similarity_results.csv')
    if HAS_PANDAS:
        cosine_df = pd.DataFrame([
            {
                'n_samples': r['n_samples'],
                'mean_cosine_similarity': r['mean_cosine_similarity'],
                'std_cosine_similarity': r['std_cosine_similarity'],
                'min_cosine_similarity': r['min_cosine_similarity'],
                'max_cosine_similarity': r['max_cosine_similarity']
            }
            for r in results
        ])
        cosine_df.to_csv(cosine_csv_path, index=False)
    else:
        # Manual CSV writing if pandas is not available
        with open(cosine_csv_path, 'w') as f:
            # Write header
            f.write('n_samples,mean_cosine_similarity,std_cosine_similarity,min_cosine_similarity,max_cosine_similarity\n')
            # Write data rows
            for r in results:
                f.write(f"{r['n_samples']},{r['mean_cosine_similarity']:.15f},"
                       f"{r['std_cosine_similarity']:.15f},{r['min_cosine_similarity']:.15f},"
                       f"{r['max_cosine_similarity']:.15f}\n")
    print(f"✅ Saved cosine similarity results CSV to: {cosine_csv_path}")
    
    # 打印总结
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    if reference_exact is not None:
        print(f"Exact Shapley reference (from {reference_exact_path}):")
        print(f"  Sum: {np.sum(reference_exact):.6f}")
        print(f"  Mean: {np.mean(reference_exact):.6f}")
        print(f"  Std: {np.std(reference_exact):.6f}")
    if len(exact_seed_files) > 0:
        print(f"  Note: Using seed-specific exact Shapley files for each seed")
    print(f"\nCosine Similarity Analysis (cos(θ) = (A·B) / (||A|| * ||B||)):")
    print(f"Higher values indicate better similarity (range: -1 to 1, 1 = identical)")
    print(f"{'n_samples':<12} {'mean':<15} {'std':<15} {'min':<15} {'max':<15} {'n_seeds':<10}")
    print("-" * 85)
    for r in results:
        print(f"{r['n_samples']:<12} {r['mean_cosine_similarity']:<15.6f} "
              f"{r['std_cosine_similarity']:<15.6f} {r['min_cosine_similarity']:<15.6f} "
              f"{r['max_cosine_similarity']:<15.6f} {len(r['cosine_similarities']):<10}")
    print("="*60)
    
    print(f"\nAll results saved to: {args.results_dir}")
    print(f"  - Cosine similarity results JSON: {results_json_path}")
    print(f"  - Cosine similarity results CSV: {cosine_csv_path}")


if __name__ == '__main__':
    main()
