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
from typing import List, Dict

# Try to import pandas, fallback to manual CSV writing if not available
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def compute_cosine_similarity(phi_mc: np.ndarray, phi_exact: np.ndarray) -> float:
    """
    计算余弦相似度：cos(θ) = (A·B) / (||A|| * ||B||)
    
    Args:
        phi_mc: 蒙特卡罗Shapley值 (num_agents, num_dates)
        phi_exact: 完全Shapley值 (num_agents, num_dates)
    
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


def find_exact_shapley_seed_files(output_dir: Path, seeds: List[int]) -> Dict[int, Path]:
    """
    查找每个seed对应的exact Shapley值文件
    
    Args:
        output_dir: 输出目录路径
        seeds: 种子列表
    
    Returns:
        exact_seed_files: dict of {seed: file_path}，如果某个seed没有对应的文件，则不在dict中
    """
    exact_seed_files = {}
    
    for seed in seeds:
        seed_file = output_dir / f'exact_shapley_seed{seed}.npy'
        if seed_file.exists():
            exact_seed_files[seed] = seed_file
    
    return exact_seed_files


def main():
    parser = argparse.ArgumentParser(description='Compute cosine similarity between exact and MC Shapley values')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory containing exact and MC Shapley value files')
    parser.add_argument('--base_path', type=str, default=None,
                       help='Base path of the project (default: auto-detect)')
    parser.add_argument('--n_samples_list', type=int, nargs='+', required=True,
                       help='List of MC sample sizes to process')
    parser.add_argument('--seeds', type=int, nargs='+', required=True,
                       help='List of random seeds')
    parser.add_argument('--metric_name', type=str, default='risk_indicator_simple',
                       help='Metric name (for metadata)')
    parser.add_argument('--baseline_type', type=str, default='hold',
                       help='Baseline type (for metadata)')
    
    args = parser.parse_args()
    
    # 确定项目根目录
    if args.base_path is None:
        base_path = Path(__file__).parent.parent.parent
    else:
        base_path = Path(args.base_path)
    
    # 确定输出目录
    if Path(args.output_dir).is_absolute():
        output_dir = Path(args.output_dir)
    elif args.output_dir.startswith('results/'):
        output_dir = base_path / args.output_dir
    else:
        output_dir = base_path / "results" / args.output_dir
    
    print("="*60)
    print("Cosine Similarity Analysis")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Sample sizes: {args.n_samples_list}")
    print(f"Seeds: {args.seeds}")
    
    # 检查结果目录是否存在
    if not output_dir.exists():
        print(f"❌ Error: Output directory does not exist: {output_dir}")
        sys.exit(1)
    
    # 查找exact Shapley值文件（每个seed对应的文件）
    print("\nFinding exact Shapley value files...")
    exact_seed_files = find_exact_shapley_seed_files(output_dir, args.seeds)
    
    if len(exact_seed_files) == 0:
        print(f"❌ Error: Cannot find exact Shapley values files in {output_dir}")
        print(f"   Expected files: exact_shapley_seed{{seed}}.npy")
        sys.exit(1)
    
    print(f"✅ Found {len(exact_seed_files)} seed-specific exact Shapley files")
    print(f"   Seeds with exact values: {sorted(exact_seed_files.keys())}")
    
    # 计算余弦相似度
    print("\n" + "="*60)
    print("Computing Cosine Similarity")
    print("="*60)
    
    all_seed_results = {}  # {n_samples: [similarity1, similarity2, ...]}
    
    for n_samples in args.n_samples_list:
        similarities_list = []
        
        for seed in args.seeds:
            # 检查MC文件是否存在
            mc_file = output_dir / f'mc_shapley_n{n_samples}_seed{seed}.npy'
            if not mc_file.exists():
                print(f"⚠️  警告: MC文件不存在: {mc_file}")
                continue
            
            # 检查exact文件是否存在
            if seed not in exact_seed_files:
                print(f"⚠️  警告: Exact文件不存在 (seed={seed}), skipping...")
                continue
            
            # 加载MC和Exact值
            phi_mc = np.load(mc_file)
            phi_exact = np.load(exact_seed_files[seed])
            
            # 计算余弦相似度（使用相同seed的exact和MC值）
            cosine_sim = compute_cosine_similarity(phi_mc, phi_exact)
            similarities_list.append(cosine_sim)
            
            print(f"  n_samples={n_samples}, seed={seed}: cosine_similarity={cosine_sim:.6f}")
        
        if len(similarities_list) > 0:
            all_seed_results[n_samples] = similarities_list
    
    # 汇总结果
    print("\n" + "="*60)
    print("Summary: Aggregating Results Across All Seeds")
    print("="*60)
    
    results = []
    
    for n_samples in args.n_samples_list:
        similarities_list = all_seed_results.get(n_samples, [])
        
        if len(similarities_list) == 0:
            print(f"⚠️  警告: No similarities found for n_samples={n_samples}, skipping...")
            continue
        
        mean_similarity = np.mean(similarities_list)
        std_similarity = np.std(similarities_list)
        
        results.append({
            'n_samples': n_samples,
            'mean_cosine_similarity': float(mean_similarity),
            'std_cosine_similarity': float(std_similarity),
            'min_cosine_similarity': float(np.min(similarities_list)),
            'max_cosine_similarity': float(np.max(similarities_list)),
            'cosine_similarities': [float(s) for s in similarities_list]
        })
        
        print(f"n_samples={n_samples}: mean={mean_similarity:.6f} ± {std_similarity:.6f} "
              f"(range: [{np.min(similarities_list):.6f}, {np.max(similarities_list):.6f}], "
              f"n_seeds={len(similarities_list)})")
    
    if len(results) == 0:
        print("❌ Error: No valid results computed")
        sys.exit(1)
    
    # 构建结果摘要（使用第一个seed的exact值作为参考）
    reference_seed = sorted(exact_seed_files.keys())[0]
    reference_exact = np.load(exact_seed_files[reference_seed])
    
    results_summary = {
        'exact_shapley_path': f'exact_shapley_seed{reference_seed}.npy',
        'exact_shapley_sum': float(np.sum(reference_exact)),
        'exact_shapley_mean': float(np.mean(reference_exact)),
        'exact_shapley_std': float(np.std(reference_exact)),
        'num_agents': reference_exact.shape[0] if len(reference_exact.shape) >= 1 else None,
        'episode_length': reference_exact.shape[1] if len(reference_exact.shape) >= 2 else None,
        'use_seed_specific_exact': True,
        'seeds': args.seeds,
        'metric_name': args.metric_name,
        'baseline_type': args.baseline_type,
        'computation_method': 'cosine_similarity',
        'results': results
    }
    
    # 保存JSON格式的结果
    results_json_path = output_dir / 'cosine_similarity_results.json'
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved cosine similarity results JSON to: {results_json_path}")
    
    # 保存CSV格式的结果
    cosine_csv_path = output_dir / 'cosine_similarity_results.csv'
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
        with open(cosine_csv_path, 'w', encoding='utf-8') as f:
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
    print(f"Exact Shapley reference (from exact_shapley_seed{reference_seed}.npy):")
    print(f"  Sum: {np.sum(reference_exact):.6f}")
    print(f"  Mean: {np.mean(reference_exact):.6f}")
    print(f"  Std: {np.std(reference_exact):.6f}")
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
    
    print(f"\nAll results saved to: {output_dir}")
    print(f"  - Cosine similarity results JSON: {results_json_path}")
    print(f"  - Cosine similarity results CSV: {cosine_csv_path}")


if __name__ == '__main__':
    main()
