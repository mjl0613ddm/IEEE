"""
保存batch plot脚本中画图所需要的数据

该脚本从各个数据源加载数据，并保存为npy格式，用于后续的画图。
保存的数据包括：
1. shapley_values: Shapley值数组
2. risk_evolution: 风险演化曲线
3. agent_aggregated: 按agent聚合的shapley值（用于散点图）
4. behaviour_aggregated: 按behavior聚合的shapley值
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# 导入batch_plot中的辅助函数
from batch_plot import (
    find_latest_shapley_files,
    load_shapley_data,
    load_risk_features_data
)


def save_model_seed_data(base_path, model_name, seed_name):
    """保存单个模型-种子组合的数据.
    
    Args:
        base_path: 项目根路径
        model_name: 模型名称
        seed_name: 种子名称
        
    Returns:
        success: 是否成功保存
    """
    # 构建路径
    shapley_dir = os.path.join(base_path, 'results', model_name, seed_name, 'shapley')
    risk_features_dir = os.path.join(base_path, 'results', 'risk_feature', model_name, seed_name)
    market_metrics_path = os.path.join(base_path, 'results', model_name, seed_name, 'analysis', 'market_metrics.csv')
    data_output_dir = os.path.join(base_path, 'results', model_name, seed_name, 'data')
    
    # 检查必要目录是否存在
    if not os.path.exists(shapley_dir):
        print(f"Warning: Shapley directory not found: {shapley_dir}, skipping...")
        return False
    
    if not os.path.exists(risk_features_dir):
        print(f"Warning: Risk features directory not found: {risk_features_dir}, skipping...")
        return False
    
    if not os.path.exists(market_metrics_path):
        print(f"Warning: Market metrics file not found: {market_metrics_path}, skipping...")
        return False
    
    # 创建输出目录
    os.makedirs(data_output_dir, exist_ok=True)
    
    try:
        # 加载数据（使用和batch_plot.py相同的逻辑）
        shapley_values, stats = load_shapley_data(shapley_dir)
        
        # 获取shapley的日期范围，用于对齐risk_evolution
        date_range = stats.get('date_range', None)
        
        agent_aggregated, time_aggregated, behaviour_aggregated, risk_evolution, risk_features = \
            load_risk_features_data(risk_features_dir, market_metrics_path, date_range=date_range)
        
        # 保存数据为npy格式
        # 1. shapley_values
        shapley_output_path = os.path.join(data_output_dir, 'shapley_values.npy')
        np.save(shapley_output_path, shapley_values)
        print(f"  Saved shapley_values: {shapley_output_path} (shape: {shapley_values.shape})")
        
        # 2. risk_evolution
        risk_evolution_output_path = os.path.join(data_output_dir, 'risk_evolution.npy')
        np.save(risk_evolution_output_path, risk_evolution)
        print(f"  Saved risk_evolution: {risk_evolution_output_path} (shape: {risk_evolution.shape})")
        
        # 3. agent_aggregated (用于散点图)
        agent_aggregated_output_path = os.path.join(data_output_dir, 'agent_aggregated.npy')
        np.save(agent_aggregated_output_path, agent_aggregated)
        print(f"  Saved agent_aggregated: {agent_aggregated_output_path} (shape: {agent_aggregated.shape})")
        
        # 4. behaviour_aggregated
        behaviour_aggregated_output_path = os.path.join(data_output_dir, 'behaviour_aggregated.npy')
        np.save(behaviour_aggregated_output_path, behaviour_aggregated)
        print(f"  Saved behaviour_aggregated: {behaviour_aggregated_output_path} (shape: {behaviour_aggregated.shape})")
        
        # 可选：保存stats信息（用于参考）
        stats_output_path = os.path.join(data_output_dir, 'stats.json')
        with open(stats_output_path, 'w') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"  Saved stats: {stats_output_path}")
        
        print(f"Successfully saved data for {model_name}/{seed_name}")
        return True
        
    except Exception as e:
        print(f"Error saving data for {model_name}/{seed_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数."""
    parser = argparse.ArgumentParser(description='Save plot data script for TwinMarket')
    parser.add_argument('--base_path', type=str, 
                       default='/mnt/shared-storage-user/meijilin/Economic_System_Attribution/TwinMarket',
                       help='Base path of the project (default: /mnt/shared-storage-user/meijilin/Economic_System_Attribution/TwinMarket)')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='List of models to process (default: all models in results/)')
    parser.add_argument('--seeds', type=str, nargs='+', default=None,
                       help='List of seeds to process (default: all seeds excluding *_rm)')
    
    args = parser.parse_args()
    
    base_path = args.base_path
    results_dir = os.path.join(base_path, 'results')
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    # 获取所有模型
    if args.models:
        models = args.models
    else:
        models = [d for d in os.listdir(results_dir) 
                 if os.path.isdir(os.path.join(results_dir, d)) 
                 and d != 'risk_feature' 
                 and d != 'accuracy' 
                 and d != 'faithfulness_exp'
                 and not d.startswith('logs_')
                 and not d.startswith('shapley_error')
                 and not d.startswith('risk_timestep')]
    
    print(f"Processing models: {models}")
    
    total_processed = 0
    total_success = 0
    
    # 遍历每个模型
    for model_name in models:
        model_dir = os.path.join(results_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
        
        # 获取所有种子（排除*_rm后缀的）
        if args.seeds:
            seeds = args.seeds
        else:
            seeds = [d for d in os.listdir(model_dir) 
                    if os.path.isdir(os.path.join(model_dir, d)) and not d.endswith('_rm')]
        
        print(f"  Processing {len(seeds)} seeds for model {model_name}")
        
        # 遍历每个种子
        for seed_name in seeds:
            seed_dir = os.path.join(model_dir, seed_name)
            if not os.path.isdir(seed_dir):
                continue
            
            total_processed += 1
            print(f"\nProcessing {model_name}/{seed_name}...")
            
            success = save_model_seed_data(base_path, model_name, seed_name)
            if success:
                total_success += 1
    
    # 打印总结
    print("\n" + "=" * 60)
    print(f"Summary:")
    print(f"  Total processed: {total_processed}")
    print(f"  Successful: {total_success}")
    print(f"  Failed: {total_processed - total_success}")
    print("=" * 60)


if __name__ == "__main__":
    main()
