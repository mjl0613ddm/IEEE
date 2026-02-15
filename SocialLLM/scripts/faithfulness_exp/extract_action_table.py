#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SocialLLM Action Table提取脚本

从actions.json中提取每个(agent_id, timestep)对的action信息，生成action_table.csv文件
用于LLM和MAST方法打分

使用方法:
    python scripts/faithfulness_exp/extract_action_table.py --result_dir results/gpt-4o-mini/gpt-4o-mini_42
    python scripts/faithfulness_exp/extract_action_table.py --result_dir results/gpt-4o-mini/gpt-4o-mini_42 --skip-existing
"""

import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_actions(actions_file: Path) -> Dict:
    """加载actions.json文件"""
    if not actions_file.exists():
        raise FileNotFoundError(f"Actions文件不存在: {actions_file}")
    
    with open(actions_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_results(results_file: Path) -> Dict:
    """加载results.json文件，获取max_risk_timestep"""
    if not results_file.exists():
        raise FileNotFoundError(f"Results文件不存在: {results_file}")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_action_table(result_dir: Path, max_timestep: Optional[int] = None) -> pd.DataFrame:
    """
    从actions.json中提取action table
    
    Args:
        result_dir: 结果目录路径
        max_timestep: 最大时间步（如果为None，则从results.json中读取max_risk_timestep）
    
    Returns:
        DataFrame with columns: agent_id, timestep, posted, view_count, like_count, dislike_count, belief
    """
    actions_file = result_dir / "actions.json"
    results_file = result_dir / "results.json"
    
    # 加载数据
    actions_dict = load_actions(actions_file)
    results_data = load_results(results_file)
    
    # 确定最大时间步
    if max_timestep is None:
        max_timestep = results_data.get('max_risk_timestep')
        if max_timestep is None:
            # 如果没有max_risk_timestep，使用num_steps
            max_timestep = results_data.get('num_steps', 30)
    
    num_agents = results_data.get('num_agents', 20)
    
    # 构建belief值查找字典 {(timestep, agent_id): belief_value}
    belief_dict = {}
    timestep_results = results_data.get('timestep_results', [])
    for timestep_result in timestep_results:
        timestep = timestep_result.get('timestep')
        beliefs = timestep_result.get('beliefs', [])
        if timestep is not None and beliefs:
            for agent_id, belief_value in enumerate(beliefs):
                belief_dict[(timestep, agent_id)] = belief_value
    
    # 提取action数据
    action_rows = []
    
    for agent_id in range(num_agents):
        for timestep in range(max_timestep):
            key = f"({agent_id}, {timestep})"
            action_data = actions_dict.get(key, {})
            
            # 是否发帖
            posted = 1 if action_data.get('post') is not None else 0
            
            # 统计互动
            interactions = action_data.get('interactions', [])
            view_count = len(interactions)
            like_count = sum(1 for i in interactions if i.get('action') == 'like')
            dislike_count = sum(1 for i in interactions if i.get('action') == 'dislike')
            
            # 获取belief值
            belief_value = belief_dict.get((timestep, agent_id))
            if belief_value is None:
                # 如果找不到belief值，尝试使用初始belief值
                initial_beliefs = results_data.get('initial_beliefs', [])
                if agent_id < len(initial_beliefs):
                    belief_value = initial_beliefs[agent_id]
                else:
                    belief_value = 0.0
            
            action_rows.append({
                'agent_id': agent_id,
                'timestep': timestep,
                'posted': posted,
                'view_count': view_count,
                'like_count': like_count,
                'dislike_count': dislike_count,
                'belief': float(belief_value)
            })
    
    # 创建DataFrame
    df = pd.DataFrame(action_rows)
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='从actions.json中提取action table，生成action_table.csv文件',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--result_dir',
        type=str,
        required=True,
        help='结果目录路径（包含actions.json和results.json）'
    )
    
    parser.add_argument(
        '--max_timestep',
        type=int,
        default=None,
        help='最大时间步（可选，默认从results.json的max_risk_timestep读取）'
    )
    
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='如果action_table.csv已存在，则跳过'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='输出详细信息'
    )
    
    args = parser.parse_args()
    
    # 转换为Path对象
    result_dir = Path(args.result_dir).resolve()
    
    if not result_dir.exists():
        print(f"错误: 结果目录不存在: {result_dir}", file=sys.stderr)
        sys.exit(1)
    
    # 检查输出文件
    output_dir = result_dir / "action_table"
    output_file = output_dir / "action_table.csv"
    
    if args.skip_existing and output_file.exists():
        print(f"跳过已存在的文件: {output_file}")
        return
    
    # 提取action table
    if args.verbose:
        print(f"处理: {result_dir}")
        print(f"提取action table...")
    
    try:
        df = extract_action_table(result_dir, args.max_timestep)
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存CSV文件
        df.to_csv(output_file, index=False)
        
        if args.verbose:
            print(f"✓ Action table已保存到: {output_file}")
            print(f"  总行数: {len(df)}")
            print(f"  列: {', '.join(df.columns.tolist())}")
            print(f"  发帖总数: {df['posted'].sum()}")
            print(f"  总浏览数: {df['view_count'].sum()}")
            print(f"  总点赞数: {df['like_count'].sum()}")
            print(f"  总点踩数: {df['dislike_count'].sum()}")
            print(f"  Belief值范围: [{df['belief'].min():.6f}, {df['belief'].max():.6f}]")
        else:
            print(f"✓ Action table已保存: {output_file}")
    
    except Exception as e:
        print(f"错误: 提取action table失败: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
