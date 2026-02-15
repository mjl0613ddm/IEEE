#!/usr/bin/env python3
"""
绘制SocialLLM模拟结果的风险折线图

用法:
    python3 plot_risk.py <results_dir>
    
示例:
    python3 plot_risk.py results/claude-3-haiku-20240307/claude-3-haiku-20240307_42
"""

import json
import sys
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir):
    """加载results.json文件"""
    results_file = Path(results_dir) / "results.json"
    
    if not results_file.exists():
        raise FileNotFoundError(f"结果文件不存在: {results_file}")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def plot_risk_timeseries(data, output_dir):
    """绘制风险时间序列图"""
    timestep_results = data.get('timestep_results', [])
    
    if not timestep_results:
        raise ValueError("timestep_results为空")
    
    # 提取时间步和风险值
    timesteps = [t['timestep'] for t in timestep_results]
    risks = [t['risk'] for t in timestep_results]
    
    # 提取其他信息用于标题
    num_agents = data.get('num_agents', 'N/A')
    num_steps = data.get('num_steps', 'N/A')
    initial_risk = data.get('initial_risk', 0)
    max_risk = data.get('max_risk', 0)
    max_risk_timestep = data.get('max_risk_timestep', 0)
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    
    # 绘制折线图
    plt.plot(timesteps, risks, marker='o', linewidth=2, markersize=4, color='#2E86AB', label='Polarization Risk')
    
    # 标记初始风险
    plt.axhline(y=initial_risk, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Initial Risk: {initial_risk:.4f}')
    
    # 标记最高风险点
    plt.plot(max_risk_timestep, max_risk, marker='*', markersize=15, color='red', 
             label=f'Max Risk: {max_risk:.4f} (timestep {max_risk_timestep})', zorder=5)
    
    # 设置标签和标题
    plt.xlabel('Timestep', fontsize=12, fontweight='bold')
    plt.ylabel('Polarization Risk', fontsize=12, fontweight='bold')
    
    # 从目录名提取模型名称和seed
    result_dir_name = Path(output_dir).name
    title = f'Polarization Risk Over Time\n{result_dir_name}'
    if num_agents != 'N/A' and num_steps != 'N/A':
        title += f' ({num_agents} agents, {num_steps} steps)'
    plt.title(title, fontsize=13, fontweight='bold')
    
    # 设置网格
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 设置图例
    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_file = Path(output_dir) / "plot" / "risk_timeseries.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"风险折线图已保存到: {output_file}")
    return output_file


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python3 plot_risk.py <results_dir>")
        print("示例: python3 plot_risk.py results/claude-3-haiku-20240307/claude-3-haiku-20240307_42")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    results_path = Path(results_dir)
    
    # 如果是相对路径，假设相对于脚本所在目录的父目录的父目录（项目根目录）
    if not results_path.is_absolute():
        script_dir = Path(__file__).parent.parent.parent
        results_path = script_dir / results_dir
    
    # 转换为绝对路径并标准化
    results_path = results_path.resolve()
    
    if not results_path.exists():
        print(f"错误: 结果目录不存在: {results_path}")
        sys.exit(1)
    
    if not results_path.is_dir():
        print(f"错误: 路径不是目录: {results_path}")
        sys.exit(1)
    
    try:
        # 加载结果
        data = load_results(results_path)
        
        # 绘制风险折线图
        plot_risk_timeseries(data, results_path)
        
        print("绘图完成!")
        
    except Exception as e:
        print(f"错误: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
