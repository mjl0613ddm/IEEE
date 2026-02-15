#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行反事实模拟分析脚本
计算 ds-verify 和 qwen-verify 相对于 baseline-verify-2 的反事实模拟
"""

import os
import sys
import subprocess
from pathlib import Path

# 设置基本参数
NUM_AGENTS = 4
EPISODE_LENGTH = 5
BASELINE_TYPE = "average"
METRIC_NAME = "risk_indicator_naive"
N_JOBS = -1  # 使用所有可用CPU核心

# 设置10个种子用于计算平均和方差
# 脚本会自动为每个种子计算exact和MC Shapley值，然后汇总计算平均误差和标准差
SEEDS = list(range(42, 52))  # 种子42-51，共10个

# 项目根目录（EconAgent）
project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)

# 配置列表：每个配置包含 (real_actions_json, baseline_actions_json, output_dir, name)
configs = [
    (
        "data/ds-verify/actions_json/all_actions.json",
        "data/baseline-verify-2/actions_json/all_actions.json",
        "results/ds_shapley_error_analyse",
        "ds-verify"
    ),
    (
        "data/qwen-verify/actions_json/all_actions.json",
        "data/baseline-verify-2/actions_json/all_actions.json",
        "results/qwen_shapley_error_analyse",
        "qwen-verify"
    ),
]

def run_counterfactual_analysis(real_actions_json, baseline_actions_json, output_dir, name):
    """运行单个反事实模拟分析"""
    print("=" * 60)
    print(f"计算 {name} 相对于 baseline-verify-2 的反事实模拟")
    print("=" * 60)
    
    cmd = [
        sys.executable,
        "scripts/accuracy_exp/compute_shapley_error.py",
        "--real_actions_json", real_actions_json,
        "--baseline_actions_json", baseline_actions_json,
        "--num_agents", str(NUM_AGENTS),
        "--episode_length", str(EPISODE_LENGTH),
        "--baseline_type", BASELINE_TYPE,
        "--metric_name", METRIC_NAME,
        "--output_dir", output_dir,
        "--n_jobs", str(N_JOBS),
        "--seeds", *[str(seed) for seed in SEEDS],  # 添加10个种子
    ]
    
    print(f"运行命令: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, cwd=project_root)
        print(f"✅ {name} 反事实模拟计算完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {name} 反事实模拟计算失败: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  {name} 反事实模拟计算被用户中断")
        return False

def main():
    """主函数"""
    print("开始运行反事实模拟分析...")
    print(f"项目根目录: {project_root}")
    print(f"使用种子: {SEEDS} (共{len(SEEDS)}个种子，将计算平均和方差)")
    print()
    
    results = []
    for real_actions_json, baseline_actions_json, output_dir, name in configs:
        success = run_counterfactual_analysis(
            real_actions_json, baseline_actions_json, output_dir, name
        )
        results.append((name, success))
        print()
    
    # 打印总结
    print("=" * 60)
    print("所有反事实模拟计算完成！")
    print("=" * 60)
    print()
    print("结果保存在：")
    for (real_actions_json, baseline_actions_json, output_dir, name), success in zip(configs, [s for _, s in results]):
        status = "✅" if success else "❌"
        print(f"  {status} {name}: {output_dir}/")
    
    # 检查是否有失败的任务
    failed = [name for name, success in results if not success]
    if failed:
        print(f"\n⚠️  以下任务失败: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("\n✅ 所有任务成功完成！")

if __name__ == "__main__":
    main()

