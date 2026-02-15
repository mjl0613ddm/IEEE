#!/bin/bash
# 运行反事实模拟分析脚本
# 计算 ds-verify 和 qwen-verify 相对于 baseline-verify-2 的反事实模拟
# 从 EconAgent 根目录运行

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# 设置基本参数
NUM_AGENTS=4
EPISODE_LENGTH=5
BASELINE_TYPE="average"
METRIC_NAME="risk_indicator_naive"
N_JOBS=-1  # 使用所有可用CPU核心

# 设置10个种子用于计算平均和方差（种子42-51）
SEEDS=(42 43 44 45 46 47 48 49 50 51)

# 1. 计算 ds-verify 相对于 baseline-verify-2 的反事实模拟
echo "=========================================="
echo "计算 ds-verify 相对于 baseline-verify-2 的反事实模拟"
echo "使用种子: ${SEEDS[@]} (共${#SEEDS[@]}个种子，将计算平均和方差)"
echo "=========================================="
python scripts/accuracy_exp/compute_shapley_error.py \
    --real_actions_json data/ds-verify/actions_json/all_actions.json \
    --baseline_actions_json data/baseline-verify-2/actions_json/all_actions.json \
    --num_agents $NUM_AGENTS \
    --episode_length $EPISODE_LENGTH \
    --baseline_type $BASELINE_TYPE \
    --metric_name $METRIC_NAME \
    --output_dir results/ds_shapley_error_analyse \
    --n_jobs $N_JOBS \
    --seeds "${SEEDS[@]}"

if [ $? -eq 0 ]; then
    echo "✅ ds-verify 反事实模拟计算完成"
else
    echo "❌ ds-verify 反事实模拟计算失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "计算 qwen-verify 相对于 baseline-verify-2 的反事实模拟"
echo "=========================================="

# 2. 计算 qwen-verify 相对于 baseline-verify-2 的反事实模拟
echo "使用种子: ${SEEDS[@]} (共${#SEEDS[@]}个种子，将计算平均和方差)"
python scripts/accuracy_exp/compute_shapley_error.py \
    --real_actions_json data/qwen-verify/actions_json/all_actions.json \
    --baseline_actions_json data/baseline-verify-2/actions_json/all_actions.json \
    --num_agents $NUM_AGENTS \
    --episode_length $EPISODE_LENGTH \
    --baseline_type $BASELINE_TYPE \
    --metric_name $METRIC_NAME \
    --output_dir results/qwen_shapley_error_analyse \
    --n_jobs $N_JOBS \
    --seeds "${SEEDS[@]}"

if [ $? -eq 0 ]; then
    echo "✅ qwen-verify 反事实模拟计算完成"
else
    echo "❌ qwen-verify 反事实模拟计算失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "所有反事实模拟计算完成！"
echo "=========================================="
echo ""
echo "结果保存在："
echo "  - results/ds_shapley_error_analyse/"
echo "  - results/qwen_shapley_error_analyse/"

