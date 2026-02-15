#!/bin/bash
# Example: Shapley approximation accuracy (exact vs Monte Carlo)
# Copy and modify REAL_ACTIONS_JSON, OUTPUT_DIR for your model (e.g. gpt-4o, claude, llama)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_SCRIPT="$PROJECT_DIR/scripts/accuracy_exp/compute_shapley_error.py"

# ==================== 修改以下参数 ====================
NUM_AGENTS=4
EPISODE_LENGTH=5
REAL_ACTIONS_JSON="data/gpt-4o-verify/actions_json/all_actions.json"   # 改为你的模型路径
BASELINE_ACTIONS_JSON="data/baseline-verify/actions_json/all_actions.json"
OUTPUT_DIR="results/shapley_error_analysis"                             # 改为你的输出路径

SEEDS=(42 43 44 45 46)
MC_SAMPLES_LIST=(10 100 1000 10000)
N_JOBS=8

# ==================== 运行 ====================
cd "$PROJECT_DIR"
python "$PYTHON_SCRIPT" \
    --num_agents $NUM_AGENTS \
    --episode_length $EPISODE_LENGTH \
    --real_actions_json "$REAL_ACTIONS_JSON" \
    --baseline_actions_json "$BASELINE_ACTIONS_JSON" \
    --metric_name risk_indicator_naive \
    --baseline_type fixed \
    --baseline_work 1.0 \
    --baseline_consumption 0.8 \
    --seeds "${SEEDS[@]}" \
    --mc_samples_list "${MC_SAMPLES_LIST[@]}" \
    --output_dir "$OUTPUT_DIR" \
    --n_jobs $N_JOBS
