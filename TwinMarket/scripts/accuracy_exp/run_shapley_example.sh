#!/bin/bash

# 单模型 Shapley 值计算与误差分析示例
# 修改 MODEL 以运行不同模型；依赖 compute_exact_shapley.py、compute_mc_shapley.py

# ==================== 配置参数 ====================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
RESULTS_DIR="${BASE_DIR}/results/accuracy"

# 模型名称（根据需要修改）
MODEL="gpt-4o"

# 交易日范围
START_DATE="2023-06-26"
TARGET_DATE="2023-06-30"

# MC 采样参数
N_SAMPLES_LIST=(10 100 1000 10000)
SEEDS=(42 43 401 45 46 613 48 49 50 51)

# 其他参数
METRIC_NAME="risk_indicator_simple"
BASELINE_TYPE="hold"
N_JOBS=64  # 0 表示使用 CPU 核心数

# ==================== 检查环境 ====================
echo "=========================================="
echo "计算模型 ${MODEL} 的 Shapley 值并分析误差"
echo "=========================================="
echo "项目根目录: ${BASE_DIR}"
echo "结果目录: ${RESULTS_DIR}"
echo "=========================================="
echo ""

if [ ! -d "$RESULTS_DIR" ]; then
    echo "❌ 结果目录不存在: $RESULTS_DIR"
    exit 1
fi

for py in compute_exact_shapley compute_mc_shapley analyze_error compute_cosine_similarity; do
    if [ ! -f "${SCRIPT_DIR}/${py}.py" ]; then
        echo "❌ ${py}.py 不存在"
        exit 1
    fi
done

# ==================== 处理模型 ====================
MODEL_DIR="${RESULTS_DIR}/${MODEL}"
if [ ! -d "$MODEL_DIR" ]; then
    echo "⚠️  模型目录不存在: $MODEL_DIR"
    exit 1
fi

MODEL_NAME_SAFE=$(echo "${MODEL}" | sed 's/\//_/g')
OUTPUT_DIR="${RESULTS_DIR}/${MODEL}/${MODEL_NAME_SAFE}_shapley_error_analyse"
mkdir -p "$OUTPUT_DIR"

# 步骤1: Exact Shapley
for seed in "${SEEDS[@]}"; do
    EXACT_OUTPUT="${OUTPUT_DIR}/exact_shapley_seed${seed}.npy"
    if [ ! -f "$EXACT_OUTPUT" ]; then
        python3 "${SCRIPT_DIR}/compute_exact_shapley.py" \
            --log_dir "accuracy/${MODEL}" \
            --base_path "${BASE_DIR}" \
            --start_date "${START_DATE}" \
            --target_date "${TARGET_DATE}" \
            --metric_name "${METRIC_NAME}" \
            --baseline_type "${BASELINE_TYPE}" \
            --seed "${seed}" \
            --n_jobs "${N_JOBS}" \
            --output_dir "${OUTPUT_DIR}" \
            --model_name "${MODEL_NAME_SAFE}" \
            2>&1 | tee "${OUTPUT_DIR}/exact_shapley_seed${seed}_computation.log"
    fi
done

# 步骤2: MC Shapley
python3 "${SCRIPT_DIR}/compute_mc_shapley.py" \
    --log_dir "accuracy/${MODEL}" \
    --base_path "${BASE_DIR}" \
    --start_date "${START_DATE}" \
    --target_date "${TARGET_DATE}" \
    --metric_name "${METRIC_NAME}" \
    --baseline_type "${BASELINE_TYPE}" \
    --n_samples_list "${N_SAMPLES_LIST[@]}" \
    --seeds "${SEEDS[@]}" \
    --n_threads "${N_JOBS}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_name "${MODEL_NAME_SAFE}" \
    2>&1 | tee "${OUTPUT_DIR}/mc_shapley_computation.log"

# 步骤3: 误差分析
python3 "${SCRIPT_DIR}/analyze_error.py" \
    --output_dir "${OUTPUT_DIR}" \
    --base_path "${BASE_DIR}" \
    --n_samples_list "${N_SAMPLES_LIST[@]}" \
    --seeds "${SEEDS[@]}" \
    --metric_name "${METRIC_NAME}" \
    --baseline_type "${BASELINE_TYPE}" \
    2>&1 | tee "${OUTPUT_DIR}/error_analysis.log"

# 步骤4: 余弦相似度
python3 "${SCRIPT_DIR}/compute_cosine_similarity.py" \
    --output_dir "${OUTPUT_DIR}" \
    --base_path "${BASE_DIR}" \
    --n_samples_list "${N_SAMPLES_LIST[@]}" \
    --seeds "${SEEDS[@]}" \
    --metric_name "${METRIC_NAME}" \
    --baseline_type "${BASELINE_TYPE}" \
    2>&1 | tee "${OUTPUT_DIR}/cosine_similarity_computation.log"

echo ""
echo "✅ 模型 ${MODEL} 处理完成"
echo "结果: ${OUTPUT_DIR}/"
