#!/bin/bash

# 计算qwen-plus模型的Shapley值并分析误差
# 这是从batch_compute_shapley.sh提取的单模型版本，可以并行运行

# ==================== 配置参数 ====================
# 项目根目录
BASE_DIR="/mnt/shared-storage-user/meijilin/Economic_System_Attribution/TwinMarket"
SCRIPT_DIR="${BASE_DIR}/scripts/accuracy_exp"
RESULTS_DIR="${BASE_DIR}/results/accuracy"

# 模型名称
MODEL="qwen-plus"

# 交易日范围
START_DATE="2023-06-26"
TARGET_DATE="2023-06-30"  # 最后一个交易日

# MC采样参数
N_SAMPLES_LIST=(10 100 1000 10000)
SEEDS=(42 43 401 45 46 613 48 49 50 51)

# 其他参数
METRIC_NAME="risk_indicator_simple"
BASELINE_TYPE="hold"
N_JOBS=64  # 并行进程数（0表示使用CPU核心数）

# ==================== 检查环境 ====================
echo "=========================================="
echo "计算模型 ${MODEL} 的Shapley值并分析误差"
echo "=========================================="
echo "项目根目录: ${BASE_DIR}"
echo "结果目录: ${RESULTS_DIR}"
echo "交易日范围: ${START_DATE} 到 ${TARGET_DATE}"
echo "MC采样次数: ${N_SAMPLES_LIST[@]}"
echo "种子数量: ${#SEEDS[@]}"
echo "=========================================="
echo ""

# 检查脚本目录
if [ ! -d "$SCRIPT_DIR" ]; then
    echo "❌ 脚本目录不存在: $SCRIPT_DIR"
    exit 1
fi

# 检查结果目录
if [ ! -d "$RESULTS_DIR" ]; then
    echo "❌ 结果目录不存在: $RESULTS_DIR"
    exit 1
fi

# 检查Python脚本是否存在
if [ ! -f "${SCRIPT_DIR}/compute_exact_shapley.py" ]; then
    echo "❌ compute_exact_shapley.py 不存在"
    exit 1
fi

if [ ! -f "${SCRIPT_DIR}/compute_mc_shapley.py" ]; then
    echo "❌ compute_mc_shapley.py 不存在"
    exit 1
fi

if [ ! -f "${SCRIPT_DIR}/analyze_error.py" ]; then
    echo "❌ analyze_error.py 不存在"
    exit 1
fi

if [ ! -f "${SCRIPT_DIR}/compute_cosine_similarity.py" ]; then
    echo "❌ compute_cosine_similarity.py 不存在"
    exit 1
fi

# ==================== 处理模型 ====================
echo "=========================================="
echo "处理模型: ${MODEL}"
echo "=========================================="

# 确定模型目录路径
MODEL_DIR="${RESULTS_DIR}/${MODEL}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "⚠️  警告: 模型目录不存在: $MODEL_DIR"
    echo "   退出"
    exit 1
fi

# 确定输出目录（将斜杠替换为下划线，确保目录名称安全）
MODEL_NAME_SAFE=$(echo "${MODEL}" | sed 's/\//_/g')
OUTPUT_DIR="${RESULTS_DIR}/${MODEL}/${MODEL_NAME_SAFE}_shapley_error_analyse"
mkdir -p "$OUTPUT_DIR"

echo "模型目录: ${MODEL_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
echo ""

# 步骤1: 计算Exact Shapley值（为每个seed计算一次，确保exact和MC使用相同的基准）
echo "----------------------------------------"
echo "步骤1: 计算Exact Shapley值（每个seed一次）"
echo "----------------------------------------"
echo "  种子数: ${#SEEDS[@]}"
echo "  将计算 ${#SEEDS[@]} 个Exact Shapley值"
echo ""

EXACT_COMPUTATION_FAILED=false
for seed in "${SEEDS[@]}"; do
    EXACT_OUTPUT_FILE="${OUTPUT_DIR}/exact_shapley_seed${seed}.npy"
    
    if [ -f "$EXACT_OUTPUT_FILE" ]; then
        echo "✅ Exact Shapley值已存在（seed=${seed}），跳过计算: $EXACT_OUTPUT_FILE"
    else
        echo "🚀 开始计算Exact Shapley值（seed=${seed}）..."
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
        
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "✅ Exact Shapley值计算完成（seed=${seed}）"
        else
            echo "❌ Exact Shapley值计算失败（seed=${seed}）"
            EXACT_COMPUTATION_FAILED=true
        fi
    fi
    echo ""
done

if [ "$EXACT_COMPUTATION_FAILED" = true ]; then
    echo "⚠️  警告: 部分Exact Shapley值计算失败，但继续后续计算"
fi

echo ""

# 步骤2: 计算MC Shapley值（10个seed × 4个采样次数 = 40次计算）
echo "----------------------------------------"
echo "步骤2: 计算MC Shapley值"
echo "----------------------------------------"
echo "  采样次数: ${N_SAMPLES_LIST[@]}"
echo "  种子数: ${#SEEDS[@]}"
echo "  总计算次数: $((${#N_SAMPLES_LIST[@]} * ${#SEEDS[@]}))"
echo ""

# 检查是否所有MC结果都已存在
ALL_MC_EXIST=true
for n_samples in "${N_SAMPLES_LIST[@]}"; do
    for seed in "${SEEDS[@]}"; do
        MC_FILE="${OUTPUT_DIR}/mc_shapley_n${n_samples}_seed${seed}.npy"
        if [ ! -f "$MC_FILE" ]; then
            ALL_MC_EXIST=false
            break 2
        fi
    done
done

if [ "$ALL_MC_EXIST" = true ]; then
    echo "✅ 所有MC Shapley值已存在，跳过计算"
else
    echo "🚀 开始计算MC Shapley值..."
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
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✅ MC Shapley值计算完成"
    else
        echo "❌ MC Shapley值计算失败"
        echo "   继续计算误差（使用已存在的结果）"
    fi
fi

echo ""

# 步骤3: 计算误差
echo "----------------------------------------"
echo "步骤3: 计算误差"
echo "----------------------------------------"

ERROR_JSON="${OUTPUT_DIR}/error_analysis_results.json"

if [ -f "$ERROR_JSON" ]; then
    echo "✅ 误差分析结果已存在，跳过计算: $ERROR_JSON"
else
    echo "🚀 开始计算误差..."
    python3 "${SCRIPT_DIR}/analyze_error.py" \
        --output_dir "${OUTPUT_DIR}" \
        --base_path "${BASE_DIR}" \
        --n_samples_list "${N_SAMPLES_LIST[@]}" \
        --seeds "${SEEDS[@]}" \
        --metric_name "${METRIC_NAME}" \
        --baseline_type "${BASELINE_TYPE}" \
        2>&1 | tee "${OUTPUT_DIR}/error_analysis.log"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✅ 误差计算完成"
    else
        echo "❌ 误差计算失败"
    fi
fi

echo ""

# 步骤4: 计算余弦相似度
echo "----------------------------------------"
echo "步骤4: 计算余弦相似度"
echo "----------------------------------------"

COSINE_JSON="${OUTPUT_DIR}/cosine_similarity_results.json"

if [ -f "$COSINE_JSON" ]; then
    echo "✅ 余弦相似度分析结果已存在，跳过计算: $COSINE_JSON"
else
    echo "🚀 开始计算余弦相似度..."
    python3 "${SCRIPT_DIR}/compute_cosine_similarity.py" \
        --output_dir "${OUTPUT_DIR}" \
        --base_path "${BASE_DIR}" \
        --n_samples_list "${N_SAMPLES_LIST[@]}" \
        --seeds "${SEEDS[@]}" \
        --metric_name "${METRIC_NAME}" \
        --baseline_type "${BASELINE_TYPE}" \
        2>&1 | tee "${OUTPUT_DIR}/cosine_similarity_computation.log"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✅ 余弦相似度计算完成"
    else
        echo "❌ 余弦相似度计算失败"
    fi
fi

echo ""
echo "✅ 模型 ${MODEL} 处理完成"
echo "结果保存在: ${OUTPUT_DIR}/"
echo "=========================================="
