#!/bin/bash

# 批量计算Shapley值并分析误差
# 对5个模型分别计算exact_shapley和MC_shapley，然后计算误差并汇总

# ==================== 配置参数 ====================
# 项目根目录（脚本目录的上级的上级）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
RESULTS_DIR="${BASE_DIR}/results/accuracy"

# 模型列表（根据需要修改）
MODELS=(
    "gpt-4o"
    "llama-3.1-70b-instruct"
    "anthropic/claude-sonnet-4.5"
    "deepseek-ai/DeepSeek-V3.1"
    # "qwen3-235b-a22b-instruct-2507"
    "qwen-plus"
)

# 交易日范围
START_DATE="2023-06-26"
TARGET_DATE="2023-06-30"  # 最后一个交易日

# MC采样参数
N_SAMPLES_LIST=(10 100 1000 10000)
SEEDS=(42 43 401 45 46 613 48 49 50 51)
EXACT_SEED=42  # 用于计算exact_shapley的种子

# 其他参数
METRIC_NAME="risk_indicator_simple"
BASELINE_TYPE="hold"
N_JOBS=64  # 并行进程数（0表示使用CPU核心数）

# 汇总结果目录（在 BASE_DIR 设置后定义）
AGGREGATE_DIR="${BASE_DIR}/results/shapley_error_aggregate"

# ==================== 检查环境 ====================
echo "=========================================="
echo "批量计算Shapley值并分析误差"
echo "=========================================="
echo "项目根目录: ${BASE_DIR}"
echo "结果目录: ${RESULTS_DIR}"
echo "交易日范围: ${START_DATE} 到 ${TARGET_DATE}"
echo "模型数量: ${#MODELS[@]}"
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

# ==================== 对每个模型进行计算 ====================
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_INDEX=$((i+1))
    
    echo "=========================================="
    echo "处理模型 ${MODEL_INDEX}/${#MODELS[@]}: ${MODEL}"
    echo "=========================================="
    
    # 确定模型目录路径
    MODEL_DIR="${RESULTS_DIR}/${MODEL}"
    
    if [ ! -d "$MODEL_DIR" ]; then
        echo "⚠️  警告: 模型目录不存在: $MODEL_DIR"
        echo "   跳过此模型"
        continue
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
    echo ""
done

# ==================== 汇总所有模型的结果 ====================
echo "=========================================="
echo "汇总所有模型的结果"
echo "=========================================="

mkdir -p "$AGGREGATE_DIR"

# 创建汇总脚本
AGGREGATE_SCRIPT="${SCRIPT_DIR}/aggregate_shapley_errors.py"

# 检查汇总脚本是否存在，如果不存在则创建
if [ ! -f "$AGGREGATE_SCRIPT" ]; then
    echo "创建汇总脚本..."
    cat > "$AGGREGATE_SCRIPT" << 'PYTHON_EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总多个模型的Shapley误差分析结果
"""

import json
import pandas as pd
from pathlib import Path
import sys
import argparse

def aggregate_results(base_dir, models, n_samples_list, seeds):
    """汇总所有模型的结果"""
    results = []
    all_model_results = {}
    
    for model in models:
        # 将斜杠替换为下划线，确保目录名称安全
        model_safe = model.replace('/', '_')
        output_dir = base_dir / "results" / "accuracy" / model / f"{model_safe}_shapley_error_analyse"
        error_json = output_dir / "error_analysis_results.json"
        
        if not error_json.exists():
            print(f"⚠️  警告: 模型 {model} 的误差分析结果不存在: {error_json}")
            continue
        
        with open(error_json, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        all_model_results[model] = model_data
        
        # 提取每个采样次数的统计信息
        for result in model_data['results']:
            n_samples = result['n_samples']
            results.append({
                'model': model,
                'n_samples': n_samples,
                'mean_error': result['mean_error'],
                'std_error': result['std_error'],
                'min_error': result['min_error'],
                'max_error': result['max_error'],
                'num_seeds': len(result['errors'])
            })
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 保存详细结果
    detailed_csv = base_dir / "results" / "shapley_error_aggregate" / "detailed_errors.csv"
    df.to_csv(detailed_csv, index=False)
    print(f"✅ 详细结果已保存到: {detailed_csv}")
    
    # 按采样次数汇总
    summary_data = []
    for n_samples in n_samples_list:
        subset = df[df['n_samples'] == n_samples]
        if len(subset) > 0:
            summary_data.append({
                'n_samples': n_samples,
                'mean_error_across_models': subset['mean_error'].mean(),
                'std_error_across_models': subset['mean_error'].std(),
                'min_error_across_models': subset['mean_error'].min(),
                'max_error_across_models': subset['mean_error'].max(),
                'num_models': len(subset)
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv = base_dir / "results" / "shapley_error_aggregate" / "summary_by_n_samples.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"✅ 汇总结果已保存到: {summary_csv}")
    
    # 按模型汇总
    model_summary_data = []
    for model in models:
        model_subset = df[df['model'] == model]
        if len(model_subset) > 0:
            model_summary_data.append({
                'model': model,
                'mean_error_across_samples': model_subset['mean_error'].mean(),
                'std_error_across_samples': model_subset['mean_error'].std(),
                'min_error_across_samples': model_subset['mean_error'].min(),
                'max_error_across_samples': model_subset['mean_error'].max(),
                'num_n_samples': len(model_subset)
            })
    
    model_summary_df = pd.DataFrame(model_summary_data)
    model_summary_csv = base_dir / "results" / "shapley_error_aggregate" / "summary_by_model.csv"
    model_summary_df.to_csv(model_summary_csv, index=False)
    print(f"✅ 按模型汇总结果已保存到: {model_summary_csv}")
    
    # 保存完整的JSON结果
    aggregate_json = {
        'models': models,
        'n_samples_list': n_samples_list,
        'seeds': seeds,
        'model_results': all_model_results,
        'summary_by_n_samples': summary_data,
        'summary_by_model': model_summary_data
    }
    
    aggregate_json_file = base_dir / "results" / "shapley_error_aggregate" / "aggregate_results.json"
    with open(aggregate_json_file, 'w', encoding='utf-8') as f:
        json.dump(aggregate_json, f, indent=2, ensure_ascii=False)
    print(f"✅ 完整汇总JSON已保存到: {aggregate_json_file}")
    
    # 打印摘要
    print("\n" + "="*60)
    print("汇总结果摘要")
    print("="*60)
    print("\n按采样次数汇总:")
    print(summary_df.to_string(index=False))
    print("\n按模型汇总:")
    print(model_summary_df.to_string(index=False))
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='汇总多个模型的Shapley误差分析结果')
    parser.add_argument('--base_dir', type=str, required=True, help='项目根目录')
    parser.add_argument('--models', type=str, nargs='+', required=True, help='模型列表')
    parser.add_argument('--n_samples_list', type=int, nargs='+', default=[10, 100, 1000, 10000], help='MC采样次数列表')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 401, 45, 46, 613, 48, 49, 50, 51], help='随机种子列表')
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    aggregate_results(base_dir, args.models, args.n_samples_list, args.seeds)
PYTHON_EOF
    chmod +x "$AGGREGATE_SCRIPT"
fi

echo "🚀 开始汇总结果..."
python3 "$AGGREGATE_SCRIPT" \
    --base_dir "${BASE_DIR}" \
    --models "${MODELS[@]}" \
    --n_samples_list "${N_SAMPLES_LIST[@]}" \
    --seeds "${SEEDS[@]}" \
    2>&1 | tee "${AGGREGATE_DIR}/aggregate.log"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ 结果汇总完成"
else
    echo "❌ 结果汇总失败"
fi

echo ""
echo "=========================================="
echo "所有计算完成！"
echo "=========================================="
echo "汇总结果保存在: ${AGGREGATE_DIR}/"
echo "各模型结果保存在: ${RESULTS_DIR}/{model}/{model}_shapley_error_analyse/"
echo "=========================================="

