#!/bin/bash
# 批量计算所有模型的risk features
# 使用方法:
#   bash batch_calculate_risk_features.sh [--max_workers N] [--models model1 model2 ...]
#   或
#   python batch_calculate_risk_features.py --max_workers N --models model1 model2

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_ROOT="$PROJECT_ROOT/results"
SCRIPT_PATH="$SCRIPT_DIR/calculate_risk_features.py"

# 检查脚本是否存在
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: calculate_risk_features.py not found at $SCRIPT_PATH"
    exit 1
fi

# 默认参数
MAX_WORKERS=1
MODELS=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --max_workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        --models)
            shift
            MODELS="$@"
            break
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--max_workers N] [--models model1 model2 ...]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "批量计算Risk Features"
echo "=========================================="
echo "Results root: $RESULTS_ROOT"
echo "Max workers: $MAX_WORKERS"
echo "Models filter: ${MODELS:-all}"
echo "=========================================="
echo ""

# 如果指定了max_workers且大于1，使用Python并行脚本
if [ "$MAX_WORKERS" -gt 1 ]; then
    echo "Using Python parallel script (max_workers > 1)..."
    python3 "$SCRIPT_DIR/batch_calculate_risk_features.py" \
        --results_root "$RESULTS_ROOT" \
        --max_workers "$MAX_WORKERS" \
        ${MODELS:+--models} $MODELS
else
    echo "Using sequential bash script..."
    # 遍历所有模型目录
    for model_dir in "$RESULTS_ROOT"/*/; do
        if [ ! -d "$model_dir" ]; then
            continue
        fi
        
        model_name=$(basename "$model_dir")
        
        # 如果指定了模型过滤，跳过不在列表中的
        if [ -n "$MODELS" ]; then
            skip=true
            for filter_model in $MODELS; do
                if [[ "$model_name" == *"$filter_model"* ]]; then
                    skip=false
                    break
                fi
            done
            if [ "$skip" = true ]; then
                continue
            fi
        fi
        
        # 遍历该模型下的所有子目录（如 model_42, model_43等）
        for seed_dir in "$model_dir"*/; do
            if [ ! -d "$seed_dir" ]; then
                continue
            fi
            
            seed_name=$(basename "$seed_dir")
            shapley_dir="$seed_dir/shapley"
            
            # 检查是否存在shapley目录和文件
            if [ ! -d "$shapley_dir" ]; then
                continue
            fi
            
            shapley_matrix=$(find "$shapley_dir" -name "shapley_matrix_*.npy" | head -1)
            if [ -z "$shapley_matrix" ]; then
                continue
            fi
            
            # 设置输出目录
            output_base="$PROJECT_ROOT/results/risk_feature/$model_name/$seed_name"
            
            echo "Processing: $model_name/$seed_name"
            
            # 运行计算脚本
            python3 "$SCRIPT_PATH" \
                --results_dir "$seed_dir" \
                --output_dir "$output_base" || {
                echo "  ERROR: Failed to process $model_name/$seed_name"
                continue
            }
            
            echo "  SUCCESS: $model_name/$seed_name"
            echo ""
        done
    done
fi

echo "=========================================="
echo "批量计算完成！"
echo "=========================================="
