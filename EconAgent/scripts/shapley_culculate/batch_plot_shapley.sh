#!/bin/bash
# 批量生成Shapley值可视化图表
# 该脚本会调用 plot_shapley.py 来为多个数据集生成可视化图表

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_SCRIPT="$PROJECT_DIR/scripts/plot/plot_shapley.py"

# ==================== 配置参数（可修改） ====================
# 数据根目录
DATA_DIR="$PROJECT_DIR/datas"

# 数据集列表（可以根据需要修改）
# GPT模型数据集
GPT_DATASETS=(
    "gpt-42"
    "gpt-45"
    "gpt-47"
    "gpt-48"
    "gpt-50"
    "gpt-52"
    "gpt-54"
)

# Claude模型数据集
CLAUDE_DATASETS=(
    "claude-42"
    "claude-43"
    "claude-46"
    "claude-48"
    "claude-49"
    "claude-50"
    "claude-52"
)

# LLaMA模型数据集
LLAMA_DATASETS=(
    "llama-42"
    "llama-44"
    "llama-46"
    "llama-48"
    "llama-50"
    "llama-52"
)

QWEN_DATASETS=( "qwen-42" "qwen-44" "qwen-46" "qwen-47" "qwen-48" "qwen-50" "qwen-52" )
DS_DATASETS=( "ds-42" "ds-46" "ds-52" "ds-51" "ds-48" )

# 合并所有数据集
ALL_DATASETS=(
    "${GPT_DATASETS[@]}"
    "${CLAUDE_DATASETS[@]}"
    "${LLAMA_DATASETS[@]}"
    "${QWEN_DATASETS[@]}"
    "${DS_DATASETS[@]}"
)

# 绘图选项（默认全部启用）
PLOT_HEATMAP=true
PLOT_CUMULATIVE=true
PLOT_AGENT_TOTAL=true
PLOT_BASELINE=true

# Baseline相关参数
BASELINE_VARIABLE="price_inflation_rate"
BASELINE_THRESHOLD=""

# ==================== 命令行参数解析 ====================
show_help() {
    cat << EOF
用法: $0 [选项]

批量生成Shapley值可视化图表

选项:
    -d, --data-dir DIR          数据根目录路径 (默认: $DATA_DIR)
    -a, --all                    处理所有数据集（GPT + Claude + LLaMA + Qwen + DS）
    --gpt-only                   只处理GPT数据集
    --claude-only                只处理Claude数据集
    --llama-only                 只处理LLaMA数据集
    --qwen-only                  只处理Qwen数据集
    --ds-only                    只处理DS数据集
    --datasets DATASET1 DATASET2 ...  指定要处理的数据集列表
    --plot-heatmap BOOL          是否生成热力图 (默认: true)
    --plot-cumulative BOOL       是否生成累计风险图 (默认: true)
    --plot-agent-total BOOL      是否生成agent总归因图 (默认: true)
    --plot-baseline BOOL         是否生成baseline指标图 (默认: true)
    --baseline-variable VAR      要绘制的baseline变量名 (默认: price_inflation_rate)
    --baseline-threshold VAL     baseline阈值（可选）
    -h, --help                   显示此帮助信息

示例:
    # 处理所有数据集，生成所有图表
    $0 --all

    # 只处理GPT数据集
    $0 --gpt-only

    # 处理指定数据集，只生成热力图
    $0 --datasets gpt-42 gpt-45 --plot-cumulative false --plot-agent-total false --plot-baseline false

    # 处理所有数据集，自定义baseline变量
    $0 --all --baseline-variable price_inflation_rate --baseline-threshold 0.1

EOF
}

# 默认参数
USE_ALL=false
USE_GPT_ONLY=false
USE_CLAUDE_ONLY=false
USE_LLAMA_ONLY=false
USE_QWEN_ONLY=false
USE_DS_ONLY=false
CUSTOM_DATASETS=()

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -a|--all)
            USE_ALL=true
            shift
            ;;
        --gpt-only)
            USE_GPT_ONLY=true
            shift
            ;;
        --claude-only)
            USE_CLAUDE_ONLY=true
            shift
            ;;
        --llama-only)
            USE_LLAMA_ONLY=true
            shift
            ;;
        --qwen-only)
            USE_QWEN_ONLY=true
            shift
            ;;
        --ds-only)
            USE_DS_ONLY=true
            shift
            ;;
        --datasets)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
                CUSTOM_DATASETS+=("$1")
                shift
            done
            ;;
        --plot-heatmap)
            if [[ "$2" =~ ^(true|false|True|False|1|0)$ ]]; then
                PLOT_HEATMAP="$2"
                shift 2
            else
                PLOT_HEATMAP=true
                shift
            fi
            ;;
        --plot-cumulative)
            if [[ "$2" =~ ^(true|false|True|False|1|0)$ ]]; then
                PLOT_CUMULATIVE="$2"
                shift 2
            else
                PLOT_CUMULATIVE=true
                shift
            fi
            ;;
        --plot-agent-total)
            if [[ "$2" =~ ^(true|false|True|False|1|0)$ ]]; then
                PLOT_AGENT_TOTAL="$2"
                shift 2
            else
                PLOT_AGENT_TOTAL=true
                shift
            fi
            ;;
        --plot-baseline)
            if [[ "$2" =~ ^(true|false|True|False|1|0)$ ]]; then
                PLOT_BASELINE="$2"
                shift 2
            else
                PLOT_BASELINE=true
                shift
            fi
            ;;
        --baseline-variable)
            BASELINE_VARIABLE="$2"
            shift 2
            ;;
        --baseline-threshold)
            BASELINE_THRESHOLD="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "错误: 未知参数 '$1'"
            show_help
            exit 1
            ;;
    esac
done

# ==================== 确定要处理的数据集 ====================
if [[ ${#CUSTOM_DATASETS[@]} -gt 0 ]]; then
    # 使用自定义数据集列表
    DATASETS_TO_PROCESS=("${CUSTOM_DATASETS[@]}")
elif [[ "$USE_ALL" == true ]]; then
    # 处理所有数据集
    DATASETS_TO_PROCESS=("${ALL_DATASETS[@]}")
elif [[ "$USE_GPT_ONLY" == true ]]; then
    # 只处理GPT数据集
    DATASETS_TO_PROCESS=("${GPT_DATASETS[@]}")
elif [[ "$USE_CLAUDE_ONLY" == true ]]; then
    # 只处理Claude数据集
    DATASETS_TO_PROCESS=("${CLAUDE_DATASETS[@]}")
elif [[ "$USE_LLAMA_ONLY" == true ]]; then
    # 只处理LLaMA数据集
    DATASETS_TO_PROCESS=("${LLAMA_DATASETS[@]}")
elif [[ "$USE_QWEN_ONLY" == true ]]; then
    # 只处理Qwen数据集
    DATASETS_TO_PROCESS=("${QWEN_DATASETS[@]}")
elif [[ "$USE_DS_ONLY" == true ]]; then
    # 只处理DS数据集
    DATASETS_TO_PROCESS=("${DS_DATASETS[@]}")
else
    # 默认处理所有数据集
    DATASETS_TO_PROCESS=("${ALL_DATASETS[@]}")
fi

# ==================== 检查Python脚本是否存在 ====================
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "错误: Python脚本不存在: $PYTHON_SCRIPT"
    exit 1
fi

# ==================== 检查数据目录是否存在 ====================
if [[ ! -d "$DATA_DIR" ]]; then
    echo "错误: 数据目录不存在: $DATA_DIR"
    exit 1
fi

# ==================== 显示配置信息 ====================
echo "=========================================="
echo "批量生成Shapley值可视化图表"
echo "=========================================="
echo "数据目录: $DATA_DIR"
echo "数据集数量: ${#DATASETS_TO_PROCESS[@]}"
echo "数据集列表:"
for dataset in "${DATASETS_TO_PROCESS[@]}"; do
    echo "  - $dataset"
done
echo ""
echo "绘图选项:"
echo "  热力图: $PLOT_HEATMAP"
echo "  累计风险图: $PLOT_CUMULATIVE"
echo "  Agent总归因图: $PLOT_AGENT_TOTAL"
echo "  Baseline指标图: $PLOT_BASELINE"
if [[ "$PLOT_BASELINE" == "true" ]]; then
    echo "  Baseline变量: $BASELINE_VARIABLE"
    if [[ -n "$BASELINE_THRESHOLD" ]]; then
        echo "  Baseline阈值: $BASELINE_THRESHOLD"
    fi
fi
echo "=========================================="
echo ""

# ==================== 处理每个数据集 ====================
cd "$PROJECT_DIR" || exit 1

SUCCESS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

for i in "${!DATASETS_TO_PROCESS[@]}"; do
    dataset="${DATASETS_TO_PROCESS[$i]}"
    dataset_num=$((i + 1))
    total_datasets=${#DATASETS_TO_PROCESS[@]}
    
    echo "[$dataset_num/$total_datasets] 处理: $dataset"
    
    # 检查数据集目录是否存在
    dataset_dir="$DATA_DIR/$dataset"
    if [[ ! -d "$dataset_dir" ]]; then
        echo "  ✗ 数据集目录不存在: $dataset_dir"
        ((SKIP_COUNT++))
        echo ""
        continue
    fi
    
    # 检查shapley目录和文件是否存在
    shapley_dir="$dataset_dir/shapley"
    shapley_values_file="$shapley_dir/shapley_values.npy"
    stats_json_file="$shapley_dir/shapley_stats.json"
    
    if [[ ! -d "$shapley_dir" ]]; then
        echo "  ✗ Shapley目录不存在: $shapley_dir"
        ((SKIP_COUNT++))
        echo ""
        continue
    fi
    
    if [[ ! -f "$shapley_values_file" ]]; then
        echo "  ✗ Shapley值文件不存在: $shapley_values_file"
        ((SKIP_COUNT++))
        echo ""
        continue
    fi
    
    if [[ ! -f "$stats_json_file" ]]; then
        echo "  ✗ 统计文件不存在: $stats_json_file"
        ((SKIP_COUNT++))
        echo ""
        continue
    fi
    
    # 构建Python命令
    PYTHON_CMD=(
        python "$PYTHON_SCRIPT"
        --shapley_values "$shapley_values_file"
        --stats_json "$stats_json_file"
        --plot_heatmap "$PLOT_HEATMAP"
        --plot_cumulative "$PLOT_CUMULATIVE"
        --plot_agent_total "$PLOT_AGENT_TOTAL"
        --plot_baseline "$PLOT_BASELINE"
        --baseline_variable "$BASELINE_VARIABLE"
    )
    
    # 添加baseline阈值（如果提供）
    if [[ -n "$BASELINE_THRESHOLD" ]]; then
        PYTHON_CMD+=("--baseline_threshold" "$BASELINE_THRESHOLD")
    fi
    
    # 执行命令
    if "${PYTHON_CMD[@]}" > /dev/null 2>&1; then
        echo "  ✓ 完成 $dataset"
        ((SUCCESS_COUNT++))
    else
        echo "  ✗ 失败 $dataset"
        ((FAIL_COUNT++))
        # 显示错误信息（可选，取消注释下面一行以查看详细错误）
        # "${PYTHON_CMD[@]}"
    fi
    echo ""
done

# ==================== 显示汇总信息 ====================
echo "=========================================="
echo "批量处理完成"
echo "=========================================="
echo "成功: $SUCCESS_COUNT"
echo "失败: $FAIL_COUNT"
echo "跳过: $SKIP_COUNT"
echo "总计: ${#DATASETS_TO_PROCESS[@]}"
echo "=========================================="

# 返回适当的退出码
if [[ $FAIL_COUNT -gt 0 ]]; then
    exit 1
else
    exit 0
fi

